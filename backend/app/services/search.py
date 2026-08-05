"""Lexical code search — one engine, ripgrep.

GrepRAG's result is the reason this module exists as a first-class layer rather
than a helper: lightweight lexical search matches far heavier graph/RAG
baselines on retrieval quality, provided the matching is identifier-aware and
the results are deduplicated. It is also the only channel that sees what an AST
index cannot model — string literals, comments, config, templates, generated
code — so it is the refinement pass at the end of retrieval, not a fallback.

Everything the pipeline searches goes through here:

  * ``lines``       — file:line:text hits (the Dev/Planner/jury `grep` tool)
  * ``files``       — which files match (test localization, existence checks)
  * ``definitions`` — where a symbol is DEFINED, language-aware
  * ``mentions``    — fixed-string existence check
  * ``count_files`` — how many files match, without listing them

Two engines, one contract. ripgrep is the engine; when the binary is missing,
a private ``git grep`` implementation answers the same calls so an install
without ripgrep degrades in fidelity, never in function (``probe()`` reports
which one is live).

Searching a COMMITTED tree (e.g. origin/main at scoping time, so unmerged code
on a stale agent branch never looks real) is not a flag here: ripgrep searches
directories, so the caller passes the path of a ref-pinned checkout instead —
see ``git_ops.ref_worktree``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

from ..config import settings
from . import lang

logger = logging.getLogger(__name__)

_TIMEOUT = 60

# Result caps that apply before anything reaches a prompt.
_MAX_SCAN_LINES = 4000


def binary() -> str | None:
    """Absolute path of the ripgrep binary, or None when unavailable/disabled."""
    if not settings.ripgrep_enabled:
        return None
    configured = (settings.ripgrep_path or "").strip() or "rg"
    return shutil.which(configured) or (configured if Path(configured).is_file() else None)


def available() -> bool:
    return binary() is not None


def probe() -> dict:
    """Prove the search engine runs, and say which one it is. {ok, output} —
    surfaced by the Settings 'Test code search' row."""
    exe = binary()
    if exe is None:
        reason = ("ripgrep is disabled (settings.ripgrep_enabled=false)"
                  if not settings.ripgrep_enabled else
                  f"ripgrep binary '{settings.ripgrep_path}' not found on PATH")
        return {"ok": True, "output":
                f"{reason}.\nFalling back to `git grep` — searches still work over "
                "tracked files; definition searches retain type filters, but the "
                "fallback has no .gitignore-aware walk. "
                "Install ripgrep (`apt install ripgrep`, `brew install ripgrep`, "
                "`cargo install ripgrep`) for the full engine."}
    try:
        proc = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15)
        ver = (proc.stdout or proc.stderr or "").strip().splitlines()
        return {"ok": proc.returncode == 0,
                "output": f"{exe}\n{ver[0] if ver else '(no version output)'}"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "output": f"{exe}\nfailed to run: {exc}"}


# --- Pathspec → glob ----------------------------------------------------------

def _globs(pathspec: str | None) -> list[str]:
    """Translate a git pathspec to ripgrep globs.

    Callers historically passed git pathspecs, where `*` crosses directory
    boundaries ('*test*' means 'any path containing test'). ripgrep globs are
    path-segment-scoped by default, so a bare '*test*' would match only in the
    top directory — silently losing every `tests/foo/test_bar.py` a caller was
    looking for. Anchoring with '**/' restores the git meaning.
    """
    spec = (pathspec or "").strip()
    if not spec or spec == ".":
        return []
    if spec.startswith(("*", "!")) or "/" not in spec:
        return [f"**/{spec.lstrip('/')}"]
    return [spec.lstrip("/")]


def _base_args(exe: str, *, ignore_case: bool, fixed: bool,
               globs: list[str], types: list[str]) -> list[str]:
    """Flags shared by every ripgrep call.

    ``--sort=path`` is not cosmetic: callers pin work to ``definitions()[0]``,
    and ripgrep's default parallel walk returns files in whatever order threads
    finish. A localization that changes between two identical runs is not a
    localization. Sorting costs the parallel walk; the result sets here are
    capped and small.

    ``--hidden`` with an explicit ``!.git/`` exclusion matches what git grep
    saw: tracked dot-directories like `.github/` are real source, ripgrep's
    default hidden-file skip would drop them.
    """
    args = [exe, "--no-messages", "--color", "never", "--sort", "path", "--hidden",
            "-g", "!.git/"]
    if ignore_case:
        args.append("-i")
    if fixed:
        args.append("-F")
    for g in globs:
        args += ["-g", g]
    for t in types:
        args += ["-t", t]
    return args


def _run(args: list[str], cwd: str) -> str:
    """Run a search process. Exit 1 means 'no matches' and exit 2 can mean
    'some paths were unreadable' — neither is an error worth raising, so this
    returns stdout for any exit code and "" only when the process could not run
    at all. errors='replace': a latin-1 fixture must not crash a scoping turn."""
    try:
        proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                              errors="replace", timeout=_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("search: %s failed to run: %s", args[0], exc)
        return ""
    return proc.stdout


# --- ripgrep JSON parsing -----------------------------------------------------

def _text_of(field: dict | None) -> str:
    """ripgrep emits {"text": ...} for UTF-8 and {"bytes": base64} otherwise."""
    if not isinstance(field, dict):
        return ""
    if "text" in field:
        return str(field["text"])
    raw = field.get("bytes")
    if raw:
        import base64

        try:
            return base64.b64decode(raw).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            return ""
    return ""


def _parse_json_matches(out: str, max_lines: int) -> list[str]:
    """ripgrep --json → 'path:line:text' rows, in the engine's sorted order."""
    rows: list[str] = []
    for i, raw in enumerate(out.splitlines()):
        if i > _MAX_SCAN_LINES or len(rows) >= max_lines:
            break
        if not raw.startswith('{"type":"match"'):
            continue
        try:
            data = json.loads(raw).get("data") or {}
        except json.JSONDecodeError:
            continue
        path = _text_of(data.get("path"))
        text = _text_of(data.get("lines")).rstrip("\n")
        line = data.get("line_number")
        if path and line:
            rows.append(f"{path}:{line}:{text}")
    return rows


def _split_null(out: str, limit: int) -> list[str]:
    """`--null`-separated path list. Paths can legally contain newlines, so the
    NUL separator is the only safe split."""
    return [p for p in out.split("\0") if p.strip()][:limit]


# --- Public API ---------------------------------------------------------------

def lines(root: str, pattern: str, *, max_lines: int = 40, pathspec: str = ".",
          ignore_case: bool = False, fixed: bool = False) -> str:
    """Matching lines as 'file:line:text', capped. Returns '(no matches)' when
    nothing matched — an invalid regex is a no-match, not an exception, because
    the caller is usually a model that guessed at a pattern."""
    if not (root and pattern):
        return "(no matches)"
    exe = binary()
    if exe is None:
        rows = _git_grep(root, "-nI" + ("i" if ignore_case else "") + ("F" if fixed else "E"),
                         pattern, pathspec)
    else:
        args = _base_args(exe, ignore_case=ignore_case, fixed=fixed,
                          globs=_globs(pathspec), types=[])
        args += ["--json", "--max-count", str(max(1, max_lines)), "--", pattern]
        rows = _parse_json_matches(_run(args, root), max_lines + 1)
    body = "\n".join(rows[:max_lines])
    if len(rows) > max_lines:
        body += f"\n… ({len(rows) - max_lines} more matches)"
    return body or "(no matches)"


def files(root: str, pattern: str, *, pathspec: str = ".", max_files: int = 5,
          ignore_case: bool = False, fixed: bool = False,
          types: list[str] | None = None) -> list[str]:
    """Paths of files containing a match, sorted, capped."""
    if not (root and pattern):
        return []
    exe = binary()
    if exe is None:
        flags = "-l" + ("i" if ignore_case else "") + ("F" if fixed else "E")
        return _git_grep(root, flags, pattern, pathspec, types=types)[:max_files]
    args = _base_args(exe, ignore_case=ignore_case, fixed=fixed,
                      globs=_globs(pathspec), types=types or [])
    args += ["--files-with-matches", "--null", "--", pattern]
    return _split_null(_run(args, root), max_files)


def count_files(root: str, pattern: str, *, ignore_case: bool = False,
                fixed: bool = False, limit: int = 200) -> int:
    """How many files match — the cheap way to answer 'is this term too common
    to be a useful pin?' without materializing the file list."""
    return len(files(root, pattern, max_files=limit, ignore_case=ignore_case, fixed=fixed))


def mentions(root: str, text: str, *, max_files: int = 3) -> list[str]:
    """Files merely CONTAINING `text` (fixed string) — an existence check for a
    symbol or flag, without assuming it is defined there."""
    return files(root, text, max_files=max_files, fixed=True)


# --- Definition search --------------------------------------------------------

# Per-language definition patterns. The generic pattern this replaces
# (`(def|class|function|const|var|let)\s+NAME`) reported a Go method as a
# JavaScript const and missed Rust `impl` blocks, Ruby `def self.`, and every
# TypeScript `interface`/`type`. Matching the language's actual declaration
# syntax — and restricting the search to that language's files with ripgrep's
# type filters — is both more precise and considerably faster.
_DEFINITION_PATTERNS: dict[str, tuple[str, tuple[str, ...]]] = {
    # language: (regex template with {n} for the escaped name, rg --type names)
    "Python": (r"^\s*(?:async\s+)?(?:def|class)\s+{n}\b|^\s*{n}\s*(?::[^=]+)?=", ("py",)),
    "JavaScript": (r"(?:function\*?|class)\s+{n}\b|(?:const|let|var)\s+{n}\s*[=:]"
                   r"|^\s*{n}\s*\([^)]*\)\s*\{{|^\s*{n}\s*[:=]\s*(?:async\s*)?\(", ("js",)),
    "TypeScript": (r"(?:function\*?|class|interface|type|enum)\s+{n}\b"
                   r"|(?:const|let|var)\s+{n}\s*[=:]|^\s*(?:public|private|protected|readonly|abstract|static|\s)*"
                   r"{n}\s*[(<:]", ("ts",)),
    "Go": (r"^func\s+(?:\([^)]*\)\s*)?{n}\b|^(?:type|var|const)\s+{n}\b", ("go",)),
    "Rust": (r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
             r"(?:fn|struct|enum|trait|type|const|static|mod)\s+{n}\b|^\s*impl\b.*\b{n}\b", ("rust",)),
    "Java": (r"^\s*(?:public|private|protected|static|final|abstract|synchronized|\s)*"
             r"(?:class|interface|enum|record)\s+{n}\b|^\s*(?:public|private|protected|static|final|\s)+"
             r"[\w<>\[\],.?\s]+\s+{n}\s*\(", ("java",)),
    "Ruby": (r"^\s*(?:def\s+(?:self\.)?{n}\b|class\s+{n}\b|module\s+{n}\b|{n}\s*=)", ("ruby",)),
    "PHP": (r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+{n}\b"
            r"|function\s+{n}\s*\(", ("php",)),
    "C": (r"^[\w\s*]*\b{n}\s*\([^;]*$|^\s*(?:struct|enum|union|typedef)\s+.*\b{n}\b", ("c", "cpp")),
    "C++": (r"^[\w\s*:&<>]*\b{n}\s*\([^;]*$|^\s*(?:class|struct|enum|namespace|using)\s+{n}\b",
            ("cpp",)),
    "C#": (r"^\s*(?:public|private|protected|internal|static|sealed|abstract|partial|\s)*"
           r"(?:class|interface|struct|enum|record)\s+{n}\b|\b{n}\s*\([^)]*\)\s*\{{", ("csharp",)),
}

# Used when the repo's language is unknown or has no entry above: the union of
# the common declaration keywords, unrestricted by file type.
_GENERIC_DEFINITION = (r"\b(?:def|class|function|func|fn|interface|type|struct|enum|trait|module"
                       r"|const|var|let|val|public|private)\s+{n}\b|^\s*{n}\s*[:=]")


def definitions(root: str, name: str, *, max_files: int = 3,
                hint_path: str = "") -> list[str]:
    """Files that DEFINE `name`, language-aware. Empty when the symbol is
    defined nowhere — i.e. it is new, or the caller invented it.

    `hint_path` (a file the symbol is believed to live in) picks the language
    to search; without it every configured language pattern is tried and the
    results merged, so a polyglot repo still resolves.
    """
    if not (root and name):
        return []
    esc = re.escape(name)
    found: list[str] = []
    langs = [lang.language_of(hint_path)] if hint_path else list(_DEFINITION_PATTERNS)
    for language in langs:
        entry = _DEFINITION_PATTERNS.get(language)
        if entry is None:
            continue
        pattern, types = entry
        # Types are passed to BOTH engines: ripgrep filters with --type, the
        # fallback with equivalent pathspecs. Handing them only to ripgrep is
        # what let the C pattern match Python files on a box without it.
        for f in files(root, pattern.format(n=esc), max_files=max_files,
                       types=list(types)):
            if f not in found:
                found.append(f)
        if len(found) >= max_files:
            return found[:max_files]
    # Nothing from the typed passes (unknown language, or a definition form the
    # per-language pattern doesn't cover) — fall back to the keyword union.
    if not found:
        found = files(root, _GENERIC_DEFINITION.format(n=esc), max_files=max_files)
    return found[:max_files]


# --- Fallback engine: git grep ------------------------------------------------

# ripgrep --type name → the globs git needs to mean the same thing. Only the
# types the definition patterns actually ask for; an unmapped type simply doesn't
# narrow the search, which is the safe direction.
_TYPE_GLOBS: dict[str, tuple[str, ...]] = {
    "py": ("*.py", "*.pyi"),
    "js": ("*.js", "*.jsx", "*.mjs", "*.cjs"),
    "ts": ("*.ts", "*.tsx", "*.mts", "*.cts"),
    "go": ("*.go",),
    "rust": ("*.rs",),
    "java": ("*.java",),
    "ruby": ("*.rb", "*.rake", "Rakefile", "Gemfile"),
    "php": ("*.php", "*.phtml"),
    "c": ("*.c", "*.h"),
    "cpp": ("*.cc", "*.cpp", "*.cxx", "*.hpp", "*.hh", "*.hxx"),
    "csharp": ("*.cs",),
}


def _type_pathspecs(types: list[str] | None) -> list[str]:
    """git pathspecs equivalent to a set of ripgrep `--type` names."""
    globs: list[str] = []
    for t in types or []:
        for glob in _TYPE_GLOBS.get(t, ()):
            if glob not in globs:
                globs.append(f"*/{glob}" if not glob.startswith("*") else glob)
                globs.append(glob)
    # git matches a bare '*.py' against the full path, so both the plain and the
    # */-prefixed form are needed to catch top-level and nested files alike.
    return list(dict.fromkeys(globs))


def _git_grep(root: str, flags: str, pattern: str, pathspec: str,
              *, types: list[str] | None = None) -> list[str]:
    """The pre-ripgrep engine, kept as the degraded tier.

    Only tracked files and no .gitignore semantics — but it is present wherever
    git is, which is everywhere this pipeline runs. Language narrowing *is*
    honoured, via pathspecs: without it the C pattern is applied to Python files
    and `definitions()` answers with whatever over-matches, so the two tiers
    would disagree on the one call the Planner pins work to.
    """
    # git cannot intersect two pathspecs, so an explicit one from the caller wins
    # over the language narrowing. No caller supplies both today.
    explicit = pathspec not in ("", ".")
    limits = [pathspec] if explicit else (_type_pathspecs(types) or ["."])

    if "F" not in flags and _supports_pcre(root):
        # Preferred: PCRE understands the same syntax ripgrep does, so the two
        # tiers agree exactly rather than approximately.
        args = ["git", "grep", flags.replace("E", "P"), pattern, "--", *limits]
        out = _run(args, root)
        if out.strip():
            return [l for l in out.splitlines() if l.strip()]
        # An empty result is ambiguous — genuinely no matches, or a pattern this
        # build choked on. Falling through to ERE costs one extra process on a
        # miss and removes the ambiguity.
    args = ["git", "grep", flags, _to_ere(pattern), "--", *limits]
    out = _run(args, root)
    return [l for l in out.splitlines() if l.strip()]


_pcre_support: bool | None = None


def _supports_pcre(root: str) -> bool:
    """Whether this git was built with PCRE (``git grep -P``).

    Probed once per process — the answer is a property of the git binary, not of
    the repository — inside a real repo, because ``git grep`` needs one. Builds
    without PCRE say so on stderr and are the *good* case: they fail loudly. The
    ERE tier below is the one that fails quietly, which is why it is second.
    """
    global _pcre_support
    if _pcre_support is None:
        proc = subprocess.run(["git", "grep", "-qP", r"\d", "--", "."], cwd=root,
                              capture_output=True, text=True, errors="replace",
                              timeout=_TIMEOUT, check=False)
        _pcre_support = "not compiled with PCRE" not in (proc.stderr or "")
    return _pcre_support


# A GNU shorthand means two different things depending on where it sits: inside
# a bracket expression it contributes to that class, outside it *is* the class.
# Substituting the outside form everywhere turns `[\w\s*]` into nested brackets,
# which POSIX reads as a class of `[`, `:`, `]`… — a pattern that matches nearly
# any line. So both forms are spelled out and the walker below picks one.
_ERE_CLASS_INNER = {"s": "[:space:]", "w": "[:alnum:]_", "d": "[:digit:]",
                    "S": "^[:space:]", "W": "^[:alnum:]_", "D": "^[:digit:]"}
_ERE_STANDALONE = {"s": "[[:space:]]", "w": "[[:alnum:]_]", "d": "[[:digit:]]",
                   "S": "[^[:space:]]", "W": "[^[:alnum:]_]", "D": "[^[:digit:]]"}
# `\b` has no ERE equivalent. These consume the boundary character, which is
# harmless here: every caller wants the matching *file* or *line*, never the
# match extent.
_ERE_WORD_START = "(^|[^[:alnum:]_])"
_ERE_WORD_END = "([^[:alnum:]_]|$)"


def _to_ere(pattern: str) -> str:
    """Make a ripgrep (Rust regex) pattern survive `git grep -E`.

    `-E` is POSIX ERE *plus whatever the platform's regex library adds*, and that
    difference is a trap. On a GNU/glibc box `\\b`, `\\s` and `\\w` work, so the
    fallback tier looks correct; on a build without those extensions they are read
    as the literal characters `b`, `s`, `w`, and every definition pattern returns
    **zero hits with no error at all**. Non-capturing groups fail the same way —
    `git grep -E '(?:def|class)'` hunts for the literal text `?:`.

    Both failures are silent, which is the worst possible shape: localization just
    quietly stops finding anything on someone else's machine. It has now bitten
    twice — once via `(?:`, and once via `\\s` on macOS, caught by CI — so this
    walks the pattern and translates every extension to strict ERE rather than
    trusting the platform. Nothing reads the capture groups, so degrading them is
    lossless.
    """
    out: list[str] = []
    i, n, in_class = 0, len(pattern), False
    while i < n:
        ch = pattern[i]
        if ch == "\\" and i + 1 < n:
            nxt = pattern[i + 1]
            if in_class and nxt in _ERE_CLASS_INNER:
                out.append(_ERE_CLASS_INNER[nxt])
            elif not in_class and nxt in _ERE_STANDALONE:
                out.append(_ERE_STANDALONE[nxt])
            elif nxt == "b" and not in_class:
                # A boundary opening an alternative anchors to the left; anywhere
                # else it closes a symbol and anchors to the right.
                tail = "".join(out).rstrip()
                out.append(_ERE_WORD_START if not tail or tail[-1] in "(|"
                           else _ERE_WORD_END)
            else:
                out.append(ch + nxt)       # a genuine escape — pass it through
            i += 2
            continue
        if not in_class and pattern.startswith("(?:", i):
            out.append("(")
            i += 3
            continue
        if in_class and pattern.startswith("[:", i):
            # A POSIX class already written by hand: copy it whole, so its
            # closing ']' is not mistaken for the end of the bracket expression.
            end = pattern.find(":]", i)
            if end != -1:
                out.append(pattern[i:end + 2])
                i = end + 2
                continue
        if not in_class and ch == "[":
            in_class = True
        elif in_class and ch == "]":
            in_class = False
        out.append(ch)
        i += 1
    return "".join(out)
