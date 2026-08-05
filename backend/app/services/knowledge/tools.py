"""Retrieval tools the Dev agent can CALL — one implementation, every backend.

The pipeline indexes a code graph and a dense-embedding channel, then used to
hand the Dev agent a pre-computed text blob and let it `grep` its way around
anyway. This module closes that gap: the same queries the PM issues at scoping
time (`lookup`, `callers`, `search`, `outline`, `grep`) become tools the Dev
agent invokes itself, mid-run, against the same index.

ONE dispatcher, three surfaces — no tool is exclusive to a particular vendor's
agent:

  * ``call()`` — the dispatcher. Pure text in, pure text out.
  * ``cli_main()`` + ``install()`` — a tiny executable shim (``.codeverdict/kb``)
    dropped into the working copy and git-ignored, so ANY headless CLI backend
    (claude-code, codex, cursor, aider, gemini-cli, …) reaches the tools through
    the shell it already has. No MCP, no per-vendor plumbing.
  * ``PROTOCOL_BLOCK`` — the ``<<<LOOKUP>>>`` / ``<<<CALLERS>>>`` /
    ``<<<SEARCH>>>`` / ``<<<OUTLINE>>>`` request blocks the HTTP SEARCH/REPLACE
    loop (services/openai_agent.py) parses, so a Groq/Gemini/OpenAI model with
    no tool-calling support gets the identical capability.

Everything degrades: with no graph binary, `lookup` falls back to the symbol map
and ripgrep, `search` returns "" and the caller sees an honest empty result.
"""

from __future__ import annotations

import os
import re
import secrets
import shlex
import sys
from pathlib import Path

from ...config import settings
from .. import git_ops
from .. import search as code_search  # aliased: `search` is a tool name here
from . import expand as expand_mod
from . import graph, retriever, symbol_map

# Tools any agent may call, in the order they're advertised.
TOOL_NAMES = ("search", "lookup", "callers", "expand", "outline", "snippet", "grep")

# Result caps: a tool reply is fed straight back into the model's context.
_MAX_CHARS = 2400
# `snippet` returns source on purpose, so it gets its own larger cap — capping it
# at the others' size would truncate mid-function, which is worse than not
# offering the tool.
_MAX_SNIPPET_CHARS = 6000


def _fmt_hits(rows: list[dict]) -> list[str]:
    return [f"  {graph.render_hit(r)}" for r in rows if r.get("file_path")]


# --- The tools ----------------------------------------------------------------

def search(repo: str, query: str, *, limit: int = 8) -> str:
    """Ranked code locations for a natural-language or mechanism query — the
    hybrid BM25 + dense-embedding channel the PM uses at scoping time."""
    out = retriever.localize(repo, query, limit=limit)
    return out or f"(no indexed code matched “{query}” — try `grep`, or different terms)"


def lookup(repo: str, name: str, cwd: str = "", *, limit: int = 5) -> str:
    """Where is this symbol DEFINED? Graph first (AST-verified), then the symbol
    map, then a live language-aware ripgrep for a definition — so the answer is
    real even on a repo the graph never indexed."""
    if graph.available():
        rows = graph.lookup(repo, name, limit=limit)
        if rows:
            return f"{name} — defined at:\n" + "\n".join(_fmt_hits(rows))
    smap = symbol_map.load_slug(repo)
    hits = (smap.lookup(name) or []) if smap else []
    if hits:
        return f"{name} — defined at:\n" + "\n".join(
            f"  {f}:{m['l']} ({m['k']})" for f, m in hits[:limit])
    if cwd:
        files = code_search.definitions(cwd, name, max_files=limit)
        if files:
            return f"{name} — defined in: " + ", ".join(files[:limit])
    close = smap.suggest(name) if smap else []
    hint = f" Similar existing symbols: {', '.join(close)}." if close else ""
    return (f"{name} — NOT FOUND as a definition anywhere in the repo. Treat it as "
            f"new (create it) rather than hunting for it.{hint}")


def callers(repo: str, name: str, *, limit: int = 8) -> str:
    """Who calls this symbol — the blast radius of changing its behavior."""
    if not graph.available():
        return "(call graph unavailable — use `grep` for the symbol name instead)"
    rows = graph.callers(repo, name, limit=limit)
    if not rows:
        return (f"{name} — no recorded callers. Either it's an entry point / "
                "dynamically dispatched, or the name is wrong (try `lookup`).")
    return f"{name} — called from:\n" + "\n".join(
        f"  {c['file_path']}:{c['start_line'] or '?'} ({c['name']})" for c in rows)


def outline(repo: str, file_path: str, cwd: str = "", *, limit: int = 40) -> str:
    """The symbols defined in one file, with line numbers — cheaper than reading
    the file when all you need is where to look inside it."""
    rows = graph.outline(repo, file_path, limit=limit) if graph.available() else []
    if rows:
        return f"{file_path} defines:\n" + "\n".join(
            f"  {o['name']}:{o.get('start_line') or '?'}" for o in rows if o.get("name"))
    smap = symbol_map.load_slug(repo)
    text = smap.outline(file_path, max_symbols=limit) if smap else ""
    if text:
        return f"{file_path} defines: {text}"
    if cwd and git_ops.read_file(cwd, file_path, max_chars=1) is None:
        return f"{file_path} — does not exist in this working copy."
    return f"{file_path} — no symbols indexed (read the file directly)."


def expand(repo: str, name: str, *, limit: int = 12) -> str:
    """A symbol's 1-hop neighbourhood: what it calls, what calls it, what it
    inherits, its class members, and the tests that already cover it.

    `callers` answers half of "what breaks if I change this"; this answers the
    whole question in one call, which is the question that actually decides
    whether an edit is safe."""
    return expand_mod.ego(repo, name, limit=limit)


def snippet(repo: str, name: str, cwd: str = "") -> str:
    """The exact source of one symbol, by name — targeted reading without
    guessing a line range or paying for a whole-file read."""
    if graph.available():
        rows = graph.lookup(repo, name, limit=4)
        for i, row in enumerate(rows):
            qn = row.get("qualified_name") or ""
            body = (graph.snippet(repo, qn) or {}).get("source") if qn else ""
            if not body:
                continue
            out = f"{name} — {row.get('file_path')}:{row.get('start_line') or '?'}\n{body}"
            # A bare name is often ambiguous (`cell_len` is both a module
            # function and a Text method here). Showing one and staying silent
            # is how an agent edits the wrong one; naming the alternatives costs
            # a line and lets it choose.
            others = [f"{o['file_path']}:{o.get('start_line') or '?'}"
                      for j, o in enumerate(rows) if j != i]
            if others:
                out += ("\n\n(also defined at " + "; ".join(others)
                        + " — if you meant one of those, `outline` that file)")
            return out
    # No graph (or no node): fall back to the definition site's surrounding lines
    # so the tool still answers rather than sending the agent to read a file.
    return lookup(repo, name, cwd) + "\n(no indexed source for this symbol — read the file)"


def grep(repo: str, pattern: str, cwd: str, *, pathspec: str = "", max_lines: int = 20) -> str:
    """Literal/regex search over the working copy (ripgrep) — the channel for
    text the index doesn't model: strings, comments, config, templates,
    generated code. Not a fallback; the refinement pass."""
    if not cwd:
        return "(no working copy available for grep)"
    hits = code_search.lines(cwd, pattern, max_lines=max_lines, pathspec=pathspec or ".")
    return hits or "(no matches)"


_DISPATCH = {
    "search": lambda repo, cwd, arg: search(repo, arg),
    "lookup": lambda repo, cwd, arg: lookup(repo, arg, cwd),
    "callers": lambda repo, cwd, arg: callers(repo, arg),
    "expand": lambda repo, cwd, arg: expand(repo, arg),
    "outline": lambda repo, cwd, arg: outline(repo, arg, cwd),
    "snippet": lambda repo, cwd, arg: snippet(repo, arg, cwd),
    "grep": lambda repo, cwd, arg: grep(repo, arg, cwd),
}


def _base_url() -> str:
    """Where the endpoint answers tool calls (0.0.0.0 binds are reached over
    loopback). Reporting only — starting the endpoint is install()'s job."""
    host = os.environ.get("HOST", "127.0.0.1")
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return f"http://{host}:{os.environ.get('PORT', '8017')}"


# Live shim tokens → the (repo, cwd) they may query. Minted per install, held in
# the server process only, dropped on restart. The shim never gets to name its
# own repo/cwd — the token does.
_TOKENS: dict[str, tuple[str, str]] = {}


def resolve_token(token: str) -> tuple[str, str] | None:
    return _TOKENS.get(token or "")


def call(repo: str, cwd: str, name: str, arg: str) -> str:
    """Run one tool by name. `repo` is a repo URL or slug (interchangeable —
    git_ops.slug is idempotent). Never raises: a broken index must degrade to an
    honest message, not kill the Dev round."""
    fn = _DISPATCH.get((name or "").strip().lower())
    if fn is None:
        return f"unknown tool '{name}' — available: {', '.join(TOOL_NAMES)}"
    arg = (arg or "").strip()
    if not arg:
        return f"{name}: missing argument"
    cap = _MAX_SNIPPET_CHARS if name.strip().lower() == "snippet" else _MAX_CHARS
    try:
        return (fn(repo, cwd, arg) or "(no result)")[:cap]
    except Exception as exc:  # noqa: BLE001 — a tool failure is a result, not a crash
        return f"{name} failed: {exc}"


# --- Surface 1: the HTTP loop's text protocol ---------------------------------

# Advertised inside openai_agent._CODE_SYSTEM. The same tools, expressed as
# request blocks a non-tool-calling model can emit.
PROTOCOL_BLOCK = (
    "To QUERY THE REPOSITORY INDEX (a code graph + semantic search over this exact "
    "repo — results arrive next round). Prefer these over guessing which file to "
    "edit; use them to CHECK the localization hints before trusting them:\n"
    "<<<SEARCH what the code does, in mechanism terms>>>   (ranked file:line hits)\n"
    "<<<LOOKUP SymbolName>>>                               (where it is DEFINED)\n"
    "<<<CALLERS SymbolName>>>                              (who calls it)\n"
    "<<<EXPAND SymbolName>>>                               (everything 1 hop away)\n"
    "<<<OUTLINE relative/path.py>>>                        (symbols + lines in a file)\n"
    "<<<SNIPPET SymbolName>>>                              (its exact source)\n"
    "<<<GREP regex-pattern>>>                              (raw text search)\n"
)


# --- Surface 2: the executable shim for headless CLI backends -----------------

# A `sh` wrapper rather than a `#!{python}` shebang: an interpreter path
# containing spaces (a venv under "…/AI_ML projs/…") makes the kernel's shebang
# parser fail with a bare ENOENT. Quoted exec is immune. Windows gets a `.cmd`
# launcher because a shebang file without an executable extension is not a
# directly launchable process there.
_LAUNCHER = '#!/bin/sh\nexec {python} {script} "$@"\n'
_WINDOWS_LAUNCHER = '@echo off\r\n"{python}" "{script}" %*\r\n'

_SHIM = '''"""CodeVerdict repository-index tools. Auto-generated per run; not part of the repo."""
import json
import sys
import urllib.request

sys.path.insert(0, {backend!r})

REPO = {repo!r}
CWD = {cwd!r}
URL = {url!r}
TOKEN = {token!r}
USAGE = """usage: .codeverdict/kb <command> <argument>

  search  <query>        ranked code locations (semantic + BM25 over this repo)
  lookup  <Symbol>       where a symbol is DEFINED (AST-verified)
  callers <Symbol>       every call site of a symbol
  expand  <Symbol>       everything 1 hop away (calls, callers, members, tests)
  outline <path>         symbols + line numbers in one file
  snippet <Symbol>       the exact source of one symbol
  grep    <regex>        raw text search over the working copy
"""

def remote(args):
    """Ask the running CodeVerdict server. It already holds the code graph and the
    embedded vector store — a second process cannot open them (the embedded
    Qdrant is single-writer), so going in-process would silently drop the
    semantic half of `search`."""
    body = json.dumps({{"token": TOKEN, "tool": args[0],
                        "arg": " ".join(args[1:])}}).encode()
    req = urllib.request.Request(URL + "/kb/tool", data=body,
                                 headers={{"content-type": "application/json"}})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())["result"]


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    try:
        print(remote(args))
        return 0
    except Exception:  # noqa: BLE001 — server down / not served from one
        pass
    from app.services.knowledge import tools
    print(tools.cli_main(REPO, CWD, args))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''

_SHIM_DIR = ".codeverdict"
_SHIM_NAME = "kb"


def _backend_root() -> str:
    """The directory holding the `app` package (…/backend), for the shim's path."""
    return str(Path(__file__).resolve().parents[3])


def install(cwd: str, repo: str) -> str:
    """Write `.codeverdict/kb` into the working copy and keep it out of git.

    Returns the relative command to advertise in the prompt, or "" if the shim
    could not be installed (the Dev agent then just works without tools —
    fail-open, same contract as the rest of the pipeline).

    The shim is excluded via `.git/info/exclude` rather than `.gitignore`: it
    must never show up in the Dev agent's diff, and editing a tracked file to
    hide our own tooling would itself be a diff.
    """
    try:
        # A run is starting, so bring up the endpoint the shim will post to.
        # This is the only place that needs it live, and doing it here rather
        # than expecting an operator to have started something is the point:
        # the shim used to be written pointing at a port nothing was listening
        # on, so every tool call failed and the agent fell back to grep with
        # nobody the wiser — the shim is fail-open by design.
        if not os.environ.get("CODEVERDICT_NO_TOOLS_SERVER"):
            from ..tools_server import ensure_running

            ensure_running(host=_base_url().split("//")[1].rsplit(":", 1)[0],
                           port=int(os.environ.get("PORT", "8017")))
        root = Path(cwd)
        d = root / _SHIM_DIR
        d.mkdir(parents=True, exist_ok=True)
        token = secrets.token_urlsafe(24)
        _TOKENS[token] = (repo, str(root))
        script = d / f"{_SHIM_NAME}.py"
        script.write_text(_SHIM.format(backend=_backend_root(), repo=repo, cwd=str(root),
                                       url=_base_url(), token=token), encoding="utf-8")
        if os.name == "nt":
            shim = d / f"{_SHIM_NAME}.cmd"
            shim.write_text(_WINDOWS_LAUNCHER.format(python=sys.executable,
                                                     script=script), encoding="utf-8")
            command = f"{_SHIM_DIR}/{_SHIM_NAME}.cmd"
        else:
            shim = d / _SHIM_NAME
            shim.write_text(_LAUNCHER.format(python=shlex.quote(sys.executable),
                                             script=shlex.quote(str(script))), encoding="utf-8")
            command = f"{_SHIM_DIR}/{_SHIM_NAME}"
        shim.chmod(0o755)
        exclude = root / ".git" / "info" / "exclude"
        if exclude.parent.is_dir():
            body = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
            if _SHIM_DIR not in body:
                exclude.write_text(body.rstrip("\n") + f"\n{_SHIM_DIR}/\n")
    except OSError:
        return ""
    return command


def prompt_block(command: str) -> str:
    """How the tools are described to a CLI-backed Dev agent (claude-code,
    codex, cursor, aider, gemini-cli — anything with a shell)."""
    if not command:
        return ""
    return f"""
Repository index tools (RUN THESE — this repo is pre-indexed as a code graph with
semantic search). Each is one shell command, sub-second, and returns real
AST-verified `file:line` locations. They are cheaper and more reliable than your
own Grep/Glob, which see text but not structure:
  {command} search  "<what the code does, in mechanism terms>"   ranked locations
  {command} lookup  <SymbolName>                                  where it's DEFINED
  {command} callers <SymbolName>                                  every call site
  {command} expand  <SymbolName>                                  everything 1 hop away
  {command} outline <relative/path>                               symbols + lines in a file
  {command} snippet <SymbolName>                                  its exact source
  {command} grep    "<regex>"                                     raw text search

You are expected to use them at these moments — skipping them is how wrong edits
happen:
  1. BEFORE your first edit, if the plan's pins don't obviously match the
     behavior described: `{command} search "<mechanism>"` to find where it really
     lives. Use MECHANISM wording ("strip ansi escape sequences when computing
     cell width"), never the user's symptom wording ("borders look crooked").
  2. BEFORE editing any function, `{command} expand <ThatFunction>` — it names
     the callers you'd break, the members you'd have to keep consistent, AND the
     tests that already cover it, in one call. A change to something with callers
     you never opened is how regressions ship.
  3. When you are about to claim a pin is wrong (or right), confirm it with
     `{command} lookup` rather than asserting it from memory.
  4. Need to read one function, not a whole file? `{command} snippet <Symbol>`.
"""


# --- Surface 4: one-shot evidence rounds (QA, jury) ---------------------------
# A reviewer that can only read the diff has to guess about everything outside
# it, and a wrong guess costs a full paid Dev+QA+Review round. These let a
# reviewing stage ask before it decides, in the same request-block form the HTTP
# coding loop already uses, so there is one protocol to learn and one to parse.

REQUEST_RE = re.compile(
    r"<<<(" + "|".join(t.upper() for t in TOOL_NAMES) + r")\s+(.+?)\s*>>>")


def parse_requests(text: str) -> list[tuple[str, str]]:
    """Extract `<<<TOOL argument>>>` requests from a model reply."""
    return [(t.lower(), a.strip()) for t, a in REQUEST_RE.findall(text or "") if a.strip()]


def run_requests(repo: str, cwd: str, requests: list[tuple[str, str]], *,
                 limit: int = 3) -> str:
    """Answer a capped batch of requests as one prompt-ready block ("" for none).
    The cap is the point: an unbounded evidence loop turns a review into a
    second Dev agent."""
    blocks = [f"$ {name} {arg}\n{call(repo, cwd, name, arg)}"
              for name, arg in (requests or [])[:max(0, limit)]]
    if not blocks:
        return ""
    return ("Answers to the repository lookups you requested:\n\n"
            + "\n\n".join(blocks))


def evidence_block(command_hint: str = "") -> str:
    """How a reviewing stage is told it can ask. Deliberately framed around
    *claims*: the failure this addresses is not ignorance, it is a confident
    finding about code the reviewer never saw."""
    return (
        "\nBEFORE YOU DECIDE — you may query this repository's index. A diff shows you "
        "what changed, not what depends on it, so any claim about code OUTSIDE the diff "
        "(\"this breaks callers\", \"this doesn't match the existing pattern\", \"this "
        "case isn't covered\") is a guess unless you check it. Emit up to "
        f"{settings.jury_tool_calls} request block(s) INSTEAD of your verdict and you "
        "will be called again with the answers:\n"
        "<<<LOOKUP SymbolName>>>     where it is DEFINED\n"
        "<<<CALLERS SymbolName>>>    who calls it\n"
        "<<<EXPAND SymbolName>>>     everything 1 hop away, including covering tests\n"
        "<<<OUTLINE relative/path>>> the symbols in a file\n"
        "<<<SNIPPET SymbolName>>>    its exact source\n"
        "<<<GREP regex>>>            raw text search\n"
        "<<<SEARCH mechanism>>>      ranked locations for a behaviour\n"
        "Ask only when the answer would change your verdict. If you can decide from what "
        "you already have, decide — every round costs money.\n")


def cli_main(repo: str, cwd: str, argv: list[str]) -> str:
    """Entry point used by the generated shim (kept here so the shim stays
    trivial and the logic stays testable)."""
    if not argv:
        return f"usage: kb <{'|'.join(TOOL_NAMES)}> <argument>"
    return call(repo, cwd, argv[0], " ".join(argv[1:]))
