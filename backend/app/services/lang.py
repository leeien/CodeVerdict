"""Language registry — every language-specific decision in one place.

The pipeline itself is language-agnostic (localize → edit → verify → iterate);
what varies by language is only:

  * symbol extraction  → `extract_symbols(rel, source)`  (feeds the symbol map)
  * import extraction  → `extract_imports(rel, source)`  (feeds the dep graph)
  * the edit-time parse gate → `syntax_error(rel, content)`
  * what counts as a test file → `is_test_file(rel)`
  * which test ecosystem a repo uses → `project_kind(root)` + `RUNNERS`

Design rule: FAIL OPEN. An unknown language still gets its files recorded,
edits applied ungated, and tests reported as "couldn't run" (None) rather than
FAIL — the QA agent already treats INCONCLUSIVE as not-a-pass, so degrading
gracefully never fabricates a green result.

Python uses the stdlib `ast` (exact). Other languages have two tiers, same
pattern as the embedding layer (semantic when installed, tfidf otherwise):

  * tree-sitter (optional, `pip install .[treesitter]`) — real parse trees for
    JS/TS/Go/Rust/Java/Ruby, and an edit-time parse gate for all of them.
    Grammars are lazy-loaded per language, so only what a repo actually uses
    costs memory.
  * regex fallback (always available, zero native deps) — line-anchored
    extractors with a naive brace/indent tracker for methods. Good enough for
    an index (the pipeline cross-checks every pin with live ripgrep), and it
    keeps the core install dependency-light.

Any tree-sitter failure (missing grammar, parse crash) silently degrades to the
regex tier — the fidelity ceiling is set by what's installed, never the floor.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Language identification
# ---------------------------------------------------------------------------

LANG_BY_EXT = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".go": "Go", ".java": "Java",
    ".rb": "Ruby", ".rs": "Rust", ".php": "PHP", ".c": "C", ".cc": "C++",
    ".cpp": "C++", ".h": "C", ".cs": "C#", ".kt": "Kotlin", ".swift": "Swift",
    ".scala": "Scala", ".vue": "Vue", ".svelte": "Svelte",
}

# Directories no analyzer pass ever descends into.
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build",
    ".next", ".turbo", "vendor", "target", "coverage", ".pytest_cache", ".qdrant",
    ".agent-venv",
}


def language_of(rel: str) -> str:
    return LANG_BY_EXT.get(Path(rel).suffix.lower(), "Other")


def _ext(rel: str) -> str:
    return Path(rel).suffix.lower()


# ---------------------------------------------------------------------------
# Test-file detection (path-shaped only — cheap, used in hot loops)
# ---------------------------------------------------------------------------

_TEST_PATH_RES = (
    re.compile(r"(^|/)tests?/"),
    re.compile(r"(^|/)__tests__/"),
    re.compile(r"(^|/)test_[^/]+\.py$"),
    re.compile(r"_test\.(py|go)$"),
    re.compile(r"\.(test|spec)\.[jt]sx?$"),
    re.compile(r"(^|/)src/test/"),            # Maven/Gradle layout
    re.compile(r"Test\.java$"),
    re.compile(r"_spec\.rb$"),
)


def is_test_file(rel: str) -> bool:
    p = rel.replace("\\", "/")
    return any(r.search(p) for r in _TEST_PATH_RES)


# ---------------------------------------------------------------------------
# Symbol extraction → [{"n": name, "k": class|function|method|const, "l": line,
#                       "p": parent}] — the raw material of the symbol map.
# ---------------------------------------------------------------------------

_MAX_SYMBOLS = 400  # per file — bounds symbols.json on generated/minified files


def extract_symbols(rel: str, source: str) -> list[dict]:
    """Line-numbered symbol facts for one file; [] when the language has no
    extractor or the source doesn't parse. Dispatches on extension: Python via
    stdlib ast, other languages via tree-sitter when installed (exact), the
    regex extractor otherwise. An empty tree-sitter result also falls through
    to regex — a walker gap must not lose symbols the fallback would find."""
    ext = _ext(rel)
    if ext != ".py":
        ts = _ts_symbols(ext, source)
        if ts:
            return ts[:_MAX_SYMBOLS]
    fn = _SYMBOL_EXTRACTORS.get(ext)
    if fn is None:
        return []
    try:
        return fn(source)[:_MAX_SYMBOLS]
    except Exception:  # noqa: BLE001 — an index must never take the pipeline down
        return []


# ---------------------------------------------------------------------------
# Optional tree-sitter tier — real parse trees when `.[treesitter]` is
# installed. Grammars are lazy-loaded per language and cached, so memory scales
# with the languages a repo actually contains, not with what the pack ships.
# Every failure path (extra not installed, grammar missing, parse crash)
# returns None and the caller degrades to the regex tier.
# ---------------------------------------------------------------------------

_TS_LANG_BY_EXT = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
}

_ts_parsers: dict[str, object] = {}  # grammar name → Parser | None (load failed)


def _ts_parser(ts_lang: str):
    if ts_lang not in _ts_parsers:
        try:
            from tree_sitter_language_pack import get_parser
            _ts_parsers[ts_lang] = get_parser(ts_lang)
        except Exception:  # noqa: BLE001 — optional tier; regex covers it
            _ts_parsers[ts_lang] = None
    return _ts_parsers[ts_lang]


def treesitter_available(ext: str = ".js") -> bool:
    """Whether the optional tree-sitter tier can serve this extension."""
    ts_lang = _TS_LANG_BY_EXT.get(ext)
    return ts_lang is not None and _ts_parser(ts_lang) is not None


def _ts_symbols(ext: str, source: str) -> list[dict] | None:
    """Symbols via tree-sitter; None = tier unavailable or the parse crashed
    (caller falls back to the regex extractor)."""
    ts_lang = _TS_LANG_BY_EXT.get(ext)
    if ts_lang is None:
        return None
    parser = _ts_parser(ts_lang)
    if parser is None:
        return None
    try:
        tree = parser.parse(source.encode("utf-8"))
        symbols: list[dict] = []
        _TS_WALKERS[ts_lang](tree.root_node, symbols)
        return symbols
    except Exception:  # noqa: BLE001 — an index must never take the pipeline down
        return None


def _ts_name(node) -> str | None:
    n = node.child_by_field_name("name")
    return n.text.decode("utf-8", "replace") if n is not None else None


def _ts_sym(node, kind: str, name: str, parent: str | None = None) -> dict:
    d = {"n": name, "k": kind, "l": node.start_point[0] + 1}
    if parent:
        d["p"] = parent
    return d


_TS_JS_CLASS = {"class_declaration", "abstract_class_declaration"}
_TS_JS_FUNC = {"function_declaration", "generator_function_declaration"}
_TS_JS_TYPE = {"interface_declaration", "enum_declaration", "type_alias_declaration"}
_TS_JS_FN_VALUES = {"arrow_function", "function_expression", "generator_function", "function"}
_TS_UPPER_RE = re.compile(r"[A-Z_][A-Z0-9_]*")


def _ts_walk_js(root, symbols: list[dict]) -> None:
    for node in root.named_children:
        if node.type == "export_statement":
            decl = node.child_by_field_name("declaration")
            if decl is not None:
                _ts_js_decl(decl, symbols, exported=True)
        else:
            _ts_js_decl(node, symbols, exported=False)


def _ts_js_decl(node, symbols: list[dict], exported: bool) -> None:
    t = node.type
    if t in _TS_JS_CLASS:
        name = _ts_name(node)
        if name:
            symbols.append(_ts_sym(node, "class", name))
            body = node.child_by_field_name("body")
            for m in (body.named_children if body is not None else []):
                if m.type == "method_definition" and (mn := _ts_name(m)):
                    symbols.append(_ts_sym(m, "method", mn, parent=name))
    elif t in _TS_JS_FUNC or t in _TS_JS_TYPE:
        name = _ts_name(node)
        if name:
            symbols.append(_ts_sym(node, "function" if t in _TS_JS_FUNC else "class", name))
    elif t in ("lexical_declaration", "variable_declaration"):
        for d in node.named_children:
            if d.type != "variable_declarator" or not (name := _ts_name(d)):
                continue
            value = d.child_by_field_name("value")
            if value is not None and value.type in _TS_JS_FN_VALUES:
                symbols.append(_ts_sym(d, "function", name))
            elif exported or _TS_UPPER_RE.fullmatch(name):
                symbols.append(_ts_sym(d, "const", name))


def _ts_walk_go(root, symbols: list[dict]) -> None:
    for node in root.named_children:
        t = node.type
        if t == "function_declaration":
            if name := _ts_name(node):
                symbols.append(_ts_sym(node, "function", name))
        elif t == "method_declaration":
            name = _ts_name(node)
            recv = node.child_by_field_name("receiver")
            recv_type = None
            if recv is not None and recv.named_children:
                tn = recv.named_children[0].child_by_field_name("type")
                if tn is not None:
                    recv_type = tn.text.decode("utf-8", "replace").lstrip("*").split("[")[0]
            if name:
                symbols.append(_ts_sym(node, "method", name, parent=recv_type)
                               if recv_type else _ts_sym(node, "function", name))
        elif t == "type_declaration":
            for spec in node.named_children:
                if spec.type in ("type_spec", "type_alias") and (name := _ts_name(spec)):
                    symbols.append(_ts_sym(spec, "class", name))
        elif t in ("const_declaration", "var_declaration"):
            for spec in node.named_children:
                if spec.type in ("const_spec", "var_spec") and (name := _ts_name(spec)):
                    symbols.append(_ts_sym(spec, "const", name))


def _ts_walk_rust(root, symbols: list[dict]) -> None:
    for node in root.named_children:
        t = node.type
        if t in ("struct_item", "enum_item", "trait_item", "union_item"):
            if name := _ts_name(node):
                symbols.append(_ts_sym(node, "class", name))
        elif t in ("const_item", "static_item"):
            if name := _ts_name(node):
                symbols.append(_ts_sym(node, "const", name))
        elif t == "function_item":
            if name := _ts_name(node):
                symbols.append(_ts_sym(node, "function", name))
        elif t == "impl_item":
            tn = node.child_by_field_name("type")
            parent = tn.text.decode("utf-8", "replace").split("<")[0] if tn is not None else None
            body = node.child_by_field_name("body")
            for m in (body.named_children if body is not None else []):
                if m.type == "function_item" and (mn := _ts_name(m)):
                    symbols.append(_ts_sym(m, "method", mn, parent=parent))


_TS_JAVA_TYPES = {"class_declaration", "interface_declaration", "enum_declaration",
                  "record_declaration", "annotation_type_declaration"}


def _ts_walk_java(root, symbols: list[dict]) -> None:
    for node in root.named_children:
        if node.type in _TS_JAVA_TYPES and (name := _ts_name(node)):
            symbols.append(_ts_sym(node, "class", name))
            body = node.child_by_field_name("body")
            if body is not None:
                _ts_java_body(body, symbols, name)


def _ts_java_body(body, symbols: list[dict], parent: str) -> None:
    for m in body.named_children:
        # constructors read as the class itself — skip, like the regex tier
        if m.type == "method_declaration" and (mn := _ts_name(m)):
            symbols.append(_ts_sym(m, "method", mn, parent=parent))
        elif m.type in _TS_JAVA_TYPES and (mn := _ts_name(m)):
            symbols.append(_ts_sym(m, "class", mn))
            inner = m.child_by_field_name("body")
            if inner is not None:
                _ts_java_body(inner, symbols, mn)


def _ts_walk_ruby(root, symbols: list[dict], parent: str | None = None) -> None:
    for node in root.named_children:
        t = node.type
        if t in ("class", "module"):
            name = _ts_name(node)
            if name:
                symbols.append(_ts_sym(node, "class", name))
            body = node.child_by_field_name("body")
            if body is not None:
                _ts_walk_ruby(body, symbols, parent=name or parent)
        elif t in ("method", "singleton_method"):
            name = _ts_name(node)
            if name:
                symbols.append(_ts_sym(node, "method", name, parent=parent)
                               if parent else _ts_sym(node, "function", name))
        elif t == "body_statement":
            _ts_walk_ruby(node, symbols, parent=parent)


_TS_WALKERS = {
    "javascript": _ts_walk_js, "typescript": _ts_walk_js, "tsx": _ts_walk_js,
    "go": _ts_walk_go, "rust": _ts_walk_rust,
    "java": _ts_walk_java, "ruby": _ts_walk_ruby,
}


def _python_symbols(source: str) -> list[dict]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    symbols: list[dict] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append({"n": node.name, "k": "class", "l": node.lineno})
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append({"n": child.name, "k": "method",
                                    "l": child.lineno, "p": node.name})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append({"n": node.name, "k": "function", "l": node.lineno})
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                    symbols.append({"n": tgt.id, "k": "const", "l": node.lineno})
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                symbols.append({"n": node.target.id, "k": "const", "l": node.lineno})
    return symbols


def _brace_delta(line: str) -> int:
    """Net {…} depth change of a line, ignoring braces inside quotes and after
    line comments. Naive (no multi-line strings) — good enough for an index."""
    depth = 0
    quote: str | None = None
    prev = ""
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                prev = ""
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "/" and prev == "/":
            break
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        prev = ch
        i += 1
    return depth


_ID = r"[A-Za-z_$][\w$]*"
_JS_CLASS = re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+({_ID})")
_JS_FUNC = re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*({_ID})")
_JS_ARROW = re.compile(
    rf"^\s*(?:export\s+)?(?:const|let|var)\s+({_ID})\s*(?::[^=\n]+)?=\s*"
    rf"(?:async\s+)?(?:\([^)]*\)|{_ID})\s*=>")
_JS_CONST = re.compile(rf"^\s*export\s+(?:const|let|var)\s+({_ID})"
                       r"|^(?:const|let|var)\s+([A-Z_][A-Z0-9_]*)\s*=")
_TS_TYPE = re.compile(rf"^\s*(?:export\s+)?(?:declare\s+)?(?:interface|enum)\s+({_ID})"
                      rf"|^\s*(?:export\s+)?type\s+({_ID})\s*=")
_JS_METHOD = re.compile(
    rf"^\s+(?:(?:public|private|protected|static|readonly|override|async|get|set)\s+|\*\s*)*"
    rf"({_ID})\s*(?:<[^>\n]*>)?\([^;]*$|"
    rf"^\s+(?:(?:public|private|protected|static|readonly|override|async|get|set)\s+|\*\s*)*"
    rf"({_ID})\s*(?:<[^>\n]*>)?\([^;{{]*\)\s*(?::[^;{{\n]+)?\s*\{{")
_JS_NOT_METHOD = {"if", "for", "while", "switch", "catch", "return", "function",
                  "else", "do", "try", "new", "await", "typeof", "super", "import"}


def _js_symbols(source: str) -> list[dict]:
    symbols: list[dict] = []
    parent: str | None = None
    parent_close = 0  # depth at which the current class body closes
    depth = 0
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _JS_CLASS.match(line):
            symbols.append({"n": m.group(1), "k": "class", "l": i})
            parent, parent_close = m.group(1), depth
        elif m := _JS_FUNC.match(line):
            symbols.append({"n": m.group(1), "k": "function", "l": i})
        elif m := _JS_ARROW.match(line):
            if parent is None or depth == 0:
                symbols.append({"n": m.group(1), "k": "function", "l": i})
        elif m := _TS_TYPE.match(line):
            symbols.append({"n": m.group(1) or m.group(2), "k": "class", "l": i})
        elif m := _JS_CONST.match(line):
            name = m.group(1) or m.group(2)
            if depth == 0:
                symbols.append({"n": name, "k": "const", "l": i})
        elif parent and depth == parent_close + 1 and (m := _JS_METHOD.match(line)):
            name = m.group(1) or m.group(2)
            if name and name not in _JS_NOT_METHOD:
                symbols.append({"n": name, "k": "method", "l": i, "p": parent})
        depth += _brace_delta(line)
        if parent is not None and depth <= parent_close:
            parent = None
    return symbols


_GO_FUNC = re.compile(r"^func\s+(?:\(\s*\w+\s+\*?(\w+)\s*\)\s+)?([A-Za-z_]\w*)")
_GO_TYPE = re.compile(r"^type\s+([A-Za-z_]\w*)\b")
_GO_VAR = re.compile(r"^(?:var|const)\s+([A-Za-z_]\w*)\s")


def _go_symbols(source: str) -> list[dict]:
    symbols: list[dict] = []
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _GO_FUNC.match(line):
            recv, name = m.group(1), m.group(2)
            if recv:
                symbols.append({"n": name, "k": "method", "l": i, "p": recv})
            else:
                symbols.append({"n": name, "k": "function", "l": i})
        elif m := _GO_TYPE.match(line):
            symbols.append({"n": m.group(1), "k": "class", "l": i})
        elif m := _GO_VAR.match(line):
            symbols.append({"n": m.group(1), "k": "const", "l": i})
    return symbols


_RS_VIS = r"(?:pub(?:\([^)]*\))?\s+)?"
_RS_IMPL = re.compile(r"^impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?([\w]+)")
_RS_FN = re.compile(rf"^(\s*){_RS_VIS}(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+([A-Za-z_]\w*)")
_RS_TYPE = re.compile(rf"^{_RS_VIS}(?:struct|enum|trait|union)\s+([A-Za-z_]\w*)")
_RS_CONST = re.compile(rf"^{_RS_VIS}(?:const|static)\s+([A-Za-z_]\w*)")


def _rust_symbols(source: str) -> list[dict]:
    symbols: list[dict] = []
    parent: str | None = None
    parent_close = 0
    depth = 0
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _RS_IMPL.match(line):
            parent, parent_close = m.group(1), depth
        elif m := _RS_TYPE.match(line):
            symbols.append({"n": m.group(1), "k": "class", "l": i})
        elif m := _RS_CONST.match(line):
            symbols.append({"n": m.group(1), "k": "const", "l": i})
        elif m := _RS_FN.match(line):
            if parent and m.group(1):
                symbols.append({"n": m.group(2), "k": "method", "l": i, "p": parent})
            else:
                symbols.append({"n": m.group(2), "k": "function", "l": i})
        depth += _brace_delta(line)
        if parent is not None and depth <= parent_close:
            parent = None
    return symbols


_JAVA_TYPE = re.compile(
    r"^\s*(?:(?:public|protected|private|final|abstract|static|sealed)\s+)*"
    r"(?:class|interface|enum|record)\s+([A-Za-z_]\w*)")
_JAVA_METHOD = re.compile(
    r"^\s+(?:(?:public|protected|private|static|final|synchronized|abstract|native|default)\s+)+"
    r"[\w<>\[\],.?\s]+?\s+(\w+)\s*\([^;]*$"
    r"|^\s+(?:(?:public|protected|private|static|final|synchronized|abstract|native|default)\s+)+"
    r"[\w<>\[\],.?\s]+?\s+(\w+)\s*\(")


def _java_symbols(source: str) -> list[dict]:
    symbols: list[dict] = []
    parent: str | None = None
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _JAVA_TYPE.match(line):
            symbols.append({"n": m.group(1), "k": "class", "l": i})
            if parent is None:  # methods attach to the file's outermost type
                parent = m.group(1)
        elif parent and (m := _JAVA_METHOD.match(line)):
            name = m.group(1) or m.group(2)
            if name != parent:  # constructors already read as the class itself
                symbols.append({"n": name, "k": "method", "l": i, "p": parent})
    return symbols


_RB_CLASS = re.compile(r"^\s*(?:class|module)\s+([A-Z][\w:]*)")
_RB_DEF = re.compile(r"^(\s*)def\s+(?:self\.)?([\w]+[?!]?)")


def _ruby_symbols(source: str) -> list[dict]:
    symbols: list[dict] = []
    parent: str | None = None
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _RB_CLASS.match(line):
            symbols.append({"n": m.group(1), "k": "class", "l": i})
            parent = m.group(1)
        elif m := _RB_DEF.match(line):
            if m.group(1) and parent:
                symbols.append({"n": m.group(2), "k": "method", "l": i, "p": parent})
            else:
                symbols.append({"n": m.group(2), "k": "function", "l": i})
    return symbols


_SYMBOL_EXTRACTORS = {
    ".py": _python_symbols,
    ".js": _js_symbols, ".jsx": _js_symbols, ".mjs": _js_symbols,
    ".cjs": _js_symbols, ".ts": _js_symbols, ".tsx": _js_symbols,
    ".go": _go_symbols,
    ".rs": _rust_symbols,
    ".java": _java_symbols,
    ".rb": _ruby_symbols,
}


# ---------------------------------------------------------------------------
# Import extraction (feeds the module dependency graph)
# ---------------------------------------------------------------------------

_PY_HANDLED_SEPARATELY = object()  # analyzer keeps its exact ast-based pass

_JS_IMPORT = re.compile(r"""(?:^\s*import\b[^'"]*|\brequire\s*\(\s*|^\s*export\b[^'"]*\bfrom\s*)["']([^"']+)["']""")
_GO_IMPORT = re.compile(r'^\s*(?:import\s+)?(?:\w+\s+)?"([\w./-]+)"')
_RS_USE = re.compile(r"^\s*(?:pub\s+)?use\s+([\w:]+)")
_JAVA_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)")
_RB_REQUIRE = re.compile(r"""^\s*require(?:_relative)?\s+["']([^"']+)["']""")


def extract_imports(rel: str, source: str) -> list[str]:
    """Imported module/path strings for non-Python files (Python keeps its exact
    ast pass in the analyzer). Used only for the module dependency graph, so a
    plain string per import is enough."""
    ext = _ext(rel)
    if ext in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".vue", ".svelte"):
        pat = _JS_IMPORT
    elif ext == ".go":
        out: list[str] = []
        in_block = False
        for line in source.splitlines():
            s = line.strip()
            if s.startswith("import ("):
                in_block = True
                continue
            if in_block and s.startswith(")"):
                in_block = False
                continue
            if (in_block or s.startswith("import")) and (m := _GO_IMPORT.match(line)):
                out.append(m.group(1))
        return out[:100]
    elif ext == ".rs":
        pat = _RS_USE
    elif ext == ".java":
        pat = _JAVA_IMPORT
    elif ext == ".rb":
        pat = _RB_REQUIRE
    else:
        return []
    return [m.group(1) for line in source.splitlines() if (m := pat.search(line))][:100]


# ---------------------------------------------------------------------------
# Edit-time parse gate (Dev loop rejects edits whose result no longer parses)
# ---------------------------------------------------------------------------

def syntax_error(rel: str, content: str) -> str | None:
    """Error string when `content` no longer parses; None = fine OR ungateable
    (no checker for this language / toolchain missing — fail open, the test run
    still catches real breakage later)."""
    ext = _ext(rel)
    if ext == ".py":
        try:
            ast.parse(content)
            return None
        except SyntaxError as exc:
            return f"line {exc.lineno}: {exc.msg}"
    if ext in (".js", ".mjs", ".cjs") and shutil.which("node"):
        return _node_check(content, ext)
    if ext == ".go" and shutil.which("gofmt"):
        try:
            p = subprocess.run(["gofmt", "-e"], input=content, capture_output=True,
                               text=True, timeout=15)
            if p.returncode != 0:
                first = (p.stderr or "").strip().splitlines()
                return re.sub(r"^<standard input>:", "line ", first[0])[:200] if first else "does not parse"
        except Exception:  # noqa: BLE001 — gate is best-effort
            return None
        return None
    # Everything else: tree-sitter when installed (gates TS/Rust/Java/Ruby, and
    # JS/Go when node/gofmt are missing); no tier available — fail open, the
    # test run still catches real breakage later.
    return _ts_gate(ext, content)


def _ts_gate(ext: str, content: str) -> str | None:
    """Parse-gate via tree-sitter; None = parses fine OR the tier can't judge
    (not installed / grammar missing / parser crash — fail open)."""
    ts_lang = _TS_LANG_BY_EXT.get(ext)
    parser = _ts_parser(ts_lang) if ts_lang else None
    if parser is None:
        return None
    try:
        tree = parser.parse(content.encode("utf-8"))
    except Exception:  # noqa: BLE001 — gate is best-effort
        return None
    node = _ts_first_error(tree.root_node)
    if node is None:
        return None
    return f"line {node.start_point[0] + 1}: does not parse ({ts_lang})"


def _ts_first_error(node):
    """Deepest-first ERROR/missing node, for a usable line number."""
    if node.type == "ERROR" or node.is_missing:
        return node
    if not node.has_error:
        return None
    for child in node.children:
        found = _ts_first_error(child)
        if found is not None:
            return found
    return node  # has_error with no localizable child — report the node itself


def _node_check(content: str, ext: str) -> str | None:
    """Parse-check JS via node using explicit temp files.

    A bare `.js` file may be either dialect (package.json "type" decides,
    and we don't know it here), so it is checked as CommonJS AND as ESM and
    rejected only when BOTH fail — valid in either dialect passes.
    """
    suffixes = {".cjs": (".cjs",), ".mjs": (".mjs",)}.get(ext, (".cjs", ".mjs"))
    last_err = "does not parse"
    for suffix in suffixes:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False,
                                             mode="w", encoding="utf-8") as temp:
                temp.write(content)
                temp_path = temp.name
            p = subprocess.run(["node", "--check", temp_path],
                               capture_output=True, text=True, timeout=15)
        except Exception:  # noqa: BLE001 — gate is best-effort
            return None
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        if p.returncode == 0:
            return None
        line = next((ln.strip() for ln in (p.stderr or "").splitlines()
                     if "Error" in ln or "SyntaxError" in ln), last_err)
        last_err = line[:200]
    return last_err


# ---------------------------------------------------------------------------
# Test ecosystems — detection + per-runner facts. Subprocess execution stays in
# git_ops; this layer only says WHAT to run and how to read the output.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Runner:
    kind: str                       # python | node | go | cargo | maven | gradle | rspec
    label: str                      # command string advertised to the Dev agent
    cmd: list[str]                  # full-suite command (python filled by caller)
    fail_re: re.Pattern | None      # per-test failure ids; None = can't baseline
    accepts_paths: bool = True      # can targeted test files be appended?
    setup: str = ""                 # env note for git_ops.ensure_test_env
    no_tests_markers: tuple[str, ...] = field(default_factory=tuple)


_PYTEST = Runner(
    kind="python", label="pytest -q", cmd=["-m", "pytest", "-q"],
    fail_re=re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE), setup="venv")
_NODE = Runner(
    kind="node", label="npm test --silent", cmd=["npm", "test", "--silent"],
    fail_re=None, accepts_paths=False, setup="npm",
    no_tests_markers=("missing script: \"test\"", "missing script: test",
                      "no test specified"))
_GO = Runner(
    kind="go", label="go test ./...", cmd=["go", "test", "./..."],
    fail_re=re.compile(r"^--- FAIL: (\S+)", re.MULTILINE),
    no_tests_markers=("no test files",))
_CARGO = Runner(
    kind="cargo", label="cargo test", cmd=["cargo", "test"],
    fail_re=re.compile(r"^test (\S+) \.\.\. FAILED", re.MULTILINE),
    accepts_paths=False)
# Surefire's [ERROR] lines vary too much across plugin versions to parse into
# stable per-test ids — no baselining (QA still reads the full output).
_MAVEN = Runner(
    kind="maven", label="mvn -q test", cmd=["mvn", "-q", "test"],
    fail_re=None, accepts_paths=False,
    no_tests_markers=("no tests to run",))
_GRADLE = Runner(
    kind="gradle", label="./gradlew test", cmd=["./gradlew", "test", "--console=plain"],
    fail_re=re.compile(r"^(\S+ > \S+) FAILED\s*$", re.MULTILINE),
    accepts_paths=False)
_RSPEC = Runner(
    kind="rspec", label="bundle exec rspec", cmd=["bundle", "exec", "rspec"],
    fail_re=re.compile(r"^rspec (\S+)", re.MULTILINE), accepts_paths=False,
    setup="bundle", no_tests_markers=("no examples found",))


_RUNNERS_BY_KIND = {r.kind: r for r in (_PYTEST, _NODE, _GO, _CARGO, _MAVEN, _GRADLE, _RSPEC)}


def detect_runner_by_kind(kind: str) -> Runner | None:
    return _RUNNERS_BY_KIND.get(kind)


def detect_runner(root: Path) -> Runner | None:
    """Which test ecosystem this repo uses. Python first (preserves existing
    behavior on mixed repos, e.g. a Go tool with a docs/ node project); then by
    manifest. None = unknown → tests report as inconclusive, never FAIL.

    A manifest alone is not evidence of an ecosystem — polyglot repos carry
    manifests for their *tooling*. gitea ships a pyproject.toml whose entire
    contents pin three Python linters (djlint, yamllint, zizmor), and has zero
    .py files against 3,024 .go files; on the manifest alone this returned
    `pytest -q`, so QA would have run pytest over a Go repo, found nothing, and
    reported INCONCLUSIVE for every delivery. So the Python branch also
    requires Python source to actually exist.
    """
    if (((root / "pyproject.toml").exists() or (root / "setup.py").exists()
            or (root / "setup.cfg").exists() or (root / "pytest.ini").exists()
            or (root / "tox.ini").exists() or list(root.glob("test_*.py"))
            or ((root / "tests").exists() and _dir_has(root, "*.py")))
            and _has_source(root, ".py")):
        return _PYTEST
    if (root / "go.mod").exists() and shutil.which("go"):
        return _GO
    if (root / "Cargo.toml").exists() and shutil.which("cargo"):
        return _CARGO
    if (root / "pom.xml").exists() and shutil.which("mvn"):
        return _MAVEN
    # Gradle only via the checked-in wrapper (standard practice) — the runner's
    # command is static, so a wrapper-less repo with only a system gradle would
    # advertise a command that can't run. Fail open to None instead.
    if (((root / "build.gradle").exists() or (root / "build.gradle.kts").exists())
            and (root / "gradlew").exists()):
        return _GRADLE
    if ((root / "spec").exists() and (root / "Gemfile").exists()
            and shutil.which("bundle")):
        return _RSPEC
    # A package.json is not a test suite. gitea's declares 106 dependencies and
    # no `test` script at all (its frontend tests run via `make test-frontend` →
    # vitest), so selecting npm here would run a command whose only possible
    # output is "missing script: test" — a runner that cannot pass or fail.
    # Returning None instead is the honest answer: QA reports INCONCLUSIVE,
    # which the pipeline already refuses to treat as a pass.
    if (root / "package.json").exists() and shutil.which("npm") and _npm_test_script(root):
        return _NODE
    return None


def _npm_test_script(root: Path) -> bool:
    try:
        manifest = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    script = (manifest.get("scripts") or {}).get("test") or ""
    # npm init writes a placeholder that exits 1 with "no test specified".
    return bool(script.strip()) and "no test specified" not in script


def _dir_has(root: Path, pattern: str) -> bool:
    try:
        next((root / "tests").rglob(pattern))
        return True
    except (StopIteration, OSError):
        return False


def _has_source(root: Path, ext: str, limit: int = 4000) -> bool:
    """Does this repo contain real source of `ext`? Walks os.walk pruned by
    SKIP_DIRS — vendor/ and node_modules/ carry other ecosystems' code and would
    otherwise vouch for an ecosystem the repo doesn't use. Bounded: this runs on
    the QA path, and one hit is all the answer needs."""
    seen = 0
    for _current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.endswith(ext):
                return True
            seen += 1
            if seen > limit:
                return False
    return False


def parse_failures(runner: Runner, output: str) -> set[str] | None:
    """Stable per-test failure ids from a full-suite run, for baseline diffing
    (pre-existing failures vs regressions). None = this runner's output has no
    reliably parseable ids — callers must skip baselining, not assume clean."""
    if runner.fail_re is None:
        return None
    return set(runner.fail_re.findall(output))


def no_tests_found(runner: Runner, returncode: int, output: str) -> bool:
    """True when the run means 'nothing to run' rather than pass/fail."""
    if runner.kind == "python" and returncode == 5:
        return True
    low = output.lower()
    return bool(runner.no_tests_markers) and any(m in low for m in runner.no_tests_markers)
