"""Walks a cloned repository and extracts raw facts (never knowledge).

Python files are parsed with the stdlib `ast` (top-level classes, functions,
imports); other languages go through the regex extractors in services/lang.py,
so every supported language now yields classes/functions/imports — not just a
path. A small regex pass finds HTTP endpoints (FastAPI/Flask decorators and
Express-style `app.get("/…")`). Output is `ExtractedFacts` — facts only.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

from .. import git_ops, lang
from .facts import Endpoint, ExtractedFacts, FileFacts

logger = logging.getLogger(__name__)

# Kept as aliases — language knowledge itself lives in services/lang.py.
_SKIP_DIRS = lang.SKIP_DIRS
_LANG_BY_EXT = lang.LANG_BY_EXT

_MAX_FILE_BYTES = 300_000

# @app.get("/path"), @router.post('/path')  (Python decorators) and
# app.get("/path", …), router.post('/path', …)  (Express/koa-router style).
# The leading-slash requirement keeps map.get("key")-style lookups out.
_ENDPOINT_RE = re.compile(
    r"""@?\w+\.(get|post|put|patch|delete|route)\(\s*["'](/[^"']*)["']""",
    re.IGNORECASE,
)
_METHODS_RE = re.compile(r"methods\s*=\s*\[([^\]]*)\]", re.IGNORECASE)


def analyze_repo(repo_url: str) -> ExtractedFacts:
    """Extract facts from the already-cloned working copy for `repo_url`."""
    root = git_ops.workdir(repo_url)
    facts = ExtractedFacts(root=str(root))
    if not (root / ".git").exists():
        return facts

    for rel in _iter_files(repo_url, root):
        ext = Path(rel).suffix.lower()
        language = _LANG_BY_EXT.get(ext, "")
        fp = root / rel
        try:
            if fp.stat().st_size > _MAX_FILE_BYTES:
                continue
            source = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if ext == ".py":
            facts.files.append(_analyze_python(rel, source))
        else:
            facts.files.append(_analyze_other(rel, source, language or "Other"))

    logger.info("knowledge.analyzer: %s — %d files", repo_url, len(facts.files))
    return facts


def _iter_files(repo_url: str, root: Path) -> list[str]:
    try:
        tracked = git_ops.list_files(str(root), limit=5000)
    except Exception:  # noqa: BLE001
        tracked = []
    out: list[str] = []
    for rel in tracked:
        p = Path(rel)
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in _LANG_BY_EXT:
            continue
        out.append(rel)
    return out


def _analyze_python(rel: str, source: str) -> FileFacts:
    classes: list[str] = []
    functions: list[str] = []
    imports: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return FileFacts(path=rel, language="Python")

    for node in tree.body:  # module-level only → top-level defs
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.Import):
            imports.append("import " + ", ".join(a.name for a in node.names))
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * (node.level or 0)) + (node.module or "")
            imports.append(f"from {mod} import " + ", ".join(a.name for a in node.names))

    return FileFacts(
        path=rel,
        language="Python",
        classes=classes,
        functions=functions,
        imports=imports,
        endpoints=_endpoints(rel, source),
    )


def _analyze_other(rel: str, source: str, language: str) -> FileFacts:
    """Non-Python facts via the lang registry's regex extractors. Languages
    without an extractor still get path + language recorded (fail open)."""
    symbols = lang.extract_symbols(rel, source)
    return FileFacts(
        path=rel,
        language=language,
        classes=[s["n"] for s in symbols if s["k"] == "class"],
        functions=[s["n"] for s in symbols if s["k"] == "function"],
        imports=lang.extract_imports(rel, source),
        endpoints=_endpoints(rel, source),
    )


def extract_symbols(rel: str, source: str) -> list[dict]:
    """Line-numbered symbol facts for one file — the raw material of the
    localization symbol map. Unlike the FileFacts passes (names only, for the
    LLM views), this keeps line numbers and descends into class bodies for
    methods: [{"n": name, "k": class|function|method|const, "l": line,
    "p": parent}]. Dispatches by extension via services/lang.py; returns []
    for unparseable source or unsupported languages."""
    return lang.extract_symbols(rel, source)


def analyzable(rel: str) -> bool:
    """Whether a repo-relative path is one the analyzer would record at all."""
    p = Path(rel)
    if any(part in _SKIP_DIRS for part in p.parts):
        return False
    return p.suffix.lower() in _LANG_BY_EXT


def language_of(rel: str) -> str:
    return lang.language_of(rel)


def _endpoints(rel: str, source: str) -> list[Endpoint]:
    endpoints: list[Endpoint] = []
    for line in source.splitlines():
        m = _ENDPOINT_RE.search(line)
        if not m:
            continue
        verb, path = m.group(1).lower(), m.group(2)
        if verb == "route":  # Flask: methods=[...], default GET
            mm = _METHODS_RE.search(line)
            methods = re.findall(r"['\"](\w+)['\"]", mm.group(1)) if mm else ["GET"]
            for method in methods:
                endpoints.append(Endpoint(method=method.upper(), path=path, file=rel))
        else:
            endpoints.append(Endpoint(method=verb.upper(), path=path, file=rel))
    return endpoints
