"""Per-repo symbol map — the localization layer of the knowledge base.

A deterministic, line-numbered index of every symbol the analyzer can see
(classes, methods, functions, module constants), persisted as
`<knowledge_dir>/<repo-slug>/symbols.json` next to the knowledge docs. Zero LLM
cost to build or update, so it can always be kept exactly current with the repo
(updated incrementally per changed file by knowledge/freshness.py).

Consumers:
  * pm_agent.ground_tickets — instant definition lookup instead of one
    ripgrep subprocess per symbol, plus "did you mean" suggestions.
  * orchestrator._verified_locations — exact file:line pins and per-file
    outlines injected into the Dev prompt, so Dev reads targeted slices
    instead of exploring.

The map is only trusted when its `built_at_sha` matches the working copy the
caller is on; callers fall back to live ripgrep otherwise.
"""

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path

from ...config import settings
from .. import git_ops
from . import analyzer

logger = logging.getLogger(__name__)

_KIND_LABEL = {"class": "class", "function": "def", "method": "def", "const": "="}


def _path(repo_url: str) -> Path:
    return Path(settings.knowledge_dir).resolve() / git_ops.slug(repo_url) / "symbols.json"


class SymbolMap:
    """In-memory view over symbols.json with an inverted name index."""

    def __init__(self, sha: str, files: dict[str, dict]):
        self.sha = sha
        self.files = files  # rel path -> {"language": str, "symbols": [{n,k,l,p?}]}
        self._defs: dict[str, list[tuple[str, dict]]] = {}
        for rel, info in files.items():
            for sym in info.get("symbols", []):
                self._defs.setdefault(sym["n"], []).append((rel, sym))

    def lookup(self, name: str) -> list[tuple[str, dict]]:
        """[(file, {n,k,l,p?})] everywhere `name` is defined (empty if nowhere)."""
        return list(self._defs.get(name, []))

    def suggest(self, name: str, n: int = 3) -> list[str]:
        """Closest existing symbol names — for PM-invented identifiers."""
        return difflib.get_close_matches(name, self._defs.keys(), n=n, cutoff=0.75)

    def outline(self, rel: str, max_symbols: int = 40) -> str:
        """One-line-per-symbol outline of a file: 'class Foo (L10)' etc."""
        info = self.files.get(rel)
        if not info or not info.get("symbols"):
            return ""
        rows: list[str] = []
        for sym in info["symbols"][:max_symbols]:
            label = _KIND_LABEL.get(sym["k"], sym["k"])
            name = f"{sym['p']}.{sym['n']}" if sym.get("p") else sym["n"]
            rows.append(f"{label} {name} (L{sym['l']})")
        more = len(info["symbols"]) - max_symbols
        if more > 0:
            rows.append(f"… +{more} more")
        return ", ".join(rows)

    def symbol_count(self) -> int:
        return sum(len(i.get("symbols", [])) for i in self.files.values())


def load(repo_url: str) -> SymbolMap | None:
    return _load_file(_path(repo_url))


def load_slug(slug: str) -> SymbolMap | None:
    """Load by repo slug — the workdir's directory name IS the slug (both are
    produced by git_ops.slug), so callers holding only a clone path can use
    `load_slug(Path(path).name)`."""
    return _load_file(Path(settings.knowledge_dir).resolve() / slug / "symbols.json")


def _load_file(fp: Path) -> SymbolMap | None:
    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
        return SymbolMap(raw.get("built_at_sha", ""), raw.get("files", {}))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _save(repo_url: str, sha: str, files: dict[str, dict]) -> None:
    fp = _path(repo_url)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps({"built_at_sha": sha, "files": files}), encoding="utf-8")


def _analyze_one(root: Path, rel: str) -> dict | None:
    """Symbol facts for one file in the working tree (None = skip/gone)."""
    if not analyzer.analyzable(rel):
        return None
    fp = root / rel
    try:
        source = fp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lang = analyzer.language_of(rel)
    symbols = analyzer.extract_symbols(rel, source)
    return {"language": lang, "symbols": symbols}


def build(repo_url: str, sha: str) -> SymbolMap:
    """Full build from the current working tree. Deterministic and cheap
    (pure AST — ~seconds even on large repos), so safe to call freely."""
    root = git_ops.workdir(repo_url)
    files: dict[str, dict] = {}
    for rel in git_ops.list_files(str(root), limit=10000):
        info = _analyze_one(root, rel)
        if info is not None:
            files[rel] = info
    _save(repo_url, sha, files)
    logger.info("symbol_map: built %s — %d files, %d symbols @ %.9s",
                repo_url, len(files), sum(len(i["symbols"]) for i in files.values()), sha)
    return SymbolMap(sha, files)


def update(repo_url: str, changed: list[str], deleted: list[str], sha: str) -> SymbolMap:
    """Incremental update: re-analyze only `changed`, drop `deleted`, stamp the
    new sha. Falls back to a full build when there's no existing map."""
    existing = load(repo_url)
    if existing is None:
        return build(repo_url, sha)
    root = git_ops.workdir(repo_url)
    files = dict(existing.files)
    for rel in deleted:
        files.pop(rel, None)
    for rel in changed:
        info = _analyze_one(root, rel)
        if info is not None:
            files[rel] = info
        else:
            files.pop(rel, None)
    _save(repo_url, sha, files)
    logger.info("symbol_map: updated %s — %d changed, %d deleted @ %.9s",
                repo_url, len(changed), len(deleted), sha)
    return SymbolMap(sha, files)
