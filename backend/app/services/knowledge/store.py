"""Reads and writes the knowledge JSON files that hold cross-run knowledge —
the accumulating delivery notes and distilled lessons written by write_back.py.
The files are the source of truth; ranking (retriever.notes) reads them
directly, so there is no separate index/vector store to keep in sync.

Layout under `settings.knowledge_dir/<repo-slug>/`:
    index.json
    functional/deliveries/<id>.json
    functional/lessons/<id>.json
    symbols.json      (symbol-map fallback tier)
    graph_sha         (code-graph watermark)
    meta.json         (freshness watermark)
    gaps.jsonl        (PM retrieval-gap telemetry)
"""

from __future__ import annotations

import json
from pathlib import Path

from ...config import settings
from .. import git_ops
from .facts import KnowledgeDocument

_SUBDIR = {"delivery_note": "functional/deliveries", "lesson": "functional/lessons"}


def _base(repo_url: str) -> Path:
    return Path(settings.knowledge_dir).resolve() / git_ops.slug(repo_url)


def _relative_path(doc: KnowledgeDocument) -> str:
    subdir = _SUBDIR.get(doc.type, "functional/misc")
    return f"{subdir}/{doc.id}.json"


def _write_index(repo_url: str, entries: list[dict]) -> None:
    base = _base(repo_url)
    base.mkdir(parents=True, exist_ok=True)
    (base / "index.json").write_text(
        json.dumps({"documents": entries}, indent=2), encoding="utf-8"
    )


def save(repo_url: str, doc: KnowledgeDocument) -> None:
    """Upsert ONE document without touching the rest of the store."""
    base = _base(repo_url)
    rel = _relative_path(doc)
    full = base / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(doc.to_dict(), indent=2), encoding="utf-8")
    entries = [e for e in load_index(repo_url) if e["id"] != doc.id]
    entries.append({"id": doc.id, "type": doc.type, "name": doc.name, "path": rel})
    _write_index(repo_url, entries)


def remove(repo_url: str, doc_id: str) -> None:
    base = _base(repo_url)
    entries = load_index(repo_url)
    for e in entries:
        if e["id"] == doc_id:
            try:
                (base / e["path"]).unlink(missing_ok=True)
            except OSError:
                pass
    _write_index(repo_url, [e for e in entries if e["id"] != doc_id])


def load_index(repo_url: str) -> list[dict]:
    index_file = _base(repo_url) / "index.json"
    if not index_file.exists():
        return []
    try:
        return json.loads(index_file.read_text(encoding="utf-8")).get("documents", [])
    except (json.JSONDecodeError, OSError):
        return []


def load(repo_url: str, doc_id: str) -> KnowledgeDocument | None:
    for entry in load_index(repo_url):
        if entry["id"] == doc_id:
            try:
                raw = (_base(repo_url) / entry["path"]).read_text(encoding="utf-8")
            except OSError:
                return None
            return KnowledgeDocument.from_dict(json.loads(raw))
    return None


def load_all(repo_url: str) -> list[KnowledgeDocument]:
    docs = [load(repo_url, e["id"]) for e in load_index(repo_url)]
    return [d for d in docs if d is not None]


def has_knowledge(repo_url: str) -> bool:
    return bool(load_index(repo_url))
