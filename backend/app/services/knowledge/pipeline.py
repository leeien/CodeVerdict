"""End-to-end knowledge build for one repository.

    clone → graph index (codebase-memory-mcp, seconds) → symbol map → meta

Called from ingest on a background thread. Reports progress through a callback
so the UI's `kb_step` reflects each phase. The build is fully deterministic —
no LLM, no embedding service, no per-repo cost: the graph binary parses the
AST of every file (158 languages) into a persistent SQLite knowledge graph,
and the symbol map is kept as the fallback localization tier for when the
binary is unavailable. Cross-run knowledge (delivery notes / lessons written
by write_back.py) lives in the JSON store and is never touched by a rebuild.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .. import git_ops
from . import analyzer, embed, graph, retriever, symbol_map

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeResult:
    generated: bool = False
    doc_count: int = 0
    overview: str = ""
    counts: dict = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    error: str | None = None


def enabled() -> bool:
    """The knowledge build is deterministic (graph + symbol map) — always
    runnable. Kept as a function because callers gate on it."""
    return True


def generate(repo_url: str, progress=None) -> KnowledgeResult:
    """Build (or rebuild) the knowledge layers for a repo. Never raises —
    returns a KnowledgeResult with `error` set on failure."""
    try:
        if progress:
            progress("Indexing code graph…", 55)
        sha = git_ops.rev_parse(str(git_ops.workdir(repo_url)))
        graph_info: dict = {}
        if graph.available():
            if graph.ensure_indexed(repo_url, sha):
                graph_info = graph.architecture(repo_url)
            else:
                logger.warning("knowledge.pipeline: graph index failed for %s "
                               "(falling back to symbol map only)", repo_url)
        else:
            logger.info("knowledge.pipeline: graph binary unavailable — "
                        "symbol-map tier only for %s", repo_url)

        if progress:
            progress("Building symbol map…", 80)
        facts = analyzer.analyze_repo(repo_url)
        if facts.files:
            symbol_map.build(repo_url, sha)
        elif not graph_info:
            return KnowledgeResult(error="no analyzable files")

        # Local dense-embedding index over the graph's nodes (optional, free,
        # CPU) — the real semantic-recall channel fused with BM25 at query time.
        vectors = 0
        if graph_info and embed.available():
            if progress:
                progress("Embedding code graph nodes…", 85)

            # Embedding is the long pole (~20 min on a big repo), so it owns a
            # real slice of the bar — 85→99 — instead of one static tick. The
            # node count is the only honest denominator we have.
            def _embed_progress(done: int, total: int) -> None:
                if progress and total:
                    pct = 85 + int(14 * min(done, total) / total)
                    progress(f"Embedding code graph nodes… {done:,}/{total:,}", pct)

            vectors = embed.build(repo_url, on_progress=_embed_progress)
            if vectors:
                logger.info("knowledge.pipeline: embedded %d nodes for %s", vectors, repo_url)
            else:
                # A zero here is not "nothing to embed" — the graph just told us
                # it has nodes. It means the build failed, and it used to fail
                # silently: `graph._int` was renamed out from under `_nodes`, the
                # AttributeError died inside the build subprocess, and retrieval
                # ran BM25-only for months with nothing in any log to say so.
                logger.warning(
                    "knowledge.pipeline: dense index EMPTY for %s despite %s graph "
                    "nodes — semantic recall is off, retrieval is BM25-only",
                    repo_url, graph_info.get("total_nodes"))
        elif graph_info and not embed.available():
            logger.info("knowledge.pipeline: semantic extras not installed — "
                        "BM25-only retrieval for %s", repo_url)

        from . import freshness
        freshness.write_meta(repo_url, views_sha=sha,
                             last_full_rebuild_at=datetime.now(UTC).isoformat())

        nodes = int(graph_info.get("total_nodes") or 0)
        edges = int(graph_info.get("total_edges") or 0)
        counts = {"graph_nodes": nodes, "graph_edges": edges,
                  "symbol_map_files": len(facts.files), "vectors": vectors}
        result = KnowledgeResult(
            generated=True, doc_count=nodes, overview=retriever.overview(repo_url),
            counts=counts,
        )
        logger.info("knowledge.pipeline: %s — %d graph nodes / %d edges, "
                    "%d files in symbol map, %d vectors (deterministic, $0)",
                    repo_url, nodes, edges, len(facts.files), vectors)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("knowledge.pipeline failed for %s", repo_url)
        return KnowledgeResult(error=str(exc)[:300])
