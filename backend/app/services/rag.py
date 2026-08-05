"""Repo ingestion + knowledge-base chat — the RAG entry point for the agents.

Everything here is local and self-contained: the code graph and symbol map built
by services/knowledge, queried in-process. There is no external knowledge
service to configure, run or pay for.
"""

import logging
import re

from sqlmodel import Session

from ..database import engine
from ..models import KBStatus, Repo, utcnow
from . import git_ops

logger = logging.getLogger(__name__)

_SOURCE_RE = re.compile(r"`([A-Za-z0-9_][\w./\-]*\.[A-Za-z0-9]+)`")


def ask(repo_url: str, messages: list[dict], *, timeout: int = 600) -> str:
    """Ask the repo's knowledge base a question. `messages` is [{role, content}]
    with roles 'user'/'assistant'. Returns the full answer text."""
    from .knowledge import retriever as knowledge_retriever

    return knowledge_retriever.answer(repo_url, messages, timeout=timeout)


def extract_sources(text: str, limit: int = 6) -> list[str]:
    """Pull cited repo file paths (backticked) out of a KB answer."""
    seen: list[str] = []
    for m in _SOURCE_RE.findall(text):
        if ("/" in m or "." in m) and m not in seen:
            seen.append(m)
    return seen[:limit]


def ingest(repo_id: int) -> None:
    """Clone a working copy + build the code graph. Runs on a background thread."""
    with Session(engine) as db:
        repo = db.get(Repo, repo_id)
        if repo is None:
            return
        repo_url = repo.git_url
        repo.kb_status = KBStatus.indexing.value
        repo.kb_progress = 5
        repo.kb_step = "Cloning working copy…"
        repo.kb_error = None
        db.add(repo)
        db.commit()

    try:
        path = git_ops.ensure_clone(repo_url)
        with Session(engine) as db:
            repo = db.get(Repo, repo_id)
            repo.kb_progress = 35
            repo.kb_step = "Analyzing structure (AST)…"
            db.add(repo)
            db.commit()

        knowledge_count = 0
        knowledge_views: dict = {}
        kb_tokens_in = kb_tokens_out = 0
        kb_cost = 0.0
        overview = ""
        # The knowledge build is deterministic and free: the code graph (AST of
        # every file → definitions, calls, routes; built by the
        # codebase-memory-mcp binary in seconds) plus the symbol-map fallback
        # tier. No LLM cost, no embedding service.
        from .knowledge import pipeline as knowledge_pipeline

        def _progress(step: str, pct: int) -> None:
            with Session(engine) as db2:
                r = db2.get(Repo, repo_id)
                if r is not None:
                    r.kb_step, r.kb_progress = step, pct
                    db2.add(r)
                    db2.commit()

        kr = knowledge_pipeline.generate(repo_url, progress=_progress)
        if kr.generated:
            knowledge_count = kr.doc_count
            knowledge_views = kr.counts
            if kr.overview:
                overview = kr.overview  # prefer the structured overview
            logger.info("Knowledge: %d graph nodes for %s", kr.doc_count, repo_url)
        elif kr.error:
            logger.info("Knowledge build skipped/failed for %s: %s", repo_url, kr.error)
        doc_count = git_ops.count_files(path)

        with Session(engine) as db:
            repo = db.get(Repo, repo_id)
            repo.kb_status = KBStatus.ready.value
            repo.kb_progress = 100
            # Name the dense channel explicitly. "Ready" said nothing about
            # whether semantic search actually got built, so a repo whose
            # embedding step failed looked identical to one where it worked —
            # which is how the graph._int regression stayed invisible.
            vectors = (knowledge_views or {}).get("vectors", 0)
            if knowledge_count:
                dense = (f", {vectors:,} vectors" if vectors
                         else ", no vectors (BM25-only)")
                repo.kb_step = (f"Ready — code graph: {knowledge_count} nodes "
                                f"over {doc_count} files{dense}")
            else:
                repo.kb_step = "Knowledge base ready"
            repo.kb_doc_count = doc_count
            repo.kb_knowledge_count = knowledge_count
            repo.kb_views = knowledge_views
            # Cumulative KB-build spend (rebuilds add to the one-time investment).
            repo.kb_tokens_in = (repo.kb_tokens_in or 0) + kb_tokens_in
            repo.kb_tokens_out = (repo.kb_tokens_out or 0) + kb_tokens_out
            repo.kb_cost_usd = round((repo.kb_cost_usd or 0.0) + kb_cost, 6)
            repo.kb_overview = (overview or "")[:1000]
            repo.last_indexed_at = utcnow()
            db.add(repo)
            db.commit()
        logger.info("Ingest complete for %s (%d files, %d views)",
                    repo_url, doc_count, knowledge_count)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ingest failed for %s: %s", repo_url, exc)
        with Session(engine) as db:
            repo = db.get(Repo, repo_id)
            repo.kb_status = KBStatus.failed.value
            repo.kb_error = str(exc)[:500]
            db.add(repo)
            db.commit()
