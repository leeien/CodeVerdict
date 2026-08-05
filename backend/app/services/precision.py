"""Precision retrieval — pull the RIGHT knowledge, not more.

Instead of dumping a broad KB blob into agent prompts, return a ranked,
token-budgeted slice with the exact source files to touch. Served by the
built-in local RAG by default (services/local_rag.py); an external "Deep
Analysis" (AST+LLM) service can be enabled via use_deep_analysis for richer,
concept-level cards. Returns "" when unavailable (caller then falls back).
"""

from __future__ import annotations

import logging

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


def _latest_done_job() -> str | None:
    """Newest completed Deep Analysis job (the repo it auto-analyzed on boot)."""
    try:
        d = httpx.get(f"{settings.functional_analysis_url}/api/latest", timeout=8).json()
        if d.get("status") == "done" and d.get("id"):
            return d["id"]
    except Exception as exc:  # noqa: BLE001
        logger.debug("precision: /api/latest failed: %s", exc)
    return None


def retrieve(query: str, use_case: str = "task-breakdown", budget: int | None = None,
             repo_url: str | None = None, plan_symbols: list[str] | None = None) -> str:
    """Return a compact, use-case-scoped context slice (or "" if unavailable).

    use_case ∈ story-breakdown | task-breakdown | architecture | ui-test |
    unit-test | design | default — each has its own token budget.

    `plan_symbols` are a Planner's verified targets when one has run; they steer
    the reranker so the slice leads with the code the plan commits to.
    """
    if not settings.precision_retrieval or not query.strip():
        return ""

    # Default path: the retrieval pipeline — fused hybrid search, call-graph
    # expansion, lexical refinement, rerank, plus exact code hits and the
    # relevant cross-run delivery notes for this repo.
    if not settings.use_deep_analysis:
        if not repo_url:
            return ""
        from .knowledge import retriever as knowledge_retriever

        return knowledge_retriever.scope_context(repo_url, query,
                                                 plan_symbols=plan_symbols)

    # Optional external Deep Analysis service.
    job = _latest_done_job()
    if not job:
        return ""
    body: dict = {"query": query.strip()[:500], "use_case": use_case}
    if budget:
        body["budget"] = budget
    try:
        pack = httpx.post(
            f"{settings.functional_analysis_url}/api/jobs/{job}/search", json=body, timeout=60
        ).json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("precision: search failed: %s", exc)
        return ""

    cards = pack.get("cards") or []
    if not cards:
        return ""
    lines = [f"Relevant code knowledge ({use_case}, ~{pack.get('total_tokens', '?')} tokens, "
             f"ranked most-relevant first):"]
    for rc in cards:
        c = rc.get("card") or {}
        title = c.get("title", "")
        ctype = c.get("type", "")
        summary = (c.get("summary") or c.get("description") or "").strip()
        files = ", ".join(sorted({s.get("file", "") for s in (c.get("sources") or []) if s.get("file")}))
        line = f"- [{ctype}] {title}"
        if summary:
            line += f" — {summary[:300]}"
        if files:
            line += f"  (files: {files})"
        lines.append(line)
    return "\n".join(lines)
