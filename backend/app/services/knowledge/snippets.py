"""Snippet preparation — turning ranked nodes into context an agent can use.

A ranked list of `file:line` pins tells an agent where to look. It does not tell
it what is there, so the agent spends a tool call (and a large fraction of the
run's input tokens) reading each one back. Attaching the source closes that
loop — but attaching *all* of it reopens a worse one: RepoGraph's measurement is
that a flattened subgraph large enough to be complete is large enough to drown
the model, while the same subgraph summarized fits and improves the result.

So each node takes one of two forms, decided by size:

  * small enough → its real source, verbatim. Nothing beats the actual lines.
  * too large    → a one-paragraph summary from the knowledge-stage model,
    cached on disk against the graph's commit so it is written once per node
    per commit, not once per retrieval.

The whole set is bounded by a character budget, spent best-first, because the
caller's prompt has a hard ceiling and the alternative to a budget is a
truncation that lands mid-function.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from ...config import settings
from . import graph

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = (
    "You summarize one unit of source code for an engineer who is about to modify "
    "code near it. In 3-4 sentences state: what it does, its inputs and outputs, "
    "any side effect or state it touches, and the one thing that would break if it "
    "changed. Name the identifiers involved. No preamble, no restating the question."
)


def _cache_path(repo_url: str, qualified_name: str, sha: str) -> Path:
    key = hashlib.sha256(f"{qualified_name}@{sha}".encode()).hexdigest()[:32]
    return (Path(settings.knowledge_dir).resolve() / graph.project(repo_url)
            / "summaries" / f"{key}.txt")


def _cached(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _store(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass


def summarize(repo_url: str, qualified_name: str, source: str) -> str:
    """One cached LLM summary of an oversized node. Returns "" when the model
    can't run — the caller then falls back to truncated source, which is worse
    but honest, rather than to nothing."""
    sha = graph.indexed_sha(repo_url) or "nosha"
    cache = _cache_path(repo_url, qualified_name, sha)
    hit = _cached(cache)
    if hit:
        return hit
    from .. import llm, providers

    if not providers.can_chat(settings.knowledge_provider):
        return ""
    r = llm.chat(_SUMMARY_SYSTEM, f"```\n{source[:24000]}\n```",
                 provider=settings.knowledge_provider,
                 model=settings.knowledge_model, timeout=90)
    text = (r.get("text") or "").strip()
    if text and not r.get("error"):
        _store(cache, text)
    return text


def _render(hit: dict, body: str, kind: str) -> str:
    loc = f"{hit.get('file_path')}:{hit.get('start_line') or '?'}"
    head = f"--- {hit.get('label', '')} {hit.get('name')} — {loc}"
    if hit.get("via"):
        head += f"  (reached via {hit['via']})"
    return f"{head} [{kind}] ---\n{body}"


def prepare(repo_url: str, hits: list[dict], *, budget: int | None = None,
            max_nodes: int = 6) -> str:
    """Source (or summaries) for the top-ranked nodes, within a char budget.

    Only nodes the graph can quote are included: a hit without a qualified name
    came from a channel that knows a location but not a node, and inventing a
    line range for it would be guessing at content — the one thing this layer
    exists to avoid.
    """
    if not settings.snippet_context or not graph.available() or not hits:
        return ""
    remaining = budget if budget is not None else settings.snippet_budget_chars
    threshold = settings.snippet_max_chars
    blocks: list[str] = []
    for hit in hits[:max_nodes]:
        if remaining <= 200:
            break
        qn = hit.get("qualified_name")
        if not qn:
            continue
        try:
            source = (graph.snippet(repo_url, str(qn)) or {}).get("source") or ""
        except Exception:  # noqa: BLE001 — a missing snippet is not a failed retrieval
            continue
        if not source.strip():
            continue
        if len(source) <= threshold:
            body, kind = source, "source"
        else:
            summary = summarize(repo_url, str(qn), source) if settings.snippet_summarize else ""
            if summary:
                body, kind = summary, "summary — too large to inline"
            else:
                body = source[:threshold] + "\n… (truncated — read the file for the rest)"
                kind = "source, truncated"
        block = _render(hit, body[:remaining], kind)
        blocks.append(block)
        remaining -= len(block)
    if not blocks:
        return ""
    return ("Source of the top-ranked locations (already read for you — do NOT spend a "
            "tool call re-reading these):\n\n" + "\n\n".join(blocks))
