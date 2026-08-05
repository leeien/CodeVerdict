"""Gap log — signals that the KB lacked something a request needed.

Every PM scoping turn that had to run on-demand retrieval rounds is appended to
`<knowledge_dir>/<repo-slug>/gaps.jsonl` (survives full rebuilds). Zero rounds
means bootstrap knowledge was enough; repeated high-round entries for the same
area are the signal that a knowledge view is missing there — the data that a
future gap-driven view refresh (and the benchmark write-up) consumes.

Append-only JSONL, no LLM, never raises.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from ...config import settings
from .. import git_ops

logger = logging.getLogger(__name__)

_MAX_LINES = 500  # oldest entries dropped past this — it's a signal, not an archive


def _path(repo_url: str) -> Path:
    return Path(settings.knowledge_dir).resolve() / git_ops.slug(repo_url) / "gaps.jsonl"


def record(repo_url: str, query: str, rounds: int, retrieved: list[str] | None = None) -> None:
    """Log one PM turn's retrieval effort. `rounds` == 0 is logged too — the
    baseline 'KB was sufficient' rate is half the signal."""
    try:
        path = _path(repo_url)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "at": datetime.now(UTC).isoformat(),
            "query": (query or "")[:300],
            "rounds": int(rounds or 0),
            "retrieved": [str(r)[:120] for r in (retrieved or [])][:8],
        }
        lines = []
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()[-(_MAX_LINES - 1):]
        lines.append(json.dumps(entry, ensure_ascii=False))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 — telemetry must never break a scoping turn
        logger.exception("knowledge.gaps: failed to record for %s", repo_url)


def summary(repo_url: str) -> dict:
    """Aggregate view: turns logged, share needing retrieval, hottest lookups."""
    try:
        path = _path(repo_url)
        if not path.exists():
            return {"turns": 0}
        entries = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        needed = [e for e in entries if e.get("rounds", 0) > 0]
        hot: dict[str, int] = {}
        for e in needed:
            for r in e.get("retrieved", []):
                hot[r] = hot.get(r, 0) + 1
        return {
            "turns": len(entries),
            "turns_needing_retrieval": len(needed),
            "avg_rounds": round(sum(e.get("rounds", 0) for e in entries) / max(1, len(entries)), 2),
            "hottest_lookups": sorted(hot.items(), key=lambda kv: -kv[1])[:10],
        }
    except Exception:  # noqa: BLE001
        logger.exception("knowledge.gaps: summary failed for %s", repo_url)
        return {"turns": 0}
