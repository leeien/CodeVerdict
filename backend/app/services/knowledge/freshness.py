"""Keeps the knowledge base current as the repo evolves — free, so it always
is. Called at pipeline-run entry (right after the clone is reset to origin's
default branch, so the working tree == origin/HEAD).

Both knowledge layers are deterministic and cheap, so freshness is simple:

  * **Symbol map** (fallback localization tier): re-analyzed for changed files
    only, every time it drifts.
  * **Code graph** (knowledge/graph.py): reindexed whenever the working copy's
    SHA differs from the graph's watermark. The binary's RAM-first pipeline
    indexes typical repos in seconds, so a plain reindex beats incremental
    bookkeeping.

Delivery notes recorded before their branch merged are also reconciled here
(write_back.reconcile_unmerged). Never raises — a knowledge refresh must not
be able to break a pipeline run.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ...config import settings
from .. import git_ops
from . import embed, graph, symbol_map, write_back

logger = logging.getLogger(__name__)


def _meta_path(repo_url: str) -> Path:
    return Path(settings.knowledge_dir).resolve() / git_ops.slug(repo_url) / "meta.json"


def read_meta(repo_url: str) -> dict:
    try:
        return json.loads(_meta_path(repo_url).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_meta(repo_url: str, **fields) -> None:
    meta = read_meta(repo_url)
    meta.update(fields)
    fp = _meta_path(repo_url)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _log(on_event, level: str, msg: str) -> None:
    logger.info("knowledge.freshness: %s", msg)
    if on_event:
        try:
            on_event(level, msg)
        except Exception:  # noqa: BLE001
            pass


def _bar(pct: int, width: int = 24) -> str:
    filled = int(width * max(0, min(100, pct)) / 100)
    return "█" * filled + "░" * (width - filled)


def _embed_reporter(on_event, *, step: int = 5, min_gap: float = 3.0):
    """A throttled progress line for the re-embed.

    Re-embedding is the one stage of a sync with no output of its own, and it is
    also the longest — minutes on a large repo, with the model loaded and the CPU
    pinned. Silence of that shape is indistinguishable from a hang, and reads as
    one: the run looks wedged right after "code graph reindexed".

    Throttled on a percentage step *and* a floor on the interval, because every
    line this emits is a row in the run log.
    """
    # -inf, not 0.0: time.monotonic() counts from an arbitrary origin, which on
    # Linux is boot. Against a 0.0 sentinel the first line's interval check reads
    # "has the machine been up longer than min_gap" — so on a fresh CI runner the
    # opening 0% was swallowed and on a laptop it was not.
    state = {"pct": -step, "at": float("-inf")}

    def _report(done: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, int(done * 100 / total))
        now = time.monotonic()
        if pct < 100 and (pct < state["pct"] + step or now - state["at"] < min_gap):
            return
        state["pct"], state["at"] = pct, now
        _log(on_event, "info",
             f"KB: re-embedding  {_bar(pct)} {pct:3d}%   {done:,}/{total:,} nodes")

    return _report


def refresh_if_stale(repo_url: str, on_event=None) -> dict:
    """Bring the knowledge layers up to origin's default branch. Cheap when
    fresh (one rev-parse). Returns a summary dict; never raises."""
    try:
        return _refresh(repo_url, on_event)
    except Exception as exc:  # noqa: BLE001
        logger.exception("knowledge.freshness failed for %s", repo_url)
        return {"action": "error", "error": str(exc)[:200]}


def _refresh(repo_url: str, on_event) -> dict:
    if not settings.kb_auto_refresh:
        return {"action": "disabled"}
    path = git_ops.workdir(repo_url)
    if not (path / ".git").exists():
        return {"action": "no_clone"}
    head = git_ops.rev_parse(str(path), f"origin/{git_ops.default_branch(str(path))}")
    if not head:
        return {"action": "no_head"}

    actions: list[str] = []

    # --- Layer 1: symbol map (fallback tier, incremental) --------------------
    smap = symbol_map.load(repo_url)
    if smap is None:
        smap = symbol_map.build(repo_url, head)
        _log(on_event, "info",
             f"KB: symbol map built — {len(smap.files)} files, {smap.symbol_count()} symbols")
        actions.append("symbol_map_built")
    elif smap.sha != head:
        delta = git_ops.changed_files(str(path), smap.sha, head)
        if delta is None:
            smap = symbol_map.build(repo_url, head)
        else:
            smap = symbol_map.update(repo_url, delta[0], delta[1], head)
        _log(on_event, "info", f"KB: symbol map synced to {head[:9]}")
        actions.append("symbol_map_synced")

    # --- Layer 2: code graph (reindex on drift — seconds, $0) ----------------
    if graph.available():
        before = graph.indexed_sha(repo_url)
        if before != head:
            if graph.ensure_indexed(repo_url, head):
                _log(on_event, "info", f"KB: code graph reindexed at {head[:9]}")
                actions.append("graph_reindexed")
                # Dense index rides on the graph's nodes — rebuild it whenever
                # the graph moves (CPU-only, no API; skipped when the feature is
                # off or the stack isn't installed).
                if embed.available():
                    # Announced before it starts, not after it finishes: the model
                    # has to load before the first batch reports, so there is a
                    # silent head on this stage no progress line can cover.
                    _log(on_event, "info",
                         "KB: re-embedding changed nodes — the slow part of a sync")
                    n = embed.build(repo_url,
                                    on_progress=_embed_reporter(on_event))
                    if n:
                        _log(on_event, "info", f"KB: re-embedded {n} nodes")
                        actions.append("re_embedded")
                    else:
                        # build() swallows its own failures (low RAM, a locked
                        # store, the timeout) and returns 0. Saying so beats
                        # letting the announcement above be the last word.
                        _log(on_event, "warn",
                             "KB: re-embed produced nothing — semantic search will "
                             "answer from the previous index until the next sync")
                        actions.append("re_embed_failed")
            else:
                _log(on_event, "warn", "KB: code graph reindex failed — "
                                       "falling back to symbol map for this run")
                actions.append("graph_failed")

    # Delivery notes recorded before their branch merged carry a "do not assume
    # this exists" disclaimer; upgrade any whose work has since landed on the
    # default branch (deterministic, one merge-base check each — never raises).
    upgraded = write_back.reconcile_unmerged(repo_url, str(path), on_event)
    if upgraded:
        actions.append(f"notes_reconciled_{upgraded}")

    write_meta(repo_url, views_sha=head)
    return {"action": ",".join(actions) or "fresh", "sha": head}
