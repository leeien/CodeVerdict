"""Write-back: persist what a delivered scope LEARNED into the knowledge base.

Every pipeline run discovers validated, expensive knowledge — the real wiring
points, repo conventions, what QA/Review flagged — that used to be thrown away,
leaving run #50 as ignorant as run #1. After a scope delivers, this module
synthesizes a compact `delivery_note` knowledge doc and indexes it in the
`deliveries` domain, so future PM retrieval and Dev KB context start from what
previous runs already proved.

Factual fields (files touched, symbols added, verdicts) are computed from the
git diff — deterministic, can't hallucinate. Only the short prose (summary,
gotchas, wiring notes) uses one cheap knowledge_model call, with a
deterministic fallback when no LLM is available. Notes live in the JSON store
and are ranked in-process at query time (retriever.notes) — no vector index
to maintain. Never raises.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

from ...config import settings
from .. import git_ops, llm, providers
from . import facts_view, store
from .facts import KnowledgeDocument

logger = logging.getLogger(__name__)

_ADDED_SYMBOL_RE = re.compile(r"^\+\s*(?:async\s+)?(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
# INCONCLUSIVE = the QA/Review agent could not run at all (stamped by the
# orchestrator) — recorded as-is so the note never implies a verified delivery.
_VERDICT_RE = re.compile(r"VERDICT\s*:?\s*(PASS|CONCERNS|FAIL|INCONCLUSIVE)", re.IGNORECASE)

_SYNTH_SYSTEM = (
    "You are a senior engineer writing a short implementation note after a code "
    "change was delivered, for future engineers (and agents) working on the same "
    "repository. From the material provided, respond ONLY as JSON: "
    '{"summary": string (2-3 sentences: what was implemented and HOW it is wired '
    "through the codebase), "
    '"gotchas": [string] (0-4 non-obvious pitfalls, conventions, or pre-existing '
    "issues discovered — only ones future work on this repo benefits from), "
    '"wiring": [string] (0-4 notes of the form "X in file Y is where Z happens" '
    "capturing WHERE key things connect)}. Be concrete, cite file paths, never "
    "invent facts not present in the material."
)


def _load_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        if text and "{" in text:
            try:
                return json.loads(text[text.find("{"): text.rfind("}") + 1])
            except (json.JSONDecodeError, ValueError):
                pass
    return {}


def _str_list(data: dict, key: str, cap: int) -> list[str]:
    v = data.get(key)
    return [str(x)[:300] for x in v[:cap]] if isinstance(v, list) else []


def _review_verdict(review_text: str) -> str:
    up = (review_text or "").upper()
    if "VERDICT: INCONCLUSIVE" in up:
        return "INCONCLUSIVE"
    if "CHANGES REQUESTED" in up:
        return "CHANGES REQUESTED"
    return "APPROVED" if "APPROVED" in up else ""


def record_delivery(repo_url: str, key: str, title: str, path: str, branch: str,
                    dev_summary: str = "", qa_text: str = "", review_text: str = "",
                    pr_url: str = "", on_event=None) -> dict:
    """Synthesize + persist a delivery note for the scope on `branch`.
    Returns a small summary dict; never raises."""
    try:
        return _record(repo_url, key, title, path, branch, dev_summary,
                       qa_text, review_text, pr_url, on_event)
    except Exception as exc:  # noqa: BLE001
        logger.exception("knowledge.write_back failed for %s", key)
        return {"written": False, "error": str(exc)[:200]}


def _record(repo_url: str, key: str, title: str, path: str, branch: str,
            dev_summary: str, qa_text: str, review_text: str,
            pr_url: str, on_event) -> dict:
    if not settings.kb_write_back:
        return {"written": False, "error": "disabled"}
    files = git_ops.diff_names(path, ref=branch)
    if not files:
        return {"written": False, "error": "empty diff"}
    diff = git_ops.diff(path, ref=branch)
    symbols = list(dict.fromkeys(
        f"{name} ({kind})" for kind, name in _ADDED_SYMBOL_RE.findall(diff)))[:20]

    qa_match = _VERDICT_RE.search(qa_text or "")
    qa_verdict = qa_match.group(1).upper() if qa_match else ""
    review_verdict = _review_verdict(review_text)
    summary, gotchas, wiring, cost = _synthesize(
        title, dev_summary, qa_text, review_text, files)
    if qa_verdict == "INCONCLUSIVE" or review_verdict == "INCONCLUSIVE":
        gotchas = ["This delivery was NEVER verified: QA/Review could not run "
                   "(agent failure) — treat its correctness as unconfirmed."] + gotchas

    # A "delivered" scope is NOT the same as "part of the code future runs
    # start from" — under DEMO_MODE (or a real PR nobody merged yet) the
    # branch never reaches origin's default branch. Writing that back as
    # confident, HIGH-confidence knowledge poisons future PM scoping: it will
    # believe unmerged work already exists (observed: scope-4's --no-color
    # flag, never merged, made a later scope assume the flag was already
    # there, which cost a full extra Dev exploration pass to un-learn).
    merged = git_ops.is_merged(path, branch)
    if not merged:
        disclaimer = (f"[NOT MERGED — exists only on branch '{branch}', not in the default "
                      f"branch yet; do not assume this code is present until it lands there] ")
        summary = disclaimer + summary
        gotchas = [f"This change is NOT merged into the default branch yet ({branch} only)."] + gotchas

    now = datetime.now(UTC).isoformat()
    doc = KnowledgeDocument(
        id=f"delivery_{re.sub(r'[^a-z0-9]+', '_', key.lower()).strip('_')}",
        type="delivery_note",
        name=("[UNMERGED] " if not merged else "") + f"Delivered: {title[:70]}",
        summary=summary,
        tags=["delivery", key] + ([] if merged else ["unmerged"]),
        # Link to the touched modules' docs so retrieval expansion connects
        # past deliveries with current module knowledge (and vice versa).
        related=list(dict.fromkeys(
            re.sub(r"[^a-z0-9]+", "_", facts_view.module_of(f).lower()).strip("_")
            for f in files))[:6],
        content={
            "files": files[:20], "symbols": symbols, "gotchas": gotchas,
            "wiring": wiring, "qa_verdict": qa_verdict,
            "review_verdict": review_verdict,
            "branch": branch, "pr_url": pr_url, "delivered_at": now, "merged": merged,
            # Tip sha survives branch deletion (a merged PR's branch usually
            # gets deleted) — reconcile_unmerged() checks ancestry against it.
            "tip_sha": git_ops.rev_parse(path, branch),
        },
        # Only trust this as ground truth once it's actually reachable from
        # the default branch — otherwise it's a plan/reference, not fact.
        confidence="HIGH" if merged else "LOW",
    )
    store.save(repo_url, doc)
    _prune(repo_url)
    msg = (f"KB write-back: delivery note stored ({len(files)} files, "
           f"{len(symbols)} new symbols)")
    logger.info("knowledge.write_back: %s — %s", key, msg)
    if on_event:
        try:
            on_event("success", msg)
        except Exception:  # noqa: BLE001
            pass
    return {"written": True, "doc_id": doc.id, "files": len(files), "cost": cost}


def _synthesize(title: str, dev_summary: str, qa_text: str, review_text: str,
                files: list[str]) -> tuple[str, list[str], list[str], float]:
    """(summary, gotchas, wiring, cost) — one cheap LLM call, deterministic
    fallback when unavailable."""
    fallback = (f"Implemented '{title[:120]}' across {len(files)} file(s): "
                f"{', '.join(files[:6])}.")
    if not providers.can_chat(settings.knowledge_provider):
        return fallback, [], [], 0.0
    user = (
        f"Change delivered: {title}\n\nFiles touched:\n"
        + "\n".join(f"- {f}" for f in files[:20])
        + f"\n\nDev agent's own summary:\n{(dev_summary or '(none)')[:2500]}"
        + f"\n\nQA findings:\n{(qa_text or '(none)')[:2000]}"
        + f"\n\nReview findings:\n{(review_text or '(none)')[:2000]}"
    )
    r = llm.chat(_SYNTH_SYSTEM, user, provider=settings.knowledge_provider,
                 model=settings.knowledge_model, json_mode=True, timeout=120)
    data = _load_json(r.get("text", ""))
    summary = str(data.get("summary") or "").strip() or fallback
    return (summary[:600], _str_list(data, "gotchas", 4), _str_list(data, "wiring", 4),
            r.get("cost", 0.0) or 0.0)


def _prune(repo_url: str) -> None:
    """Keep the newest N delivery notes; retire the rest from store + index.
    Before deletion, retiring notes are DISTILLED into per-module lesson docs
    (see _consolidate) so pruning drops the paperwork, not the learning."""
    notes = [d for d in store.load_all(repo_url) if d.type == "delivery_note"]
    if len(notes) <= settings.kb_delivery_notes_max:
        return
    notes.sort(key=lambda d: d.content.get("delivered_at", ""), reverse=True)
    retired = notes[settings.kb_delivery_notes_max:]
    if settings.kb_consolidate:
        try:
            _consolidate(repo_url, retired)
        except Exception:  # noqa: BLE001 — consolidation must never block pruning
            logger.exception("knowledge.write_back: consolidation failed for %s", repo_url)
    for doc in retired:
        store.remove(repo_url, doc.id)


_CONSOL_SYSTEM = (
    "You maintain a software repository's knowledge base. You are given the "
    "existing durable lessons for one module plus delivery notes about to be "
    "retired. Merge them into ONE updated lesson list. Respond ONLY as JSON: "
    '{"lessons": [string]} — at most 8 entries, deduplicated, each a concrete, '
    "durable fact, pitfall, or wiring note (cite file paths where present) that "
    "future work on this module benefits from. Drop anything transient: ticket "
    "ids, delivery status, one-off circumstances."
)


def _consolidate(repo_url: str, retiring: list[KnowledgeDocument]) -> None:
    """Fold retiring delivery notes' gotchas/wiring into per-module `lesson`
    docs (id ``lessons_<module>``). Lesson docs survive full rebuilds, so the
    KB keeps getting *smarter* with every request without accumulating one
    note per delivery forever. One cheap LLM call per affected module, with a
    deterministic dedupe fallback."""
    by_module: dict[str, list[str]] = {}
    for note in retiring:
        if not note.content.get("merged"):
            continue  # unmerged work never became ground truth — don't distill it
        items = [str(g) for g in (note.content.get("gotchas") or [])
                 if not _UNMERGED_GOTCHA_RE.match(str(g))]
        items += [str(w) for w in (note.content.get("wiring") or [])]
        if not items:
            continue
        for module in (note.related or ["repository"])[:3]:
            by_module.setdefault(module, []).extend(items)

    now = datetime.now(UTC).isoformat()
    for module, items in by_module.items():
        doc_id = f"lessons_{re.sub(r'[^a-z0-9]+', '_', module.lower()).strip('_')}"
        existing = store.load(repo_url, doc_id)
        old = [str(x) for x in ((existing.content.get("lessons") if existing else None) or [])]

        merged = _merge_lessons(module, old, items)
        doc = KnowledgeDocument(
            id=doc_id, type="lesson", name=f"Lessons: {module}",
            summary=f"Durable lessons distilled from delivered work touching '{module}' "
                    f"({len(merged)} entries, updated {now[:10]}).",
            tags=["lessons", module],
            related=[re.sub(r"[^a-z0-9]+", "_", module.lower()).strip("_")],
            content={"lessons": merged, "updated_at": now},
            confidence="MEDIUM",
        )
        store.save(repo_url, doc)
        logger.info("knowledge.write_back: consolidated %d retiring note item(s) into %s",
                    len(items), doc_id)


def _merge_lessons(module: str, old: list[str], new: list[str]) -> list[str]:
    """LLM-merge old + new lessons (dedup, drop transient); deterministic
    order-preserving dedupe capped at 8 when no LLM is available."""
    fallback = list(dict.fromkeys(old + new))[:8]
    if not providers.can_chat(settings.knowledge_provider):
        return fallback
    user = (f"Module: {module}\n\nExisting lessons:\n"
            + ("\n".join(f"- {x}" for x in old) or "(none)")
            + "\n\nItems from retiring delivery notes:\n"
            + "\n".join(f"- {x}" for x in new[:24]))
    r = llm.chat(_CONSOL_SYSTEM, user, provider=settings.knowledge_provider,
                 model=settings.knowledge_model, json_mode=True, timeout=120)
    data = _load_json(r.get("text", ""))
    merged = [str(x)[:300] for x in data.get("lessons", []) if str(x).strip()][:8]
    return merged or fallback


_UNMERGED_SUMMARY_RE = re.compile(r"^\[NOT MERGED[^\]]*\]\s*")
_UNMERGED_GOTCHA_RE = re.compile(r"^This (change|delivery) is NOT merged", re.IGNORECASE)


def reconcile_unmerged(repo_url: str, path: str, on_event=None) -> int:
    """Upgrade delivery notes recorded as [UNMERGED] once their work has actually
    landed on origin's default branch. Without this the merge-check is one-way:
    a note written before its PR merged would carry the 'do not assume this
    exists' disclaimer (and LOW confidence) forever, making the PM re-verify —
    or worse, re-plan — work that IS in the code now. Called at pipeline-run
    entry (freshness time); one merge-base check per unmerged note, no LLM.
    Returns the number upgraded; never raises."""
    try:
        upgraded = 0
        for doc in store.load_all(repo_url):
            if doc.type != "delivery_note" or doc.content.get("merged"):
                continue
            # Prefer the recorded tip sha (survives post-merge branch deletion);
            # fall back to the branch name for notes written before tip_sha existed.
            ref = doc.content.get("tip_sha") or doc.content.get("branch") or ""
            if not ref or not git_ops.is_merged(path, ref):
                continue
            doc.name = doc.name.removeprefix("[UNMERGED] ")
            doc.summary = _UNMERGED_SUMMARY_RE.sub("", doc.summary or "")
            doc.tags = [t for t in doc.tags if t != "unmerged"]
            doc.content["gotchas"] = [g for g in (doc.content.get("gotchas") or [])
                                      if not _UNMERGED_GOTCHA_RE.match(str(g))]
            doc.content["merged"] = True
            doc.confidence = "HIGH"
            store.save(repo_url, doc)
            upgraded += 1
            msg = f"KB: delivery note '{doc.id}' merged into default branch — upgraded to trusted knowledge"
            logger.info("knowledge.write_back: %s", msg)
            if on_event:
                try:
                    on_event("success", msg)
                except Exception:  # noqa: BLE001
                    pass
        return upgraded
    except Exception:  # noqa: BLE001
        logger.exception("knowledge.reconcile_unmerged failed for %s", repo_url)
        return 0
