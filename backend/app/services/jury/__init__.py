"""The review jury: an ensemble of specialized judges instead of one reviewer.

A single reviewer model evaluating a single coding model's output has a known
shape of failure. It is confident, it is fast, and it is blind in exactly the
places the author was blind — same training distribution, same idea of what
"looks right", same tendency to accept a plausible-looking diff that never wires
the feature through. Adding a second pass from the same model does not fix this;
it produces the same opinion twice.

So the review stage is a panel:

    personas.py   — the briefs. Each juror is told what to look for AND what to
                    leave to the others, because uncorrelated reviewers are the
                    entire mechanism.
    panel.py      — polls the jurors in parallel, independently. They never see
                    one another's opinions. Failures abstain loudly.
    synthesis.py  — the foreperson merges duplicates, resolves conflicts, drops
                    low-confidence guesses, and decides.
    ../judges.py  — the roster: which judges are seated, in what order, on which
                    models. Operator-editable at runtime.

``review()`` below is the whole public surface. It returns the same shape the
old single-reviewer path returned (text + usage + error) plus the structured
decision, so ``orchestrator`` can swap between them on one setting.
"""

from __future__ import annotations

import logging

from sqlmodel import Session

from ...config import settings
from ...database import engine
from .. import judges as roster
from .. import providers
from . import panel, personas, prompts, synthesis

logger = logging.getLogger(__name__)

__all__ = ["review", "enabled", "panel", "personas", "prompts", "synthesis"]


def enabled() -> bool:
    """Whether the jury runs at all. Off = the classic single-reviewer path."""
    return bool(settings.jury_enabled)


def seated() -> int:
    """How many judges would be polled right now (for logs and cost warnings)."""
    with Session(engine) as db:
        return len(roster.enabled_judges(db))


def review(task_key: str, title: str, criteria: list[str], diff: str, *, workdir: str = "",
           context: str = "", impact: str = "", dev_summary: str = "", test_output: str = "",
           request: str = "", description: str = "", localization: str = "",
           on_event=None, on_opinion=None) -> dict:
    """Run the full panel over one change.

    ``on_opinion(opinion)`` is called once per juror as the panel finishes, so
    the caller can bill each judge to its own run row. Returns::

        {text, decision, opinions, tokens_in, tokens_out, cost, error}

    ``text`` is the rendered prose review (ending in a VERDICT line, as the rest
    of the pipeline expects) and ``decision["verdict"]`` is the authoritative
    machine-readable outcome: APPROVED, CHANGES REQUESTED, or INCONCLUSIVE.
    """
    with Session(engine) as db:
        roster.ensure_seeded(db)
        judge_rows = roster.enabled_judges(db)
        # Detach: the panel runs across threads and must not share a Session.
        for j in judge_rows:
            db.expunge(j)

    if not judge_rows:
        # An empty panel is a configuration mistake, not an approval.
        decision = {
            "verdict": "INCONCLUSIVE",
            "rationale": "No judges are enabled on the review jury, so this change has NOT "
                         "been code-reviewed. Seat at least one judge under Settings → Jury.",
            "blocking": [], "observations": [], "dismissed": [], "synthesis": "none",
        }
        if on_event:
            on_event("error", "Jury has no enabled judges — delivery is UNREVIEWED")
        return {"text": synthesis.render(decision, []), "decision": decision, "opinions": [],
                "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
                "error": "no judges enabled"}

    case = {"task_key": task_key, "title": title, "criteria": criteria, "diff": diff,
            "context": context, "impact": impact, "dev_summary": dev_summary,
            "test_output": test_output, "request": request, "description": description,
            "localization": localization,
            # Computed, not asked for: a rewritten assertion is a fact about the
            # diff, and every juror should see it before it forms an opinion.
            "alerts": panel.tampering_brief(diff)}
    opinions = panel.empanel(judge_rows, case, workdir=workdir, on_event=on_event)
    if on_opinion:
        for op in opinions:
            on_opinion(op)

    decision = synthesis.deliberate(opinions, task_key, title, criteria,
                                    on_event=on_event, request=request)
    # Every juror's own opinion IN FULL, carried on the decision and persisted
    # with it. The synthesis is the decision, but it is not the evidence: a
    # reader has to be able to see what each judge actually said, including the
    # findings the foreperson merged away or dismissed, and including that (say)
    # the security perspective abstained on a delivery that was still approved.
    decision["jurors"] = [
        {"name": o.name, "persona": o.persona, "provider": o.provider, "model": o.model,
         "model_label": providers.label(o.provider, o.model), "verdict": o.verdict,
         "summary": o.summary, "findings": o.findings, "error": o.error,
         "tokens_in": o.tokens_in, "tokens_out": o.tokens_out, "cost": o.cost}
        for o in opinions
    ]
    text = synthesis.render(decision, opinions)

    if on_event:
        agreed = len(decision.get("blocking") or [])
        if decision["verdict"] == "INCONCLUSIVE":
            on_event("error", "Jury INCONCLUSIVE — this delivery has NOT been code-reviewed")
        else:
            on_event("success" if decision["verdict"] == "APPROVED" else "warn",
                     f"Jury verdict: {decision['verdict']} "
                     f"({agreed} blocking, {len(decision.get('observations') or [])} observations, "
                     f"{len(decision.get('dismissed') or [])} dismissed)")
        on_event("info", text)

    # The jurors' usage is billed to their own runs by the caller; only the
    # foreperson's usage is reported here so nothing is double-counted.
    return {
        "text": text,
        "decision": decision,
        "opinions": [o.as_dict() for o in opinions],
        "tokens_in": decision.get("tokens_in", 0),
        "tokens_out": decision.get("tokens_out", 0),
        "cost": decision.get("cost", 0.0),
        "error": None if decision["verdict"] != "INCONCLUSIVE" else decision["rationale"][:300],
    }
