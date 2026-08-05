"""Prompts for the review panel: one brief per juror, one charge to the foreperson.

Both stages ask for JSON. Prose reviews were what the single-reviewer pipeline
used, and they cost us the ability to do anything mechanical with a review —
you cannot dedupe, weight by confidence, or detect agreement across five prose
essays. The parser in ``panel.py`` is deliberately forgiving about what comes
back, because a juror that returns malformed JSON should abstain, not crash the
delivery.
"""

from __future__ import annotations

from ...config import settings
from ..prompts import clip

# Budgets. The diff is the expensive part and every juror pays for it, so it is
# capped harder here than in the single-reviewer prompt.
DIFF_CHARS = 11000
CONTEXT_CHARS = 4000
DEV_SUMMARY_CHARS = 1500
# The original request is the cheapest and highest-value block in the prompt —
# it is short, and it is the only thing in the case file that cannot be wrong.
REQUEST_CHARS = 3000
DESCRIPTION_CHARS = 2000
LOCALIZATION_CHARS = 1200


JUDGE_SYSTEM = (
    "You are one juror on a multi-perspective code review panel. Several other "
    "jurors are reviewing this same change from different angles, independently "
    "and in parallel; a foreperson will combine your opinions afterwards. Review "
    "ONLY from the perspective you are assigned — trust the other jurors to cover "
    "theirs, and do not pad your review with their concerns. Report what you can "
    "point at in the code. Respond with a single JSON object and nothing else."
)


_OUTPUT_CONTRACT = """\
Respond with ONE JSON object, no markdown fences, no prose outside it:

{
  "verdict": "APPROVE" | "REQUEST_CHANGES" | "ABSTAIN",
  "summary": "2-3 sentences: what you checked from your perspective and what you concluded.",
  "findings": [
    {
      "title": "short statement of the defect",
      "location": "path/to/file.py:123 (or the function name; \\"\\" if not pinpointable)",
      "evidence": "the specific code or diff line that shows it — quote it",
      "why_it_matters": "the concrete consequence: what breaks, for whom, when",
      "severity": "critical" | "high" | "medium" | "low",
      "confidence": 0.0-1.0,
      "suggestion": "the specific change that would resolve it"
    }
  ]
}

Rules for the fields:
- "verdict" is REQUEST_CHANGES only if at least one of your findings is a real
  defect that must be fixed before merge. Preferences, polish and "would be
  nicer" are findings at severity low — they do NOT justify REQUEST_CHANGES.
  Use ABSTAIN if this change contains nothing in your area of responsibility.
- "confidence" is your honest probability that the finding is real and not a
  misreading of code you can only partly see. Below 0.5 means you are guessing;
  guesses are filtered out by the foreperson, so mark them honestly rather than
  inflating them.
- "severity" is about impact if you are right, independent of confidence.
- Every finding needs "evidence" you can quote from the diff or the provided
  file content. A finding you cannot ground is not a finding — drop it.
- An empty "findings" list with "verdict": "APPROVE" is a perfectly good review.
  Do NOT invent issues to look thorough; every false finding costs a real,
  paid revision round and trains the pipeline to ignore you."""


def judge_user(charge: str, task_key: str, title: str, criteria: list[str], diff: str,
               context: str = "", impact: str = "", dev_summary: str = "",
               test_output: str = "", request: str = "", description: str = "",
               localization: str = "", alerts: str = "", evidence: str = "") -> str:
    """One juror's full brief: their charge, then the case file every juror sees.

    The ORIGINAL request leads, deliberately. Acceptance criteria are a PM's
    lossy restatement written before anyone looked at the code, and a panel given
    only the criteria can only check the change against that restatement — it
    cannot notice that the restatement itself missed the point. We watched a
    whole panel approve a delivery that satisfied the criteria on paper and did
    nothing at all about the user's actual bug. So the jurors get what the human
    said first, and the derived requirement second.
    """
    return judge_case(task_key, title, criteria, diff, context, impact, dev_summary,
                      test_output, request, description, localization, alerts) \
        + judge_charge(charge, evidence)


def judge_case(task_key: str, title: str, criteria: list[str], diff: str,
               context: str = "", impact: str = "", dev_summary: str = "",
               test_output: str = "", request: str = "", description: str = "",
               localization: str = "", alerts: str = "") -> str:
    """The case file every juror sees — byte-identical across the panel.

    Split out from the per-juror charge so it can be built ONCE and handed to
    every juror as a shared prompt *prefix*. That ordering is the whole point:
    prompt caching — explicit on Anthropic, automatic on OpenAI/Groq/Gemini/
    DeepSeek — only ever discounts a matching prefix. With the charge first the
    prompts diverged at character zero and ~6k tokens of identical case file
    were paid for at full price by every juror, every round.
    """
    crit = "\n".join(f"- {c}" for c in criteria) or "- (no explicit criteria — judge general fitness)"
    blocks = ["=" * 70, "", f"CASE {task_key}: {title}", ""]
    if request.strip():
        blocks += ["What the user actually asked for, in their own words — this is the "
                   "problem that has to be SOLVED, and it outranks every restatement below:",
                   '"""', request.strip()[:REQUEST_CHARS], '"""', ""]
    blocks += ["The requirement as the PM restated it (derived from the above before anyone "
               "read the code — it may have narrowed, widened or simply misread the request; "
               "if it conflicts with what the user asked for, say so):", crit]
    if description.strip():
        blocks += ["", "The PM's notes for this ticket — behaviour detail from the "
                       "conversation with the human. The PM never read the code, so treat "
                       "anything structural here as commentary, not fact:",
                   description.strip()[:DESCRIPTION_CHARS]]
    if localization.strip():
        blocks += ["", localization.strip()[:LOCALIZATION_CHARS]]
    if context.strip():
        blocks += ["", "Repository knowledge and existing architecture (retrieved for this "
                       "change; may be incomplete — prefer the code itself where they disagree):",
                   context.strip()[:CONTEXT_CHARS]]
    if impact.strip():
        blocks += ["", impact.strip()]
    if dev_summary.strip():
        blocks += ["", "What the implementing agent says it did (its own account — verify it "
                       "against the diff rather than believing it):",
                   dev_summary.strip()[:DEV_SUMMARY_CHARS]]
    if test_output.strip():
        blocks += ["", "Test run output:", "```", clip(test_output.strip(), 2500, 'test output'), "```"]
    if alerts.strip():
        blocks += ["", alerts.strip()]
    blocks += ["", "The implementation under review:", "```diff", clip(diff, DIFF_CHARS), "```"]
    return "\n".join(blocks)


def judge_charge(charge: str, evidence: str = "") -> str:
    """The per-juror tail: who this juror is, what it alone was given, and the
    output contract.

    Deliberately last. The charge is the instruction the juror must act on, and
    the reachback offer sits immediately before the output contract so it is in
    view exactly when the juror decides whether it can support the finding it is
    about to make. Everything above this point is identical panel-wide, which is
    what makes it cacheable.
    """
    from ..knowledge import tools as kb_tools

    blocks = ["", "=" * 70, ""]
    # Evidence gathered specifically for THIS juror's perspective (see
    # jury/evidence.py) — the neighbours to compare against, the reachability of
    # the change, the tests that already exist. Deterministic, from the code
    # graph, so a finding in this juror's own area can be grounded in it.
    if evidence.strip():
        blocks += [evidence.strip(), ""]
    blocks += ["YOUR ASSIGNMENT ON THIS PANEL — review the case above from this "
               "perspective and no other:", "", charge]
    if settings.jury_tool_calls > 0:
        blocks += ["", kb_tools.evidence_block()]
    blocks += ["", _OUTPUT_CONTRACT]
    return "\n".join(blocks)


FOREPERSON_SYSTEM = (
    "You are the foreperson of a code review panel. Several jurors reviewed the "
    "same change independently from different perspectives; their opinions are "
    "below. Your job is NOT to review the code again — it is to weigh what the "
    "jurors said and return a single decision the pipeline can act on. Respond "
    "with a single JSON object and nothing else."
)


_FOREPERSON_CONTRACT = """\
Respond with ONE JSON object, no markdown fences, no prose outside it:

{
  "verdict": "APPROVED" | "CHANGES REQUESTED",
  "rationale": "2-4 sentences explaining the decision and how you weighed disagreement.",
  "blocking": [
    {
      "title": "...", "location": "...", "why_it_matters": "...",
      "severity": "critical" | "high" | "medium",
      "confidence": 0.0-1.0,
      "suggestion": "the specific fix",
      "raised_by": ["juror name", ...],
      "agreement": "unanimous" | "majority" | "single" | "contested"
    }
  ],
  "observations": [ { "title": "...", "why_it_matters": "...", "raised_by": ["..."] } ],
  "dismissed": [ { "title": "...", "raised_by": ["..."], "reason": "why this does not block" } ]
}

How to decide:
1. MERGE duplicates. Two jurors describing the same defect in different words is
   ONE finding — list both in "raised_by" and mark the agreement level. Independent
   corroboration is the strongest signal this panel produces; surface it.
2. RESOLVE conflicts. When jurors disagree about the same code, prefer the one
   with specific quoted evidence over the one reasoning from general principle,
   and prefer the juror whose assigned perspective actually owns the question.
   Say in "rationale" which way you went and why. Mark it "contested".
3. DROP low-confidence and unsupported claims into "dismissed" with a reason —
   including anything a juror raised outside their assigned perspective, and
   anything whose evidence does not actually show what it claims.
4. Only "critical"/"high"/"medium" defects belong in "blocking", and "medium"
   blocks ONLY when the change is behaviourally wrong or incomplete — something
   a user or caller would observe. Style, taste, refactor ideas, type-annotation
   nits, "for consistency with the rest of the class", speculative hardening and
   nice-to-have tests go to "observations" no matter how many jurors mentioned
   them or how senior the juror sounds. A juror labelling polish as "medium"
   does not make it blocking; re-classify it.
5. REQUIREMENT DRIFT IS A REAL DEFECT, not an unsupported claim. If a juror
   shows the change satisfies the criteria but not what the user asked for, that
   BLOCKS — the criteria are a paraphrase and the paraphrase is what failed.
   Judge it against the user's own words above, and do not dismiss it on the
   grounds that it "contradicts the documented requirement": that contradiction
   is the finding. This is the one case where a single juror, unsupported by the
   others, is enough — the other jurors were checking the same paraphrase and
   would agree with each other for the same wrong reason.
6. "verdict" is CHANGES REQUESTED if and only if "blocking" is non-empty.

Calibration matters: every CHANGES REQUESTED sends the change back for a full
paid Dev + QA + Review round. A panel that blocks on polish is worse than no
panel. An empty "blocking" list with a few observations is the normal outcome
for competent work."""


def foreperson_user(task_key: str, title: str, criteria: list[str], opinions: str,
                    min_confidence: float, abstained: str = "", request: str = "") -> str:
    crit = "\n".join(f"- {c}" for c in criteria) or "- (no explicit criteria)"
    tail = (f"\n\nJurors who could not return an opinion (their perspective went "
            f"UNREVIEWED — say so in the rationale): {abstained}") if abstained else ""
    # The jurors get the original request; the foreperson used not to, and so had
    # no way to check a juror's appeal to user intent. Observed live: a juror
    # correctly reported that already-linked issues were still shown when the user
    # had twice written "it should only show issues that aren't already
    # dependencies", and the foreperson dismissed it as "unsupported by evidence"
    # and "contradicting the documented requirement" — because the only
    # requirement it could see was the PM's paraphrase, which is what drifted.
    asked = (f'''
What the user actually asked for, in their own words:
"""
{request.strip()[:REQUEST_CHARS]}
"""
The criteria below are a PM's paraphrase of that, written before anyone read the
code. When a juror says the change misses what the USER asked, check it against
this — do not dismiss it merely because it conflicts with the paraphrase.
''' if request.strip() else "")
    return f"""CASE {task_key}: {title}
{asked}
The requirement the change was accepted against:
{crit}

Confidence floor for this panel: {min_confidence:.2f}. Findings below it must be
dismissed, not promoted — however senior the juror sounds.

Juror opinions:
{opinions}{tail}

{_FOREPERSON_CONTRACT}"""
