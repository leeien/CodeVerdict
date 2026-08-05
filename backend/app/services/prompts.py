"""Prompt builders for the agents."""

from ..config import settings


def pm_tasks(conversation: str, repo_name: str) -> str:
    return f"""You are the Product Manager agent for an autonomous SDLC pipeline, working in the repository `{repo_name}`.

Read the repository to understand its structure, then turn the conversation below into concrete, implementable engineering tasks (stories).

<conversation>
{conversation}
</conversation>

Output ONLY a JSON array (no prose, no markdown fences) of 1-4 tasks. Each task:
{{"title": str, "description": str, "acceptance_criteria": [str, ...], "priority": "low"|"medium"|"high"}}

SCOPE MINIMALISM — every task and every criterion is a paid Dev+QA+Review cycle:
- Default to ONE task. Split only when the request truly contains independent
  deliverables a reviewer would want to approve separately (a fix and its tests
  are ONE task, not two; implementation and wiring are ONE task).
- Criteria must restate the USER'S observable ask and nothing more — do not
  invent adjacent requirements, extra edge cases, or nice-to-have hardening the
  user didn't imply. 3-6 criteria across the whole scope is the norm.
- The Dev implements exactly what the criteria say and the Reviewer enforces
  them, so an invented criterion becomes real paid work and a real rejection
  ground. When in doubt, leave it out and note it in the description instead.

Keep tasks small, specific, and grounded in the actual code you see. acceptance_criteria
describe observable behavior only (given X, the user sees Y) — put file/symbol specifics in
the description, never in a criterion (a wrong structural claim there becomes an
unsatisfiable requirement the reviewer enforces)."""


def _hints_block(affected_files: list[str] | None, target_symbols: list[str] | None,
                 tools_cmd: str = "", has_plan: bool = False) -> str:
    """Bare file/symbol pins, for a run with no plan.

    When a Planner ran, this is redundant — its block carries the same pins with
    the reasoning, the blast radius and the tests attached, and repeating them
    here without that context invites the agent to treat a verified location as
    another guess to second-guess."""
    if has_plan or (not affected_files and not target_symbols):
        return ""
    verify = (f"VERIFY each hint before editing (`{tools_cmd} lookup <Symbol>`, "
              f"`{tools_cmd} search \"<mechanism>\"`)"
              if tools_cmd else "VERIFY each hint (search for the symbol / read the file) before editing")
    lines = ["Localization hints — a HYPOTHESIS, not facts. No planner ran on this",
             "delivery, so these come from summaries and may name the wrong file or",
             f"invent symbols. {verify}. You are EXPECTED to disagree when the evidence",
             "says otherwise: if the behavior lives somewhere else, implement it THERE and",
             "state in your summary which hint was wrong and what the real location is.",
             "Implementing a change in the wrong file because the hint said so is a failed",
             "work order."]
    if affected_files:
        lines.append("  files: " + ", ".join(affected_files[:8]))
    if target_symbols:
        lines.append("  symbols: " + ", ".join(target_symbols[:12]))
    return "\n" + "\n".join(lines) + "\n"


def _relocalize_hint(tools_cmd: str = "") -> str:
    """What to do when a pin is missing or the hints look wrong — query the
    repository index if it's installed, else fall back to one grep."""
    if not tools_cmd:
        return "At most ONE extra grep if a pin is missing."
    return (f"When a pin is missing — or the code you find doesn't actually produce the "
            f"behavior in the work order — query the index (`{tools_cmd} search`, "
            f"`{tools_cmd} lookup`, `{tools_cmd} callers`) instead of exploring. It is "
            f"cheaper than reading files and it is the right way to overrule a wrong hint.")


def _test_block(test_cmd: str = "") -> str:
    if "pytest" in test_cmd:
        return (f"Run the tests you add or touch with:  {test_cmd} <test files>\n"
                "(that interpreter has this repo installed; plain `python` does not). "
                "Iterate until they pass.")
    if test_cmd:
        return (f"Run the repo's tests with:  {test_cmd}\n"
                "— use exactly this command. Iterate until they pass.")
    return "Add or update tests where appropriate and run them if a test runner is available."


def dev(task_key: str, title: str, description: str, criteria: list[str], context: str = "",
        affected_files: list[str] | None = None, target_symbols: list[str] | None = None,
        test_cmd: str = "", verified: str = "", tools_cmd: str = "", plan: str = "") -> str:
    crit = "\n".join(f"- {c}" for c in criteria) or "- (use your judgment)"
    ctx = (f"\nRepository knowledge (retrieved, may be incomplete — verify against the code):\n"
           f"{context.strip()[:6000]}\n" if context.strip() else "")
    ver = f"\n{verified.strip()}\n" if verified.strip() else ""
    # The plan goes ahead of the retrieved knowledge: it is the decided approach,
    # and the rest is supporting material for carrying it out.
    plan_block = f"\n{plan.strip()}\n" if plan.strip() else ""
    described = ("Description (context for the plan above; any file or line number named "
                 "here is prose, not a verified pin — the plan's pins are the verified ones):"
                 if plan.strip() else
                 "Description (WHAT to achieve is binding; any file, symbol or line number "
                 "named in here is a guess from summaries and is NOT — verify it like the "
                 "hints above, and if the behaviour lives elsewhere, implement it there and "
                 "say so):")
    from .knowledge import tools as _tools
    # The tools block goes FIRST, ahead of the (large) plan, retrieved knowledge
    # and verified-locations dumps. Live run: with it placed after them — tens of
    # thousands of characters in — the Dev agent never once invoked the tools and
    # fell back to its own `grep`, which sees text but not call structure.
    return f"""You are the Dev agent in an autonomous SDLC pipeline. Implement this work order by editing files in the current repository working copy (a dedicated branch — edits are expected). Your cwd IS the repo root: use relative paths everywhere.
{_tools.prompt_block(tools_cmd)}{plan_block}{ctx}{ver}{_hints_block(affected_files, target_symbols, tools_cmd, has_plan=bool(plan.strip()))}
Work order {task_key}: {title}
{described}
{description}
Acceptance criteria:
{crit}

How to work — BE TOKEN-EFFICIENT, every extra read costs real money:
1. START from the plan and the verified locations above (computed against this exact working copy). If full file contents are included there, that IS the current file — do not re-read it with a tool call, edit directly from what's shown. For anything not already shown, go straight to file:line pins with targeted reads (offset/limit around the pin). Do NOT explore with find/ls or read whole large files when a pinned region or included content already answers the question. {_relocalize_hint(tools_cmd)}
2. Make the smallest correct, surgical change. DIFF DISCIPLINE: touch only lines the work order requires. Do NOT refactor, reformat, merge statements, rewrite type hints, rename, or otherwise "improve" surrounding code — even when you see something better. Every changed line outside the work order is scope creep the reviewer will flag and a human must re-read. If you notice a worthwhile unrelated improvement, mention it in your summary instead of making it.
3. Wire everything END-TO-END: a new flag/option must be registered where the others are, threaded through to the code that consumes it, and observable in behavior. Before finishing, check `git diff` — every acceptance criterion actually connected, not just defined.
4. {_test_block(test_cmd)}
5. Do NOT commit, push, or create branches — just edit files; the pipeline commits.

When done, summarize in a few sentences: what changed, in which files, and how you verified it."""


def revise(task_key: str, title: str, criteria: list[str], review_text: str,
           qa_text: str, context: str = "", test_cmd: str = "", verified: str = "",
           tools_cmd: str = "", plan: str = "") -> str:
    """Dev prompt for a revision round: fix the already-committed change to
    address reviewer + QA feedback."""
    crit = "\n".join(f"- {c}" for c in criteria) or "- (use your judgment)"
    ctx = (f"\nRepository knowledge (retrieved, may be incomplete — verify against the code):\n"
           f"{context.strip()[:5000]}\n" if context.strip() else "")
    ver = f"\n{verified.strip()}\n" if verified.strip() else ""
    plan_block = f"\n{plan.strip()}\n" if plan.strip() else ""
    from .knowledge import tools as _tools
    return f"""You are the Dev agent doing a REVISION round in an autonomous SDLC pipeline. Your earlier change is already committed on the current branch — run `git diff origin/HEAD...HEAD` (or `git log -p`) to see it, then improve it in place to address the feedback below. Your cwd IS the repo root: use relative paths, targeted reads (offset/limit), and no find/ls exploration — every extra read costs real money.
{_tools.prompt_block(tools_cmd)}{plan_block}{ctx}{ver}
Task {task_key}: {title}
Acceptance criteria:
{crit}

An unbiased reviewer (a DIFFERENT model provider) requested changes. Address every concrete, correct concern by editing the files (if a point is factually wrong, skip it and say why in your summary):
<reviewer_feedback>
{(review_text or '').strip()[:6000]}
</reviewer_feedback>

QA findings:
<qa_findings>
{(qa_text or '').strip()[:4000]}
</qa_findings>

Make the smallest correct, surgical edits; keep working code intact. DIFF DISCIPLINE: change only what the feedback requires — no refactors, reformatting, or improvements to untouched code; do not expand the diff beyond the feedback. {_test_block(test_cmd)}
Do NOT commit or push — just edit the files. When done, briefly summarize what you changed to address the feedback."""


def review(task_key: str, criteria: list[str], diff: str) -> str:
    crit = "\n".join(f"- {c}" for c in criteria) or "- (general quality)"
    return f"""You are the Review agent in an autonomous SDLC pipeline. Review the code change below against the requirements. Do NOT edit files.

Task {task_key}
Acceptance criteria:
{crit}

Git diff:
```diff
{clip(diff, 12000)}
```

Calibrate the verdict — every CHANGES REQUESTED triggers a full paid Dev+QA+Review round:
- CHANGES REQUESTED only for a real defect: broken/missing behavior a criterion requires, a bug, a security problem, or dead wiring (code defined but never connected).
- The criteria were written by a PM working from repo summaries. If a criterion asserts a specific file/symbol/structure that doesn't match this repo, judge the BEHAVIOR the criterion is after (verify in the code, not just the diff) — do not enforce the wrong letter of it.
- Style preferences, refactor ideas, "would be cleaner", and extra-test suggestions are observations to note, never grounds for CHANGES REQUESTED.
- DO flag as an issue any changed lines unrelated to the work order (drive-by refactors) — but if behavior is correct, that alone is an observation, not a rejection.

Give a concise review: does it meet the criteria? List concrete issues or risks (or state it's approved). End with a one-line verdict: APPROVED or CHANGES REQUESTED."""


def scope_pr_body(scope_title: str, subtasks: list[dict], qa_summary: str,
                  review_summary: str = "") -> str:
    """PR description for a delivered scope. The review section carries the
    jury's verdict verbatim — a human merging this should see what the panel
    found (and, when a juror abstained, what went unreviewed) without opening
    the app."""
    items = "\n".join(f"- **{t['key']}** {t['title']}" for t in subtasks)
    review = f"\n### Code review\n{review_summary[:9000]}\n" if review_summary.strip() else ""
    # An unapproved delivery must say so in the first line a reviewer sees, not
    # only inside the review section further down.
    unapproved = ("> ⚠️ **This PR was NOT approved by the review panel.** Revision rounds "
                  "ran out with findings outstanding — see the code review below before "
                  "merging.\n\n" if "DELIVERED WITHOUT APPROVAL" in review_summary else "")
    return f"""{unapproved}## {scope_title}

One PR implementing the full scope as {len(subtasks)} subtasks:

{items}

### QA summary
{qa_summary[:8000]}
{review}
🤖 Opened by **{settings.agent_git_name}** — the CodeVerdict agent pipeline (scope-level)."""


def pr_body(task_key: str, title: str, qa_summary: str) -> str:
    return f"""## {task_key}: {title}

Implemented by the autonomous SDLC agent pipeline (Dev = Claude, QA = OpenAI, Review = Claude).

### QA summary
{qa_summary[:8000]}

🤖 Opened by **{settings.agent_git_name}** — the CodeVerdict agent pipeline."""


REVIEW_SYSTEM = (
    "You are a senior code reviewer in an autonomous SDLC pipeline. Review the change "
    "against the requirements. Be concrete: does it meet the acceptance criteria? List "
    "issues or risks, or state it's approved. Verdict discipline — every CHANGES "
    "REQUESTED triggers a full paid Dev+QA+Review round: request changes only for a "
    "real defect (broken/missing required behavior, bug, security, dead wiring). The "
    "criteria come from a PM working from summaries — if one asserts a file/symbol/"
    "structure that doesn't match the repo, judge the behavior it's after, not its "
    "letter. Style preferences and refactor ideas are observations, not rejections. "
    "End with a one-line verdict: APPROVED or CHANGES REQUESTED."
)


def clip(text: str, limit: int, what: str = "the diff") -> str:
    """Cut `text` to the context budget and SAY that it was cut.

    A bare slice ends wherever the budget runs out — mid-line, mid-token — and a
    reviewer reading the tail cannot tell a display cut from code that genuinely
    stops there. It does not guess conservatively either: it reports.

    Observed on a live run: QA raised a finding that a test was "cut off
    mid-assertion — `MakeRequest(t,` with no closing paren", and asked for
    coverage that was already written. The paren was in the file, 9,000
    characters in. The code was complete; the prompt was not.

    So the cut lands on a line boundary, and what follows names itself as a
    budget cut rather than leaving the reader to infer one.
    """
    text = text or ""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    at_line = cut[:cut.rfind("\n") + 1]
    cut = at_line or cut
    return cut + (
        f"\n[... {len(text) - len(cut):,} more characters of {what} omitted to fit the "
        "context budget. This is a DISPLAY CUT, not the end of the code: anything below "
        "this point is unseen, not missing. Do not report it as truncated, incomplete or "
        "absent — if you need it, read the file.]")


QA_SYSTEM = (
    "You are a senior QA engineer. You are deliberately from a DIFFERENT model provider "
    "than the engineer who wrote this code, to provide an unbiased second opinion. Be "
    "skeptical and concrete. Judge correctness against the acceptance criteria and the "
    "test output."
)


def qa_user(task_key: str, title: str, criteria: list[str], diff: str, test_output: str) -> str:
    crit = "\n".join(f"- {c}" for c in criteria) or "- (general correctness)"
    return f"""Task {task_key}: {title}
Acceptance criteria:
{crit}

Test output:
```
{clip(test_output, 3000, 'test output')}
```

Code change (git diff):
```diff
{clip(diff, 9000)}
```

Respond with:
1. VERDICT: PASS, CONCERNS, or FAIL
2. Up to 3 concrete findings (bugs, missing criteria, risky edge cases)
3. One specific edge case worth noting
4. A coverage/confidence estimate as a percentage."""
