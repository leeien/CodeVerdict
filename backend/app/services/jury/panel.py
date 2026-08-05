"""Empanelling and polling the jurors.

Each judge is an independent LLM call over the same case file, run in parallel.
Independence is the point: the jurors never see each other's opinions, so their
errors are uncorrelated in a way a single reviewer's second pass never is.

Everything here is failure-tolerant by design. A juror that errors, times out,
or returns unparseable output ABSTAINS — it is recorded as having not reviewed,
and the foreperson is told which perspectives went uncovered. What it must never
do is disappear silently, because a missing juror looks exactly like a clean
juror once the opinions are concatenated.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ...config import settings
from ...models import Judge
from .. import agent_backends, lang, llm, providers
from .. import judges as roster
from ..knowledge import tools as kb_tools
from . import evidence, personas, prompts

logger = logging.getLogger(__name__)

_SEVERITIES = ("critical", "high", "medium", "low")
_VERDICTS = ("APPROVE", "REQUEST_CHANGES", "ABSTAIN")


@dataclass
class Opinion:
    """One juror's returned review, normalized."""

    judge_id: int | None
    name: str
    persona: str
    provider: str
    model: str
    verdict: str = "ABSTAIN"
    summary: str = ""
    findings: list[dict] = field(default_factory=list)
    error: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0

    @property
    def abstained(self) -> bool:
        return self.verdict == "ABSTAIN"

    @property
    def usable(self) -> bool:
        """Did this juror actually review? An abstention caused by an error is
        NOT a clean bill of health, and is never counted as one."""
        return not self.error and self.verdict in ("APPROVE", "REQUEST_CHANGES")

    def as_dict(self) -> dict:
        return {"judge_id": self.judge_id, "name": self.name, "persona": self.persona,
                "provider": self.provider, "model": self.model, "verdict": self.verdict,
                "summary": self.summary, "findings": self.findings, "error": self.error,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out, "cost": self.cost}


def _load_json(text: str) -> dict:
    """Best-effort JSON out of a model response: raw, fenced, or embedded."""
    for candidate in _json_candidates(text or ""):
        parsed = _try_json(candidate)
        if parsed is None:
            continue
        # Some models double-encode: the whole answer is a JSON STRING whose
        # value is itself the intended JSON object (quoted, with \n / \" left
        # escaped). That parses cleanly on the first pass into a `str` — unwrap
        # it once more before giving up on this candidate.
        if isinstance(parsed, str):
            parsed = _try_json(parsed)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):        # a juror that returned bare findings
            return {"findings": parsed}
    return {}


def _try_json(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _unescape_once(text: str) -> str | None:
    """Undo one layer of string-escaping, if that is what happened.

    Some models emit the *escaped* form of the object without the quotes that
    would make it a string: ``{\\n  \\"verdict\\": \\"APPROVE\\" …}``. Every
    inner quote and newline is backslashed, but there is no surrounding string
    for those escapes to belong to, so it is not valid JSON and every plain
    parse of it fails.

    Wrapping it in quotes turns it into a valid JSON string whose *value* is the
    object's real source text; parsing that hands back the text to parse for
    real. Observed from ``gemini-flash-latest``, where it silently cost the
    panel a whole juror — the seat abstained for a formatting reason that had
    nothing to do with the review.

    Returns None when this isn't what happened (e.g. the text contains a raw,
    unescaped quote), so the caller just moves on to the next strategy.
    """
    if "\\n" not in text and '\\"' not in text:
        return None
    inner = _try_json(f'"{text}"')
    return inner if isinstance(inner, str) else None


def _json_candidates(text: str):
    text = text.strip()
    yield text
    fenced = re.findall(r"```(?:json)?\s*(.+?)```", text, re.S)
    yield from (f.strip() for f in fenced)
    braced = text[text.find("{"): text.rfind("}") + 1] if "{" in text and "}" in text else ""
    if braced:
        yield braced
    # Tried last: a real object should already have parsed, and unescaping is
    # only ever the right answer once every direct reading has failed.
    for candidate in (text, braced):
        if candidate:
            unescaped = _unescape_once(candidate)
            if unescaped:
                yield unescaped


_ASSERT_RE = re.compile(r"\b(assert|expect|should|assertEqual|assertRaises|\.to\w*\()")


def tampering_brief(diff_text: str) -> str:
    """Deterministic check: did this change REWRITE existing test expectations?

    Rewriting an assertion so a wrong implementation passes is the single most
    dangerous thing an autonomous coding agent can do, and it is invisible to a
    green test run — the suite passes precisely because the evidence was edited.
    We watched a Dev agent flip an existing `WindowsCoordinates(row=29, col=19)`
    to `row=30, col=20` to fit an off-by-one it had just introduced, and a
    four-judge panel approved it unanimously.

    So this is computed from the diff rather than asked of a model: removed
    assertion lines in test files are a fact, and facts belong in the case file.
    Added assertions are ignored — new tests are the normal, wanted case.
    """
    removed: list[str] = []
    in_test = False
    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            in_test = lang.is_test_file(line.split(" b/", 1)[-1])
        elif in_test and line.startswith("-") and not line.startswith("---"):
            body = line[1:].strip()
            # Any removed line of real test code counts, not just ones with the
            # word "assert" on them: the assertion we actually got burned by was
            # a continuation line — `WindowsCoordinates(row=29, col=19)` — inside
            # a multi-line assert_called_once_with(...). Keyword matching missed
            # it entirely. Comments, blanks and imports are excluded as noise.
            if (body and not body.startswith("#")
                    and not body.startswith(("import ", "from "))):
                removed.append(body[:160])
    if not removed:
        return ""
    asserts = sum(1 for r in removed if _ASSERT_RE.search(r))
    detail = (f" ({asserts} of them contain an assertion keyword)" if asserts else "")
    return (
        "DETERMINISTIC ALERT — this change DELETES OR REWRITES "
        f"{len(removed)} line(s) of EXISTING test code{detail}:\n"
        + "\n".join(f"  - {r}" for r in removed[:12])
        + "\n\nA passing suite proves nothing about a change that edited the assertions it "
          "is judged by. Establish which happened: (a) the assertion encoded the OLD, "
          "buggy behaviour and the requirement legitimately changes it — then the new "
          "expectation must match what the USER asked for; or (b) the implementation is "
          "wrong and the assertion was rewritten to accommodate it — which is a critical "
          "defect regardless of how green the tests are. Do not assume (a).")


def _clamp_confidence(value) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.5          # unstated confidence is neither a guess nor a certainty
    return min(1.0, max(0.0, conf))


def _normalize_finding(raw) -> dict | None:
    """Coerce one juror's finding into the panel's shape, or drop it.

    Models are inconsistent about these keys, and a finding with no title and no
    evidence carries no information for the foreperson to weigh — better dropped
    here than allowed to dilute the synthesis prompt."""
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title") or raw.get("issue") or raw.get("summary") or "").strip()
    evidence = str(raw.get("evidence") or raw.get("code") or "").strip()
    if not title and not evidence:
        return None
    severity = str(raw.get("severity") or "medium").strip().lower()
    if severity not in _SEVERITIES:
        severity = "medium"
    return {
        "title": title[:300] or "(untitled finding)",
        "location": str(raw.get("location") or raw.get("file") or "").strip()[:200],
        "evidence": evidence[:1200],
        "why_it_matters": str(raw.get("why_it_matters") or raw.get("why")
                              or raw.get("impact") or "").strip()[:1200],
        "severity": severity,
        "confidence": _clamp_confidence(raw.get("confidence")),
        "suggestion": str(raw.get("suggestion") or raw.get("fix") or "").strip()[:1200],
    }


def _normalize(payload: dict) -> tuple[str, str, list[dict]]:
    findings = [f for f in (_normalize_finding(r) for r in (payload.get("findings") or []))
                if f is not None]
    verdict = str(payload.get("verdict") or "").strip().upper().replace(" ", "_")
    if verdict in ("CHANGES_REQUESTED", "REQUEST_CHANGE", "REJECT", "REJECTED"):
        verdict = "REQUEST_CHANGES"
    elif verdict in ("APPROVED", "LGTM", "PASS"):
        verdict = "APPROVE"
    if verdict not in _VERDICTS:
        # No usable verdict field, but real findings came back: infer from them
        # rather than throwing away a review the panel paid for.
        verdict = "REQUEST_CHANGES" if any(
            f["severity"] in ("critical", "high") for f in findings) else (
            "APPROVE" if findings or payload.get("summary") else "ABSTAIN")
    summary = str(payload.get("summary") or payload.get("rationale") or "").strip()[:2000]
    return verdict, summary, findings


def _call(system: str, user: str, provider: str, model: str, workdir: str = "",
          cache_prefix: str = "") -> dict:
    """One juror's LLM call. Agentic-CLI providers get the repo working copy so
    they can read files the diff only partially shows; API providers judge from
    the case file alone.

    ``cache_prefix`` is the panel-wide case file, kept ahead of this juror's own
    charge so every provider that discounts a repeated prompt prefix can.
    """
    if workdir and providers.kind(provider) in ("claude-cli", "agent"):
        backend = providers.agent_backend(provider)
        # The adapters CALL their on_event rather than null-checking it, so a
        # None here takes the juror down with a TypeError — which reads as an
        # abstention and silently costs the panel a whole perspective. The
        # jurors run in parallel, so their step-by-step output would interleave
        # into nonsense anyway; the caller logs each opinion when it lands.
        res = agent_backends.run(backend, workdir, f"{system}\n\n{cache_prefix}{user}",
                                 lambda level, message: None, model=model)
        return {"text": res.get("text", ""), "tokens_in": res.get("tokens_in", 0),
                "tokens_out": res.get("tokens_out", 0), "cost": res.get("cost", 0.0),
                "error": res.get("error")}
    return llm.chat(system, user, provider=provider, model=model, json_mode=True,
                    cache_prefix=cache_prefix)


def poll_judge(judge: Judge, case: dict, workdir: str = "", repo: str = "",
               case_file: str | None = None) -> Opinion:
    """Run one juror to an Opinion. Never raises — an exception here would take
    down the whole panel over a single misbehaving provider.

    A juror may spend its first reply asking the repository index questions
    instead of voting (see ``tools.evidence_block``); it is then called once
    more with the answers. Only a juror that ASKS pays for the extra call, and
    there is exactly one follow-up — an unbounded evidence loop would turn a
    review into a second Dev agent. This exists because the expensive mistake is
    not an ignorant juror but a confident one: a blocking finding about code
    outside the diff costs a full paid Dev+QA+Review round, and jurors were
    being asked to judge callers, patterns and coverage they could not see.
    """
    provider, model = roster.resolve(judge)
    op = Opinion(judge_id=judge.id, name=judge.name, persona=judge.persona,
                 provider=provider, model=model)
    charge = personas.charge(judge.persona, judge.focus or "")
    # The case file is identical for every juror, so it is built once by the
    # caller and reused here as a cacheable prompt prefix. This juror's own
    # charge and its persona-specific evidence follow it — the architecture
    # juror sees the neighbours it must compare against, the security juror sees
    # whether the change is reachable at all, and so on.
    shared = case_file if case_file is not None else prompts.judge_case(**case)
    user = prompts.judge_charge(
        charge, evidence.for_persona(judge.persona, workdir, case.get("diff", "")))

    for attempt in range(2):
        try:
            res = _call(prompts.JUDGE_SYSTEM, user, provider, model, workdir=workdir,
                        cache_prefix=shared)
        except Exception as exc:  # noqa: BLE001 — a juror's crash is an abstention
            op.error = f"{type(exc).__name__}: {exc}"[:300]
            return op
        op.tokens_in += res.get("tokens_in") or 0
        op.tokens_out += res.get("tokens_out") or 0
        op.cost += res.get("cost") or 0.0
        text = res.get("text") or ""
        if res.get("error") and not text.strip():
            op.error = str(res["error"])[:300]
            return op

        # Evidence request? Answer it and re-ask — but only on the first pass, and
        # only when it asked INSTEAD of voting. A juror that returned findings has
        # made up its mind; re-asking would just pay twice for the same opinion.
        requests = kb_tools.parse_requests(text) if attempt == 0 else []
        payload = _load_json(text)
        if requests and settings.jury_tool_calls > 0 and not payload.get("findings"):
            answers = kb_tools.run_requests(repo or _repo_of(workdir), workdir, requests,
                                            limit=settings.jury_tool_calls)
            if answers:
                # Append to the TAIL, never the shared prefix — a follow-up that
                # mutated the prefix would miss the cache it just paid to write.
                user = (f"{user}\n\n{answers}\n\nNow give your verdict as JSON. "
                        "Do not request more lookups.")
                continue

        if not payload:
            op.error = ("returned no parseable JSON"
                        + (f": {text.strip()[:160]}" if text.strip() else ""))
            return op
        op.verdict, op.summary, op.findings = _normalize(payload)
        return op
    return op


def _repo_of(workdir: str) -> str:
    """The index key for a working copy — its directory name is the repo slug
    (git_ops.slug is idempotent on slugs, so this resolves to the same project)."""
    from pathlib import Path

    return Path(workdir).name if workdir else ""


def empanel(judge_rows: list[Judge], case: dict, workdir: str = "",
            on_event=None) -> list[Opinion]:
    """Poll every juror in parallel and return their opinions in roster order.

    Reviews are read-only, so several agentic CLIs sharing the working copy is
    safe — none of them is permitted to edit. ``jury_max_parallel`` caps the
    fan-out anyway, since free-tier providers rate-limit on concurrency.
    """
    if not judge_rows:
        return []
    workers = max(1, min(int(settings.jury_max_parallel), len(judge_rows)))
    if on_event:
        on_event("info", f"Empanelling {len(judge_rows)} judge(s), {workers} at a time: "
                         + ", ".join(roster.label(j) for j in judge_rows))
    repo = _repo_of(workdir)
    # Built once for the whole panel: it is byte-identical per juror, and being
    # identical is what lets every provider that discounts a repeated prompt
    # prefix charge the 2nd..Nth juror a fraction of the 1st.
    case_file = prompts.judge_case(**case)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="juror") as pool:
        opinions = list(pool.map(
            lambda j: poll_judge(j, case, workdir, repo, case_file), judge_rows))
    if on_event:
        for op in opinions:
            if op.error:
                on_event("warn", f"Juror {op.name} ({op.provider} {op.model}) ABSTAINED — "
                                 f"{op.error[:120]}")
            else:
                blocking = sum(1 for f in op.findings if f["severity"] in ("critical", "high"))
                on_event("info", f"Juror {op.name}: {op.verdict} — {len(op.findings)} finding(s)"
                                 f"{f', {blocking} high/critical' if blocking else ''}"
                                 + (f"\n{op.summary}" if op.summary else ""))
    return opinions


def render_opinions(opinions: list[Opinion]) -> str:
    """The jurors' opinions as the foreperson sees them."""
    blocks = []
    for op in opinions:
        if not op.usable:
            continue
        p = personas.get(op.persona)
        head = (f"### Juror: {op.name} (perspective: {p.name if p else 'custom'}; "
                f"model: {op.provider} {op.model})\nVerdict: {op.verdict}")
        body = [head]
        if op.summary:
            body.append(f"Summary: {op.summary}")
        if not op.findings:
            body.append("Findings: none.")
        for i, f in enumerate(op.findings, 1):
            body.append(
                f"Finding {i}: {f['title']}\n"
                f"  location: {f['location'] or '(unspecified)'}\n"
                f"  evidence: {f['evidence'] or '(none given)'}\n"
                f"  why it matters: {f['why_it_matters'] or '(not stated)'}\n"
                f"  severity: {f['severity']} | confidence: {f['confidence']:.2f}\n"
                f"  suggested fix: {f['suggestion'] or '(none given)'}")
        blocks.append("\n".join(body))
    return "\n\n".join(blocks)
