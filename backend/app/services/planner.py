"""The Planner — decides HOW the change will be made, before anyone writes code.

Between "the human and the PM agreed what to build" and "the Dev agent starts
editing" there used to be nothing. The PM guessed at files and symbols from
summarized knowledge and handed those guesses forward as localization; the Dev
agent got a large pre-computed context blob and started typing. That is the
single-agent shape the research measures as the weak one — AgentForge's ablation
removes the planner and lands back at the single-agent baseline; HyperAgent finds
specialized planning/navigation roles beating one all-purpose agent.

So this stage exists, and it is deliberately NOT the PM:

  * the PM talks to a human. It owns the requirement, the clarifying questions
    and the acceptance criteria — observable behaviour, no code.
  * the Planner talks to the repository. It runs after the scope is locked, at
    pipeline entry, when the working copy is clean and the code graph is current,
    and it answers a different question: which code actually produces this
    behaviour, what else touches it, and in what order should it change.

It works the way the research prescribes for a Planner: a short bounded loop
(3-5 rounds), retrieval on demand rather than one pre-loaded dump, and a
structured plan out. Then — and this is the part an LLM cannot be trusted with —
every symbol it named is verified against the graph, the symbol map and ripgrep
before the plan is allowed to mean anything. A step whose target cannot be found
is marked as new rather than quietly pointed at whatever matched first.

The verification here is the logic that used to live in ``pm_agent.ground_tickets``,
and it carries real scars: a bare ``__init__`` matches every class in a repo, a
protocol method like ``__rich__`` resolves to whichever file the index returns
first (a live run pinned ``Table.__rich__`` to ``rich/json.py`` and dragged two
unrelated files into the ticket). Those rules moved here intact.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from ..config import settings
from . import agent_backends, llm, providers, search
from .knowledge import graph, symbol_map
from .knowledge import retriever as knowledge_retriever
from .knowledge import tools as kb_tools

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are the Planner in an autonomous software-delivery pipeline. The scope is "
    "already agreed with a human — you do NOT re-litigate WHAT to build. You decide "
    "HOW: which code actually produces the described behaviour, what else touches it, "
    "and in what order it should change.\n\n"
    "You work over a few rounds. Each round choose EXACTLY ONE action:\n"
    '- "retrieve": you need to see repository code you haven\'t seen. Provide `queries` '
    "(1-3 phrasings) and optionally `tools`, a list of [tool, argument] pairs from: "
    "search (ranked locations for a mechanism), lookup (where a symbol is DEFINED), "
    "callers (who calls it), expand (everything 1 hop away — callers, callees, class "
    "members, covering tests), outline (symbols in a file), snippet (a symbol's exact "
    "source), grep (raw text).\n"
    "  QUERY FORMULATION — this is where planning usually fails. The index holds CODE "
    "(function names, signatures, call sites), not the user's words, so a restated "
    "symptom retrieves nothing. Ask for the MECHANISM: the operation the code performs "
    "that would produce the reported behaviour.\n"
    "    symptom (bad):    'table borders look wrong when the text is colored'\n"
    "    mechanism (good): 'measure printable cell width ignoring ansi escape sequences'\n"
    '- "plan": you understand the code well enough to commit to an approach. Fill '
    "`plan`.\n\n"
    "A GOOD PLAN:\n"
    "- names the REAL definition site of every symbol it touches — you have tools, so "
    "a guess is a choice. If you never confirmed a location, say so in `open_questions` "
    "rather than asserting it.\n"
    "- accounts for the blast radius: if you change a function's behaviour, its callers "
    "are part of the change even when they need no edit (you must have checked).\n"
    "- orders the steps so the repository is coherent at the end, and says which "
    "EXISTING tests cover the behaviour (extend those; a parallel suite is waste).\n"
    "- is MINIMAL. Every step is paid Dev+QA+Review work. Do not add refactors, "
    "hardening, or adjacent improvements the scope didn't ask for. 1-4 steps is normal; "
    "more usually means you are expanding the request.\n"
    "- states real `risks`: what could break, what you are unsure of. A plan with no "
    "risks on a non-trivial change is a plan that didn't look.\n\n"
    "Respond ONLY as JSON:\n"
    '{"action": "retrieve"|"plan", "reason": string,\n'
    ' "queries": [string]|null, "tools": [[string, string]]|null,\n'
    ' "plan": {"summary": string,\n'
    '   "steps": [{"intent": string, "edit_kind": "modify"|"create"|"wire"|"delete",\n'
    '              "files": [string], "symbols": ["path::Symbol"], "why": string,\n'
    '              "verify": [string]}],\n'
    '   "risks": [string], "open_questions": [string],\n'
    '   "tests": {"files": [string], "new_cases": [string]}} | null}\n'
    "`plan` is non-null ONLY on a 'plan' action."
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


# --- The planning loop --------------------------------------------------------

def _ask(system: str, user: str, cache_prefix: str = "") -> dict:
    """One Planner decision, on whichever provider the operator pointed the stage
    at. An agentic CLI is a legitimate choice here — planning is read-only, and a
    CLI-backed planner reaches the same tools through the shim."""
    provider = settings.planner_provider or settings.pm_provider
    model = settings.planner_model or settings.pm_model
    backend = providers.agent_backend(provider)
    if backend:
        res = agent_backends.chat(backend, system, cache_prefix + user,
                                  model=model, json_mode=True)
        # CLI backends report usage inconsistently (None = unknown, not free);
        # coerce to numbers so the run row's arithmetic stays sound.
        return {"text": res.get("text", ""), "tokens_in": res.get("tokens_in") or 0,
                "tokens_out": res.get("tokens_out") or 0, "cost": res.get("cost") or 0.0,
                "error": res.get("error")}
    return llm.chat(system, user, provider=provider, model=model, json_mode=True,
                    cache_prefix=cache_prefix)


def _tool_round(repo_url: str, workdir: str, calls: list) -> list[str]:
    """Run the Planner's requested tool calls through the same dispatcher the Dev
    agent and the jury use. Capped: a planner that spends twenty calls per round
    is not planning."""
    out: list[str] = []
    for entry in (calls or [])[:4]:
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
            continue
        name, arg = str(entry[0]), str(entry[1])
        out.append(f"$ {name} {arg}\n{kb_tools.call(repo_url, workdir, name, arg)}")
    return out


def _scope_block(scope: dict, tickets: list[dict]) -> str:
    lines = [f"Scope: {scope.get('summary') or '(none)'}", "", "Acceptance criteria:"]
    lines += [f"- {c}" for c in (scope.get("acceptance_criteria") or [])] or ["- (none)"]
    if tickets:
        lines += ["", "Approved tickets:"]
        for t in tickets:
            lines.append(f"- {t.get('key', '?')} {t.get('title', '')}: "
                         f"{str(t.get('description') or '')[:400]}")
    return "\n".join(lines)


def plan(repo_url: str, workdir: str, scope: dict, tickets: list[dict] | None = None,
         on_event=None) -> dict:
    """Produce a verified implementation plan for a locked scope.

    Returns ``{plan, rounds, retrieved, tokens_in, tokens_out, cost, error}``.
    ``plan`` is always a dict — an empty one when planning could not run, which
    the orchestrator treats as "no plan", never as "nothing to do".
    """
    def log(level: str, msg: str) -> None:
        if on_event:
            on_event(level, msg)

    tickets = tickets or []
    request = _scope_block(scope, tickets)
    context: list[str] = []
    anchor = graph.bootstrap_text(repo_url)
    if anchor:
        context.append(anchor)

    tin = tout = 0
    cost = 0.0
    err = None
    retrieved: list[str] = []
    shown: set[str] = set()     # hit keys already in `context`; see the loop below
    max_rounds = max(1, settings.planner_max_rounds)

    for round_no in range(max_rounds + 1):
        final = round_no >= max_rounds
        force = ("\n\nIMPORTANT: this is your LAST round. Do NOT choose 'retrieve' — "
                 "commit to a plan with what you have, and put anything you could not "
                 "confirm in `open_questions`." if final else "")
        # Split at the boundary between what is stable across rounds and what
        # changes. The scope and everything retrieved so far are APPEND-ONLY, so
        # each round's prefix contains the previous round's verbatim — the ideal
        # shape for prompt caching. Only the trailing instruction varies, and it
        # is a couple of lines. Measured before this: the Planner paid ~95% of
        # full retail on 322,893 input tokens while Dev, whose backend caches
        # across turns, paid 17% on 1.88M.
        # The empty-case hint lives in the TAIL, not the prefix. Putting it in
        # the prefix made round 1 end with "(nothing yet — retrieve first)" and
        # round 2 REPLACE that text with the hits, so the prefix was not
        # append-only and the cache missed at its very first boundary.
        prefix = (f"{request}\n\nWhat you know about the repository so far:\n"
                  + "\n\n".join(context))
        empty = "" if context else "\n(nothing retrieved yet — retrieve first.)"
        tail = f"{empty}\n\nDecide your next action as JSON.{force}"
        r = _ask(_SYSTEM, tail, cache_prefix=prefix)
        tin += r.get("tokens_in", 0) or 0
        tout += r.get("tokens_out", 0) or 0
        cost += r.get("cost", 0.0) or 0.0
        err = r.get("error")
        data = _load_json(r.get("text", ""))

        if data.get("action") == "retrieve" and not final:
            queries = [str(q) for q in (data.get("queries") or []) if str(q).strip()][:3]
            blocks: list[str] = []
            for q in queries:
                # `shown` carries across rounds, so a hit already in the context
                # is not rendered again. The rounds ask near-identical questions
                # by design ("dependency picker dropdown", then "dependency
                # dropdown template", then "dependency sidebar dropdown list"),
                # so their result sets overlap heavily — and the context is
                # re-sent whole every round, making each duplicate compound.
                blocks.append(knowledge_retriever.retrieve_context(
                    repo_url, q, limit=6, exclude=shown))
                retrieved.append(q)
            blocks += _tool_round(repo_url, workdir, data.get("tools") or [])
            blocks = [b for b in blocks if b]
            context.extend(blocks or ["(that lookup returned nothing — try different "
                                      "mechanism wording, or plan with what you have)"])
            log("info", f"Planner round {round_no + 1}: looked up "
                        + "; ".join(queries[:3] or ["(tools only)"]))
            continue

        raw = data.get("plan") if isinstance(data.get("plan"), dict) else None
        if raw is None:
            log("warn", f"Planner returned no plan{f' ({err})' if err else ''} — "
                        "the pipeline continues without one")
            return {"plan": {}, "rounds": round_no, "retrieved": retrieved,
                    "tokens_in": tin, "tokens_out": tout, "cost": cost,
                    "error": err or "planner returned no plan"}

        verified = verify_plan(repo_url, workdir, _normalize(raw))
        gaps = coverage_gaps(scope, verified)
        if gaps:
            verified["uncovered_criteria"] = gaps
            log("warn", f"Planner: {len(gaps)} acceptance criterion/criteria have no "
                        "step in this plan — " + "; ".join(g[:120] for g in gaps))
        log("success", describe(verified))
        return {"plan": verified, "rounds": round_no, "retrieved": retrieved,
                "tokens_in": tin, "tokens_out": tout, "cost": cost, "error": err}

    return {"plan": {}, "rounds": max_rounds, "retrieved": retrieved, "tokens_in": tin,
            "tokens_out": tout, "cost": cost, "error": err}


# --- Normalization ------------------------------------------------------------

_EDIT_KINDS = ("modify", "create", "wire", "delete")


def _str_list(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, (str, int)) and str(v).strip()][:limit]


def _normalize(raw: dict) -> dict:
    steps = []
    for i, s in enumerate(raw.get("steps") or [], start=1):
        if not isinstance(s, dict) or not str(s.get("intent") or "").strip():
            continue
        kind = str(s.get("edit_kind") or "modify").strip().lower()
        steps.append({
            "id": i,
            "intent": str(s["intent"]).strip()[:400],
            "edit_kind": kind if kind in _EDIT_KINDS else "modify",
            "files": _str_list(s.get("files"), 6),
            "symbols": _str_list(s.get("symbols"), 8),
            "why": str(s.get("why") or "").strip()[:400],
            "verify": _str_list(s.get("verify"), 4),
            "blast_radius": [],      # filled by verify_plan from the call graph
        })
    tests = raw.get("tests") if isinstance(raw.get("tests"), dict) else {}
    return {
        "summary": str(raw.get("summary") or "").strip()[:800],
        "steps": steps[:6],
        "risks": _str_list(raw.get("risks"), 6),
        "open_questions": _str_list(raw.get("open_questions"), 6),
        "tests": {"files": _str_list(tests.get("files"), 6),
                  "new_cases": _str_list(tests.get("new_cases"), 6)},
    }


# --- Deterministic verification ----------------------------------------------
# Moved from pm_agent.ground_tickets, where it grounded a guess. Here it grounds
# a decision — same rules, and they are the ones that stop a plan from pointing
# the Dev agent at a file it has no business editing.

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Method names that exist on hundreds of classes — looking one up on its own
# resolves to whichever class the index happens to return first.
_GENERIC_METHODS = frozenset(
    ["render", "run", "main", "setup", "update", "get", "set",
     "add", "remove", "close", "start", "stop", "parse", "process", "handle"])


def _is_generic(name: str) -> bool:
    """True for a name that identifies nothing on its own. EVERY dunder counts:
    `__init__`, and equally `__rich__`/`__rich_console__` (protocol methods that
    dozens of classes in a repo implement — a live run resolved
    `Table.__rich__` to `rich/json.py`)."""
    return name in _GENERIC_METHODS or (name.startswith("__") and name.endswith("__"))


def _ident_and_owner(idents: list[str]) -> tuple[str, str]:
    """Split a dotted symbol into (identifier to look up, owning class).

    'Table.__init__' must NOT be looked up as '__init__' — that is how a ticket
    got pinned to `rich/panel.py::Table.__init__` (observed live), dragging
    panel.py and rule.py into the affected files and sending the Dev agent at
    three files it had no business editing. For a dotted name we keep the last
    segment as the identifier but remember its owner, so hits can be filtered to
    that class; a generic/dunder method with a known owner looks up the OWNER
    instead, which is unambiguous.
    """
    if not idents:
        return "", ""
    if len(idents) == 1:
        return ("", "") if _is_generic(idents[0]) else (idents[0], "")
    owner, member = idents[-2], idents[-1]
    if _is_generic(member):
        return owner, owner  # pin the class; '__init__' alone means nothing
    return member, owner


def _graph_files(repo_url: str, ident: str, owner: str) -> list[str]:
    """Definition files for `ident` from the code graph, preferring hits that
    belong to `owner` when the symbol was written as 'Owner.member'."""
    hits = graph.lookup(repo_url, ident, limit=5)
    if owner and ident != owner:
        scoped = [h for h in hits if owner in str(h.get("qualified_name") or "")]
        if not scoped:
            # None of the matches belong to the class the Planner named. Falling
            # back to an arbitrary same-named method is how `Table.__rich__`
            # became `rich/json.py::Table.__rich__`. Resolve the CLASS instead —
            # unambiguous, and the file the work actually belongs in.
            return [h["file_path"] for h in graph.lookup(repo_url, owner, limit=2)]
        hits = scoped
    return [h["file_path"] for h in hits[:2]]


def verify_plan(repo_url: str, workdir: str, plan_obj: dict) -> dict:
    """Ground every symbol the plan names against the real repository.

    An LLM naming a file is a claim; this makes it a fact or marks it as one
    that could not be checked. For each symbol: resolve its definition site
    (code graph → symbol map → language-aware ripgrep), pin the step's files to
    what was actually found, attach the call-graph blast radius, and name the
    existing tests that already exercise it. A symbol found nowhere is tagged
    ``(new — not in repo yet)`` with a "did you mean", because the usual failure
    is a slightly-wrong invented name, and sending Dev hunting for it is worse
    than telling it to create one.

    Deterministic throughout — no LLM, so it cannot hallucinate the correction.
    """
    if not plan_obj.get("steps"):
        return plan_obj
    root = workdir if workdir and Path(workdir, ".git").exists() else ""
    smap = symbol_map.load(repo_url)
    use_graph = graph.available()
    unresolved = 0

    for step in plan_obj["steps"]:
        files = list(step.get("files") or [])
        symbols: list[str] = []
        radius: dict[str, None] = {}
        tests: dict[str, None] = {}
        for sym in step.get("symbols") or []:
            hint = str(sym).split("::")[0] if "::" in str(sym) else ""
            name = str(sym).split("::")[-1].strip()
            ident, owner = _ident_and_owner(_IDENT_RE.findall(name))
            if not ident:
                symbols.append(str(sym))
                continue
            defined_in = (_graph_files(repo_url, ident, owner) if use_graph else []) \
                or ([f for f, _ in smap.lookup(ident)] if smap else []) \
                or (search.definitions(root, ident, hint_path=hint) if root else [])
            if defined_in:
                for f in dict.fromkeys(defined_in[:2]):
                    if f not in files:
                        files.append(f)
                symbols.append(f"{defined_in[0]}::{name}")
                for c in (graph.callers(repo_url, ident, limit=4) if use_graph else []):
                    radius.setdefault(
                        f"{c['file_path']}:{c.get('start_line') or '?'} ({c['name']})")
                if root:
                    for tf in search.files(root, rf"\b{re.escape(ident)}\b",
                                           pathspec="*test*", max_files=3):
                        tests.setdefault(tf)
            elif root and search.mentions(root, ident):
                # It appears but isn't defined — a usage, a string, a config key.
                # Keep it as written rather than inventing a definition site.
                symbols.append(str(sym))
            else:
                close = smap.suggest(ident) if smap else []
                hint_text = f"; similar existing: {', '.join(close)}" if close else ""
                symbols.append(f"{name} (new — not in repo yet{hint_text})")
                unresolved += 1
        step["files"] = files[:8]
        step["symbols"] = symbols
        step["blast_radius"] = list(radius)[:8]
        if tests:
            step["existing_tests"] = list(tests)[:5]

    plan_obj["verified"] = True
    plan_obj["unresolved_symbols"] = unresolved
    return plan_obj


# --- Criterion coverage -------------------------------------------------------
# A plan can be entirely correct about what it covers and still deliver half the
# scope. Observed: a five-criterion scope where the plan's three steps addressed
# criteria 1-2 and never mentioned 3-5. The Planner had noticed them — it listed
# the missing behaviour verbatim under `tests.new_cases` — but a test case is not
# an implementation step, so nothing was built, and Dev, QA and four jurors all
# approved a change that met two fifths of what was asked.
#
# Hence: match criteria against STEPS ONLY. Counting the tests block would have
# scored that plan as fully covered, which is precisely the mistake being caught.

_COVERAGE_STOP = frozenset((
    "about", "above", "after", "again", "against", "also", "because", "been",
    "before", "being", "below", "between", "both", "does", "doing", "during",
    "each", "else", "from", "further", "have", "here", "into", "itself", "more",
    "most", "only", "other", "over", "same", "some", "such", "than", "that",
    "then", "there", "these", "they", "this", "those", "through", "under",
    "until", "very", "were", "what", "when", "where", "which", "while", "with",
    "within", "would", "your", "must", "should", "shall", "able", "allow",
    "allows", "user", "users", "given",
))


def _split_identifiers(text: str) -> str:
    """MaxPinned -> Max Pinned, issue_pin.go -> issue pin go.

    Criteria are written in user language and steps in code, so the two sides
    only meet once identifiers are broken into words: without this, the criterion
    "a maximum of 3 pinned issues" shares nothing with the step that verifies
    `MaxPinned` and `ErrIssueMaxPinReached`.
    """
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text or "").replace("_", " ").replace(".", " ")


def _terms(text: str, minimum: int = 4) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z]+", _split_identifiers(text).lower())
            if len(w) >= minimum and w not in _COVERAGE_STOP}


def _hit(term: str, covered: set[str]) -> bool:
    """Prefix matching in both directions — the cheapest stemmer that works on
    the pairs that actually occur: maximum/max, error/err, ordered/order."""
    return any(term.startswith(c) or c.startswith(term) for c in covered)


def coverage_gaps(scope: dict, plan_obj: dict, threshold: float = 0.34) -> list[str]:
    """Acceptance criteria that no plan step appears to address.

    Deliberately dumb and deterministic — term overlap, no model call, no cost.
    A term shared by most of the criteria ("issue", "pinned" in a pinning scope)
    says nothing about which one a step is for, so scoring uses each criterion's
    *distinctive* terms: the ones that are not spread across half the set.

    Advisory, not a veto. It reports what to look at; the decision to block
    belongs to a human at /approve, and a wrong guess here must never be able to
    stop a correct plan.
    """
    criteria = [c for c in (scope.get("acceptance_criteria") or []) if str(c).strip()]
    steps = plan_obj.get("steps") or []
    if not criteria or not steps:
        return []

    per = [_terms(c) for c in criteria]
    spread = Counter(t for terms in per for t in terms)
    common = max(1, len(criteria) // 2)

    # Three characters on the plan side: `max` has to be able to answer
    # `maximum`. Four on the criterion side, so short filler words never become
    # the thing a criterion is judged by.
    covered = set()
    for s in steps:
        for field in ("intent", "why"):
            covered |= _terms(s.get(field) or "", minimum=3)
        for field in ("files", "symbols", "verify"):
            covered |= _terms(" ".join(s.get(field) or []), minimum=3)

    gaps = []
    for criterion, terms in zip(criteria, per, strict=False):
        distinctive = {t for t in terms if spread[t] <= common} or terms
        if not distinctive:
            continue
        hit = sum(1 for t in distinctive if _hit(t, covered))
        if hit / len(distinctive) < threshold:
            gaps.append(criterion)
    return gaps


# --- Rendering ----------------------------------------------------------------

def describe(plan_obj: dict) -> str:
    """One-line log summary of a finished plan."""
    steps = plan_obj.get("steps") or []
    files = {f for s in steps for f in (s.get("files") or [])}
    unresolved = plan_obj.get("unresolved_symbols") or 0
    out = (f"Plan: {len(steps)} step(s) across {len(files)} file(s)"
           f"; {len(plan_obj.get('risks') or [])} risk(s)")
    if unresolved:
        out += f"; {unresolved} symbol(s) not found in the repo (to be created)"
    if plan_obj.get("open_questions"):
        out += f"; {len(plan_obj['open_questions'])} open question(s)"
    return out


def as_prompt(plan_obj: dict) -> str:
    """The plan as the Dev agent sees it: the ordered steps with their verified
    pins, the blast radius, the tests to extend, and — stated plainly — that the
    pins are verified but the approach is still the Planner's judgement."""
    steps = plan_obj.get("steps") or []
    if not steps:
        return ""
    lines = ["Implementation plan (from the Planner agent, which read this repository "
             "through the code graph before you started). Every `file::symbol` below was "
             "VERIFIED against the real code — trust the locations. The approach is still "
             "a judgement: if the code contradicts it, implement what's correct and say "
             "so in your summary."]
    if plan_obj.get("summary"):
        lines.append(f"\nApproach: {plan_obj['summary']}")
    for s in steps:
        lines.append(f"\nStep {s['id']} ({s['edit_kind']}): {s['intent']}")
        if s.get("why"):
            lines.append(f"    why: {s['why']}")
        if s.get("files"):
            lines.append(f"    files: {', '.join(s['files'])}")
        if s.get("symbols"):
            lines.append(f"    symbols: {', '.join(s['symbols'])}")
        if s.get("blast_radius"):
            lines.append("    callers affected (AST-verified — check you don't break "
                         "these): " + "; ".join(s["blast_radius"]))
        if s.get("existing_tests"):
            lines.append("    existing tests covering this (extend THESE): "
                         + ", ".join(s["existing_tests"]))
        if s.get("verify"):
            lines.append(f"    done when: {'; '.join(s['verify'])}")
    if plan_obj.get("uncovered_criteria"):
        # Dev is the last stage that can still add work. Below this point the
        # remaining gates only judge the diff that exists, and a criterion nobody
        # implemented leaves no trace in a diff for them to judge.
        lines.append(
            "\nAcceptance criteria with NO step above (a deterministic check, not the "
            "Planner's opinion — the plan may simply have missed these). Implement them "
            "too, or state plainly in your summary why each needs no code:\n"
            + "\n".join(f"    - {c}" for c in plan_obj["uncovered_criteria"]))
    if plan_obj.get("risks"):
        lines.append("\nRisks the Planner flagged: " + "; ".join(plan_obj["risks"]))
    if plan_obj.get("open_questions"):
        lines.append("Open questions (the Planner could NOT confirm these — verify "
                     "yourself before relying on them): "
                     + "; ".join(plan_obj["open_questions"]))
    tests = plan_obj.get("tests") or {}
    if tests.get("files"):
        lines.append("Test files to extend: " + ", ".join(tests["files"]))
    if tests.get("new_cases"):
        lines.append("Cases worth covering: " + "; ".join(tests["new_cases"]))
    return "\n".join(lines)


def targets(plan_obj: dict) -> tuple[list[str], list[str]]:
    """(files, symbols) the plan actually commits to — the verified localization
    the rest of the pipeline consumes in place of the PM's old guesses."""
    files: dict[str, None] = {}
    symbols: dict[str, None] = {}
    for s in plan_obj.get("steps") or []:
        for f in s.get("files") or []:
            files.setdefault(f)
        for y in s.get("symbols") or []:
            symbols.setdefault(y)
    return list(files), list(symbols)
