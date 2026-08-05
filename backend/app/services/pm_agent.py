"""Conversational PM agent — an agentic retrieval loop modeled on oxygen/PM_agent.

Orchestration is done by hand (no framework). Per user turn:

    1. Bootstrap (each turn, cheap): the code graph's structural summary
       (languages, packages, entry points, routes, hotspots, functional
       clusters) plus the newest cross-run delivery notes — a repo-shaped
       anchor, deterministic and free.
    2. Agent loop (capped at pm_max_retrieval_rounds): the PM decides ONE action:
         - "retrieve": it needs repo knowledge it doesn't have yet → we pull
           ranked graph localizations (real file:line hits), exact code hits and
           relevant delivery notes for its `retrieval_query`, add them to the
           context, and loop again.
         - "question": the requirement is unclear → ask ONE clarifying question.
         - "answer": respond conversationally without locking scope yet.
         - "jira": it has enough to lock the scope (summary + acceptance criteria).
       On the final allowed round we force it to stop retrieving and act.

    PM agent → retrieve knowledge → grow context → PM agent → … → question | jira

The LLM never sees whole files — only compact, AST-verified locations
(file:line + signatures) and validated delivery notes. Runs on Groq
(OpenAI-compatible) so the whole flow works without a Claude terminal.

The PM does NOT decide where the change lands. It reads the repository well
enough to ask good questions and to write criteria that are actually achievable
here, and it stops there; `services/planner.py` decides the approach and pins
the symbols, after the scope is locked and with the code in front of it. That
split is deliberate. The PM works from summaries and never opens the code, so
its file/symbol guesses were wrong often enough to matter — a live run pinned
`Table.__rich__` to `rich/json.py` and sent the Dev agent at three files it had
no business editing. Asking it to guess anyway, then verifying the guess, spends
tokens to produce something a later stage overwrites.
"""

from __future__ import annotations

import json

from ..config import settings
from . import llm
from .knowledge import graph, store
from .knowledge import retriever as knowledge_retriever

_SYSTEM = (
    "You are a sharp, senior Product Manager agent scoping a feature request for a "
    "specific software repository. You turn a requirement into a locked, "
    "unambiguous scope, using your KNOWLEDGE of the repository (never raw code).\n\n"
    "On every turn choose EXACTLY ONE action:\n"
    '- "retrieve": you need repository knowledge you don\'t have yet. Provide '
    "`retrieval_query`, `alt_queries` and `information_need` (1-2 sentences on WHAT "
    "you need to understand and which decision it informs). Describe the information "
    "itself — do NOT name storage or document types. Put a brief note in `message` "
    "about what you're looking for.\n"
    "  QUERY FORMULATION — this is where retrieval usually fails. The index holds "
    "CODE (function names, signatures, call sites), not the user's words, so a "
    "restated symptom retrieves nothing useful. Ask for the MECHANISM: the operation "
    "the code performs that would produce the reported behavior.\n"
    "    symptom (bad):   'table borders look wrong when the text is colored'\n"
    "    mechanism (good): 'measure printable cell width ignoring ansi escape sequences'\n"
    "    symptom (bad):   'login is slow for some users'\n"
    "    mechanism (good): 'session lookup query on each authenticated request'\n"
    "  `alt_queries`: 1-2 ADDITIONAL phrasings using DIFFERENT vocabulary for the same "
    "mechanism (a synonym, the likely function name, the data structure involved). "
    "They are searched together with `retrieval_query` — cheap insurance against the "
    "repo naming the concept differently than you did.\n"
    '- "question": the requirement is unclear or missing detail. Ask ONE focused '
    "clarifying question in `message`. Hunt for: vague quantifiers ('some', 'fast', "
    "'50% chance' — pin down exact deterministic rules), undefined terms, missing "
    "edge cases (idempotency, concurrency, empty/boundary values), authorization "
    "boundaries, where the result is shown/recorded, and failure modes.\n"
    '- "answer": respond conversationally (e.g. explaining part of the repo) '
    "without locking scope yet.\n"
    '- "jira": you have enough understanding to lock the scope. Fill `scope` and put '
    "a short note in `message`.\n\n"
    "Guidelines:\n"
    "- Bootstrap knowledge (the repository's code-graph structure + recent delivery "
    "notes) is loaded first. Read it before deciding whether you need more.\n"
    "- Retrieve ITERATIVELY: each round has a specific purpose in `information_need`. "
    "Stop retrieving once you have what you need. Don't retrieve everything at once.\n"
    "- Ask clarifying questions before guessing at scope — but don't manufacture "
    "questions once the scope is clear (typically 1-3 for a non-trivial feature).\n"
    "- Only choose 'jira' once the requirement is reasonably clear. When the user "
    "confirms the scope is good, choose 'jira', return it unchanged, and set "
    "`finalized` true.\n"
    "- You do NOT decide which files or symbols change. A separate Planner agent "
    "does that after the scope is locked, with the code in front of it. Use the "
    "repository knowledge to judge whether a requirement is CLEAR and ACHIEVABLE "
    "here, and to ask sharper questions — not to pick edit targets.\n"
    "- acceptance_criteria describe OBSERVABLE BEHAVIOR only (given X, the user "
    "sees Y) — never files, symbols, or code structure. You work from summaries, "
    "so a structural claim in a criterion becomes an unsatisfiable hard "
    "requirement that the reviewer enforces and the Dev cannot meet.\n"
    "- THE USER'S OWN WORDS OUTRANK YOUR CONVENTIONS. When the user delegates a "
    "decision ('use your judgement', 'you decide', 'whatever you think'), that is "
    "permission to settle what they have NOT already said — it is not permission "
    "to overrule what they HAVE. Before applying any standard pattern or best "
    "practice, re-read the user's own phrasing: if it already answers the "
    "question, follow it and say which words you followed. Choose a convention "
    "over the user's stated wording only when following them is impossible or "
    "unsafe, and then say plainly that you are departing from what they asked "
    "and why. (Observed failure: the user wrote 'it should only show issues that "
    "aren't already dependencies', was asked hide-vs-disable anyway, replied 'use "
    "your judgement', and the scope locked as 'disable' — the opposite of the "
    "sentence they had already written twice.)\n"
    "- SCOPE MINIMALISM: criteria restate the USER'S ask and nothing more. Do not "
    "invent adjacent requirements, extra edge cases, or hardening the user didn't "
    "imply — every criterion is paid Dev+QA+Review work and a rejection ground. "
    "3-6 criteria for the whole scope is the norm; more means you're expanding "
    "the request. Note worthwhile extras in `message` instead of the scope.\n\n"
    "Respond ONLY as JSON: "
    '{"action": "retrieve"|"question"|"answer"|"jira", "message": string, '
    '"retrieval_query": string|null, "alt_queries": [string]|null, '
    '"information_need": string|null, '
    '"reason": string|null, "finalized": boolean, '
    '"scope": {"summary": string, "acceptance_criteria": [string]} | null}. '
    "scope is non-null ONLY on a 'jira' action."
)

_DRAFT_SYSTEM = (
    "You are a Product Manager agent. Turn the locked scope into concrete, small, "
    "implementable engineering tickets — units of WORK a human can approve, not "
    "instructions for where to type.\n\n"
    "You do NOT localize. A Planner agent reads the actual code after these tickets "
    "are approved and decides which files and symbols change; it has the code graph "
    "and you have summaries, so anything you guessed here would be overwritten. "
    "Describe the OUTCOME and let it find the code.\n\n"
    "For EACH ticket fill:\n"
    "  - title: what gets delivered, in a line.\n"
    "  - description: what must be true when this ticket is done, and any behaviour "
    "detail from the conversation the Planner and Dev would otherwise have to guess "
    "at (edge cases, formats, error handling). Describe behaviour, not file layout.\n"
    "  - acceptance_criteria: concrete, testable conditions about OBSERVABLE "
    "BEHAVIOR only (given input X, the user sees Y). NEVER name files, symbols, "
    "decorators, or code structure in a criterion — you work from summaries and a "
    "wrong structural claim becomes a hard requirement the reviewer enforces and "
    "the Dev cannot satisfy, deadlocking the pipeline in paid revision rounds.\n"
    "    PROVENANCE: prefix every criterion with [stated] or [assumed]. [stated] "
    "means the user said it, in their words or an unambiguous restatement. "
    "[assumed] means YOU decided it — a delegated judgement call, a convention, "
    "an edge case they never mentioned. Be honest: a judgement call the user "
    "delegated is [assumed], not [stated]. This is the last point a human sees "
    "the requirement before money is spent, and an [assumed] tag is what lets "
    "them catch a decision they would have made differently.\n\n"
    "TICKET MINIMALISM — every ticket and criterion is a paid Dev+QA+Review cycle: "
    "default to ONE ticket covering the whole scope (a fix and its regression tests "
    "are one ticket, not two; implementation and wiring are one ticket). Split only "
    "when the scope truly contains independent deliverables. Do not add criteria "
    "beyond the locked scope's — the Dev implements exactly what criteria say and "
    "the Reviewer enforces them, so an invented one becomes real paid work.\n\n"
    "Respond ONLY as JSON: {\"tickets\": [{\"title\": string, \"description\": string, "
    '"acceptance_criteria": [string], "priority": "low"|"medium"|"high"}]}. '
    "Produce 1-4 tickets (default 1)."
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


def _tree_block(file_tree: list[str] | None) -> str:
    if not file_tree:
        return "(repository not indexed yet — avoid naming specific files)"
    return "\n".join(file_tree[:200])


# --- Context: accumulated knowledge the PM has pulled this turn ---------------
def _render_doc(doc) -> str:
    head = f"[{doc.type}] {doc.name}"
    # Some doc types are inferred, not extracted (business rules are LOW by
    # design) — the PM must see that, or speculation reads with the same
    # authority as AST-verified facts and gets locked into scopes unchecked.
    conf = (doc.confidence or "").upper()
    if conf and conf != "HIGH":
        head += f" (confidence: {conf} — inferred, verify before relying on it)"
    lines = [head]
    if doc.summary:
        lines.append(f"  {doc.summary}")
    desc = doc.content.get("description") or doc.content.get("purpose")
    if desc:
        lines.append(f"  {str(desc)[:400]}")
    files = doc.content.get("files") or []
    if files:
        lines.append(f"  files: {', '.join(files[:8])}")
    syms = doc.content.get("symbols") or []
    if syms:
        lines.append(f"  symbols: {', '.join(str(s) for s in syms[:20])}")
    return "\n".join(lines)


def _bootstrap_context(repo_url: str) -> tuple[list[str], set[str]]:
    """The code graph's structural summary + newest cross-run notes — the
    minimal repo-shaped anchor. Deterministic, no LLM."""
    blocks: list[str] = []
    seen: set[str] = set()
    anchor = graph.bootstrap_text(repo_url)
    if anchor:
        blocks.append(anchor)
    # Newest delivery notes/lessons: what recent delivered work already
    # established (including "NOT MERGED" caveats the PM must not ignore).
    docs = [d for d in store.load_all(repo_url) if d.type in ("delivery_note", "lesson")]
    docs.sort(key=lambda d: str(d.content.get("delivered_at")
                                or d.content.get("updated_at") or ""), reverse=True)
    for doc in docs[:4]:
        blocks.append(_render_doc(doc))
        seen.add(doc.id)
    return blocks, seen


def _retrieve_more(repo_url: str, queries: list[str], seen: set[str]) -> list[str]:
    """Pull not-yet-seen knowledge for every phrasing in `queries`: ranked graph
    localizations, exact code hits, and relevant delivery notes.

    Multiple phrasings are searched, not just the PM's first one. The measured
    failure mode is vocabulary mismatch — the PM restates the user's symptom and
    the index answers with whatever shares those words — so one extra phrasing of
    the same mechanism is the cheapest correction available (search is
    deterministic and free; a mislocalized ticket costs a full Dev+QA+Review
    round)."""
    blocks: list[str] = []
    for query in queries:
        if not query.strip():
            continue
        for part in (knowledge_retriever.localize(repo_url, query),
                     knowledge_retriever.code_hits(repo_url, query)):
            key = f"block:{hash(part)}"
            if part and key not in seen:
                seen.add(key)
                blocks.append(part)
        for doc, _score in knowledge_retriever.notes(repo_url, query, k=3):
            if doc.id in seen:
                continue
            seen.add(doc.id)
            blocks.append(_render_doc(doc))
    return blocks


def _queries(data: dict) -> list[str]:
    """The PM's search phrasings for one retrieve action: the primary query plus
    up to two alternates (capped — each extra phrasing is more context, and past
    two the marginal recall is noise)."""
    out = [str(data.get("retrieval_query") or "").strip()]
    alts = data.get("alt_queries")
    if isinstance(alts, str):
        alts = [alts]
    for a in alts or []:
        a = str(a).strip()
        # Cap AFTER filtering: a blank or duplicate alternate must not spend the
        # budget a real alternate phrasing needs.
        if a and a.lower() not in {q.lower() for q in out}:
            out.append(a)
        if len(out) >= 3:
            break
    return [q for q in out if q]


def _ask(repo_name: str, overview: str, file_tree: list[str] | None,
         context_blocks: list[str], conversation: str, force_answer: bool) -> dict:
    """One structured PM decision over the currently-loaded knowledge."""
    knowledge = "\n\n".join(context_blocks) if context_blocks else "(no structured knowledge loaded)"
    force = ("\n\nIMPORTANT: You have reached the maximum retrieval rounds for this "
             "turn. Do NOT choose 'retrieve' again — either ask a clarifying question "
             "(action='question') or lock the scope (action='jira') with what you have."
             ) if force_answer else ""
    user = (
        f"Repository: {repo_name}\n\nRepository overview:\n{overview or '(none)'}\n\n"
        f"Currently loaded repository knowledge:\n{knowledge}\n\n"
        f"Repository file tree (for orientation only — you are not choosing edit "
        f"targets):\n{_tree_block(file_tree)}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        f"Decide your next action as JSON.{force}"
    )
    return llm.chat(_SYSTEM, user, provider=settings.pm_provider, model=settings.pm_model, json_mode=True)


def scope_turn(repo_name: str, repo_url: str, overview: str, history: list[dict],
               file_tree: list[str] | None = None) -> dict:
    """Run one agentic PM turn. `history` is [{role, content}] (user/agent).

    Returns {action, message, ready, scope, retrieval_rounds, retrieved,
    tokens_in, tokens_out, cost, error}. `ready` is True on a locked scope.
    """
    conversation = "\n".join(
        f"{'PM (human)' if m['role'] == 'user' else 'PM agent'}: {m['content']}" for m in history
    )
    context_blocks, seen = _bootstrap_context(repo_url)
    tin = tout = 0
    cost = 0.0
    retrieved: list[str] = []
    last_err = None

    max_rounds = max(1, settings.pm_max_retrieval_rounds)
    for round_num in range(max_rounds + 1):
        force = round_num >= max_rounds
        r = _ask(repo_name, overview, file_tree, context_blocks, conversation, force)
        tin += r.get("tokens_in", 0) or 0
        tout += r.get("tokens_out", 0) or 0
        cost += r.get("cost", 0.0) or 0.0
        last_err = r.get("error")
        data = _load_json(r.get("text", ""))
        action = data.get("action")

        if action == "retrieve" and not force and data.get("retrieval_query"):
            queries = _queries(data)
            new_blocks = _retrieve_more(repo_url, queries, seen)
            retrieved.append(str(data.get("information_need") or queries[0]))
            if not new_blocks:
                # Nothing new to add — force the PM to act with what it has.
                context_blocks.append("(no additional knowledge found for: "
                                      + "; ".join(queries) + ")")
            else:
                context_blocks.extend(new_blocks)
            continue

        # Terminal action: question | answer | jira (or a malformed/forced turn).
        scope = data.get("scope") if (action == "jira" and isinstance(data.get("scope"), dict)) else None
        ready = scope is not None
        message = data.get("message") or last_err or "Could you tell me a bit more about what you need?"
        return {
            "action": action or "question", "message": message, "ready": ready,
            "scope": _normalize_scope(scope) if scope else None,
            "finalized": bool(data.get("finalized")),
            "retrieval_rounds": round_num, "retrieved": retrieved,
            "tokens_in": tin, "tokens_out": tout, "cost": cost, "error": last_err,
        }

    # Unreachable (loop always returns), but keep the type checker happy.
    return {"action": "question", "message": "Could you tell me a bit more?", "ready": False,
            "scope": None, "finalized": False, "retrieval_rounds": max_rounds,
            "retrieved": retrieved, "tokens_in": tin, "tokens_out": tout, "cost": cost, "error": last_err}


def _normalize_scope(scope: dict) -> dict:
    """The locked scope: what to build and how we'll know it works.

    Any `affected_files`/`target_symbols` a model volunteers here are dropped
    rather than stored. Persisting an unverified guess makes it look like a
    decision to every later reader, and the Planner is about to make the real
    one against the code."""
    criteria = scope.get("acceptance_criteria")
    return {
        "summary": scope.get("summary", ""),
        "acceptance_criteria": ([str(x) for x in criteria if isinstance(x, (str, int))]
                                if isinstance(criteria, list) else []),
    }


def draft_tickets(repo_name: str, scope: dict, context: str, file_tree: list[str] | None = None,
                  knowledge: str = "") -> dict:
    """Draft approvable tickets from the locked scope.

    Tickets carry the requirement, not the localization — the Planner fills in
    `affected_files`/`target_symbols` at pipeline entry, verified against the
    code graph, and writes them back onto these rows."""
    knowledge_block = f"Structured knowledge (architecture / modules / features):\n{knowledge}\n\n" if knowledge else ""
    user = (
        f"Repository: {repo_name}\n\nLocked scope:\n{json.dumps(scope, indent=2)}\n\n"
        f"{knowledge_block}"
        f"Repository file tree (for orientation only):\n{_tree_block(file_tree)}\n\n"
        f"Relevant repo context:\n{context or '(none)'}\n\nProduce the tickets as JSON."
    )
    r = llm.chat(_DRAFT_SYSTEM, user, provider=settings.pm_provider, model=settings.pm_model, json_mode=True)
    data = _load_json(r.get("text", ""))
    tickets = data.get("tickets") if isinstance(data.get("tickets"), list) else []
    tickets = [_normalize_ticket(t) for t in tickets
               if isinstance(t, dict) and t.get("title")]
    return {"tickets": tickets, "tokens_in": r.get("tokens_in", 0),
            "tokens_out": r.get("tokens_out", 0), "cost": r.get("cost", 0.0), "error": r.get("error")}


def _normalize_ticket(t: dict) -> dict:
    """One ticket, coerced. No files/symbols: `Task.affected_files` and
    `Task.target_symbols` stay empty until the Planner verifies them, so an
    empty pair on the board means "not localized yet" rather than "localized to
    nothing"."""
    criteria = t.get("acceptance_criteria")
    return {
        "title": t.get("title"),
        "description": t.get("description", ""),
        "acceptance_criteria": criteria if isinstance(criteria, list) else [],
        "affected_files": [],
        "target_symbols": [],
        "priority": t.get("priority"),
    }
