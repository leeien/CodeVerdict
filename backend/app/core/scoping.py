"""Scoping: the PM clarify-loop, ticket drafting, and launching a scope.

Extracted verbatim from ``routers/sessions.py``. The PM agent owns the
requirement here; localization is deliberately *not* done at drafting time — the
Planner decides which files and symbols change at pipeline entry, with the code
graph in front of it and the working copy clean.
"""

from __future__ import annotations

import logging

from sqlmodel import Session, select

from ..models import ChatMessage, MessageRole, Repo, ScopeSession, Task, TaskStatus
from ..services import background, git_ops, orchestrator, pm_agent, precision, rag
from ..services.knowledge import retriever as knowledge_retriever
from .errors import conflict, not_found, upstream

logger = logging.getLogger(__name__)

# Titles that mean "the user hasn't named this yet", so the first real message
# is allowed to rename the scope.
_PLACEHOLDER_TITLES = (None, "", "Feature scoping", "New requirement")


def next_key(db: Session, repo: Repo) -> str:
    """Next free ticket key. Derived from the MAX existing key number, not the
    row count — tickets get deleted (scope re-lock clears stale drafts), and a
    count-based key would then collide with a surviving ticket's key."""
    keys = db.exec(select(Task.key).where(Task.repo_id == repo.id)).all()
    prefix = f"{repo.key_prefix}-"
    highest = 100
    for k in keys:
        if k and k.startswith(prefix) and k[len(prefix):].isdigit():
            highest = max(highest, int(k[len(prefix):]))
    return f"{prefix}{highest + 1}"


def require(db: Session, session_id: int) -> ScopeSession:
    session = db.get(ScopeSession, session_id)
    if session is None:
        raise not_found("Session")
    return session


def create(db: Session, repo_id: int, kind: str = "feature", title: str = "") -> ScopeSession:
    if db.get(Repo, repo_id) is None:
        raise not_found("Repo")
    session = ScopeSession(repo_id=repo_id, kind=kind, title=title or "New requirement")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def listing(db: Session, repo_id: int | None = None, kind: str | None = None) -> list[ScopeSession]:
    q = select(ScopeSession).order_by(ScopeSession.created_at.desc())
    if repo_id is not None:
        q = q.where(ScopeSession.repo_id == repo_id)
    if kind is not None:
        q = q.where(ScopeSession.kind == kind)
    return list(db.exec(q).all())


def messages(db: Session, session_id: int) -> list[ChatMessage]:
    return list(db.exec(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    ).all())


def detail(db: Session, session_id: int) -> dict:
    session = require(db, session_id)
    return {"session": session, "messages": messages(db, session_id)}


def tasks_for(db: Session, session_id: int) -> list[Task]:
    return list(db.exec(
        select(Task).where(Task.session_id == session_id).order_by(Task.created_at)
    ).all())


def scope_turn(db: Session, session_id: int, content: str) -> dict:
    """One turn of the agent-led PM scoping chat: the PM agent asks a clarifying
    question or locks the scope.

    Returns the raw turn dict alongside the persisted reply, so a caller that
    wants to render *how* the PM worked (what it looked up, whether it locked)
    doesn't have to re-derive it from the chat log.
    """
    session = require(db, session_id)
    repo = db.get(Repo, session.repo_id)

    db.add(ChatMessage(session_id=session_id, role=MessageRole.user.value, content=content))
    # Name the scope after the first user message so the scope switcher is legible.
    if session.title in _PLACEHOLDER_TITLES:
        session.title = content.strip()[:60]
        db.add(session)
    db.commit()

    history = [{"role": m.role, "content": m.content} for m in messages(db, session_id)]

    tree = git_ops.tree_for(repo.git_url)
    # Agentic PM turn: bootstrap knowledge, then the PM retrieves more on demand
    # (up to pm_max_retrieval_rounds) before it asks a question or locks the scope.
    turn = pm_agent.scope_turn(f"{repo.org}/{repo.name}", repo.git_url,
                               repo.kb_overview or "", history, tree)
    scope = turn["scope"] if (turn["ready"] and turn["scope"]) else None
    # Surface what the PM looked up this turn (transparency in the chat log).
    retrieved = turn.get("retrieved") or []
    retr_note = (f"\n\n_[looked up: {'; '.join(retrieved[:4])}]_" if retrieved else "")

    # Gap signal: how hard the PM had to work the KB this turn (0 rounds = the
    # bootstrap context sufficed). Feeds gap-driven view refresh + benchmarks.
    from ..services.knowledge import gaps
    gaps.record(repo.git_url, content, turn.get("retrieval_rounds") or 0,
                turn.get("retrieved") or [])

    # Accumulate PM scoping cost on the session (no Task/AgentRun exists yet).
    session.pm_tokens_input = (session.pm_tokens_input or 0) + (turn.get("tokens_in") or 0)
    session.pm_tokens_output = (session.pm_tokens_output or 0) + (turn.get("tokens_out") or 0)
    session.pm_cost_usd = round((session.pm_cost_usd or 0.0) + (turn.get("cost") or 0.0), 6)
    db.add(session)

    note = ""
    cleared = 0
    if scope:
        session.requirement_summary = scope.get("summary")
        session.acceptance_criteria = scope.get("acceptance_criteria") or []
        session.affected_files = scope.get("affected_files") or []
        session.status = "scoped"
        db.add(session)
        # Scope (re)locked → any not-yet-started draft tickets are now stale. Clear
        # them so tickets always reflect the CURRENT scope; re-draft is required.
        stale = db.exec(
            select(Task).where(
                Task.session_id == session_id,
                Task.status.in_([TaskStatus.backlog.value, TaskStatus.scoped.value]),
            )
        ).all()
        if stale:
            cleared = len(stale)
            for t in stale:
                db.delete(t)
            note = (f"\n\n(Scope changed — cleared {cleared} previous draft ticket"
                    f"{'s' if cleared != 1 else ''}. Draft tickets again to regenerate "
                    f"from the new scope.)")

    db.add(ChatMessage(
        session_id=session_id, role=MessageRole.agent.value, agent_type="pm",
        content=turn["message"] + retr_note + note,
        sources=(scope.get("affected_files", []) if scope else []),
        tokens=turn["tokens_in"] + turn["tokens_out"],
    ))
    db.commit()
    db.refresh(session)
    return {
        "session": session,
        "message": turn["message"],
        "locked": scope is not None,
        "scope": scope,
        "retrieved": retrieved,
        "retrieval_rounds": turn.get("retrieval_rounds") or 0,
        "cleared_drafts": cleared,
        "tokens_in": turn.get("tokens_in") or 0,
        "tokens_out": turn.get("tokens_out") or 0,
        "cost": turn.get("cost") or 0.0,
    }


def draft_tickets(db: Session, session_id: int) -> list[Task]:
    """PM agent drafts tickets from the locked scope. Tickets are created
    UNAPPROVED — no pipeline can run on them until a human approves."""
    session = require(db, session_id)
    repo = db.get(Repo, session.repo_id)
    if not session.acceptance_criteria:
        raise conflict("Scope is not locked yet — keep clarifying with the PM agent first")

    scope = {
        "summary": session.requirement_summary,
        "acceptance_criteria": session.acceptance_criteria,
        "affected_files": session.affected_files,
    }
    # Ground the tickets in the RIGHT knowledge: a story-scoped slice from Deep
    # Analysis, falling back to a focused knowledge-base lookup (best-effort).
    context = ""
    try:
        context = precision.retrieve(session.requirement_summary or "",
                                     use_case="story-breakdown", repo_url=repo.git_url)
    except Exception:  # noqa: BLE001
        context = ""
    if not context:
        try:
            context = rag.ask(
                repo.git_url,
                [{"role": "user",
                  "content": f"Which files are most relevant to: {session.requirement_summary}"}],
                timeout=120,
            )
        except Exception:  # noqa: BLE001
            context = ""

    # Structured-knowledge digest for the locked scope (architecture / modules /
    # features), alongside the chunk-level file context above.
    try:
        knowledge = knowledge_retriever.scope_context(repo.git_url, session.requirement_summary or "")
    except Exception:  # noqa: BLE001
        knowledge = ""

    drafted = pm_agent.draft_tickets(f"{repo.org}/{repo.name}", scope, context,
                                     git_ops.tree_for(repo.git_url), knowledge=knowledge)
    if not drafted["tickets"]:
        raise upstream("PM agent could not draft tickets: "
                       f"{drafted.get('error') or 'no tickets returned'}")

    # Accumulate the draft-tickets PM cost on the session too.
    session.pm_tokens_input = (session.pm_tokens_input or 0) + (drafted.get("tokens_in") or 0)
    session.pm_tokens_output = (session.pm_tokens_output or 0) + (drafted.get("tokens_out") or 0)
    session.pm_cost_usd = round((session.pm_cost_usd or 0.0) + (drafted.get("cost") or 0.0), 6)
    db.add(session)
    db.commit()

    created: list[Task] = []
    for t in drafted["tickets"][:6]:
        task = Task(
            key=next_key(db, repo),
            repo_id=repo.id,
            session_id=session_id,
            title=str(t.get("title"))[:200],
            description=str(t.get("description", "")),
            acceptance_criteria=t.get("acceptance_criteria") or [],
            affected_files=t.get("affected_files") or [],
            target_symbols=t.get("target_symbols") or [],
            status=TaskStatus.scoped.value,
            priority=t.get("priority") if t.get("priority") in ("low", "medium", "high") else "medium",
            approved=False,
            current_agent="pm",
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        created.append(task)
    return created


def approved_subtasks(db: Session, session_id: int) -> list[Task]:
    """The tickets a run would pick up: approved and not yet started."""
    return list(db.exec(
        select(Task).where(
            Task.session_id == session_id, Task.approved == True,  # noqa: E712
            Task.status.in_([TaskStatus.scoped.value, TaskStatus.backlog.value]),
        )
    ).all())


def run_scope(db: Session, session_id: int, *, blocking: bool = False) -> ScopeSession:
    """Run the WHOLE scope as one deliverable: all approved subtasks on one
    branch → one PR.

    ``blocking`` decides who owns the thread. The HTTP API hands it to the
    background pool and returns immediately so the request doesn't sit open for
    the length of a pipeline; the terminal client runs it on a thread it owns so
    it can render progress and know when it's finished.
    """
    session = require(db, session_id)
    if not approved_subtasks(db, session_id):
        raise conflict("No approved subtasks to run — approve tickets first")
    if blocking:
        orchestrator.run_scope(session_id)
    else:
        background.submit(orchestrator.run_scope, session_id)
    return session
