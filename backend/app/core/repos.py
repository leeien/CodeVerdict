"""Repositories and their knowledge bases.

Extracted verbatim from ``routers/repos.py``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from sqlmodel import Session, select

from ..config import settings
from ..models import AgentRun, ChatMessage, KBStatus, LogEntry, Repo, ScopeSession, Task
from ..services import background, git_ops, rag
from .errors import conflict, not_found

logger = logging.getLogger(__name__)


def parse_git_url(url: str) -> tuple[str, str]:
    """Best-effort (org, name) from a git URL."""
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    parts = [p for p in cleaned.replace(":", "/").split("/") if p]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "unknown", parts[-1] if parts else "repo"


def derive_prefix(name: str) -> str:
    """Story-key prefix from a repo name, e.g. 'payments-api' -> 'PA'."""
    words = [w for w in name.replace("_", "-").split("-") if w]
    letters = "".join(w[0] for w in words[:3]).upper()
    return letters or name[:2].upper() or "TASK"


def listing(db: Session) -> list[Repo]:
    return list(db.exec(select(Repo).order_by(Repo.created_at.desc())).all())


def require(db: Session, repo_id: int) -> Repo:
    repo = db.get(Repo, repo_id)
    if repo is None:
        raise not_found("Repo")
    return repo


def resolve(db: Session, ref: str | int) -> Repo:
    """Accept a numeric id, a git URL, or a bare/qualified repo name.

    The terminal client is addressed by humans, who say "rich" or
    "Textualize/rich" — not "3".
    """
    if isinstance(ref, int) or str(ref).isdigit():
        return require(db, int(ref))
    needle = str(ref).strip().lower()
    repos = listing(db)
    for r in repos:
        if r.git_url.lower() == needle:
            return r
    for r in repos:
        if f"{r.org}/{r.name}".lower() == needle or r.name.lower() == needle:
            return r
    matches = [r for r in repos if needle in f"{r.org}/{r.name}".lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(f"{r.org}/{r.name}" for r in matches[:5])
        raise conflict(f"'{ref}' matches several repositories: {names}")
    raise not_found(f"Repository '{ref}'")


def ingest(db: Session, git_url: str, default_branch: str = "main") -> Repo:
    """Register a repo and kick off knowledge-base indexing off-thread."""
    # One Repo row per git URL: duplicates would share the same workspace clone
    # and knowledge slug on disk, so deleting either row would destroy the
    # other's data. Re-ingesting an existing repo is what reindex is for.
    url = git_url.strip()
    existing = db.exec(select(Repo).where(Repo.git_url == url)).first()
    if existing is not None:
        raise conflict(f"This repository is already ingested (id {existing.id}) — "
                       "reindex it to rebuild its knowledge base.")
    org, name = parse_git_url(url)
    repo = Repo(
        name=name,
        org=org,
        git_url=url,
        default_branch=default_branch,
        key_prefix=derive_prefix(name),
        kb_status=KBStatus.pending.value,
    )
    db.add(repo)
    db.commit()
    db.refresh(repo)

    background.submit(rag.ingest, repo.id)
    return repo


def reindex(db: Session, repo_id: int) -> Repo:
    repo = require(db, repo_id)
    background.submit(rag.ingest, repo.id)
    return repo


def reconcile_interrupted(db: Session) -> int:
    """Fail any repo left mid-index by a dead process. Called at startup.

    Indexing runs on this process's thread pool, so nothing can still be
    indexing when the process has only just come up: an ``indexing`` row at boot
    is always the corpse of a run that was killed (crash, Ctrl-C, OOM). Left
    alone it stays ``indexing`` forever, and `/kb` reports the ghost as live
    work — which is indistinguishable from a real ingest and sends people
    looking for a background job that isn't there. Stamping it failed with the
    remedy in the message is the only honest reading of that state.
    """
    stuck = db.exec(select(Repo).where(Repo.kb_status == KBStatus.indexing.value)).all()
    for repo in stuck:
        repo.kb_status = KBStatus.failed.value
        repo.kb_error = (f"Indexing was interrupted at {repo.kb_progress or 0}% "
                         f"({repo.kb_step or 'unknown step'}) — /kb reindex to restart.")
        repo.kb_step = "Interrupted — /kb reindex to restart"
        db.add(repo)
    if stuck:
        db.commit()
        logger.info("reconciled %d interrupted KB index run(s)", len(stuck))
    return len(stuck)


def delete(db: Session, repo_id: int) -> None:
    """Remove a repository and everything derived from it: sessions, messages,
    tasks, runs, logs, plus (best-effort) its workspace clone and knowledge dir."""
    repo = require(db, repo_id)

    task_ids = db.exec(select(Task.id).where(Task.repo_id == repo_id)).all()
    if task_ids:
        run_ids = db.exec(select(AgentRun.id).where(AgentRun.task_id.in_(task_ids))).all()
        if run_ids:
            for log in db.exec(select(LogEntry).where(LogEntry.run_id.in_(run_ids))).all():
                db.delete(log)
            for run in db.exec(select(AgentRun).where(AgentRun.id.in_(run_ids))).all():
                db.delete(run)
        for task in db.exec(select(Task).where(Task.id.in_(task_ids))).all():
            db.delete(task)

    session_ids = db.exec(select(ScopeSession.id).where(ScopeSession.repo_id == repo_id)).all()
    if session_ids:
        for msg in db.exec(select(ChatMessage).where(ChatMessage.session_id.in_(session_ids))).all():
            db.delete(msg)
        for sess in db.exec(select(ScopeSession).where(ScopeSession.id.in_(session_ids))).all():
            db.delete(sess)

    db.delete(repo)
    db.commit()

    # On-disk artifacts: the workspace clone and the structured-knowledge dir.
    # Best-effort — a failure here must not resurrect the DB rows.
    for path in (git_ops.workdir(repo.git_url),
                 Path(settings.knowledge_dir).resolve() / git_ops.slug(repo.git_url)):
        try:
            if path.exists():
                shutil.rmtree(path)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", path, exc)


def knowledge(db: Session, repo_id: int) -> dict:
    """The structured, multi-view knowledge generated for this repo, grouped by
    domain (architecture, modules, features, workflows, …)."""
    repo = require(db, repo_id)
    from ..services.knowledge import retriever as knowledge_retriever

    return knowledge_retriever.views(repo.git_url)
