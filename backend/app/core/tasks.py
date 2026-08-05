"""Tickets: the board, the approval gate, the review payload, and delivery.

Extracted verbatim from ``routers/tasks.py`` so the terminal client and the HTTP
API run the same code. ``HTTPException`` became ``CoreError`` with the same
status codes; nothing else about the behaviour changed.
"""

from __future__ import annotations

import re

from sqlmodel import Session, select

from ..config import settings
from ..models import (
    BOARD_ORDER,
    AgentRun,
    LogEntry,
    Repo,
    RunStatus,
    ScopeSession,
    Task,
    User,
    utcnow,
)
from ..services import crypto, git_ops, jira, prompts
from .errors import CoreError, conflict, not_found, refused, upstream

COLUMN_TITLES = {
    "backlog": "Backlog",
    "scoped": "Scoped",
    "approved": "Approved",
    "in_dev": "In Dev",
    "qa": "QA",
    "review": "Review",
    "blocked": "Blocked",
    "pr": "PR",
    "done": "Done",
}

# Board lanes, left→right. "approved" is a virtual lane (not a TaskStatus): it
# holds human-approved tickets that haven't started yet.
BOARD_LANES = ["backlog", "scoped", "approved", "in_dev", "qa", "review", "blocked", "pr", "done"]

# The board columns collapsed into 4 JIRA-style lanes.
PIPELINE_MAP = [
    ("todo", "To Do", ["backlog", "scoped"]),
    ("in_progress", "In Progress", ["in_dev"]),
    ("review", "Review", ["qa", "review", "blocked", "pr"]),
    ("done", "Done", ["done"]),
]


def scope_siblings(db: Session, task: Task) -> list[Task]:
    """Every ticket delivered together with `task` — the subtasks of the same
    scope session sharing its branch (they move as one unit: one branch, one PR,
    one merge). A standalone ticket (no session) is its own only sibling."""
    if not task.session_id:
        return [task]
    return list(db.exec(select(Task).where(Task.session_id == task.session_id,
                                           Task.branch == task.branch)).all())


def require(db: Session, task_id: int) -> Task:
    task = db.get(Task, task_id)
    if task is None:
        raise not_found("Task")
    return task


def by_key(db: Session, key: str, repo_id: int | None = None) -> Task:
    """Look a ticket up the way a person refers to it — ``TASK-101``, case-insensitively.

    The terminal client addresses tickets by key, not by the numeric id the web
    UI carries around in its DOM.
    """
    q = select(Task).where(Task.key == key.strip().upper())
    if repo_id is not None:
        q = q.where(Task.repo_id == repo_id)
    task = db.exec(q).first()
    if task is None:
        raise not_found(f"Ticket {key.strip().upper()}")
    return task


def resolve(db: Session, ref: str | int, repo_id: int | None = None) -> Task:
    """Accept either a numeric id or a ticket key."""
    if isinstance(ref, int) or str(ref).isdigit():
        return require(db, int(ref))
    return by_key(db, str(ref), repo_id)


def listing(db: Session, repo_id: int | None = None, status: str | None = None) -> list[Task]:
    q = select(Task).order_by(Task.updated_at.desc())
    if repo_id is not None:
        q = q.where(Task.repo_id == repo_id)
    if status is not None:
        q = q.where(Task.status == status)
    return list(db.exec(q).all())


def board(db: Session, repo_id: int | None = None) -> list[dict]:
    """Board lanes as plain dicts: ``[{status, title, tasks: [Task, …]}, …]``."""
    q = select(Task)
    if repo_id is not None:
        q = q.where(Task.repo_id == repo_id)
    tasks = db.exec(q).all()

    # Pull human-approved-but-not-yet-started tickets into their own "Approved"
    # lane so the ready-to-run queue is visible apart from Scoped/Backlog.
    approved_pending = [t for t in tasks if t.status in ("backlog", "scoped") and t.approved]
    approved_ids = {t.id for t in approved_pending}

    by_status: dict[str, list[Task]] = {s: [] for s in BOARD_ORDER}
    for t in tasks:
        if t.id in approved_ids:
            continue  # shown in the Approved lane instead
        by_status.setdefault(t.status, []).append(t)
    by_status["approved"] = approved_pending

    return [
        {"status": s, "title": COLUMN_TITLES.get(s, s.title()), "tasks": by_status.get(s, [])}
        for s in BOARD_LANES
    ]


def _progress_for(db: Session, task: Task) -> int:
    """Rough % for an in-progress task, from its running dev run's log volume."""
    run = db.exec(
        select(AgentRun)
        .where(AgentRun.task_id == task.id, AgentRun.status == RunStatus.running.value)
        .order_by(AgentRun.created_at.desc())
    ).first()
    if run is None:
        return 50
    lines = len(db.exec(select(LogEntry).where(LogEntry.run_id == run.id)).all())
    return max(15, min(95, round(lines / 6 * 100)))


def pipeline(db: Session, repo_id: int | None = None) -> dict:
    q = select(Task)
    if repo_id is not None:
        q = q.where(Task.repo_id == repo_id)
    tasks = db.exec(q).all()

    columns = []
    for lane_id, title, statuses in PIPELINE_MAP:
        cards = []
        for t in tasks:
            if t.status in statuses:
                cards.append({
                    "id": t.id,
                    "key": t.key,
                    "title": t.title,
                    "agent": t.current_agent,
                    "approved": t.approved,
                    "progress": _progress_for(db, t) if lane_id == "in_progress" else None,
                })
        columns.append({"id": lane_id, "title": title, "count": len(cards), "cards": cards})
    return {"columns": columns}


def update(db: Session, task_id: int, values: dict) -> Task:
    task = require(db, task_id)
    # Status is advanced by the pipeline gates, never by a generic PATCH.
    # Otherwise a caller could set `pr` or `done` without QA, review, or merge
    # validation and make it look like a genuine delivery.
    if "status" in values and values["status"] != task.status:
        raise conflict("Task status is pipeline-controlled; use the approval, "
                       "run, review, or delivery action instead.")
    for field, value in values.items():
        setattr(task, field, value)
    task.updated_at = utcnow()
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def approve(db: Session, task_id: int) -> Task:
    """PM approval gate: a task must be approved before any agent pipeline can run."""
    task = require(db, task_id)
    task.approved = True
    task.updated_at = utcnow()

    # On approval, push the ticket to the Jira board (once). No-op if Jira isn't
    # configured; never blocks approval if the Jira call fails.
    if jira.is_configured() and not task.jira_key:
        story = jira.push_story(task.key, task.title, task.description, task.acceptance_criteria)
        if story:
            task.jira_key = story.get("key")
            task.jira_url = story.get("url")

    db.add(task)
    db.commit()
    db.refresh(task)
    return task


# NOTE: There is deliberately no per-ticket run operation. A scope is the unit
# of work — every approved subtask is delivered together on one branch as one
# PR. Run a scope via core.scoping.run_scope().


# --- QA / PR review ----------------------------------------------------------
def parse_diff(diff_text: str, max_lines: int = 400) -> list[dict]:
    """Turn a unified git diff into renderable lines with new-side line numbers.
    Each changed file contributes a ``type: "file"`` header row so multi-file
    diffs render grouped per file instead of as one anonymous stream."""
    lines: list[dict] = []
    newno = 0
    truncated = False
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git"):
            m = re.search(r" b/(.+)$", raw)
            lines.append({"n": "", "text": m.group(1) if m else raw, "type": "file"})
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            newno = int(m.group(1)) if m else newno
            lines.append({"n": "", "text": raw, "type": "hunk"})
        elif raw.startswith(("+++", "---", "index ", "new file", "deleted file", "rename ")):
            continue
        elif raw.startswith("+"):
            lines.append({"n": newno, "text": raw[1:], "type": "add"})
            newno += 1
        elif raw.startswith("-"):
            lines.append({"n": "", "text": raw[1:], "type": "del"})
        else:
            lines.append({"n": newno, "text": raw[1:] if raw.startswith(" ") else raw, "type": "ctx"})
            newno += 1
        if len(lines) >= max_lines:
            truncated = True
            break
    if truncated:
        lines.append({"n": "", "text": "… diff truncated — see the branch for the full change", "type": "hunk"})
    return lines


def review(db: Session, task_id: int, diff_lines: int = 400) -> dict:
    """Real final-review payload: actual git diff + actual QA/Review agent output."""
    task = require(db, task_id)
    repo = db.get(Repo, task.repo_id)

    diff_text, ins, dels = "", 0, 0
    if repo is not None and task.branch:
        path = str(git_ops.workdir(repo.git_url))
        try:
            diff_text = git_ops.diff(path, ref=task.branch)
            ins, dels = git_ops.diff_stat(path, ref=task.branch)
        except Exception:  # noqa: BLE001
            diff_text = ""

    qa = task.qa_summary or ""
    review_text = task.review_summary or ""
    cov_match = re.search(r"(\d{1,3})\s*%", qa)
    coverage = min(100, int(cov_match.group(1))) if cov_match else None
    pr_number = task.pr_url.rstrip("/").split("/")[-1] if task.pr_url else "—"

    # The Planner's verified plan for this scope — what the change was SUPPOSED
    # to do, decided against the code graph before any code was written. It is
    # the only artifact that lets a human read the diff as "did it do the agreed
    # thing" rather than "does this look plausible".
    session = db.get(ScopeSession, task.session_id) if task.session_id else None
    plan = (session.plan if session else None) or {}

    checks = [
        {"label": "Implementation planned", "ok": bool(plan.get("steps"))},
        {"label": "Dev changes committed", "ok": bool(diff_text.strip())},
        {"label": "QA reviewed", "ok": bool(qa)},
        {"label": "Code review complete", "ok": bool(review_text)},
        {"label": "Pull request opened", "ok": bool(task.pr_url)},
    ]

    # The jury's structured decision, when the panel reviewed this delivery. The
    # prose review_summary is still returned alongside it — the panel is a
    # setting, so any given task may have been reviewed either way.
    jury = task.review_findings or {}
    if jury.get("verdict"):
        # Located by label, not by index: this row was addressed positionally and
        # silently relabelled the wrong check the first time a step was added
        # above it.
        for i, c in enumerate(checks):
            if c["label"] == "Code review complete":
                checks[i] = {"label": f"Jury review — {jury['verdict'].lower()}",
                             "ok": jury["verdict"] == "APPROVED"}
                break

    observations = []
    if review_text:
        verdict = jury.get("verdict") or ("CHANGES REQUESTED"
                                          if "CHANGES REQUESTED" in review_text.upper() else "")
        changes = verdict == "CHANGES REQUESTED" or "FAIL" in qa.upper()
        observations.append({
            "level": "warn" if changes else "info",
            "title": "Review verdict" if changes else "Reviewer notes",
            "text": (review_text[:400] + ("…" if len(review_text) > 400 else "")),
        })

    qa_notes = [p.strip() for p in re.split(r"\n{1,}", qa) if p.strip()] or \
        ["QA agent has not run for this task yet."]

    return {
        "task": {"id": task.id, "key": task.key, "title": task.title, "status": task.status},
        "pr": {
            "number": pr_number,
            "title": task.title,
            "status": "MERGED" if task.status == "done" else ("OPEN" if task.pr_url else "DRAFT"),
            "url": task.pr_url,
            "branch": task.branch,
            "insertions": ins,
            "deletions": dels,
            "diff": parse_diff(diff_text, diff_lines) or
                    [{"n": "", "text": "No diff yet — run the pipeline on this task.", "type": "ctx"}],
        },
        "checks": checks,
        "plan": plan,
        "observations": observations,
        "qa_notes": qa_notes,
        "review_summary": review_text,
        "coverage": coverage,
        "jury": jury,
        "ready_for_deployment": (
            task.status in ("pr", "done") and _qa_approved(qa)
            and _review_approved(review_text, jury)),
    }


def _qa_approved(text: str) -> bool:
    """Require an explicit successful QA outcome; empty/inconclusive is not pass."""
    upper = (text or "").upper()
    return bool(re.search(r"VERDICT\s*:\s*(?:PASS|APPROVED)\b", upper)) \
        and "INCONCLUSIVE" not in upper


def _review_approved(text: str, jury: dict | None = None) -> bool:
    """Require an explicit approval from either review mode or the deterministic gate."""
    verdict = (jury or {}).get("verdict")
    if verdict:
        return verdict == "APPROVED" and not (jury or {}).get("unresolved_blocking")
    upper = (text or "").upper()
    return ("FAST PATH:" in upper or
            bool(re.search(r"VERDICT\s*:\s*APPROVED\b", upper))) \
        and "INCONCLUSIVE" not in upper


def create_pr(db: Session, task_id: int, user: User) -> dict:
    """Human-triggered PR creation: push the task's delivered branch and open the
    PR. If the acting user has connected their own GitHub account, the PR is
    opened AS them (their token) — and if they don't have push access to the
    origin repo, this auto-forks it to their account first and opens a cross-repo
    PR, exactly like a normal open-source contribution. With no connected account
    it falls back to the shared bot token, then the host's gh login. Commits stay
    agent-authored either way. Covers the open_real_pr=off flow (deliver to a
    branch, a person decides per-delivery)."""
    task = require(db, task_id)
    if task.pr_url:
        return {"url": task.pr_url, "created": False}
    if settings.demo_mode:
        raise refused("Demo mode is on — the platform won't push to real repos. "
                      "Turn it off in Settings → Delivery & safety first.")
    if task.status not in ("pr", "done"):
        raise conflict("This ticket hasn't been delivered yet — run the pipeline first.")
    if not task.branch:
        raise conflict("No delivery branch is recorded on this ticket.")
    repo = db.get(Repo, task.repo_id)
    if repo is None:
        raise not_found("Repo")

    path = str(git_ops.workdir(repo.git_url))
    diff = git_ops.diff(path, ref=task.branch)
    if not diff.strip():
        raise conflict("The delivery branch has no changes to open a PR for.")

    # Scope deliveries share one branch → one PR covering all sibling subtasks.
    siblings = scope_siblings(db, task)
    session = db.get(ScopeSession, task.session_id) if task.session_id else None
    title = f"Scope: {(session.title if session else task.title)[:60]}"
    if len(siblings) > 1:
        subs = [{"key": t.key, "title": t.title} for t in siblings]
        body = prompts.scope_pr_body(session.title if session else task.title, subs,
                                     task.qa_summary or "")
    else:
        body = prompts.pr_body(task.key, task.title, task.qa_summary or "")

    # Prefer the acting user's own connected GitHub account, so the PR is theirs.
    user_token = crypto.decrypt(user.github_token) if user.github_token else None
    author = user.github_login or settings.agent_git_name

    fork_target: tuple[str, str] | None = None
    pr_repo: str | None = None
    head_owner: str | None = None
    if user_token:
        origin = git_ops.repo_owner_name(repo.git_url)
        if origin and git_ops.github_can_push(*origin, user_token) is False:
            # No write access to origin under this account — fork it (or reuse
            # an existing fork) and open a cross-repo PR, same as any GitHub
            # contributor without collaborator access would.
            try:
                fork = git_ops.fork_repo(*origin, user_token)
            except RuntimeError as e:
                raise upstream(f"Fork failed: {str(e)[:300]}")
            fork_target = (fork["owner"], fork["name"])
            pr_repo = f"{origin[0]}/{origin[1]}"
            head_owner = fork["owner"]

    def _deliver() -> str:
        git_ops.push(path, task.branch, token=user_token, target=fork_target)
        return git_ops.gh_pr_create(path, title, body, token=user_token,
                                    repo=pr_repo, head_owner=head_owner, branch=task.branch)

    try:
        url = _deliver()
    except Exception as e:
        # github_can_push() can be undeterminable (repo lookup hiccup) and let
        # a doomed direct push through — if it's a permission error and we
        # haven't already tried a fork, fork now and retry once.
        msg = str(e)
        if fork_target is None and user_token and "denied" in msg.lower():
            origin = git_ops.repo_owner_name(repo.git_url)
            try:
                fork = git_ops.fork_repo(*origin, user_token) if origin else None
            except RuntimeError:
                fork = None
            if fork:
                fork_target = (fork["owner"], fork["name"])
                pr_repo, head_owner = f"{origin[0]}/{origin[1]}", fork["owner"]
                try:
                    url = _deliver()
                except Exception as e2:  # noqa: BLE001
                    raise upstream(f"Push/PR failed: {str(e2)[:300]}")
            else:
                raise upstream(f"Push/PR failed: {msg[:300]}")
        else:
            raise upstream(f"Push/PR failed: {msg[:300]}")
    if not url.startswith("http"):
        raise upstream(f"gh did not return a PR URL: {url[:300]}")

    now = utcnow()
    for t in siblings:
        t.pr_url = url
        t.updated_at = now
        db.add(t)
    db.commit()
    return {"url": url, "created": True, "tasks": [t.key for t in siblings], "author": author,
            "forked": fork_target is not None,
            "fork_repo": f"{fork_target[0]}/{fork_target[1]}" if fork_target else None}


def merge(db: Session, task_id: int) -> dict:
    """Mark the delivery done. A scope ships as one unit, so this marks every
    sibling subtask done together — they share the same branch and PR."""
    task = require(db, task_id)
    siblings = scope_siblings(db, task)
    if all(t.status == "done" for t in siblings):
        return {"ok": True, "tasks": [t.key for t in siblings]}
    if any(t.status != "pr" for t in siblings):
        raise conflict("The whole scope must reach the PR lane before it can be merged.")
    if not settings.demo_mode and any(not t.pr_url for t in siblings):
        raise conflict("A real pull request must exist before the delivery can be merged.")
    if any(not _qa_approved(t.qa_summary or "") for t in siblings) or any(
            not _review_approved(t.review_summary or "", t.review_findings or {})
            for t in siblings):
        raise conflict("QA and code review must explicitly approve this delivery before merge.")
    now = utcnow()
    for t in siblings:
        t.status = "done"
        t.current_agent = None
        t.updated_at = now
        db.add(t)
    db.commit()
    return {"ok": True, "tasks": [t.key for t in siblings]}


def request_changes(db: Session, task_id: int, note: str = "") -> dict:
    """Send the scope back for changes: the note is recorded on the ticket it was
    raised against, and the whole scope returns to the (approved) Scoped lane so
    Run scope re-runs it as one unit. Scopes deliver together, so they re-work
    together — re-running just one subtask would desync the shared branch."""
    task = require(db, task_id)
    siblings = scope_siblings(db, task)
    if any(t.status == "done" for t in siblings):
        raise conflict("This scope is already merged/done — reopen isn't allowed")
    if any(t.status in ("in_dev", "qa", "review") for t in siblings):
        raise conflict("Pipeline is running — wait for it to finish, then request changes")
    if (note or "").strip():
        task.description = f"{task.description or ''}\n\n[Change request] {note.strip()}".strip()
    now = utcnow()
    for t in siblings:
        t.status = "scoped"        # back to the drafts lane…
        t.approved = True          # …but still approved, so Run scope picks it up
        t.current_agent = "pm"
        t.pr_url = None            # the prior PR no longer reflects the pending re-run
        t.updated_at = now
        db.add(t)
    db.commit()
    return {"ok": True, "tasks": [t.key for t in siblings]}


def detail(db: Session, task_id: int) -> dict:
    """Full ticket detail: the ticket, its scope, and its runs."""
    task = require(db, task_id)
    session = db.get(ScopeSession, task.session_id) if task.session_id else None
    runs = db.exec(
        select(AgentRun).where(AgentRun.task_id == task_id).order_by(AgentRun.created_at)
    ).all()
    siblings = scope_siblings(db, task)
    return {
        "task": {
            "id": task.id, "key": task.key, "title": task.title, "description": task.description,
            "acceptance_criteria": task.acceptance_criteria, "status": task.status,
            "priority": task.priority, "approved": task.approved, "current_agent": task.current_agent,
            "branch": task.branch, "pr_url": task.pr_url, "token_cost": task.token_cost,
            "session_id": task.session_id,
        },
        "scope": ({"title": session.title, "summary": session.requirement_summary,
                   "acceptance_criteria": session.acceptance_criteria,
                   "sibling_count": len(siblings)} if session else None),
        "runs": [{"id": r.id, "agent_type": r.agent_type, "status": r.status,
                  "tokens": r.tokens_input + r.tokens_output, "duration_ms": r.duration_ms,
                  "model": r.model} for r in runs],
    }


__all__ = [
    "BOARD_LANES", "COLUMN_TITLES", "CoreError", "approve", "board", "by_key", "create_pr",
    "detail", "listing", "merge", "parse_diff", "pipeline", "request_changes", "require",
    "resolve", "review", "scope_siblings", "update",
]
