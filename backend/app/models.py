"""SQLModel tables for the SDLC agent platform.

Statuses are stored as plain strings (see the *Enum classes for the allowed
values) to keep SQLite migrations trivial. Validation of inbound values happens
in the Pydantic schemas.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import JSON, Column, Text
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


# --- Enumerations (allowed string values) -----------------------------------
class KBStatus(str, enum.Enum):
    pending = "pending"
    indexing = "indexing"
    ready = "ready"
    failed = "failed"


class TaskStatus(str, enum.Enum):
    backlog = "backlog"
    scoped = "scoped"
    in_dev = "in_dev"
    qa = "qa"
    review = "review"
    blocked = "blocked"
    pr = "pr"
    done = "done"


# Canonical left-to-right order of the Kanban board.
BOARD_ORDER: list[str] = [s.value for s in TaskStatus]


class AgentType(str, enum.Enum):
    pm = "pm"          # Product Manager — scopes requirements, creates tasks
    plan = "plan"      # Planner — decides HOW: verified localization + step order
    dev = "dev"        # writes the code
    qa = "qa"          # runs unit tests
    review = "review"  # reviews code against the requirements
    pr = "pr"          # opens the pull request


class RunStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Severity(str, enum.Enum):
    info = "info"
    success = "success"
    warn = "warn"
    error = "error"


class MessageRole(str, enum.Enum):
    user = "user"
    agent = "agent"


class UserRole(str, enum.Enum):
    """Access levels, most → least privileged.

    admin  — everything, plus Settings and user management.
    member — operates the pipeline (ingest, scope, approve, run, merge).
    viewer — read-only: browse the board, runs, logs, and costs.
    """

    admin = "admin"
    member = "member"
    viewer = "viewer"


# Privilege order used by the role guards (higher = more privileged).
ROLE_RANK: dict[str, int] = {"viewer": 0, "member": 1, "admin": 2}


# --- Tables -----------------------------------------------------------------
class Repo(SQLModel, table=True):
    """A git repository ingested into the knowledge base."""

    id: int | None = Field(default=None, primary_key=True)
    name: str
    org: str
    git_url: str
    default_branch: str = "main"
    key_prefix: str = "TASK"          # story-key prefix, e.g. "AL" -> AL-101
    primary_language: str | None = None
    languages: list = Field(default_factory=list, sa_column=Column(JSON))

    # Knowledge-base build state
    kb_status: str = Field(default=KBStatus.pending.value, index=True)
    kb_progress: int = 0          # 0–100
    kb_step: str | None = None    # e.g. "Embedding files — 3,204 / 5,110"
    kb_doc_count: int = 0         # indexed source files (chunk RAG)
    kb_error: str | None = None
    kb_overview: str | None = Field(default=None, sa_column=Column(Text))

    # Structured knowledge (multi-view): how many view documents were generated,
    # and per-domain counts for the UI (e.g. {"features": 8, "modules": 5}).
    kb_knowledge_count: int = 0
    kb_views: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # One-time LLM cost of the structured-knowledge build (cumulative across
    # rebuilds), so the KB investment is visible next to per-task run costs.
    kb_tokens_in: int = 0
    kb_tokens_out: int = 0
    kb_cost_usd: float = 0.0

    last_indexed_at: datetime | None = None

    created_at: datetime = Field(default_factory=utcnow)


class ScopeSession(SQLModel, table=True):
    """A requirement-clarification conversation with the PM agent, grounded in
    a repo's knowledge base."""

    id: int | None = Field(default=None, primary_key=True)
    repo_id: int = Field(foreign_key="repo.id", index=True)
    title: str
    kind: str = Field(default="kb")       # kb = repo Q&A | pm = PM scoping chat
    status: str = Field(default="open")   # open | scoped | delivered | failed

    # Scope panel — filled in as the conversation converges.
    requirement_summary: str | None = Field(default=None, sa_column=Column(Text))
    acceptance_criteria: list = Field(default_factory=list, sa_column=Column(JSON))
    affected_files: list = Field(default_factory=list, sa_column=Column(JSON))

    # The Planner's verified implementation plan for this scope (services/planner.py):
    # ordered steps with real file:line pins, blast radius and the tests to extend.
    # Written at pipeline entry and read by Dev, QA and the jury.
    plan: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # PM scoping cost (clarify + draft-tickets) accumulated here, since those
    # calls happen before any Task/AgentRun exists to attribute them to.
    pm_tokens_input: int = 0
    pm_tokens_output: int = 0
    pm_cost_usd: float = 0.0

    created_at: datetime = Field(default_factory=utcnow)


class ChatMessage(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="scopesession.id", index=True)
    role: str                                   # user | agent
    agent_type: str | None = None               # pm (when role == agent)
    content: str = Field(sa_column=Column(Text))
    sources: list = Field(default_factory=list, sa_column=Column(JSON))  # KB file citations
    tokens: int = 0
    created_at: datetime = Field(default_factory=utcnow)


class Task(SQLModel, table=True):
    """A ticket that flows across the Kanban board through the agent pipeline."""

    id: int | None = Field(default=None, primary_key=True)
    key: str = Field(index=True)                # human id, e.g. "TASK-142"
    repo_id: int = Field(foreign_key="repo.id", index=True)
    session_id: int | None = Field(default=None, foreign_key="scopesession.id")

    title: str
    description: str | None = Field(default=None, sa_column=Column(Text))
    acceptance_criteria: list = Field(default_factory=list, sa_column=Column(JSON))
    # PM-supplied localization carried forward to the Dev agent so it doesn't have
    # to rediscover the repo: exact files to change and symbols to edit/call.
    affected_files: list = Field(default_factory=list, sa_column=Column(JSON))
    target_symbols: list = Field(default_factory=list, sa_column=Column(JSON))

    status: str = Field(default=TaskStatus.backlog.value, index=True)
    priority: str = "medium"                    # low | medium | high
    approved: bool = False                      # PM approval gate before any agent runs
    current_agent: str | None = None            # AgentType currently owning it

    pr_url: str | None = None
    branch: str | None = None
    qa_summary: str | None = Field(default=None, sa_column=Column(Text))
    review_summary: str | None = Field(default=None, sa_column=Column(Text))
    # The jury's structured verdict (see services/jury): per-juror opinions plus
    # the foreperson's synthesis. review_summary stays the rendered prose so the
    # board, PR body and knowledge write-back keep working unchanged.
    review_findings: dict = Field(default_factory=dict, sa_column=Column(JSON))
    token_cost: int = 0                         # rolled up from agent runs

    # Jira story created when the ticket is approved (if Jira is configured).
    jira_key: str | None = None
    jira_url: str | None = None

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class AgentRun(SQLModel, table=True):
    """One invocation of an agent against a task."""

    id: int | None = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="task.id", index=True)
    agent_type: str = Field(index=True)
    status: str = Field(default=RunStatus.queued.value, index=True)
    model: str = "claude-opus (Claude Code CLI)"

    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    # True when the backend did not report tokens and/or cost for this run — the
    # stored zeros then mean "unknown", not "free" (UI shows them as such).
    usage_unknown: bool = False

    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    error: str | None = None

    created_at: datetime = Field(default_factory=utcnow)


class LogEntry(SQLModel, table=True):
    """A single streamed line from an agent run's console."""

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="agentrun.id", index=True)
    ts: datetime = Field(default_factory=utcnow)
    severity: str = Field(default=Severity.info.value)
    message: str = Field(sa_column=Column(Text))


class User(SQLModel, table=True):
    """An operator account (see UserRole for what each role may do)."""

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    role: str = Field(default=UserRole.member.value)
    created_at: datetime = Field(default_factory=utcnow)
    # Optional per-user GitHub connection: the operator pastes their own PAT, so
    # PRs they open from the board are authored by THEIR account. github_token is
    # Fernet-encrypted at rest (see crypto.py); login/name are cached for display.
    github_login: str | None = None
    github_name: str | None = None
    github_token: str | None = None


class AuthSession(SQLModel, table=True):
    """A browser login session (opaque token stored in an httponly cookie)."""

    id: int | None = Field(default=None, primary_key=True)
    token: str = Field(index=True, unique=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime


class Judge(SQLModel, table=True):
    """One seat on the review panel.

    ``persona`` names a brief from ``services/jury/personas.py`` (or "custom",
    in which case ``focus`` carries the operator's own brief). ``provider`` /
    ``model`` are per-judge overrides — empty means "inherit the Review stage",
    which is what a fresh install falls back to before any keys are set.
    """

    id: int | None = Field(default=None, primary_key=True)
    name: str
    persona: str = "custom"
    enabled: bool = True
    position: int = 0                 # panel order, ascending
    provider: str = ""                # "" = inherit review_provider
    model: str = ""                   # "" = inherit review_model
    focus: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=utcnow)


class AppSetting(SQLModel, table=True):
    """A runtime configuration override (key → JSON-encoded value), applied on
    top of env/.env settings at startup and whenever it changes. Lets operators
    manage keys, models, and pipeline limits from the Settings screen without
    restarting the server."""

    key: str = Field(primary_key=True)
    value: str = Field(sa_column=Column(Text))  # JSON-encoded
    updated_at: datetime = Field(default_factory=utcnow)
