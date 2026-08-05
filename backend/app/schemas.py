"""Request/response schemas for the API layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .models import AgentRun, ChatMessage, ScopeSession, Task


# --- Repos ------------------------------------------------------------------
class IngestRepoRequest(BaseModel):
    git_url: str
    default_branch: str = "main"


# --- Scope chat -------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    repo_id: int
    title: str | None = None
    kind: str = "kb"  # "kb" (repo Q&A) or "pm" (PM scoping chat)


class PostMessageRequest(BaseModel):
    content: str


class SessionDetail(BaseModel):
    session: ScopeSession
    messages: list[ChatMessage]


# --- Tasks / board ----------------------------------------------------------
class CreateTasksResponse(BaseModel):
    created: list[Task]


class UpdateTaskRequest(BaseModel):
    status: Literal[
        "backlog", "scoped", "in_dev", "qa", "review", "blocked", "pr", "done"
    ] | None = None
    priority: Literal["low", "medium", "high"] | None = None
    title: str | None = None


class BoardColumn(BaseModel):
    status: str
    title: str
    tasks: list[Task]


class Board(BaseModel):
    columns: list[BoardColumn]


# --- Agents -----------------------------------------------------------------
class RunDetail(BaseModel):
    run: AgentRun
    logs: list[dict]


class AgentStats(BaseModel):
    total_tokens: int
    total_cost_usd: float
    active_agents: int
    tasks_completed_today: int
