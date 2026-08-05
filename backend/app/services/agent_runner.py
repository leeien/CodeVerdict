"""Helpers to record a real agent run: create the AgentRun, stream log lines,
and finalize with real token usage / cost.

Every write here also publishes to ``services.events`` so an in-process client
(the terminal UI) can follow a run live instead of polling the rows back out.
The DB remains the durable record; the bus is a no-op when nobody is listening.
"""

import time

from sqlmodel import Session

from ..database import engine
from ..models import AgentRun, LogEntry, RunStatus, Severity, Task, utcnow
from . import events

# Placeholder shown only until the caller resolves the real provider/model and
# calls set_model(). "pr" never calls an LLM (git/gh only) so it keeps its label.
_MODEL_LABEL = {
    "pm": "resolving…",
    "dev": "resolving…",
    "review": "resolving…",
    "pr": "git/gh (no LLM)",
    "qa": "resolving…",
}


def start_run(task_id: int, agent_type: str) -> tuple[int, float]:
    """Create a running AgentRun, mark the task's current agent. Returns (run_id, t0)."""
    with Session(engine) as db:
        run = AgentRun(
            task_id=task_id,
            agent_type=agent_type,
            status=RunStatus.running.value,
            model=_MODEL_LABEL.get(agent_type, "agent"),
            started_at=utcnow(),
        )
        db.add(run)
        task = db.get(Task, task_id)
        if task is not None:
            task.current_agent = agent_type
            task.updated_at = utcnow()
            db.add(task)
        db.commit()
        db.refresh(run)
        events.publish("run.started", run_id=run.id, task_id=task_id, agent=agent_type)
        return run.id, time.monotonic()


def set_model(run_id: int, model: str) -> None:
    """Update the run's model label once the actual provider is known."""
    with Session(engine) as db:
        run = db.get(AgentRun, run_id)
        if run is not None:
            run.model = model
            db.add(run)
            db.commit()
    events.publish("run.model", run_id=run_id, model=model)


def log(run_id: int, severity: str, message: str) -> None:
    if not message:
        return
    with Session(engine) as db:
        # 8000 keeps full QA/Review verdicts (~2-3k chars) intact in the log panel.
        db.add(LogEntry(run_id=run_id, severity=severity, message=message[:8000]))
        db.commit()
    events.publish("run.log", run_id=run_id, severity=severity, message=message)


def logger_for(run_id: int):
    """A (severity, message) callback bound to a run — passed to agent runners."""
    def _emit(severity: str, message: str) -> None:
        log(run_id, severity, message)
    return _emit


def finish_run(run_id: int, task_id: int, t0: float, *, tokens_in: int | None = 0,
               tokens_out: int | None = 0, cost: float | None = 0.0,
               error: str | None = None) -> None:
    """None for tokens/cost means the backend did not report usage — stored as 0
    but flagged usage_unknown so the UI never shows a fake $0.00."""
    with Session(engine) as db:
        run = db.get(AgentRun, run_id)
        if run is None:
            return
        run.status = RunStatus.failed.value if error else RunStatus.completed.value
        run.tokens_input = tokens_in or 0
        run.tokens_output = tokens_out or 0
        run.cost_usd = round(cost or 0.0, 4)
        run.usage_unknown = tokens_in is None or tokens_out is None or cost is None
        run.finished_at = utcnow()
        run.duration_ms = int((time.monotonic() - t0) * 1000)
        run.error = error
        db.add(run)
        if error:
            db.add(LogEntry(run_id=run_id, severity=Severity.error.value, message=error[:2000]))
        task = db.get(Task, task_id)
        if task is not None:
            task.token_cost += (tokens_in or 0) + (tokens_out or 0)
            task.updated_at = utcnow()
            db.add(task)
        db.commit()
        status, duration, model = run.status, run.duration_ms, run.model
    events.publish("run.finished", run_id=run_id, task_id=task_id, status=status,
                   model=model, duration_ms=duration, tokens_in=tokens_in,
                   tokens_out=tokens_out, cost=cost, error=error,
                   usage_unknown=tokens_in is None or tokens_out is None or cost is None)
