"""In-process event bus for live pipeline progress.

The pipeline already records everything it does to the database — ``AgentRun``
rows for stages, ``LogEntry`` rows for console lines. That is the durable
record, and the web UI polls it. But a terminal client lives in the *same*
process as the orchestrator thread, and polling SQLite every 400ms to discover
something that happened in the next thread over is both wasteful and laggy.

So the writers publish here as well as to the DB. Subscribing is optional and
has no effect on the pipeline: with no subscribers, ``publish`` is a dict
lookup and a return. Nothing in the pipeline's behaviour depends on anyone
listening, and a subscriber that raises is dropped rather than allowed to break
the run that fed it.

Event kinds
-----------
``run.started``   a stage began              — task_id, run_id, agent
``run.model``     the stage resolved a model — run_id, model
``run.log``       one console line           — run_id, severity, message
``run.finished``  a stage ended              — run_id, task_id, status, cost, …
``task.updated``  task fields changed        — task_id, fields
"""

from __future__ import annotations

import itertools
import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_subscribers: dict[int, Callable[[str, dict[str, Any]], None]] = {}
_ids = itertools.count(1)


def subscribe(handler: Callable[[str, dict[str, Any]], None]) -> int:
    """Register ``handler(kind, payload)``. Returns a token for unsubscribe.

    The handler is called on the pipeline's worker thread, so it must be quick
    and thread-safe — push onto a queue rather than rendering inline.
    """
    with _lock:
        token = next(_ids)
        _subscribers[token] = handler
        return token


def unsubscribe(token: int) -> None:
    with _lock:
        _subscribers.pop(token, None)


class listener:  # noqa: N801 — used as a context manager, reads like one
    """``with events.listener(handler):`` — subscribe for a block."""

    def __init__(self, handler: Callable[[str, dict[str, Any]], None]) -> None:
        self._handler = handler
        self._token: int | None = None

    def __enter__(self) -> listener:
        self._token = subscribe(self._handler)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            unsubscribe(self._token)


def publish(kind: str, **payload: Any) -> None:
    """Fan out to every subscriber. Never raises into the caller."""
    with _lock:
        handlers = list(_subscribers.values())
    if not handlers:
        return
    for handler in handlers:
        try:
            handler(kind, payload)
        except Exception:  # noqa: BLE001 — a bad listener must not fail the run
            logger.debug("event subscriber raised on %s", kind, exc_info=True)
