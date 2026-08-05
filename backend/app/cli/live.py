"""Watching a pipeline run.

A scope run is minutes of work by five agents across several models. The web UI
polls a log table for it; here we subscribe to ``services.events`` and render
the run as it happens.

Three things shape this module:

**The pipeline is not ours to interrupt.** It runs on its own thread with a
working git clone, a branch, and possibly a paid model call in flight. So Ctrl-C
*detaches the view*, it does not kill the run — the alternative is a half-applied
commit and a wedged working copy. The run is rejoinable, because everything it
does is also in the database.

**Events arrive on the pipeline's thread.** The subscriber does nothing but put
them on a queue; all rendering happens on the main thread, inside ``Live``.
Rendering from the worker thread would interleave with the prompt and tear.

**The interesting part is the shape of the run, not the log.** So the timeline
of stages is the primary object and the log is a tail beneath it — a full log
scroll tells you nothing about whether QA has started.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from ..services import events
from . import theme

# Stage order as the orchestrator drives it. `plan` and `dev` can recur across
# revision rounds; the timeline collapses repeats into one row with a counter
# rather than growing without bound.
STAGE_ORDER = ["plan", "dev", "qa", "review", "pr"]
STAGE_LABEL = {
    "plan": "Planner",
    "dev": "Dev",
    "qa": "QA",
    "review": "Review",
    "pr": "Pull request",
    "pm": "PM",
}
STAGE_STYLE = {
    "plan": "planner", "dev": "dev", "qa": "qa", "review": "jury", "pr": "info", "pm": "pm",
}

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_SPINNER_ASCII = ["-", "\\", "|", "/"]

# How much log to keep on screen. Enough to see what an agent is doing, little
# enough that the timeline stays visible above it on an 80x24 terminal.
TAIL_LINES = 8
# The pinned region used to show the stage timeline and nothing else, so a
# 52-turn Dev run read as one spinner next to the word "Dev" while every tool
# call scrolled past above it and was gone. These are the lines kept on the
# RUNNING stage so the window your eye rests on says what is happening now.
ACTIVITY_LINES = 6
LOG_LINES = 6           # opening lines of a multi-paragraph agent reply
TOOL_PREFIX = "→"       # claude_agent formats tool calls as "→ Name: input"


class _Stage:
    __slots__ = ("key", "runs", "state", "model", "started", "finished", "cost", "error",
                 "tokens", "tools", "activity")

    def __init__(self, key: str) -> None:
        self.key = key
        self.runs = 0
        self.state = "pending"      # pending | running | done | failed
        self.model = ""
        self.started: float | None = None
        self.finished: float | None = None
        self.cost = 0.0
        self.tokens = 0
        self.error: str | None = None
        self.tools = 0              # tool calls seen on the stream so far
        self.activity: list[str] = []   # most recent lines, for the live window


class RunView:
    """Accumulates events into something renderable."""

    def __init__(self, g: theme.Glyphs, unicode: bool) -> None:
        self.g = g
        self.frames = _SPINNER if unicode else _SPINNER_ASCII
        self.stages: dict[str, _Stage] = {}
        self.order: list[str] = []
        self.run_stage: dict[int, str] = {}   # run_id → stage key
        self.tail: list[tuple[str, str]] = []  # (severity, message)
        self.cost = 0.0
        self.tokens = 0
        self.started = time.monotonic()
        self.tick = 0

    # ── event intake ─────────────────────────────────────────────────────────
    def apply(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "run.started":
            key = payload["agent"]
            stage = self.stages.get(key)
            if stage is None:
                stage = self.stages[key] = _Stage(key)
                self.order.append(key)
            stage.runs += 1
            stage.state = "running"
            stage.started = time.monotonic()
            stage.finished = None
            self.run_stage[payload["run_id"]] = key

        elif kind == "run.model":
            stage = self._stage_for(payload.get("run_id"))
            if stage is not None:
                stage.model = payload.get("model") or ""

        elif kind == "run.log":
            message = (payload.get("message") or "").strip()
            if message:
                stage = self._stage_for(payload.get("run_id"))
                # Multi-line agent output (a full QA verdict, a review) would
                # blow the tail out in one event; keep the first lines so the
                # tail stays a tail.
                for line in message.splitlines()[:4]:
                    line = line.strip()
                    if not line:
                        continue
                    self.tail.append((payload.get("severity") or "info", line))
                    if stage is not None:
                        # claude_agent renders every tool call as "→ Name: input".
                        # Counting them live turns the Dev efficiency number —
                        # previously only reported once the stage had finished —
                        # into something you can watch while it happens.
                        if line.startswith(TOOL_PREFIX):
                            stage.tools += 1
                        stage.activity.append(line)
                        del stage.activity[:-ACTIVITY_LINES]
                del self.tail[:-TAIL_LINES]

        elif kind == "run.finished":
            stage = self._stage_for(payload.get("run_id"))
            if stage is not None:
                stage.state = "failed" if payload.get("error") else "done"
                stage.finished = time.monotonic()
                stage.error = payload.get("error")
                if not payload.get("usage_unknown"):
                    stage.cost += payload.get("cost") or 0.0
                    self.cost += payload.get("cost") or 0.0
                stage.tokens += (payload.get("tokens_in") or 0) + (payload.get("tokens_out") or 0)
                self.tokens += (payload.get("tokens_in") or 0) + (payload.get("tokens_out") or 0)

    def _stage_for(self, run_id: int | None) -> _Stage | None:
        key = self.run_stage.get(run_id) if run_id is not None else None
        return self.stages.get(key) if key else None

    def stage_label(self, run_id: int | None) -> str:
        """Which stage emitted this event. Five agents write to one scrollback,
        so without a name the transcript is anonymous — a jury opinion and a Dev
        tool call look alike once they have scrolled past the timeline."""
        key = self.run_stage.get(run_id) if run_id is not None else None
        return STAGE_LABEL.get(key, key or "") if key else ""

    # ── rendering ────────────────────────────────────────────────────────────
    def _mark(self, stage: _Stage) -> Text:
        if stage.state == "running":
            return Text(self.frames[self.tick % len(self.frames)], style="running")
        if stage.state == "done":
            return Text(self.g.ok, style="ok")
        if stage.state == "failed":
            return Text(self.g.fail, style="err")
        return Text(self.g.pending, style="pending")

    def _elapsed(self, stage: _Stage) -> str:
        if stage.started is None:
            return ""
        end = stage.finished if stage.finished is not None else time.monotonic()
        seconds = end - stage.started
        return f"{seconds:5.1f}s"

    def log_line(self, severity: str, message: str, stage: str = "") -> RenderableType:
        """One log row, styled — shared by the live region and the scrollback
        above it so a line looks the same wherever it is read.

        A two-column table, not a prefix + string, because a long line has to
        WRAP rather than be cut, and its continuation has to line up under the
        first column instead of falling back to the terminal's left edge. It
        used to be `Text(gutter) + Text(message[:400])`: a juror's paragraph
        came out as a wall of full-width text starting at column 0, with the
        gutter appearing once every few lines and no way to tell where one
        entry ended and the next began.
        """
        style = {"error": "err", "warn": "warn", "success": "ok"}.get(severity, "muted")
        row = Table(box=None, pad_edge=False, show_header=False, padding=(0, 0, 0, 0),
                    expand=True)
        row.add_column(width=10, no_wrap=True, justify="right")   # stage
        row.add_column(width=3, no_wrap=True)                     # gutter
        row.add_column(overflow="fold", ratio=1)                  # the message, wrapped
        row.add_row(
            Text(stage or "", style="brand.dim"),
            Text(f" {self.g.vbar} ", style="rule"),
            Text(message, style=style),
        )
        return row

    def renderable(self, *, include_tail: bool = True) -> RenderableType:
        self.tick += 1

        timeline = Table(box=None, pad_edge=False, show_header=False, padding=(0, 2, 0, 0))
        timeline.add_column(width=1)
        timeline.add_column(style="heading", no_wrap=True, width=12)
        timeline.add_column(overflow="ellipsis")        # model
        timeline.add_column(justify="right", style="muted", no_wrap=True)   # elapsed

        # Show every stage the pipeline knows about, so the ones still to come
        # read as "not started" rather than being invisible.
        keys = [k for k in STAGE_ORDER if k in self.stages]
        keys += [k for k in self.order if k not in STAGE_ORDER]
        for k in STAGE_ORDER:
            if k not in self.stages:
                keys.append(k)
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            stage = self.stages.get(key) or _Stage(key)
            label = Text(STAGE_LABEL.get(key, key.title()),
                         style=STAGE_STYLE.get(key, "muted") if stage.state != "pending" else "muted")
            model = Text(stage.model or "", style="muted")
            if stage.runs > 1:
                model += Text(f"  (round {stage.runs})", style="brand.dim")
            if stage.error:
                model = Text(stage.error[:60], style="err")
            if stage.tools:
                model += Text(f"   {stage.tools} tool call{'s' if stage.tools != 1 else ''}",
                              style="muted")
            timeline.add_row(self._mark(stage), label, model, self._elapsed(stage))

        body: list[RenderableType] = [timeline]

        # What the running agent is doing, right now. The scrollback above still
        # carries the full log — this is the rolling window, not the archive.
        running = next((self.stages[k] for k in seen
                        if self.stages.get(k) and self.stages[k].state == "running"), None)
        if running is not None and running.activity:
            body.append(Text(""))
            body.append(Text(f"  {STAGE_LABEL.get(running.key, running.key)} is doing",
                             style="heading"))
            for line in running.activity[-ACTIVITY_LINES:]:
                style = "brand.dim" if line.startswith(TOOL_PREFIX) else "muted"
                # No hard character cap: 120 columns truncated mid-word on any
                # terminal wider than that, which is most of them. overflow
                # trims to the width actually available, with an ellipsis so a
                # cut line is visibly cut rather than silently short.
                body.append(Text(f"    {line}", style=style,
                                 no_wrap=True, overflow="ellipsis"))

        if include_tail and self.tail:
            body.append(Text(""))
            for severity, line in self.tail[-TAIL_LINES:]:
                body.append(self.log_line(severity, line[:140]))

        total = time.monotonic() - self.started
        footer = Text(f"{total:.0f}s elapsed", style="muted")
        if self.tokens:
            footer += Text(f"   {self.g.bullet}   {self.tokens:,} tokens", style="muted")
        if self.cost:
            places = 2 if self.cost >= 1 else 4
            footer += Text(f"   {self.g.bullet}   ${self.cost:.{places}f}", style="muted")
        footer += Text(f"   {self.g.bullet}   Ctrl-C detaches (the run keeps going)", style="rule")
        body.append(Text(""))
        body.append(footer)

        from . import render
        return render.frame(Group(*body), self.g, title="Delivery",
                            subtitle=f"{total:.0f}s")


def watch(console: Console, g: theme.Glyphs, unicode: bool, work: threading.Thread,
          *, refresh_hz: int = 12) -> tuple[RunView, bool]:
    """Render ``work`` until it finishes or the user detaches.

    Returns the accumulated view and whether the run actually completed —
    a detached run is still going, and the caller must not report it as done.
    """
    inbox: queue.Queue[tuple[str, dict]] = queue.Queue()
    view = RunView(g, unicode)

    # Runs on the pipeline thread: enqueue only, never render.
    def _on_event(kind: str, payload: dict) -> None:
        inbox.put((kind, payload))

    detached = False
    with events.listener(_on_event):
        work.start()
        try:
            # The timeline is the only thing that redraws in place. Log lines are
            # PRINTED above it instead, because a Live region repaints the same
            # rows forever: everything that scrolled off was gone for good, and
            # the tail only ever held the last 8 lines anyway. Printing puts them
            # in the terminal's own scrollback, where they can be scrolled back
            # to, selected and copied — which is what a run log is for.
            with Live(view.renderable(include_tail=False), console=console,
                      refresh_per_second=refresh_hz, transient=False) as live:
                while work.is_alive() or not inbox.empty():
                    try:
                        kind, payload = inbox.get(timeout=1 / refresh_hz)
                        view.apply(kind, payload)
                        if kind == "run.log":
                            stage = view.stage_label(payload.get("run_id"))
                            for i, line in enumerate(_log_rows(payload)):
                                # Name the stage once per entry, not per wrapped
                                # line — repeating it is what made the old log
                                # read as noise rather than a transcript.
                                live.console.print(view.log_line(
                                    payload.get("severity") or "info", line,
                                    stage=stage if i == 0 else ""))
                    except queue.Empty:
                        pass
                    live.update(view.renderable(include_tail=False))
                live.update(view.renderable(include_tail=False))
        except KeyboardInterrupt:
            detached = True

    return view, not detached


def _log_rows(payload: dict) -> list[str]:
    """The printable lines of one log event.

    A full QA verdict or jury opinion is several paragraphs, and dumping all of
    it inline is what turned the run into a wall of prose with the timeline
    buried somewhere above. Keep the opening lines — the verdict is always in
    them — and say how much was withheld and where to read it, rather than
    stopping mid-sentence and leaving the reader unsure whether the agent was
    cut off or had finished.
    """
    message = (payload.get("message") or "").strip()
    if not message:
        return []
    lines = [ln.strip() for ln in message.splitlines() if ln.strip()]
    if len(lines) <= LOG_LINES:
        return lines
    return [*lines[:LOG_LINES],
            f"… {len(lines) - LOG_LINES} more line(s) — /review for the full text"]


# ── Watching a knowledge-base build ───────────────────────────────────────────
# Indexing has no event stream — it is a background job that reports through the
# Repo row (kb_status / kb_progress / kb_step). So this polls that row rather
# than subscribing, and renders the same two facts the pipeline view shows: how
# far along, and what it is doing right now.

KB_TERMINAL = ("ready", "failed")


def watch_kb(console: Console, g: theme.Glyphs, ctx, repo_id: int,
             *, poll_s: float = 1.0) -> tuple[str, str]:
    """Render a KB build until it settles. Returns (status, step).

    Ctrl-C detaches the view, exactly as it does for a pipeline run: the build
    is on this process's thread pool with a clone and a graph index in flight,
    and killing the view must not kill the work.
    """
    from ..core import repos as core_repos

    def _read() -> tuple[str, int, str]:
        with ctx.db() as db:
            repo = core_repos.require(db, repo_id)
            return (repo.kb_status or "pending", repo.kb_progress or 0,
                    repo.kb_step or "")

    status, progress, step = _read()
    started = time.monotonic()
    try:
        with Live(_kb_panel(g, status, progress, step, 0.0), console=console,
                  refresh_per_second=8, transient=False) as live:
            while status not in KB_TERMINAL:
                time.sleep(poll_s)
                status, progress, step = _read()
                live.update(_kb_panel(g, status, progress, step,
                                      time.monotonic() - started))
            live.update(_kb_panel(g, status, progress, step,
                                  time.monotonic() - started))
    except KeyboardInterrupt:
        console.print()
        return "detached", step
    return status, step


def _kb_panel(g: theme.Glyphs, status: str, progress: int, step: str,
              elapsed: float) -> RenderableType:
    done = status == "ready"
    failed = status == "failed"
    mark = g.ok if done else (g.fail if failed else g.step)
    style = "ok" if done else ("err" if failed else "running")

    bar = _kb_bar(progress, done or failed)
    head = Text.assemble(
        (f"{mark} ", theme.s(style)),
        (bar, theme.s(style)),
        (f"  {progress:3d}%", theme.s("heading")),
    )
    body: list[RenderableType] = [head]
    if step:
        body.append(Text(""))
        body.append(Text(step, style=theme.s("muted")))

    from . import render
    return render.frame(Group(*body), g, title="Knowledge base",
                        subtitle=_elapsed(elapsed),
                        style="ok" if done else ("err" if failed else "rule"))


def _kb_bar(progress: int, settled: bool, width: int = 28) -> str:
    filled = int(width * max(0, min(100, progress)) / 100)
    return "█" * filled + ("░" * (width - filled) if not settled else " " * (width - filled))


def _elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"
