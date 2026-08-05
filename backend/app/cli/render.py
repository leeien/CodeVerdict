"""Rich renderers — every recurring shape the terminal client draws.

Kept apart from the commands so that *what* an operation does and *how* it looks
stay separable, and so the same ticket or plan renders identically whether it
was produced by a slash command, a live run, or a one-shot subcommand.

Every style goes through ``S()``. Palette *names* work in the REPL, which draws
through a themed Console, but several of these renderables are also mounted
inside Textual panels — and Textual resolves styles with its own machinery,
where an unknown name is not an error, it just silently comes out uncoloured.
Resolving to literals up front means one renderer, correct in both places.
"""

from __future__ import annotations

from rich.align import Align
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models import Repo, ScopeSession, Task
from . import theme

S = theme.s

# The six pipeline stages, in the order work flows through them.
STAGES = [
    ("knowledge", "Knowledge"),
    ("pm", "PM"),
    ("planner", "Planner"),
    ("dev", "Dev"),
    ("qa", "QA"),
    ("review", "Review"),
]

_STATUS_STYLE = {
    "backlog": "muted",
    "scoped": "info",
    "approved": "ok",
    "in_dev": "running",
    "qa": "running",
    "review": "running",
    "pr": "ok",
    "done": "ok",
}

_PRIORITY_MARK = {"high": ("!", "err"), "medium": ("", "muted"), "low": ("", "muted")}


# ── Frames ────────────────────────────────────────────────────────────────────
# Every multi-row surface is drawn inside one of these. A terminal session is a
# single long scrollback with no other separation between a ticket list, a
# preflight report and an agent's reply, and flat output left the reader to find
# the seams. The frame is the seam.
#
# The border is the palette's rule colour and never a background — a filled
# panel is the fastest way to look broken on a light terminal. Box style is
# resolved from the same capability check the glyphs use, so a console that
# cannot draw box characters gets ASCII rather than mojibake.

def _box(g: theme.Glyphs):
    from rich import box

    # `g.vbar` is "│" only when the unicode probe passed, which makes it a
    # reliable proxy here without threading the flag through every caller.
    return box.ROUNDED if g.vbar == "│" else box.ASCII


def frame(body: RenderableType, g: theme.Glyphs, *, title: str = "",
          subtitle: str = "", style: str = "rule") -> RenderableType:
    """Wrap a surface in a titled panel."""
    return Padding(
        Panel(
            body,
            box=_box(g),
            border_style=S(style),
            title=Text(f" {title} ", style=S("heading")) if title else None,
            title_align="left",
            subtitle=Text(f" {subtitle} ", style=S("muted")) if subtitle else None,
            subtitle_align="right",
            padding=(1, 2),
            expand=True,
        ),
        (1, 0, 0, 2),
    )


# ── Banner ────────────────────────────────────────────────────────────────────
def banner(width: int, g: theme.Glyphs, unicode: bool) -> RenderableType:
    mark = Text("\n".join(theme.wordmark(width, unicode)), style=S("brand"))
    tagline = Text(theme.TAGLINE, style=S("muted"))
    return Padding(Align.left(Group(mark, Text(""), tagline)), (1, 0, 1, 2))


# ── Status line ───────────────────────────────────────────────────────────────
def status_line(*, repo: Repo | None, scope: ScopeSession | None, stages: dict[str, str],
                demo: bool, cost: float, g: theme.Glyphs) -> RenderableType:
    """The one-line "where am I" strip under the banner.

    Deliberately dense: a person mid-task needs to know which repo they're
    pointed at, whether the thing is allowed to touch real repositories, and
    what they've spent — without typing a command to find out.
    """
    parts: list[Text] = []

    if repo is not None:
        parts.append(Text(f"{repo.org}/{repo.name}", style=S("key")))
    else:
        parts.append(Text("no repo", style=S("muted")))

    if scope is not None:
        title = scope.title or "untitled scope"
        parts.append(Text(title[:34] + (g.ellipsis if len(title) > 34 else ""), style=S("muted")))

    # Which model families are actually in play, deduplicated — six stages named
    # individually would swamp the line, and the interesting fact is "who is
    # doing the work", not the full matrix. /models shows that.
    families: list[str] = []
    for key, _ in STAGES:
        provider = stages.get(f"{key}_provider") or ""
        if provider and provider not in families:
            families.append(provider)
    if families:
        parts.append(Text("+".join(families[:3]), style=S("brand.dim")))

    # The one status worth alarming about: demo mode off means the pipeline may
    # push to a real repository.
    parts.append(Text("demo", style=S("muted")) if demo else Text("LIVE", style=S("warn")))

    if cost > 0:
        # Dollars, read at a glance. Four fixed decimals rendered a $1.94 run as
        # "$1.9400"; sub-cent spend still needs the precision, so the scale
        # picks the places rather than one format serving neither case.
        places = 2 if cost >= 1 else (3 if cost >= 0.1 else 4)
        parts.append(Text(f"${cost:.{places}f}", style=S("muted")))

    return Padding(Text(f"  {g.bullet}  ", style=S("rule")).join(parts), (0, 0, 0, 2))


# ── Chat ──────────────────────────────────────────────────────────────────────
def speaker(name: str, style: str, body: RenderableType, g: theme.Glyphs) -> RenderableType:
    """One turn in the transcript: a labelled gutter and an indented body."""
    header = Text(f"{g.speaker} ", style=S(style)) + Text(name, style=f"bold {S(style)}")
    return Padding(Group(header, Padding(body, (0, 0, 0, 2))), (1, 0, 0, 2))


def agent_message(name: str, style: str, text: str, g: theme.Glyphs) -> RenderableType:
    return speaker(name, style, Markdown(text.strip() or "(no reply)"), g)


def user_echo(text: str, g: theme.Glyphs) -> RenderableType:
    return Padding(Text(f"{g.prompt} ", style=S("brand")) + Text(text, style=S("you")), (1, 0, 0, 2))


def note(text: str, g: theme.Glyphs, style: str = "muted") -> RenderableType:
    return Padding(Text(f"{g.bullet} {text}", style=S(style)), (0, 0, 0, 4))


def error(text: str, g: theme.Glyphs) -> RenderableType:
    return Padding(Text(f"{g.fail} {text}", style=S("err")), (1, 0, 0, 2))


def success(text: str, g: theme.Glyphs) -> RenderableType:
    return Padding(Text(f"{g.ok} {text}", style=S("ok")), (1, 0, 0, 2))


# ── Tickets ───────────────────────────────────────────────────────────────────
def tickets(tasks: list[Task], g: theme.Glyphs, *, title: str = "Tickets") -> RenderableType:
    if not tasks:
        return note("No tickets yet — describe what you want, then /tickets to draft them.", g)

    table = Table(box=None, pad_edge=False, show_header=True, header_style=S("muted"),
                  padding=(0, 2, 0, 0))
    table.add_column("", width=1)                       # the approval gate
    table.add_column("KEY", style=S("key"), no_wrap=True)
    table.add_column("TITLE", overflow="ellipsis")
    table.add_column("STATUS", no_wrap=True)
    table.add_column("PRI", no_wrap=True)

    for t in tasks:
        gate = Text(g.ok, style=S("ok")) if t.approved else Text(g.pending, style=S("muted"))
        status = Text(t.status.replace("_", " "), style=S(_STATUS_STYLE.get(t.status, "muted")))
        mark, priority_style = _PRIORITY_MARK.get(t.priority, ("", "muted"))
        table.add_row(gate, t.key, t.title, status,
                      Text(f"{mark}{t.priority}", style=S(priority_style)))

    hint = Text(f"{g.ok} approved   {g.pending} awaiting approval   "
                f"{g.bullet} /approve <KEY|all> to unlock the pipeline", style=S("muted"))
    pending = sum(1 for t in tasks if not t.approved)
    return frame(Group(table, Text(""), hint), g, title=title,
                 subtitle=f"{len(tasks)} ticket(s), {pending} awaiting approval"
                          if pending else f"{len(tasks)} ticket(s), all approved")


def ticket_detail(task: Task, g: theme.Glyphs) -> RenderableType:
    body: list[RenderableType] = [
        Text(f"{task.key}  ", style=S("key")) + Text(task.title, style="bold")
    ]
    if task.description:
        body.extend([Text(""), Markdown(task.description)])
    if task.acceptance_criteria:
        assumed = sum(1 for c in task.acceptance_criteria
                      if str(c).lstrip().lower().startswith("[assumed]"))
        header = Text("Acceptance criteria", style=S("heading"))
        if assumed:
            # This is the last screen before money is spent. A criterion the PM
            # decided rather than heard is exactly what a human should look at,
            # so it is marked, counted, and coloured — not left to be skimmed.
            header += Text(f"   {assumed} assumed by the PM — check these",
                           style=S("warn"))
        body.extend([Text(""), header])
        for criterion in task.acceptance_criteria:
            text = str(criterion).strip()
            low = text.lower()
            if low.startswith("[assumed]"):
                body.append(Text(f"  {g.pending} ", style=S("warn"))
                            + Text(text[len("[assumed]"):].strip(), style=S("warn")))
            elif low.startswith("[stated]"):
                body.append(Text(f"  {g.ok} ", style=S("ok"))
                            + Text(text[len("[stated]"):].strip()))
            else:
                body.append(Text(f"  {g.ok} ", style=S("ok")) + Text(text))
    if task.affected_files:
        body.extend([Text(""), Text("Files", style=S("heading"))])
        for path in task.affected_files[:12]:
            body.append(Text(f"  {g.bullet} ", style=S("muted")) + Text(str(path), style=S("path")))

    meta = Text(f"status {task.status}  {g.bullet}  ", style=S("muted"))
    meta += Text("approved" if task.approved else "not approved",
                 style=S("ok") if task.approved else S("warn"))
    if task.branch:
        meta += Text(f"  {g.bullet}  {task.branch}", style=S("muted"))
    if task.pr_url:
        meta += Text(f"  {g.bullet}  {task.pr_url}", style=S("url"))
    body.extend([Text(""), meta])
    return Padding(Group(*body), (1, 0, 0, 2))


# ── Board ─────────────────────────────────────────────────────────────────────
def board(lanes: list[dict], g: theme.Glyphs) -> RenderableType:
    """The Kanban lanes, as cards that reflow to the terminal width."""
    populated = [lane for lane in lanes if lane["tasks"]]
    if not populated:
        return note("The board is empty — nothing has been drafted for this repo yet.", g)

    # Card width, and the text width inside it once the border and padding are
    # taken out. Titles are clipped to fit rather than wrapped: a lane of
    # two-line cards stops reading as a column at a glance, which is the only
    # reason to draw a board instead of a list.
    card_width = 32
    cards = []
    for lane in populated:
        rows: list[RenderableType] = [
            Text(f"{lane['title']} ", style=S("heading")) +
            Text(str(len(lane["tasks"])), style=S("muted"))
        ]
        for t in lane["tasks"][:8]:
            rows.append(Text(t.key, style=S("key")))
            rows.append(Text(f"  {t.title}", style=S("muted"),
                             no_wrap=True, overflow="ellipsis"))
        if len(lane["tasks"]) > 8:
            rows.append(Text(f"  +{len(lane['tasks']) - 8} more", style=S("muted")))
        cards.append(Panel(Group(*rows), border_style=S("rule"), padding=(0, 1),
                           width=card_width))

    return Padding(Columns(cards, padding=(0, 1), expand=False), (1, 0, 0, 2))


# ── Repositories ──────────────────────────────────────────────────────────────
def repos(rows: list[Repo], active_id: int | None, g: theme.Glyphs) -> RenderableType:
    if not rows:
        return note("No repositories indexed. Add one with /kb add <git-url>.", g)

    table = Table(box=None, pad_edge=False, show_header=True, header_style=S("muted"),
                  padding=(0, 2, 0, 0))
    table.add_column("", width=1)
    table.add_column("REPOSITORY", style=S("key"), no_wrap=True)
    table.add_column("KB", no_wrap=True)
    table.add_column("SYMBOLS", justify="right", no_wrap=True)
    table.add_column("URL", style=S("muted"), overflow="ellipsis")

    kb_styles = {"ready": "ok", "indexing": "running", "failed": "err"}
    # Indexing writes kb_progress/kb_step continuously; a bare "indexing" throws
    # that away and leaves a multi-minute clone+AST pass looking identical to a
    # dead one. Show the number the job is already reporting.
    for r in rows:
        active = Text(g.step, style=S("brand")) if r.id == active_id else Text(" ")
        count = getattr(r, "kb_knowledge_count", 0) or 0
        status = r.kb_status
        if status == "indexing":
            status = f"{status} {r.kb_progress or 0}%"
        table.add_row(active, f"{r.org}/{r.name}",
                      Text(status, style=S(kb_styles.get(r.kb_status, "muted"))),
                      f"{count:,}" if count else "—", r.git_url)

    # The step is the half that says *what* it is doing — cloning a 1GB history
    # and walking an AST look the same from a percentage alone.
    steps = [r for r in rows if r.kb_status == "indexing" and r.kb_step]
    body: list[RenderableType] = [table]
    for r in steps:
        body.append(Text(""))
        body.append(Padding(Text(f"{r.org}/{r.name}: {r.kb_step}", style=S("muted")), (0, 0, 0, 1)))
    ready = sum(1 for r in rows if r.kb_status == "ready")
    return frame(Group(*body), g, title="Repositories",
                 subtitle=f"{ready}/{len(rows)} indexed")


# ── Plan ──────────────────────────────────────────────────────────────────────
def _plan_item_lines(item) -> list[str]:
    """Readable lines for one blast-radius / test-plan entry.

    The Planner is free-form JSON, so an entry may be a plain string or an
    object like ``{"files": [...], "new_cases": [...]}``. Rendering it with
    ``str()`` printed the raw Python dict — braces, quotes and all — into the
    one view a human reads to decide whether the plan is worth approving.
    """
    if isinstance(item, dict):
        lines: list[str] = []
        for key, value in item.items():
            if not value:
                continue
            joined = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            lines.append(f"{str(key).replace('_', ' ')}: {joined}")
        return lines or [str(item)]
    if isinstance(item, list):
        return [", ".join(str(v) for v in item)]
    return [str(item)]


def plan(plan_obj: dict, g: theme.Glyphs) -> RenderableType:
    """The Planner's verified implementation plan.

    Verification status leads, because it is the whole point of this view: every
    symbol the Planner named was checked against the code graph and ripgrep, so
    a reader can tell a *located* plan from a plausible-sounding guess.
    """
    steps = plan_obj.get("steps") or []
    if not steps:
        return note("No plan recorded for this scope.", g)

    verified = plan_obj.get("verified")
    badge = (Text(f"{g.ok} verified against the code graph", style=S("ok")) if verified
             else Text(f"{g.pending} unverified", style=S("warn")))
    body: list[RenderableType] = [badge, Text("")]

    for i, step in enumerate(steps, 1):
        if isinstance(step, str):
            body.append(Text(f"{i}. ", style=S("brand")) + Text(step))
            continue
        title = str(step.get("action") or step.get("title") or step.get("summary") or "step")
        body.append(Text(f"{i}. ", style=S("brand")) + Text(title, style="bold"))
        for path in (step.get("files") or [])[:6]:
            body.append(Text(f"     {g.bullet} ", style=S("muted")) + Text(str(path), style=S("path")))
        for symbol in (step.get("symbols") or [])[:6]:
            body.append(Text(f"     {g.step} ", style=S("muted")) + Text(str(symbol), style=S("path")))

    for label, key in (("Blast radius", "blast_radius"), ("Tests to extend", "tests")):
        value = plan_obj.get(key) or plan_obj.get({"blast_radius": "impact",
                                                   "tests": "test_plan"}[key])
        if not value:
            continue
        body.extend([Text(""), Text(label, style=S("heading"))])
        for item in (value if isinstance(value, list) else [value])[:8]:
            for line in _plan_item_lines(item):
                body.append(Text(f"  {g.bullet} ", style=S("muted")) + Text(line, style=S("path")))

    return frame(Group(*body), g, title="Implementation plan",
                 subtitle=f"{len(steps)} step(s)",
                 style="ok" if verified else "warn")


# ── Per-stage model matrix ────────────────────────────────────────────────────
def stage_models(values: dict, backends: dict[str, dict], g: theme.Glyphs) -> RenderableType:
    """Which provider and model owns each stage.

    This is the product's core promise, so it gets a first-class view rather
    than living only inside Settings — and it reports whether a stage pointed at
    a coding CLI can actually run here, since "configured" and "installed" are
    different facts.
    """
    table = Table(box=None, pad_edge=False, show_header=True, header_style=S("muted"),
                  padding=(0, 2, 0, 0))
    table.add_column("STAGE", style=S("heading"), no_wrap=True)
    table.add_column("PROVIDER", style=S("brand"), no_wrap=True)
    table.add_column("MODEL", overflow="ellipsis")
    table.add_column("", no_wrap=True)

    for key, label in STAGES:
        provider = values.get(f"{key}_provider") or "—"
        model = values.get(f"{key}_model") or "(provider default)"
        detail = backends.get(provider) or {}
        if not detail:
            status = Text("")
        elif detail.get("available"):
            status = Text(f"{g.ok} {detail.get('version') or ''}".strip(), style=S("ok"))
        else:
            status = Text(f"{g.fail} not installed", style=S("warn"))
        table.add_row(label, provider, Text(str(model), style=S("muted")), status)

    hint = Text(f"/model <stage> <provider> [model]   {g.bullet}   "
                f"/settings for everything else", style=S("muted"))
    return frame(Group(table, Text(""), hint), g, title="Agent models",
                 subtitle=f"{len(STAGES)} stages")


# ── Preflight ─────────────────────────────────────────────────────────────────
def doctor(report: dict, g: theme.Glyphs) -> RenderableType:
    """The preflight report: every check, then the verdict.

    Hints are printed only for rows that are not OK, and only once at the bottom
    of their group — a wall of advice next to working things is what makes a
    diagnostic unreadable.
    """
    blocks: list[RenderableType] = []
    mark = {"ok": (g.ok, "ok"), "warn": (g.fail, "warn"), "fail": (g.fail, "err")}

    for group in report["groups"]:
        table = Table(box=None, pad_edge=False, show_header=False, padding=(0, 2, 0, 0))
        table.add_column(no_wrap=True)
        table.add_column(style=S("heading"), no_wrap=True)
        table.add_column(overflow="fold")
        hints: list[tuple[str, str]] = []
        for c in group["checks"]:
            glyph, style = mark[c["status"]]
            table.add_row(Text(glyph, style=S(style)), c["name"],
                          Text(c["detail"], style=S("muted")))
            if c["status"] != "ok" and c["hint"]:
                hints.append((c["name"], c["hint"]))

        rows: list[RenderableType] = [Text(group["name"], style=S("brand")),
                                      Padding(table, (0, 0, 0, 2))]
        for name, hint in hints:
            rows.append(Padding(Text(f"{g.bullet} {name}: {hint}", style=S("muted")), (0, 0, 0, 4)))
        blocks.append(Group(*rows))
        blocks.append(Text(""))

    if report["ready"]:
        verdict = Text(f"{g.ok} Ready to run.", style=S("ok"))
        if report["warnings"]:
            verdict.append(f"  {report['warnings']} optional component(s) degraded "
                           f"— see above.", style=S("muted"))
    else:
        verdict = Text(f"{g.fail} {report['failures']} blocking problem(s) — "
                       f"fix those before /run.", style=S("err"))
    blocks.append(verdict)
    return frame(Group(*blocks), g, title="Preflight",
                 subtitle="ready" if report["ready"] else "blocked",
                 style="ok" if report["ready"] else "err")


# ── Help ──────────────────────────────────────────────────────────────────────
def help_panel(commands: list[tuple[str, str, str]], g: theme.Glyphs) -> RenderableType:
    """``commands`` is [(name, args, description), …] in registration order."""
    table = Table(box=None, pad_edge=False, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(style=S("brand"), no_wrap=True)
    # Not no_wrap: `/kb`'s arg list is 38 characters and, pinned, it stole the
    # width the descriptions needed — every one of them wrapped early against a
    # half-empty line. Let it wrap in place instead.
    table.add_column(style=S("muted"), max_width=20)
    table.add_column(ratio=1)
    for name, args, description in commands:
        table.add_row(name, args, description)
    intro = Text("Type plainly to talk to the PM agent. Slash commands drive everything else.",
                 style=S("muted"))
    return frame(Group(intro, Text(""), table), g, title="Commands",
                 subtitle=f"{len(commands)} available")
