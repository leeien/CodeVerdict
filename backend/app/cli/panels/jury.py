"""The jury roster: who sits on the panel, in what order, on which model.

Editing this changes what every future delivery is judged against, and each
seated judge is a paid model call per review round — so the panel makes both
costs legible: how many seats are enabled, and whether each one can actually
run with the providers currently configured. A judge pointed at a provider with
no key is not a stricter review, it is an abstention waiting to happen.

Ordering matters too, and not only cosmetically: the roster order is the order
opinions reach the foreperson.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
    TextArea,
)

from .. import theme
from . import BRAND_CSS

S = theme.s


def judge_detail(judge: dict, g: theme.Glyphs) -> RenderableType:
    body: list[RenderableType] = []
    head = Text(str(judge["name"]), style="bold")
    if not judge["enabled"]:
        head += Text("   not seated", style=S("muted"))
    body.append(head)
    body.append(Text(str(judge.get("persona_name") or "Custom"), style=S("brand")))

    model = judge.get("effective_model") or "(provider default)"
    provider = judge.get("effective_provider") or "—"
    line = Text(f"{provider} / {model}", style=S("muted"))
    if judge.get("inherits"):
        line += Text("   inherited from the Review stage", style=S("brand.dim"))
    body.extend([Text(""), line])

    if not judge.get("runnable"):
        body.extend([
            Text(""),
            Text(f"{g.fail} This seat cannot run: no usable key or CLI for "
                 f"'{provider}'. It will abstain, and its perspective will be "
                 f"missing from every verdict.", style=S("err")),
        ])

    body.extend([Text(""), Text("Brief", style=S("heading")), Text("")])
    body.append(Text(str(judge.get("focus") or judge.get("persona_summary") or "")))
    return Group(*body)


class JuryPanel(App[int]):
    """Returns the number of edits made."""

    CSS = BRAND_CSS + """
    #roster { width: 40; border-right: solid $panel; }
    #detail { padding: 1 2; }
    """

    BINDINGS = [
        Binding("space,e", "toggle", "Seat / unseat"),
        Binding("m", "set_model", "Model"),
        Binding("a", "add", "Add judge"),
        Binding("d,delete", "remove", "Remove"),
        Binding("shift+up", "move_up", "Move up"),
        Binding("shift+down", "move_down", "Move down"),
        Binding("R", "reset", "Reset to defaults"),
        Binding("q,escape", "close", "Close"),
        Binding("t", "toggle_theme", "Light/dark"),
    ]

    def __init__(self, ctx, g: theme.Glyphs) -> None:
        super().__init__()
        self.ctx = ctx
        self.g = g
        self.data: dict = {}
        self.edits = 0

    # ── data ─────────────────────────────────────────────────────────────────
    def _reload(self) -> None:
        from ...services import judges

        with self.ctx.db() as db:
            self.data = judges.view(db)

    @property
    def judges(self) -> list[dict]:
        return self.data.get("judges") or []

    def _selected(self) -> dict | None:
        roster = self.query_one("#roster", ListView)
        index = roster.index or 0
        return self.judges[index] if 0 <= index < len(self.judges) else None

    # ── layout ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        self._reload()
        self.title = "CodeVerdict — review jury"
        yield Header(show_clock=False)
        with Horizontal():
            yield ListView(id="roster")
            yield VerticalScroll(Static(Text(""), id="detail"))
        yield Footer()

    async def on_mount(self) -> None:
        await self._refresh_roster()
        self.query_one("#roster", ListView).focus()

    def _rebuild(self, keep: int = 0) -> None:
        """Redraw from a synchronous handler (see ``_refresh_roster``)."""
        self.call_later(self._refresh_roster, keep)

    async def _refresh_roster(self, keep: int = 0) -> None:
        """Rebuild the roster list.

        Awaited because ``ListView.clear()`` is: appending without waiting for
        it leaves the old rows in place and the detail pane bound to a judge
        that is no longer at that index.
        """
        roster = self.query_one("#roster", ListView)
        await roster.clear()
        for judge in self.judges:
            await roster.append(ListItem(Static(self._row(judge))))
        if self.judges:
            index = min(keep, len(self.judges) - 1)
            roster.index = index
            self._show(self.judges[index])
        self.sub_title = (f"{self.data.get('enabled_count', 0)} of {len(self.judges)} seated "
                          f"— each is a model call per review round")

    def _row(self, judge: dict) -> RenderableType:
        if judge["enabled"]:
            mark = Text(f"{self.g.ok} ", style=S("ok"))
            name = Text(str(judge["name"]))
        else:
            mark = Text(f"{self.g.pending} ", style=S("muted"))
            name = Text(str(judge["name"]), style=S("muted"))
        if not judge.get("runnable"):
            name += Text("  cannot run", style=S("err"))
        model = judge.get("effective_model") or judge.get("effective_provider") or ""
        return Group(mark + name, Text(f"  {str(model)[:34]}", style=S("muted")))

    def _show(self, judge: dict) -> None:
        self.query_one("#detail", Static).update(judge_detail(judge, self.g))

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id == "roster" and self.judges:
            index = event.list_view.index or 0
            if 0 <= index < len(self.judges):
                self._show(self.judges[index])

    # ── actions ──────────────────────────────────────────────────────────────
    def _mutate(self, fn) -> None:
        from ...services import judges

        judge = self._selected()
        if judge is None:
            return
        index = self.query_one("#roster", ListView).index or 0
        try:
            with self.ctx.db() as db:
                fn(judges, db, judge)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self.edits += 1
        self._reload()
        self._rebuild(index)

    def action_toggle(self) -> None:
        self._mutate(lambda judges, db, j:
                     judges.update(db, j["id"], {"enabled": not j["enabled"]}))

    def action_move_up(self) -> None:
        index = self.query_one("#roster", ListView).index or 0
        self._mutate(lambda judges, db, j: judges.move(db, j["id"], -1))
        self.query_one("#roster", ListView).index = max(0, index - 1)

    def action_move_down(self) -> None:
        index = self.query_one("#roster", ListView).index or 0
        self._mutate(lambda judges, db, j: judges.move(db, j["id"], 1))
        self.query_one("#roster", ListView).index = min(len(self.judges) - 1, index + 1)

    def action_remove(self) -> None:
        self._mutate(lambda judges, db, j: judges.delete(db, j["id"]))

    def action_set_model(self) -> None:
        judge = self._selected()
        if judge is None:
            return
        self.push_screen(
            _ModelPicker(judge, self.data.get("provider_options") or []),
            self._apply_model)

    def _apply_model(self, result: dict | None) -> None:
        from ...services import judges

        if not result:
            return
        index = self.query_one("#roster", ListView).index or 0
        judge = self._selected()
        if judge is None:
            return
        with self.ctx.db() as db:
            judges.update(db, judge["id"], result)
        self.edits += 1
        self._reload()
        self._rebuild(index)

    def action_add(self) -> None:
        self.push_screen(_NewJudge(self.data.get("personas") or []), self._create)

    def _create(self, result: dict | None) -> None:
        from ...services import judges

        if not result:
            return
        try:
            with self.ctx.db() as db:
                judges.create(db, **result)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self.edits += 1
        self._reload()
        self._rebuild(len(self.judges) - 1)

    def action_reset(self) -> None:
        from ...services import judges

        with self.ctx.db() as db:
            judges.reset_to_defaults(db)
        self.edits += 1
        self._reload()
        self._rebuild()
        self.notify("Roster reset to the default panel.")

    def action_toggle_theme(self) -> None:
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"

    def action_close(self) -> None:
        self.exit(self.edits)


class _ModelPicker(ModalScreen[dict | None]):
    CSS = """
    _ModelPicker { align: center middle; }
    #box { width: 64; height: auto; border: round $primary; background: $surface; padding: 1 2; }
    """

    def __init__(self, judge: dict, provider_options: list[str]) -> None:
        super().__init__()
        self.judge = judge
        self.provider_options = provider_options

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="box"):
            yield Label(f"Model for {self.judge['name']}")
            yield Static(Text("Blank provider inherits the Review stage's provider — which "
                              "is usually what you want for most seats, and deliberately "
                              "not what you want for all of them.", style=S("muted")))
            options = [(p or "(inherit from Review stage)", p) for p in self.provider_options
                      if p]
            current = self.judge.get("provider") or ""
            # The blank option's real value is Select.NULL, not "" — the
            # empty string is only its on-screen label. Constructing with
            # value="" for a judge that inherits (the common case) would
            # try to validate "" as a legal option and raise.
            yield Select(options, value=current if current else Select.NULL,
                         allow_blank=True, id="provider")
            yield Input(value=self.judge.get("model") or "",
                        placeholder="model id (blank = provider default)", id="model")
            with Horizontal():
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        provider = self.query_one("#provider", Select).value
        self.dismiss({
            "provider": "" if provider is Select.NULL else str(provider or ""),
            "model": self.query_one("#model", Input).value.strip(),
        })


class _NewJudge(ModalScreen[dict | None]):
    CSS = """
    _NewJudge { align: center middle; }
    #box { width: 70; height: auto; border: round $primary; background: $surface; padding: 1 2; }
    #focus { height: 6; }
    """

    def __init__(self, personas: list[dict]) -> None:
        super().__init__()
        self.personas = personas

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="box"):
            yield Label("Seat a new judge")
            yield Input(placeholder="name, e.g. Concurrency", id="name")
            yield Select([(p["name"], p["id"]) for p in self.personas],
                         allow_blank=True, id="persona")
            yield Static(Text("Brief — what this juror should look for, and what to leave "
                              "to the others. Uncorrelated reviewers are the whole "
                              "mechanism.", style=S("muted")))
            yield TextArea(id="focus")
            with Horizontal():
                yield Button("Seat", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        name = self.query_one("#name", Input).value.strip()
        if not name:
            self.notify("A judge needs a name.", severity="error")
            return
        persona = self.query_one("#persona", Select).value
        self.dismiss({
            "name": name,
            "persona": "custom" if persona is Select.NULL else str(persona),
            "focus": self.query_one("#focus", TextArea).text.strip(),
            "enabled": True,
        })


def open_panel(ctx, *, unicode: bool = True) -> int:
    return JuryPanel(ctx, theme.Glyphs(unicode)).run() or 0
