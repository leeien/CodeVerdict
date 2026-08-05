"""Settings.

Not hand-written. ``runtime_settings.view()`` already describes every field the
product has — its group, type, options, secrecy, conditional visibility, and for
model fields, which provider field it depends on. The web form is generated from
that description; so is this. Which means a new setting appears in the terminal
the moment it appears in ``FIELDS``, with no second UI to update and no chance
of the two surfaces disagreeing about what is configurable.

The parts that can't be generated are the ones that aren't really fields:

* **presets** — point every stage at one provider in a keystroke, because the
  common case is "I have one API key" and making that person fill in twelve
  fields is a bad first five minutes;
* **model lists fetched live** from the provider's own API, so the picker only
  ever offers models that currently exist;
* **coding-CLI detection and install**, because whether ``claude`` is on this
  machine is a fact about the host, not a value to type;
* **probes** for the code graph and the lexical engine, which answer "will the
  knowledge base actually work here" — a question no settings value can.
"""

from __future__ import annotations

import asyncio

from rich.console import Group, RenderableType
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

from .. import theme
from . import BRAND_CSS

S = theme.s

# runtime_settings masks secrets with this; typing over it sets a new value,
# leaving it alone must not overwrite the stored one with the mask itself.
SECRET_MASK = "••••••••"


def _visible(field: dict, values: dict) -> bool:
    """Honour a field's ``show_if`` the same way the web form does.

    Format is ``name=value``, alternatives separated by ``|``, bools as
    true/false. A field that only makes sense in one mode shouldn't clutter the
    others.
    """
    condition = (field.get("show_if") or "").strip()
    if not condition:
        return True
    name, _, expected = condition.partition("=")
    current = values.get(name.strip())
    if isinstance(current, bool):
        current = "true" if current else "false"
    return str(current).lower() in [v.strip().lower() for v in expected.split("|")]


class ProbeResult(ModalScreen[None]):
    """A modal for the answer to a question, not for editing anything."""

    CSS = """
    ProbeResult { align: center middle; }
    #box { width: 78; max-height: 22; border: round $primary; background: $surface; padding: 1 2; }
    """

    def __init__(self, title: str, body: RenderableType) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="box"):
            yield Static(Text(self._title, style="bold"))
            yield Static(Text(""))
            yield Static(self._body)
            yield Static(Text(""))
            yield Static(Text("press escape to close", style=S("muted")))

    def on_key(self, event) -> None:
        if event.key in ("escape", "q", "enter"):
            self.dismiss(None)


class SettingsPanel(App[int]):
    """Returns the number of settings changed."""

    CSS = BRAND_CSS + """
    #groups { width: 26; border-right: solid $panel; }
    #fields { padding: 1 2; }
    .field-label { text-style: bold; padding: 1 0 0 0; }
    .field-help { color: $text-muted; padding: 0 0 0 0; }
    .actions { height: auto; padding: 1 0; }
    .actions Button { margin-right: 2; }
    /* Width is explicit because these are mounted into the pane after layout
       rather than composed with it, and an inherited auto width resolves to
       zero — the field is then present, focusable and completely invisible. */
    #fields Input, #fields Select { width: 100%; margin: 0 0 1 0; }
    #fields Checkbox { margin: 0 0 1 0; }
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("q,escape", "close", "Close"),
        Binding("p", "preset", "One-provider preset"),
        Binding("r", "recheck", "Re-check CLIs"),
        Binding("g", "probe_graph", "Test code graph"),
        Binding("f", "probe_search", "Test search"),
        Binding("t", "toggle_theme", "Light/dark"),
    ]

    def __init__(self, ctx, g: theme.Glyphs) -> None:
        super().__init__()
        self.ctx = ctx
        self.g = g
        self.view: dict = {}
        self.values: dict = {}          # current form state, by field name
        self.pending: dict = {}         # edits not yet saved
        self.saved_count = 0
        self.group_index = 0
        self._model_cache: dict[str, list[str]] = {}
        self._render_lock = asyncio.Lock()

    # ── data ─────────────────────────────────────────────────────────────────
    def _reload(self) -> None:
        from ...services import runtime_settings

        self.view = runtime_settings.view()
        self.values = {f["name"]: f["value"]
                       for group in self.view["groups"] for f in group["fields"]}

    def _known_models(self, provider_id: str) -> list[str]:
        """Whatever we can offer *right now*, without blocking.

        Either the live list a worker has already fetched, or the registry's
        curated list. Never a network call: this runs inside the render path,
        and an HTTP round trip there freezes the event loop — which does not
        look like slowness, it looks like the fields failed to appear.
        """
        from ...services import providers

        if provider_id in self._model_cache:
            return self._model_cache[provider_id]
        entry = providers.PROVIDERS.get(provider_id)
        return list(entry.models) if entry else []

    def _fetch_models(self, provider_ids: list[str]) -> None:
        """Refresh the model lists off-thread, then redraw.

        The registry's list is curated, not current — providers add and retire
        models constantly, and offering one that 404s is worse than offering
        none. So the form opens instantly on the curated list and quietly
        upgrades to the provider's own answer.
        """
        wanted = [p for p in dict.fromkeys(provider_ids) if p and p not in self._model_cache]
        if not wanted:
            return

        def _work() -> None:
            from ...services import providers

            improved = False
            for provider_id in wanted:
                try:
                    models = list(providers.fetch_models(provider_id) or [])
                except Exception:  # noqa: BLE001 — a dead endpoint must not break the form
                    models = []
                # Cache the answer either way, so a provider with no key or no
                # model endpoint is asked once per session rather than on every
                # redraw. An empty list means "we asked; use the curated list".
                self._model_cache[provider_id] = models or self._known_models(provider_id)
                improved = improved or bool(models)
            if improved:
                self.call_from_thread(self._rerender)

        self.run_worker(_work, thread=True, exit_on_error=False,
                        name=f"models:{','.join(wanted)}")

    # ── layout ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        self._reload()
        self.title = "CodeVerdict — Settings"
        self.sub_title = "ctrl+s saves"
        yield Header(show_clock=False)
        with Horizontal():
            yield ListView(
                *[ListItem(Static(Text(group["label"]))) for group in self.view["groups"]],
                id="groups")
            yield VerticalScroll(id="fields")
        yield Footer()

    async def on_mount(self) -> None:
        await self._render_group(0)

    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id == "groups":
            await self._render_group(event.list_view.index or 0)

    async def _render_group(self, index: int) -> None:
        """Rebuild the field pane for one group.

        Awaited, and serialised behind a lock, because field widgets carry
        stable ids derived from the setting name. Two overlapping rebuilds — the
        mount and the list's own initial highlight both fire on startup — would
        mount `f-openai_base_url` twice before the first removal had completed,
        and Textual rejects the duplicate id rather than silently showing two.
        """
        async with self._render_lock:
            self.group_index = index
            group = self.view["groups"][index]
            pane = self.query_one("#fields", VerticalScroll)
            await pane.remove_children()
            # One awaited batch, not a mount per widget: a Select composes its
            # own overlay as a child, and mounting them one at a time delivers
            # its Mount event before that overlay exists, which it treats as a
            # fatal missing node.
            await pane.mount_all(self._group_widgets(group))

        # After the group is on screen, not before: this upgrades the curated
        # model lists to each provider's live one, and must never be what the
        # user waits on to see the form.
        self._fetch_models([
            str(self.pending.get(f["provider_field"], self.values.get(f["provider_field"], "")))
            for f in group["fields"] if f["type"] == "model" and f.get("provider_field")
        ])

    def _group_widgets(self, group: dict) -> list:
        widgets: list = [
            Static(Text(group["label"], style="bold")),
            Static(Text(group["help"], style=S("muted"))),
        ]
        if group["id"] == "models":
            widgets.append(Static(Text(
                f"\n{self.g.bullet} press p to point every stage at one provider",
                style=S("brand.dim"))))
        if group["id"] == "providers":
            widgets.append(Static(self._backends_summary()))

        section = ""
        for field in group["fields"]:
            if not _visible(field, self.values):
                continue
            if field.get("section") and field["section"] != section:
                section = field["section"]
                widgets.append(Static(Text(f"\n{section}", style=S("heading"))))
            widgets.append(Static(Text(field["label"], style="bold"), classes="field-label"))
            if field.get("help"):
                widgets.append(Static(Text(field["help"], style=S("muted")), classes="field-help"))
            widgets.append(self._widget_for(field))
        return widgets

    def _widget_for(self, field: dict):
        name, kind = field["name"], field["type"]
        value = self.pending.get(name, self.values.get(name))

        if kind == "bool":
            return Checkbox(value=bool(value), id=f"f-{name}")

        if kind in ("enum", "provider"):
            options = [(o, o) for o in (field.get("options") or [])]
            allow_blank = not options or value in ("", None)
            # Select.NULL, not Select.BLANK — this Textual build has no BLANK
            # of its own on Select, so the name resolves up the MRO to
            # Widget.BLANK (an unrelated flag, = False) instead of raising,
            # and every "no selection" here was silently constructing with a
            # literal False. NULL is the real sentinel.
            return Select(options, value=value if value in dict(options) else Select.NULL,
                          allow_blank=allow_blank, id=f"f-{name}")

        if kind == "model":
            provider = self.pending.get(field["provider_field"],
                                        self.values.get(field["provider_field"], ""))
            models = self._known_models(str(provider)) if provider else []
            if models:
                options = [(m, m) for m in models]
                # A configured model that isn't in the list is still the current
                # setting — offer it rather than silently blanking the field.
                if value and value not in models:
                    options.insert(0, (f"{value}  (current)", value))
                return Select(options, value=value or Select.NULL,
                              allow_blank=True, id=f"f-{name}")
            # Nothing to offer (a CLI provider with no model endpoint, or a
            # provider with no key yet) — free text is right, not a disabled box.
            return Input(value=str(value or ""), placeholder="model id", id=f"f-{name}")

        if field.get("secret"):
            return Input(value="", password=True,
                         placeholder=SECRET_MASK if field.get("set") else "not set",
                         id=f"f-{name}")

        return Input(value="" if value is None else str(value), id=f"f-{name}")

    def _backends_summary(self) -> RenderableType:
        """Which coding CLIs this machine can actually run."""
        rows: list[RenderableType] = [Text("\nCoding CLIs on this machine", style=S("heading"))]
        any_cli = False
        for provider in self.view.get("providers") or []:
            if not provider.get("backend"):
                continue
            any_cli = True
            if provider.get("available"):
                line = Text(f"  {self.g.ok} ", style=S("ok")) + Text(provider["name"])
                line += Text(f"  {provider.get('version') or ''}", style=S("muted"))
            else:
                line = Text(f"  {self.g.fail} ", style=S("muted")) + Text(provider["name"],
                                                                          style=S("muted"))
                reason = provider.get("unavailable_reason") or "not installed"
                line += Text(f"  {reason}", style=S("muted"))
            rows.append(line)
        if not any_cli:
            rows.append(Text("  none detected", style=S("muted")))
        rows.append(Text(f"  {self.g.bullet} press r to re-check after installing one",
                         style=S("brand.dim")))
        return Group(*rows)

    # ── edits ────────────────────────────────────────────────────────────────
    def _rerender(self) -> None:
        """Redraw the current group from a synchronous handler.

        Event handlers and actions are sync, but the rebuild has to be
        awaited to keep widget ids unique — so hand it to the event loop
        rather than firing and forgetting it.
        """
        self.call_later(self._render_group, self.group_index)

    def _record(self, widget_id: str | None, value) -> None:
        # Mounting a widget with value=X emits its own Changed(X) message —
        # Textual queues it and delivers it on a later pump tick, which lands
        # AFTER this render's _loading window has already closed (mount_all
        # returns once widgets are structurally attached, not once every
        # descendant's own posted messages have drained). A timing flag can't
        # reliably catch that race, so the real signal is content instead: a
        # widget echoing the value it was already constructed with is not an
        # edit, no matter when the message arrives. Without this, every field
        # looks edited the moment a group is drawn, Ctrl-S writes back all 59
        # settings nobody touched, and worse — every provider field's own
        # echo re-renders the group, which reconstructs all 6 provider
        # fields, which echo again: an unbounded render loop, observed
        # pegging a full core.
        if not widget_id or not widget_id.startswith("f-"):
            return
        name = widget_id[2:]
        if name in self.pending:
            if self.pending[name] == value:
                return                                   # exact repeat
        else:
            baseline = self.values.get(name)
            # Input always reports a str; a number/None-typed setting isn't
            # stored as one. Compare in the representation the widget was
            # actually constructed with (see _widget_for's str(value)),
            # not raw equality — 2.0 != "2.0" would otherwise read every
            # numeric field as edited the instant its group is drawn.
            if str(value) == ("" if baseline is None else str(baseline)):
                return                                    # construction echo
        self.pending[name] = value
        # A provider change invalidates the model list beside it, so re-render
        # the group: the old provider's models must not stay on offer.
        if name.endswith("_provider"):
            self.values[name] = value
            self._rerender()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._record(event.input.id, event.value)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self._record(event.checkbox.id, event.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        value = "" if event.value is Select.NULL else event.value
        self._record(event.select.id, value)

    # ── actions ──────────────────────────────────────────────────────────────
    def action_save(self) -> None:
        from ...services import runtime_settings

        index = {f["name"]: f for group in self.view["groups"] for f in group["fields"]}
        payload = {}
        for name, value in self.pending.items():
            field = index.get(name)
            if field is None:
                continue
            # An untouched secret input is empty, which must not be read as
            # "clear the stored key".
            if field.get("secret") and not str(value).strip():
                continue
            payload[name] = value
        if not payload:
            self.notify("Nothing changed.")
            return
        try:
            with self.ctx.db() as db:
                changed = runtime_settings.update(db, payload)
        except ValueError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.saved_count += len(changed)
        self.pending.clear()
        self._reload()
        self._rerender()
        self.notify(f"Saved {len(changed)} setting(s).")

    def action_preset(self) -> None:
        self.push_screen(_PresetPicker(self.view.get("preset_providers") or []),
                         self._apply_preset)

    def _apply_preset(self, provider_id: str | None) -> None:
        if not provider_id:
            return
        from ...services import runtime_settings

        try:
            with self.ctx.db() as db:
                changed = runtime_settings.apply_provider_preset(db, provider_id)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self.saved_count += len(changed)
        self._reload()
        self._rerender()
        self.notify(f"Every stage now points at {provider_id} ({len(changed)} fields).")

    def action_recheck(self) -> None:
        """Re-probe the coding CLIs — off-thread, because probing runs a
        subprocess per tool and takes seconds (see ``warm_up``)."""
        def _work() -> None:
            from ...services import agent_backends

            agent_backends.refresh()
            self.call_from_thread(self._after_recheck)

        self.notify("Re-probing the coding CLIs…")
        self.run_worker(_work, thread=True, exit_on_error=False, name="recheck-backends")

    def _after_recheck(self) -> None:
        self._reload()
        self._rerender()
        self.notify("Re-probed the coding CLIs.")

    def action_probe_graph(self) -> None:
        from ...services.knowledge import graph

        result = graph.probe()
        ok = result.get("ok") or result.get("available")
        body = Text(f"{self.g.ok} " if ok else f"{self.g.fail} ",
                    style=S("ok") if ok else S("err"))
        body += Text(str(result.get("version") or result.get("error") or result))
        if result.get("path"):
            body += Text(f"\n\n{result['path']}", style=S("muted"))
        if not ok:
            body += Text("\n\nWithout it the knowledge base falls back to the built-in "
                         "symbol map plus ripgrep — still works, measurably coarser.",
                         style=S("muted"))
        self.push_screen(ProbeResult("Code graph engine", body))

    def action_probe_search(self) -> None:
        from ...services import search

        result = search.probe()
        engine = result.get("engine") or result.get("tool") or "unknown"
        body = Text("lexical engine: ", style=S("muted")) + Text(str(engine), style="bold")
        if engine != "ripgrep":
            body += Text("\n\nFalling back to git grep: tracked files only, with "
                         "language filters retained for definition searches, but no "
                         ".gitignore-aware walk. Install ripgrep for the full engine.",
                         style=S("warn"))
        self.push_screen(ProbeResult("Lexical search", body))

    def action_toggle_theme(self) -> None:
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"

    def action_close(self) -> None:
        if self.pending:
            self.notify("Unsaved changes — ctrl+s to save, escape again to discard.",
                        severity="warning")
            self.pending.clear()
            return
        self.exit(self.saved_count)


class _PresetPicker(ModalScreen[str | None]):
    CSS = """
    _PresetPicker { align: center middle; }
    #box { width: 56; height: auto; border: round $primary; background: $surface; padding: 1 2; }
    """

    def __init__(self, providers: list[dict]) -> None:
        super().__init__()
        self.providers = providers

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="box"):
            yield Label("Point every stage at one provider")
            yield Static(Text("Recommended per-stage models are applied automatically.",
                              style=S("muted")))
            for provider in self.providers:
                yield Button(provider["name"], id=f"p-{provider['id']}")
            yield Button("Cancel", id="p-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        self.dismiss(None if button_id == "p-cancel" else button_id[2:])


def warm_up() -> None:
    """Probe the coding CLIs *before* the full-screen app takes over.

    ``runtime_settings.view()`` reports which agentic CLIs are installed, and
    finding that out means running a subprocess per tool — about five seconds
    on a cold cache. The view is needed by ``compose()``, so left alone that
    work lands on Textual's event loop at startup and stalls it: widgets that
    compose children (every ``Select`` here) never finish mounting, and the
    form draws its labels with nothing underneath them.

    The result is cached process-wide, so doing it out here — where the caller
    can show a spinner on an ordinary terminal — makes the panel's own startup
    instant. Costs nothing if something already warmed it.
    """
    try:
        from ...services import agent_backends

        agent_backends.availability()
    except Exception:  # noqa: BLE001 — a probe failure must not stop Settings opening
        pass


def open_panel(ctx, *, unicode: bool = True) -> int:
    warm_up()
    return SettingsPanel(ctx, theme.Glyphs(unicode)).run() or 0
