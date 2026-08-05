"""The review panel: what the jury decided, and why.

The jury's whole argument is that four independent perspectives beat one
confident one. A UI that collapses them into a single verdict throws away the
thing you paid four model calls for. So the panel keeps the evidence next to
the decision:

* **Verdict** — the foreperson's synthesis, and what it *dismissed*. Dismissals
  are shown, not hidden: "three jurors raised this and the foreperson dropped it
  as low-confidence" is a fact a reviewer may reasonably disagree with.
* **Jurors** — every seat, in roster order, including the ones that abstained.
  An abstention is a coverage gap; a panel reporting "3/4 approved" while
  silently omitting the fourth seat is lying about how well the change was
  reviewed.
* **Diff**, **Plan**, **QA** — the change, what it was supposed to be, and
  whether the tests agreed.

The panel decides nothing itself. It returns an intent and the shell performs
it, so every mutation still goes through ``core``.
"""

from __future__ import annotations

from typing import Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, ListItem, ListView, Static, TabbedContent, TabPane

from .. import render, theme
from . import BRAND_CSS

S = theme.s

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_SEVERITY_STYLE = {
    "critical": "sev.critical", "high": "sev.high",
    "medium": "sev.medium", "low": "sev.low",
}
_JUROR_VERDICT_STYLE = {"APPROVE": "ok", "REQUEST_CHANGES": "warn", "ABSTAIN": "muted"}


def _severity_style(severity: str) -> str:
    return S(_SEVERITY_STYLE.get((severity or "medium").lower(), "sev.medium"))


def _verdict_style(verdict: str) -> str:
    v = (verdict or "").upper()
    if v == "APPROVED":
        return S("ok")
    if v in ("CHANGES REQUESTED", "CHANGES_REQUESTED", "REQUEST_CHANGES"):
        return S("warn")
    return S("err")          # INCONCLUSIVE, and anything unrecognised


# ── renderables ───────────────────────────────────────────────────────────────
def _finding_block(finding: dict, g: theme.Glyphs) -> RenderableType:
    severity = (finding.get("severity") or "medium").lower()
    style = _severity_style(severity)
    confidence = finding.get("confidence")

    head = Text(f"{g.bullet} ", style=style)
    head += Text(finding.get("title") or "(untitled finding)", style="bold")

    meta = Text(f"    {severity}", style=style)
    if isinstance(confidence, (int, float)):
        meta += Text(f"   {g.bullet}   confidence {confidence:.0%}", style=S("muted"))
    if finding.get("location"):
        meta += Text(f"   {g.bullet}   ", style=S("muted"))
        meta += Text(str(finding["location"]), style=S("path"))

    body: list[RenderableType] = [head, meta]
    for label, key in (("why it matters", "why_it_matters"), ("suggestion", "suggestion")):
        if finding.get(key):
            body.append(Text(f"    {label}: ", style=S("muted")) + Text(str(finding[key])))
    if finding.get("evidence"):
        body.append(Text(""))
        for line in str(finding["evidence"]).splitlines()[:8]:
            body.append(Text(f"      {line}", style=S("muted")))
    body.append(Text(""))
    return Group(*body)


def verdict_view(payload: dict, g: theme.Glyphs) -> RenderableType:
    jury = payload.get("jury") or {}

    if not jury.get("verdict"):
        # The panel is a setting, so a delivery reviewed by the single-reviewer
        # path has prose and no structured decision. Show the prose rather than
        # an empty panel implying it was never reviewed at all.
        summary = payload.get("review_summary") or ""
        return Group(
            Text("Reviewed by a single reviewer — the jury was off for this run",
                 style=S("muted")),
            Text(""),
            Markdown(summary) if summary else Text("No review recorded.", style=S("muted")),
        )

    jurors = jury.get("jurors") or []
    voted = [j for j in jurors
             if (j.get("verdict") or "").upper() != "ABSTAIN" and not j.get("error")]
    approving = [j for j in voted if (j.get("verdict") or "").upper() == "APPROVE"]

    banner = Text(str(jury["verdict"]).upper(), style=_verdict_style(jury["verdict"]))
    if jurors:
        banner += Text(f"    {len(approving)}/{len(jurors)} jurors approving", style=S("muted"))
    if len(voted) < len(jurors):
        banner += Text(f"    ({len(jurors) - len(voted)} abstained — coverage gap)",
                       style=S("warn"))
    body: list[RenderableType] = [banner]

    if jury.get("synthesis") == "deterministic":
        body.append(Text(""))
        body.append(Text(f"{g.fail} Synthesised WITHOUT the foreperson model — merged "
                         "mechanically by title overlap, and deliberately more permissive "
                         "than a full review.", style=S("warn")))

    if jury.get("rationale"):
        body.extend([Text(""), Markdown(str(jury["rationale"]))])

    for label, key, style_name in (("Blocking", "blocking", "err"),
                                   ("Observations", "observations", "warn"),
                                   ("Dismissed by the foreperson", "dismissed", "muted")):
        entries = jury.get(key) or []
        if not entries:
            continue
        body.extend([
            Text(""),
            Text(f"{label}  ({len(entries)})", style=f"bold {S(style_name)}"),
            Text(""),
        ])
        body.extend(_finding_block(entry, g) for entry in entries)

    return Group(*body)


def checks_view(payload: dict, g: theme.Glyphs) -> RenderableType:
    body: list[RenderableType] = [Text("Delivery checks", style=S("heading")), Text("")]
    for check in payload.get("checks") or []:
        mark = (Text(f"  {g.ok} ", style=S("ok")) if check["ok"]
                else Text(f"  {g.pending} ", style=S("warn")))
        body.append(mark + Text(check["label"], style="" if check["ok"] else S("muted")))

    pr = payload.get("pr") or {}
    stat = Text(f"  +{pr.get('insertions', 0)}", style=S("diff.add"))
    stat += Text(f"  -{pr.get('deletions', 0)}", style=S("diff.del"))
    if pr.get("branch"):
        stat += Text(f"    {pr['branch']}", style=S("muted"))
    body.extend([Text(""), stat])
    if pr.get("url"):
        body.append(Text(f"  {pr['url']}", style=S("url")))
    return Group(*body)


def juror_view(juror: dict, g: theme.Glyphs) -> RenderableType:
    verdict = (juror.get("verdict") or "ABSTAIN").upper()
    head = Text(str(juror.get("name") or "juror"), style="bold")
    head += Text(f"    {verdict}", style=S(_JUROR_VERDICT_STYLE.get(verdict, "muted")))
    body: list[RenderableType] = [
        head,
        Text(str(juror.get("model_label") or juror.get("model") or ""), style=S("muted")),
    ]

    if juror.get("error"):
        # Loudly. A seat that failed to return an opinion is not a neutral
        # event — it is a perspective the verdict above was decided without.
        body.extend([
            Text(""),
            Text(f"{g.fail} This seat did not return a usable opinion:", style=S("err")),
            Text(f"  {juror['error']}", style=S("muted")),
            Text(""),
            Text("Its perspective is missing from the verdict.", style=S("warn")),
        ])
        return Group(*body)

    if juror.get("summary"):
        body.extend([Text(""), Markdown(str(juror["summary"]))])

    findings = sorted(
        juror.get("findings") or [],
        key=lambda f: _SEVERITY_ORDER.get((f.get("severity") or "medium").lower(), 9))
    body.append(Text(""))
    if not findings:
        body.append(Text("No findings from this seat.", style=S("muted")))
    else:
        body.extend([Text(f"Findings ({len(findings)})", style=S("heading")), Text("")])
        body.extend(_finding_block(finding, g) for finding in findings)
    return Group(*body)


def diff_view(payload: dict) -> RenderableType:
    lines = (payload.get("pr") or {}).get("diff") or []
    table = Table(box=None, pad_edge=False, show_header=False, padding=(0, 1, 0, 0))
    table.add_column(justify="right", style=S("diff.lineno"), width=5, no_wrap=True)
    table.add_column(width=1, no_wrap=True)
    table.add_column(overflow="fold")

    for line in lines:
        kind, text = line.get("type"), str(line.get("text", ""))
        number = str(line.get("n") or "")
        if kind == "file":
            table.add_row("", "", Text(f"\n{text}", style=S("diff.file")))
        elif kind == "hunk":
            table.add_row("", "", Text(text, style=S("diff.hunk")))
        elif kind == "add":
            table.add_row(number, Text("+", style=S("diff.add")), Text(text, style=S("diff.add")))
        elif kind == "del":
            table.add_row("", Text("-", style=S("diff.del")), Text(text, style=S("diff.del")))
        else:
            table.add_row(number, " ", Text(text, style=S("muted")))
    return table


def qa_view(payload: dict) -> RenderableType:
    body: list[RenderableType] = [Text("QA", style=S("heading")), Text("")]
    coverage = payload.get("coverage")
    if coverage is not None:
        body.extend([Text(f"coverage {coverage}%", style=S("muted")), Text("")])
    for note in payload.get("qa_notes") or []:
        body.extend([Text(str(note)), Text("")])
    return Group(*body)


# ── the app ───────────────────────────────────────────────────────────────────
class ReviewPanel(App[dict]):
    """Returns an intent dict; the shell performs it."""

    CSS = BRAND_CSS + """
    #jurors { width: 34; border-right: solid $panel; }
    #juror-detail { padding: 0 1; }
    .scroll { padding: 1 2; }
    """

    BINDINGS = [
        Binding("q,escape", "close", "Close"),
        Binding("a", "open_pr", "Open PR"),
        Binding("c", "request_changes", "Request changes"),
        Binding("m", "mark_done", "Mark done"),
        Binding("o", "open_url", "Browser"),
        Binding("t", "toggle_theme", "Light/dark"),
    ]

    def __init__(self, payload: dict, g: theme.Glyphs) -> None:
        super().__init__()
        self.payload = payload
        self.g = g
        self.jurors: list[dict] = (payload.get("jury") or {}).get("jurors") or []

    def compose(self) -> ComposeResult:
        task = self.payload.get("task") or {}
        verdict = (self.payload.get("jury") or {}).get("verdict") or "not reviewed"
        self.title = f"{task.get('key', '')} — {task.get('title', '')}"[:70]
        self.sub_title = str(verdict)

        yield Header(show_clock=False)

        with TabbedContent(initial="tab-verdict"):
            with TabPane("Verdict", id="tab-verdict"), VerticalScroll(classes="scroll"):
                yield Static(verdict_view(self.payload, self.g))
                yield Static(Text(""))
                yield Static(checks_view(self.payload, self.g))

            with TabPane(f"Jurors ({len(self.jurors)})", id="tab-jurors"), Horizontal():
                yield ListView(*[ListItem(Static(self._juror_label(j))) for j in self.jurors],
                               id="jurors")
                with VerticalScroll(classes="scroll"):
                    yield Static(
                        juror_view(self.jurors[0], self.g) if self.jurors
                        else Text("No juror opinions recorded.", style=S("muted")),
                        id="juror-detail")

            with TabPane("Diff", id="tab-diff"), VerticalScroll(classes="scroll"):
                yield Static(diff_view(self.payload))

            with TabPane("Plan", id="tab-plan"), VerticalScroll(classes="scroll"):
                yield Static(render.plan(self.payload.get("plan") or {}, self.g))

            with TabPane("QA", id="tab-qa"), VerticalScroll(classes="scroll"):
                yield Static(qa_view(self.payload))

        yield Footer()

    def _juror_label(self, juror: dict) -> RenderableType:
        verdict = (juror.get("verdict") or "ABSTAIN").upper()
        mark = {"APPROVE": self.g.ok, "REQUEST_CHANGES": self.g.fail}.get(verdict, self.g.pending)
        style = S(_JUROR_VERDICT_STYLE.get(verdict, "muted"))
        line = Text(f"{mark} ", style=style) + Text(str(juror.get("name") or "juror"))
        count = len(juror.get("findings") or [])
        if count:
            line += Text(f"   {count}", style=S("muted"))
        return Group(line, Text(f"  {(juror.get('model_label') or '')[:28]}", style=S("muted")))

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Give the juror list the keyboard as soon as its tab opens.

        Without this the arrows still belong to the tab bar, and moving between
        jurors — the main thing you do on this tab — silently does nothing.
        """
        if event.pane.id == "tab-jurors" and self.jurors:
            self.query_one("#jurors", ListView).focus()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "jurors" or not self.jurors:
            return
        index = event.list_view.index or 0
        if 0 <= index < len(self.jurors):
            self.query_one("#juror-detail", Static).update(juror_view(self.jurors[index], self.g))

    # ── actions ──────────────────────────────────────────────────────────────
    def action_close(self) -> None:
        self.exit({})

    def action_open_pr(self) -> None:
        self.exit({"action": "pr"})

    def action_mark_done(self) -> None:
        self.exit({"action": "merge"})

    def action_open_url(self) -> None:
        url = (self.payload.get("pr") or {}).get("url")
        if url:
            self.exit({"action": "open", "url": url})
        else:
            self.notify("No pull request URL on this ticket yet.", severity="warning")

    def action_request_changes(self) -> None:
        # Collecting the note belongs to the prompt, which already has history
        # and real line editing — a modal here would be worse at the same job.
        # Close with the intent and let the shell ask.
        self.exit({"action": "changes", "ask_note": True})

    def action_toggle_theme(self) -> None:
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"


def open_panel(payload: dict, *, unicode: bool = True) -> dict[str, Any]:
    """Run the panel; return the intent the user chose (possibly empty)."""
    return ReviewPanel(payload, theme.Glyphs(unicode)).run() or {}
