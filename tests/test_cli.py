"""The terminal client: dispatch, selection state, and the rendering contract.

These cover the seams where the CLI could quietly diverge from the rest of the
system — command lookup, which repo/scope a command acts on, and the style
resolution that Textual gets wrong silently rather than loudly.
"""

from __future__ import annotations

import pytest
from app.cli import commands, theme
from app.cli.context import Context
from app.cli.panels.settings import _visible
from app.core import CoreError
from app.models import Repo, ScopeSession
from sqlmodel import Session


@pytest.fixture()
def repos(db: Session) -> tuple[Repo, Repo]:
    first = Repo(name="rich", org="Textualize", git_url="https://github.com/Textualize/rich",
                 key_prefix="R", default_branch="main")
    second = Repo(name="click", org="pallets", git_url="https://github.com/pallets/click",
                  key_prefix="C", default_branch="main")
    db.add(first)
    db.add(second)
    db.commit()
    db.refresh(first)
    db.refresh(second)
    return first, second


# ── command dispatch ─────────────────────────────────────────────────────────
class TestRegistry:
    def test_every_command_is_reachable_by_its_own_name(self):
        for name in commands.ORDER:
            assert commands.resolve(name) is not None, name

    def test_resolves_without_the_leading_slash(self):
        """Subcommands dispatch through the same registry as slash commands."""
        assert commands.resolve("board") is commands.resolve("/board")

    def test_resolves_an_unambiguous_prefix(self):
        assert commands.resolve("/appr").name == "/approve"

    def test_refuses_an_ambiguous_prefix(self):
        # /model and /models both exist; guessing between them would silently
        # repoint a pipeline stage when the user meant to look at one.
        assert commands.resolve("/mode") is None

    def test_aliases_reach_the_same_command(self):
        assert commands.resolve("/q") is commands.resolve("/quit")
        assert commands.resolve("/b") is commands.resolve("/board")

    def test_unknown_command_resolves_to_nothing(self):
        assert commands.resolve("/definitely-not-a-command") is None

    def test_help_text_exists_for_every_command(self):
        """The registry is what /help and tab-completion render from."""
        for name in commands.ORDER:
            assert commands.REGISTRY[name].help.strip(), f"{name} has no help"


# ── selection state ──────────────────────────────────────────────────────────
class TestContext:
    def test_switching_repo_clears_the_scope(self, db: Session, repos):
        """A scope belongs to one repo; carrying it across would have the PM
        agent answering about the wrong codebase."""
        first, second = repos
        scope = ScopeSession(repo_id=first.id, kind="pm", title="add a flag")
        db.add(scope)
        db.commit()
        db.refresh(scope)

        ctx = Context()
        ctx.select_scope(scope)
        assert ctx.repo_id == first.id and ctx.session_id == scope.id

        ctx.select_repo(second)
        assert ctx.repo_id == second.id
        assert ctx.session_id is None

    def test_reselecting_the_same_repo_keeps_the_scope(self, db: Session, repos):
        first, _ = repos
        scope = ScopeSession(repo_id=first.id, kind="pm", title="add a flag")
        db.add(scope)
        db.commit()
        db.refresh(scope)

        ctx = Context()
        ctx.select_scope(scope)
        ctx.select_repo(first)
        assert ctx.session_id == scope.id

    def test_ensure_repo_auto_selects_when_only_one_exists(self, db: Session):
        repo = Repo(name="rich", org="Textualize", git_url="https://x/rich", key_prefix="R")
        db.add(repo)
        db.commit()

        ctx = Context()
        assert ctx.ensure_repo(db).name == "rich"

    def test_ensure_repo_asks_rather_than_guessing_between_several(self, db: Session, repos):
        ctx = Context()
        with pytest.raises(CoreError) as excinfo:
            ctx.ensure_repo(db)
        assert excinfo.value.status == 409
        # The message has to name the alternatives, or it isn't actionable.
        assert "rich" in str(excinfo.value)

    def test_ensure_repo_explains_the_empty_case_differently(self, db: Session):
        """'Nothing indexed yet' and 'several, pick one' need different advice."""
        ctx = Context()
        with pytest.raises(CoreError) as excinfo:
            ctx.ensure_repo(db)
        assert "/kb add" in str(excinfo.value)


# ── rendering contract ───────────────────────────────────────────────────────
class TestTheme:
    def test_palette_names_resolve_to_literal_styles(self):
        """Textual renders widgets with its own machinery and never consults a
        Rich theme, so a bare palette name there is not an error — it just comes
        out uncoloured. Everything shared between the two must resolve first."""
        assert theme.s("ok") == theme.STYLES["ok"]
        assert theme.s("sev.critical").startswith("bold ")
        assert theme.s("ok") != "ok"

    def test_unknown_style_passes_through_untouched(self):
        # "bold" and "" are real Rich styles that aren't in our palette.
        assert theme.s("bold") == "bold"
        assert theme.s("") == ""

    def test_every_palette_entry_is_a_parseable_rich_style(self):
        from rich.style import Style

        for _name, value in theme.STYLES.items():
            Style.parse(value)      # raises if the palette holds a typo

    def test_glyphs_have_ascii_fallbacks(self):
        """--ascii has to change every glyph, not most of them."""
        unicode_glyphs = theme.Glyphs(True)
        ascii_glyphs = theme.Glyphs(False)
        for slot in theme.Glyphs.__slots__ if hasattr(theme.Glyphs, "__slots__") else vars(ascii_glyphs):
            value = getattr(ascii_glyphs, slot)
            assert value.isascii(), f"{slot} is not ASCII in fallback mode: {value!r}"
            assert value != getattr(unicode_glyphs, slot) or value.isascii()

    def test_wordmark_degrades_on_a_narrow_terminal(self):
        wide = theme.wordmark(100, unicode=True)
        narrow = theme.wordmark(40, unicode=True)
        assert len(wide) > 1                       # the block wordmark
        assert len(narrow) == 1                    # the lockup
        assert all(len(line) <= 72 for line in wide)


# ── settings form spec ───────────────────────────────────────────────────────
class TestShowIf:
    def test_field_with_no_condition_is_always_visible(self):
        assert _visible({"show_if": ""}, {})

    def test_condition_matches_the_driver_value(self):
        field = {"show_if": "rag_embeddings=api"}
        assert _visible(field, {"rag_embeddings": "api"})
        assert not _visible(field, {"rag_embeddings": "local"})

    def test_condition_accepts_alternatives(self):
        field = {"show_if": "engine=api|local"}
        assert _visible(field, {"engine": "local"})
        assert not _visible(field, {"engine": "tfidf"})

    def test_booleans_are_compared_as_words(self):
        """The spec writes bools as true/false; the values are real bools."""
        field = {"show_if": "jury_enabled=true"}
        assert _visible(field, {"jury_enabled": True})
        assert not _visible(field, {"jury_enabled": False})


# ── the settings panel's edit detection ──────────────────────────────────────
class TestBlankSentinel:
    """`Select.BLANK` is not part of Select's API in the installed Textual.

    The name still resolves — up the MRO to ``Widget.BLANK``, an unrelated flag
    whose value is ``False`` — so using it as "no selection" constructs a Select
    with a literal ``False`` and blows up in ``_validate_value`` the first time
    a field has no configured value. It fails at mount, not at import, so only
    an assertion like this catches it before a user does.
    """

    def test_blank_is_not_the_no_selection_sentinel(self):
        from textual.widgets import Select

        assert Select.BLANK is not Select.NULL
        assert Select.BLANK is False        # noqa: E712 — that is the whole point

    def test_no_panel_uses_blank_as_a_sentinel(self):
        """Comments may name it (they explain the trap); code may not use it."""
        import pathlib

        panels = pathlib.Path(__file__).resolve().parents[1] / "backend/app/cli/panels"
        offenders = []
        for path in panels.glob("*.py"):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                if "Select.BLANK" in code:
                    offenders.append(f"{path.name}:{lineno}")
        assert not offenders, f"Select.BLANK used as a sentinel at: {offenders}"


class TestRecordIgnoresConstructionEchoes:
    """Mounting a widget with a value makes it post its own Changed message.

    Textual delivers that on a later pump tick, so a "are we still mounting?"
    flag can't catch it. If those echoes count as edits, two things break: every
    field looks edited (Ctrl-S then writes back all 59 settings nobody touched),
    and each provider field's echo re-renders the group, which rebuilds all six
    provider fields, which echo again — an unbounded loop that pegs a core.
    """

    @staticmethod
    def _panel(values: dict):
        from app.cli.panels.settings import SettingsPanel

        panel = SettingsPanel.__new__(SettingsPanel)   # no App bootstrap needed
        panel.values = dict(values)
        panel.pending = {}
        panel.group_index = 0
        panel._rerender_calls = 0
        panel._rerender = lambda: setattr(          # type: ignore[method-assign]
            panel, "_rerender_calls", panel._rerender_calls + 1)
        return panel

    def test_echo_of_the_constructed_value_is_not_an_edit(self):
        panel = self._panel({"dev_provider": "claude-cli"})
        panel._record("f-dev_provider", "claude-cli")
        assert panel.pending == {}
        assert panel._rerender_calls == 0, "an echo triggered a re-render — this is the loop"

    def test_numeric_echo_is_not_an_edit(self):
        """Input reports a str; the setting is stored as a float. Comparing them
        raw reads every numeric field as edited the instant its group is drawn."""
        panel = self._panel({"claude_max_budget_usd": 2.0})
        panel._record("f-claude_max_budget_usd", "2.0")
        assert panel.pending == {}

    def test_unset_field_echoing_empty_is_not_an_edit(self):
        panel = self._panel({"jury_synthesis_model": None})
        panel._record("f-jury_synthesis_model", "")
        assert panel.pending == {}

    def test_a_real_change_is_still_recorded(self):
        panel = self._panel({"dev_provider": "claude-cli"})
        panel._record("f-dev_provider", "anthropic")
        assert panel.pending == {"dev_provider": "anthropic"}
        assert panel._rerender_calls == 1, "a provider change must refresh its model list"

    def test_repeating_a_real_change_does_not_re_render(self):
        panel = self._panel({"dev_provider": "claude-cli"})
        panel._record("f-dev_provider", "anthropic")
        panel._record("f-dev_provider", "anthropic")
        assert panel._rerender_calls == 1


class TestModelCommandValidation:
    """`/model` joins everything after the provider into the model id, because
    real ids contain slashes ('openai/gpt-oss-120b'). That made a trailing typo
    invisible: `/model planner claude-cli sonnet /models` stored the model as
    "sonnet /models" and the only symptom was the Planner failing mid-run."""

    def test_a_trailing_slash_command_is_rejected(self, db: Session, repos):
        from app.cli.repl import Shell

        shell = Shell(Context(), console=theme.console())
        with pytest.raises(CoreError) as exc:
            commands.resolve("/model").run(shell, "planner claude-cli sonnet /models")
        assert "doesn't look like a model id" in str(exc.value)
        assert "/model planner claude-cli sonnet" in str(exc.value)

    def test_a_slashed_model_id_still_works(self, db: Session, repos):
        """Provider-prefixed ids are the normal case and must not be blocked."""
        from app.cli.repl import Shell

        shell = Shell(Context(), console=theme.console())
        commands.resolve("/model").run(shell, "qa groq openai/gpt-oss-120b")
        from app.config import settings as cfg
        assert cfg.qa_model == "openai/gpt-oss-120b"


class TestKbViews:
    """`/kb views` renders the same envelope the web Analysis screen consumes —
    {slug, total, labels, order, domains} — not a bare {domain: items} mapping.
    Iterating the envelope's top level walked `slug` (a str) and then `total`
    (an int), so the command died on `len(15)` the moment a repo had any
    knowledge at all."""

    def _views(self, monkeypatch, payload):
        from app.core import repos as core_repos

        monkeypatch.setattr(core_repos, "knowledge", lambda db, repo_id: payload)

    def test_counts_come_from_the_domains_map(self, db: Session, monkeypatch, capsys):
        from app.cli.repl import Shell

        self._views(monkeypatch, {
            "slug": "go-gitea__gitea", "total": 3,
            "labels": {"structure": "Code graph", "deliveries": "Delivery notes"},
            "order": ["structure", "deliveries"],
            "domains": {"structure": [{"id": "graph_architecture"}],
                        "deliveries": [{"id": "a"}, {"id": "b"}]},
        })
        repo = Repo(name="gitea", org="go-gitea", git_url="https://x/gitea", key_prefix="G")
        db.add(repo)
        db.commit()

        shell = Shell(Context(), console=theme.console())
        commands.resolve("/kb").run(shell, "views")
        out = capsys.readouterr().out
        # Labelled by domain, counted from the lists — never from `total`/`slug`.
        assert "Code graph: 1 entries" in out
        assert "Delivery notes: 2 entries" in out
        assert "slug:" not in out

    def test_an_empty_envelope_reports_nothing_generated(self, db: Session, monkeypatch, capsys):
        """`total`/`slug` are always present, so truthiness of the envelope says
        nothing about whether any knowledge exists — the domains do."""
        from app.cli.repl import Shell

        self._views(monkeypatch, {
            "slug": "go-gitea__gitea", "total": 0, "labels": {}, "order": [],
            "domains": {"structure": [], "deliveries": []},
        })
        repo = Repo(name="gitea", org="go-gitea", git_url="https://x/gitea", key_prefix="G")
        db.add(repo)
        db.commit()

        shell = Shell(Context(), console=theme.console())
        commands.resolve("/kb").run(shell, "views")
        assert "No structured knowledge" in capsys.readouterr().out


class TestReplSurvivesABadCommand:
    """The catch-all in Shell.run is the last thing between a buggy command and
    a lost session — including a background KB ingest running in-process. It
    consulted `cfg.log_level`, which does not exist on Settings, so the
    AttributeError escaped the handler and killed the shell."""

    def test_the_debug_toggle_does_not_read_a_missing_setting(self):
        from app.config import settings as cfg

        assert not hasattr(cfg, "log_level")

    def test_no_cli_module_reads_a_setting_that_does_not_exist(self):
        """Same class of bug, one grep wider: any `cfg.<name>` in the CLI that
        Settings doesn't define is a crash waiting for the line to execute."""
        import re
        from pathlib import Path

        from app.config import settings as cfg

        cli = Path(commands.__file__).parent
        missing = [
            (path.name, name)
            for path in cli.rglob("*.py")
            for name in re.findall(r"\bcfg\.([a-z_][a-z0-9_]*)",
                                       path.read_text(encoding="utf-8"))
            if not hasattr(cfg, name)
        ]
        assert missing == []


class TestInterruptedIndexIsReconciled:
    """Indexing runs on this process's thread pool, so an `indexing` row at boot
    is always a corpse. Left alone it reads as live work forever — which is what
    sent a real gitea ingest looking like it was running when the clone had died
    with the shell twenty minutes earlier."""

    def test_boot_fails_a_row_left_indexing(self, db: Session):
        from app.core import repos as core_repos

        repo = Repo(name="gitea", org="go-gitea", git_url="https://x/gitea", key_prefix="G",
                    kb_status="indexing", kb_progress=5, kb_step="Cloning working copy…")
        db.add(repo)
        db.commit()

        assert core_repos.reconcile_interrupted(db) == 1
        db.refresh(repo)
        assert repo.kb_status == "failed"
        # The remedy has to be in the message, or the state is a dead end.
        assert "/kb reindex" in repo.kb_error
        assert "5%" in repo.kb_error

    def test_it_leaves_every_other_status_alone(self, db: Session):
        from app.core import repos as core_repos

        rows = [Repo(name=n, org="o", git_url=f"https://x/{n}", key_prefix="X", kb_status=s)
                for n, s in (("a", "ready"), ("b", "pending"), ("c", "failed"))]
        for r in rows:
            db.add(r)
        db.commit()

        assert core_repos.reconcile_interrupted(db) == 0
        for r in rows:
            db.refresh(r)
        assert [r.kb_status for r in rows] == ["ready", "pending", "failed"]


class TestRepoTableShowsIndexingProgress:
    """A multi-minute clone and a dead job both rendered as the bare word
    'indexing'. kb_progress/kb_step are written continuously by the ingest job —
    the table just discarded them."""

    def _render(self, rows):
        from app.cli import render
        from rich.console import Console

        console = Console(width=200, no_color=True)
        with console.capture() as cap:
            console.print(render.repos(rows, None, theme.Glyphs(False)))
        return cap.get()

    def test_percentage_and_step_are_shown_while_indexing(self):
        out = self._render([Repo(id=1, name="gitea", org="go-gitea", git_url="https://x/gitea",
                                 key_prefix="G", kb_status="indexing", kb_progress=35,
                                 kb_step="Analyzing structure (AST)…")])
        assert "indexing 35%" in out
        assert "Analyzing structure (AST)" in out

    def test_a_settled_repo_gets_no_percentage_or_step_line(self):
        out = self._render([Repo(id=1, name="rich", org="Textualize", git_url="https://x/rich",
                                 key_prefix="R", kb_status="ready", kb_progress=100,
                                 kb_step="done")])
        assert "ready" in out
        assert "100%" not in out
        assert "done" not in out


class TestKbLiveWatch:
    """`/kb add` used to print "runs in the background" and hand the prompt
    back, leaving `/kb status` — a bare word, no percentage, no step — as the
    only way to learn anything about a multi-minute build."""

    def test_it_polls_until_the_build_settles(self, db: Session, monkeypatch):
        from app.cli import live, theme
        from app.cli.context import Context
        from rich.console import Console

        repo = Repo(name="gitea", org="go-gitea", git_url="https://x/gitea", key_prefix="G",
                    kb_status="indexing", kb_progress=5, kb_step="Cloning working copy…")
        db.add(repo)
        db.commit()
        db.refresh(repo)

        # Each poll advances the build one step, ending ready.
        steps = iter([
            ("indexing", 55, "Indexing code graph…"),
            ("indexing", 91, "Embedding code graph nodes… 7,000/15,991"),
            ("ready", 100, "Ready — 120,521 nodes over 6,174 files, 15,991 vectors"),
        ])

        def advance(_db, _id):
            state = next(steps, None)
            if state:
                repo.kb_status, repo.kb_progress, repo.kb_step = state
            return repo

        monkeypatch.setattr(live, "time", type("T", (), {
            "sleep": staticmethod(lambda _s: None),
            "monotonic": staticmethod(lambda: 0.0)}))
        from app.core import repos as core_repos
        monkeypatch.setattr(core_repos, "require", advance)

        console = Console(width=100, no_color=True)
        with console.capture() as cap:
            status, step = live.watch_kb(console, theme.Glyphs(False), Context(),
                                         repo.id, poll_s=0)

        assert status == "ready"
        assert "15,991 vectors" in step
        # Live repaints in place, so only the settled frame is in the capture.
        out = cap.get()
        assert "100%" in out
        assert "15,991 vectors" in out

    def test_a_frame_names_the_step_not_just_a_number(self):
        """Cloning a 400MB history and embedding 15,991 nodes both sit in the
        middle of the bar; the percentage alone cannot tell them apart."""
        from app.cli import live, theme

        frame = live._kb_panel(theme.Glyphs(False), "indexing", 91,
                               "Embedding code graph nodes… 7,000/15,991", 137.0)
        from rich.console import Console
        console = Console(width=100, no_color=True)
        with console.capture() as cap:
            console.print(frame)
        out = cap.get()
        assert "Embedding code graph nodes" in out
        assert "7,000/15,991" in out
        assert " 91%" in out
        assert "2m17s" in out

    def test_the_bar_tracks_the_percentage(self):
        from app.cli import live

        assert live._kb_bar(0, False).startswith("░")
        assert live._kb_bar(100, True).strip() == "█" * 28
        half = live._kb_bar(50, False)
        assert half.count("█") == 14 and half.count("░") == 14


class TestFrames:
    """Every multi-row surface is drawn in a panel. A terminal session is one
    long scrollback with nothing separating a ticket list from a preflight
    report, so the frame is the seam."""

    def _out(self, renderable, width=92, ascii_mode=False):
        from rich.console import Console

        console = Console(width=width, no_color=True, legacy_windows=False)
        with console.capture() as cap:
            console.print(renderable)
        return cap.get()

    def test_a_frame_carries_its_title_and_subtitle(self):
        from app.cli import render, theme
        from rich.text import Text

        out = self._out(render.frame(Text("body"), theme.Glyphs(True),
                                     title="Preflight", subtitle="ready"))
        assert "Preflight" in out and "ready" in out
        assert "╭" in out and "╰" in out

    def test_it_degrades_to_ascii_when_glyphs_are_unavailable(self):
        """A console that cannot draw box characters must get ASCII, not
        mojibake — same capability probe the glyph vocabulary uses."""
        from app.cli import render, theme
        from rich.text import Text

        out = self._out(render.frame(Text("body"), theme.Glyphs(False), title="Preflight"))
        assert "╭" not in out and "│" not in out
        assert "+-" in out or "|" in out

    def test_no_frame_sets_a_background(self):
        """A filled panel is the fastest way to look broken on a light
        terminal; the palette rule is accents only, never a background."""
        from app.cli import render, theme
        from rich.text import Text

        panel = render.frame(Text("body"), theme.Glyphs(True), title="X")
        assert panel.renderable.style in ("", "none", None)


class TestCostFormatting:
    """Four fixed decimals rendered a $1.94 run as "$1.9400"; sub-cent spend
    still needs the precision, so the scale picks the places."""

    def _status(self, cost):
        from app.cli import render, theme
        from rich.console import Console

        line = render.status_line(repo=None, scope=None, stages={}, demo=True,
                                  cost=cost, g=theme.Glyphs(True))
        console = Console(width=100, no_color=True)
        with console.capture() as cap:
            console.print(line)
        return cap.get()

    def test_dollars_read_at_a_glance(self):
        assert "$1.94" in self._status(1.9400)
        assert "$1.9400" not in self._status(1.9400)

    def test_sub_cent_spend_keeps_its_precision(self):
        assert "$0.0042" in self._status(0.0042)

    def test_nothing_is_shown_before_anything_is_spent(self):
        assert "$" not in self._status(0.0)


class TestLiveActivityWindow:
    """The pinned region showed the stage timeline and nothing else, so a
    52-turn Dev run read as one spinner beside the word "Dev" while every tool
    call scrolled past above it. The scrollback is still the archive; this is
    the rolling window that says what is happening now."""

    def _view(self):
        from app.cli import live, theme

        view = live.RunView(theme.Glyphs(True), True)
        view.apply("run.started", {"agent": "dev", "run_id": 2})
        view.apply("run.model", {"run_id": 2, "model": "claude-cli/sonnet"})
        return view

    def _out(self, view):
        from app.cli import theme

        console = theme.console(width=100, no_color=True)
        with console.capture() as cap:
            console.print(view.renderable(include_tail=False))
        return cap.get()

    def test_the_running_stage_shows_what_it_is_doing(self):
        view = self._view()
        view.apply("run.log", {"run_id": 2, "severity": "info",
                               "message": '→ Edit: {"file_path":"repo-issue-sidebar.ts"}'})
        out = self._out(view)
        assert "Dev is doing" in out
        assert "repo-issue-sidebar.ts" in out

    def test_tool_calls_are_counted_live(self):
        """The Dev efficiency number used to arrive only once the stage had
        finished; watching it climb is the point."""
        view = self._view()
        for name in ("Read", "Grep", "Edit"):
            view.apply("run.log", {"run_id": 2, "severity": "info",
                                   "message": f'→ {name}: {{}}'})
        view.apply("run.log", {"run_id": 2, "severity": "info",
                               "message": "just prose, not a tool call"})
        assert view.stages["dev"].tools == 3
        assert "3 tool calls" in self._out(view)

    def test_the_window_rolls_rather_than_growing(self):
        view = self._view()
        for i in range(20):
            view.apply("run.log", {"run_id": 2, "severity": "info",
                                   "message": f"→ Read: file{i}.go"})
        from app.cli import live
        assert len(view.stages["dev"].activity) == live.ACTIVITY_LINES
        out = self._out(view)
        assert "file19.go" in out and "file0.go" not in out

    def test_activity_is_attributed_to_the_stage_that_logged_it(self):
        """Two stages can be mid-flight across a revision round; a line must not
        land on whichever happens to be rendered."""
        view = self._view()
        view.apply("run.started", {"agent": "qa", "run_id": 3})
        view.apply("run.log", {"run_id": 3, "severity": "info", "message": "→ Bash: go test"})
        assert view.stages["qa"].tools == 1
        assert view.stages["dev"].tools == 0

    def test_a_finished_stage_stops_showing_its_activity(self):
        view = self._view()
        view.apply("run.log", {"run_id": 2, "severity": "info", "message": "→ Read: x.go"})
        view.apply("run.finished", {"run_id": 2})
        assert "Dev is doing" not in self._out(view)


class TestAssumedCriteriaAreVisible:
    """PM tags each criterion [stated] or [assumed]. /approve is the last point
    a human sees the requirement before money is spent, so the ones PM decided
    rather than heard have to stand out — that is the gate that would have
    caught the gitea drift, and both of us skimmed past it."""

    def _out(self, criteria):
        from app.cli import render, theme
        from app.models import Task

        task = Task(id=1, key="G-101", title="t", status="scoped", repo_id=1,
                    acceptance_criteria=criteria, approved=False, priority="medium")
        console = theme.console(width=100, no_color=True)
        with console.capture() as cap:
            console.print(render.ticket_detail(task, theme.Glyphs(True)))
        return cap.get()

    def test_assumed_criteria_are_counted_in_the_header(self):
        out = self._out(["[stated] the dropdown excludes already-added issues",
                         "[assumed] they appear greyed out instead of hidden"])
        assert "1 assumed by the PM" in out

    def test_the_tags_are_stripped_from_the_text(self):
        out = self._out(["[stated] excludes already-added issues"])
        assert "excludes already-added issues" in out
        assert "[stated]" not in out

    def test_untagged_criteria_still_render(self):
        """Older tickets predate the tags; they must not lose their criteria."""
        out = self._out(["plain old criterion"])
        assert "plain old criterion" in out
        assert "assumed by the PM" not in out

    def test_all_stated_shows_no_warning(self):
        out = self._out(["[stated] a", "[stated] b"])
        assert "assumed by the PM" not in out


class TestSourceIsReadAsUtf8:
    """Windows defaults to cp1252, not UTF-8. Reading a file with no explicit
    encoding works on Linux and macOS and dies on Windows the moment that file
    contains a box glyph or an arrow — which every CLI source file does. CI
    caught it there and only there:

        UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f

    Cheap to state as a rule, so it is stated rather than rediscovered.
    """

    def test_no_read_text_without_an_explicit_encoding(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        offenders = [
            f"{path.relative_to(root)}:{i}"
            for folder in ("tests", "backend/app")
            for path in (root / folder).rglob("*.py")
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"\.read_text\(\s*\)", line)
        ]
        assert offenders == [], f"read_text() without encoding: {offenders}"


class TestLogRowsAreReadable:
    """From a real gitea run: the scrollback was a wall of full-width prose
    starting at column 0, because a log row was `Text(gutter) + Text(msg[:400])`
    — the gutter printed once and every wrapped continuation fell back to the
    terminal's left edge, with five agents' output indistinguishable."""

    def _render(self, severity="info", message="x", stage="", width=70):
        from app.cli import live, theme

        view = live.RunView(theme.Glyphs(True), True)
        console = theme.console(width=width, no_color=True)
        with console.capture() as cap:
            console.print(view.log_line(severity, message, stage=stage))
        return cap.get().rstrip("\n").split("\n")

    def test_wrapped_lines_hang_under_the_gutter(self):
        lines = self._render(message="w " * 80, stage="Review")
        assert len(lines) > 1, "a long message must wrap, not be cut"
        gutter = lines[0].index("│")
        for cont in lines[1:]:
            # Continuations start past the gutter column, not at column 0.
            assert cont[:gutter].strip() == "", f"continuation not indented: {cont!r}"

    def test_the_stage_names_the_entry(self):
        assert "Review" in self._render(stage="Review", message="a verdict")[0]

    def test_nothing_is_cut_mid_word(self):
        """The old 400-char slice ended sentences mid-token, so a reader could
        not tell a truncated agent from a terminated one."""
        message = "supercalifragilistic " * 40
        joined = " ".join(l.split("│", 1)[-1].strip() for l in self._render(message=message))
        assert "supercalifragilisti " not in joined
        assert joined.count("supercalifragilistic") == 40

    def test_a_long_reply_says_how_much_it_withheld(self):
        from app.cli import live

        rows = live._log_rows({"message": "\n".join(f"line {i}" for i in range(30))})
        assert len(rows) == live.LOG_LINES + 1
        assert "24 more line(s)" in rows[-1]
        assert "/review" in rows[-1], "the reader needs somewhere to go for the rest"

    def test_a_short_reply_is_left_whole(self):
        from app.cli import live

        rows = live._log_rows({"message": "one\ntwo\nthree"})
        assert rows == ["one", "two", "three"]
