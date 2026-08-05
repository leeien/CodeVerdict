"""The interactive shell.

The home surface is a conversation, not a set of screens. You describe a change in
plain English and the PM agent works it; slash commands drive everything the
conversation can't express. Full-screen panels exist for the three things that
genuinely need two dimensions — the jury's review, the settings matrix, and the
jury roster — and they open *over* the transcript and hand control back, so the
scrollback remains the record of the session.

Input is prompt_toolkit (history, completion, sane editing on every platform);
output is Rich. They cooperate as long as only one of them owns the screen at a
time, which is why long operations render inside ``thinking()`` and panels take
the alternate screen.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console, RenderableType

from ..config import settings as cfg
from ..core import CoreError
from . import commands, render, theme
from .context import Context

_PROMPT_STYLE = Style.from_dict({
    "prompt": f"bold {theme.GOLD}",
    "": "",           # typed text keeps the terminal's own colour
})


class SlashCompleter(Completer):
    """Completes command names, then whatever that command declares.

    Driven entirely off the command registry, so a new command is completable
    the moment it is registered — there is no second list to forget to update.
    """

    def __init__(self, shell: Shell) -> None:
        self.shell = shell

    def get_completions(self, document, complete_event) -> Iterator[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        head, _, tail = text.partition(" ")
        if not _:
            for name in commands.ORDER:
                if name.startswith(head):
                    cmd = commands.REGISTRY[name]
                    yield Completion(name, start_position=-len(head),
                                     display=name, display_meta=cmd.help)
            return

        cmd = commands.resolve(head)
        if cmd is None:
            return
        source = cmd.completions
        try:
            options = source(self.shell) if callable(source) else list(source)
        except Exception:  # noqa: BLE001 — completion must never break the prompt
            return
        word = tail.rsplit(" ", 1)[-1]
        for option in options:
            if option.lower().startswith(word.lower()):
                yield Completion(option, start_position=-len(word))


class Shell:
    """One interactive session."""

    def __init__(self, ctx: Context, console: Console | None = None) -> None:
        self.ctx = ctx
        self.unicode = theme.supports_unicode()
        self.g = theme.Glyphs(self.unicode)
        self.console = console or theme.console()
        self._session: PromptSession | None = None

    # ── output helpers ───────────────────────────────────────────────────────
    def print(self, renderable: RenderableType) -> None:
        self.console.print(renderable)

    def note(self, text: str, style: str = "muted") -> None:
        self.console.print(render.note(text, self.g, style=style))

    def ask(self, question: str, default: str = "") -> str:
        """One follow-up answer, on the same prompt stack as everything else.

        Panels close before asking rather than opening a modal, so this always
        runs with the terminal back in line-editing mode.
        """
        try:
            answer = self._prompt_session().prompt(
                FormattedText([("class:prompt", f"  {self.g.prompt} "),
                               ("", f"{question} ")]),
                default=default)
        except (KeyboardInterrupt, EOFError):
            return default
        return answer.strip()

    @contextlib.contextmanager
    def thinking(self, message: str):
        """A spinner for work that blocks the prompt (an LLM turn, an index build)."""
        with self.console.status(f"[muted]{message}{self.g.ellipsis}[/muted]",
                                 spinner="dots", spinner_style="brand"):
            yield

    # ── chrome ───────────────────────────────────────────────────────────────
    def demo_mode(self) -> bool:
        return bool(getattr(cfg, "demo_mode", True))

    def banner(self) -> None:
        self.print(render.banner(self.console.width, self.g, self.unicode))
        self.status()
        self.note(self._opening_hint())

    def _opening_hint(self) -> str:
        """The first thing to type — which is not the same on a fresh install.

        "Describe a change in plain English" is the right instruction only once a
        repository is indexed; before that it is advice the user cannot follow,
        which is how a first run turns into reading the README.
        """
        from ..core import repos as core_repos

        with contextlib.suppress(Exception), self.ctx.db() as db:
            if not core_repos.listing(db):
                return ("start by indexing a repository: "
                        "/kb add https://github.com/pallets/click"
                        f"   {self.g.bullet}   /doctor checks your setup")
        return "describe a change in plain English, or /help for commands"

    def status(self) -> None:
        from ..core import costs as core_costs

        with self.ctx.db() as db:
            repo = self.ctx.repo(db)
            scope = self.ctx.scope(db)
            total = 0.0
            if repo is not None:
                with contextlib.suppress(Exception):
                    total = core_costs.breakdown(db, repo.id)["totals"]["cost"]
        stages = {f"{k}_{s}": getattr(cfg, f"{k}_{s}", "")
                  for k, _ in render.STAGES for s in ("provider", "model")}
        self.print(render.status_line(repo=repo, scope=scope, stages=stages,
                                      demo=self.demo_mode(), cost=total, g=self.g))

    def _prompt_fragments(self) -> FormattedText:
        return FormattedText([("class:prompt", f"  {self.g.prompt} ")])

    # ── dispatch ─────────────────────────────────────────────────────────────
    def handle(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if not line.startswith("/"):
            commands.talk(self, line)
            return

        head, _, tail = line.partition(" ")
        cmd = commands.resolve(head)
        if cmd is None:
            self.print(render.error(
                f"Unknown command {head}. /help lists everything.", self.g))
            return
        cmd.run(self, tail.strip())

    # ── loop ─────────────────────────────────────────────────────────────────
    def _prompt_session(self) -> PromptSession:
        if self._session is None:
            bindings = KeyBindings()

            @bindings.add("escape", "enter")     # Alt-Enter: newline, for pasted detail
            def _(event) -> None:
                event.current_buffer.insert_text("\n")

            history_path = _history_path()
            self._session = PromptSession(
                completer=SlashCompleter(self),
                history=FileHistory(str(history_path)) if history_path else None,
                key_bindings=bindings,
                style=_PROMPT_STYLE,
                complete_while_typing=False,
            )
        return self._session

    def run(self) -> int:
        self.console.clear()
        self.banner()
        session = self._prompt_session()

        while True:
            try:
                line = session.prompt(self._prompt_fragments())
            except KeyboardInterrupt:
                # Ctrl-C at an empty prompt is "clear this line", not "quit" —
                # quitting on it is how CLIs lose people's work.
                continue
            except EOFError:
                self.console.print()
                self.note("bye")
                return 0

            try:
                self.handle(line)
            except SystemExit as exit_:
                self.console.print()
                self.note("bye")
                return int(exit_.code or 0)
            except CoreError as err:
                self.print(render.error(str(err), self.g))
            except KeyboardInterrupt:
                self.print(render.note("interrupted", self.g, style="warn"))
            except Exception as err:  # noqa: BLE001 — a bad command must not end the session
                self.print(render.error(f"{type(err).__name__}: {err}", self.g))
                # Read the toggle from the environment, not from Settings: this
                # handler is the last thing standing between a buggy command and
                # a lost session, so it must not depend on a config field that
                # can go missing. It did — this line used to read a log-level
                # setting that Settings never defined, and the AttributeError
                # escaped the very handler it lived in, killing the shell and
                # taking a running background KB ingest down with it.
                if os.environ.get("CODEVERDICT_DEBUG"):
                    self.console.print_exception()


def _history_path():
    """Command history beside the database, for the same reason selection state is."""
    from pathlib import Path

    url = cfg.database_url
    if url.startswith("sqlite:///"):
        return Path(url[len("sqlite:///"):]).with_suffix(".history")
    return Path.home() / ".codeverdict_history"


def start(ctx: Context) -> int:
    return Shell(ctx).run()
