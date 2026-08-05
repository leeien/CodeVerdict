"""Argument parsing and dispatch.

Most subcommands are the corresponding slash command run once against a
non-interactive shell. That is not a shortcut — it is the only way the two
modes stay honest with each other. ``codeverdict approve TASK-101`` and typing
``/approve TASK-101`` at the prompt are the same call, so they cannot drift,
and a new command is scriptable the moment it exists.

Only the machine-readable paths (``--json``) and ``settings get/set`` are
handled separately, because they have no conversational equivalent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from ..core import CoreError
from . import theme
from .context import Context

# Subcommands that are simply a slash command with its arguments. The tuple is
# (slash command, positional metavar or None, help text).
PASSTHROUGH: dict[str, tuple[str, str | None, str]] = {
    "repos":    ("/repo",     None,      "list indexed repositories"),
    "board":    ("/board",    None,      "show the delivery board"),
    "tickets":  ("/tickets",  None,      "list this scope's tickets"),
    "approve":  ("/approve",  "TICKET",  "approve a ticket (or 'all')"),
    "run":      ("/run",      None,      "run the selected scope end to end"),
    "plan":     ("/plan",     None,      "show the Planner's verified plan"),
    "models":   ("/models",   None,      "show the provider and model per stage"),
    "costs":    ("/costs",    None,      "token and dollar breakdown"),
    "pr":       ("/pr",       "TICKET",  "open a pull request for a delivered ticket"),
    "jury":     ("/jury",     None,      "open the jury roster"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeverdict",
        description="A jury of agents for your codebase. Run with no arguments for the shell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `codeverdict` with no arguments to open the interactive shell.",
    )
    parser.add_argument("--repo", metavar="NAME",
                        help="act on this repository (name, org/name, or git URL)")
    parser.add_argument("--no-color", action="store_true", help="disable colour output")
    parser.add_argument("--ascii", action="store_true",
                        help="use ASCII glyphs only (also: CODEVERDICT_ASCII=1)")
    parser.add_argument("--version", action="store_true", help="print the version and exit")

    subs = parser.add_subparsers(dest="command", metavar="<command>")

    ingest = subs.add_parser("ingest", help="index a repository into the knowledge base")
    ingest.add_argument("git_url", help="repository to index")
    ingest.add_argument("--branch", default="main", help="default branch (default: main)")
    ingest.add_argument("--wait", action="store_true", help="block until indexing finishes")

    scope = subs.add_parser("scope", help="one turn with the PM agent")
    scope.add_argument("text", nargs="+", help="what you want, in plain English")
    scope.add_argument("--new", action="store_true", help="start a fresh scope first")

    draft = subs.add_parser("draft", help="draft tickets from the locked scope")
    draft.add_argument("--json", action="store_true", help="emit the tickets as JSON")

    review = subs.add_parser("review", help="review a delivered ticket")
    review.add_argument("ticket", nargs="?", help="ticket key (default: most recent delivery)")
    review.add_argument("--json", action="store_true",
                        help="emit the review payload as JSON instead of opening the panel")

    doctor = subs.add_parser("doctor", help="preflight: what's installed, what blocks a run")
    doctor.add_argument("--json", action="store_true", help="machine-readable report")

    settings_cmd = subs.add_parser("settings", help="read or change configuration")
    settings_cmd.add_argument("action", nargs="?", default="open",
                              choices=["open", "list", "get", "set"],
                              help="default: open the settings panel")
    settings_cmd.add_argument("params", nargs="*", metavar="NAME[=VALUE]")
    settings_cmd.add_argument("--json", action="store_true", help="machine-readable output")

    for name, (_, metavar, help_text) in PASSTHROUGH.items():
        sub = subs.add_parser(name, help=help_text)
        if metavar:
            sub.add_argument("target", nargs="?", metavar=metavar)

    return parser


def _make_shell(args: argparse.Namespace):
    from .repl import Shell

    ctx = Context()
    ctx.boot()

    console_kwargs = {}
    if args.no_color:
        console_kwargs["no_color"] = True
    shell = Shell(ctx, console=theme.console(**console_kwargs))

    if args.ascii:
        shell.unicode = False
        shell.g = theme.Glyphs(False)

    if args.repo:
        from ..core import repos as core_repos

        with ctx.db() as db:
            ctx.select_repo(core_repos.resolve(db, args.repo))
    return shell


def main(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.ascii:
        os.environ["CODEVERDICT_ASCII"] = "1"

    if args.version:
        from importlib.metadata import PackageNotFoundError, version
        try:
            print(f"codeverdict {version('codeverdict')}")
        except PackageNotFoundError:
            print("codeverdict (development checkout)")
        return 0

    try:
        shell = _make_shell(args)
    except CoreError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    try:
        return _dispatch(shell, args)
    except CoreError as err:
        shell.console.print(f"[err]{shell.g.fail} {err}[/err]")
        return 1
    except KeyboardInterrupt:
        return 130


def _dispatch(shell, args: argparse.Namespace) -> int:
    from . import commands

    command = args.command

    if command is None:
        from .repl import start
        return start(shell.ctx)

    if command in PASSTHROUGH:
        slash, metavar, _ = PASSTHROUGH[command]
        target = getattr(args, "target", None) if metavar else None
        shell.handle(f"{slash} {target}".strip() if target else slash)
        return 0

    if command == "ingest":
        return _ingest(shell, args)

    if command == "scope":
        if args.new:
            shell.handle("/scope new")
        commands.talk(shell, " ".join(args.text))
        return 0

    if command == "draft":
        return _draft(shell, args)

    if command == "review":
        return _review(shell, args)

    if command == "doctor":
        return _doctor(shell, args)

    if command == "settings":
        return _settings(shell, args)

    shell.console.print(f"[err]unknown command {command}[/err]")
    return 2


# ── subcommands with no conversational equivalent ─────────────────────────────
def _doctor(shell, args: argparse.Namespace) -> int:
    """Preflight. Unlike the slash command this returns a real exit code, so it
    can gate a CI job or a Makefile target — 0 ready, 1 something blocks a run."""
    from ..core import doctor as core_doctor

    report = core_doctor.check()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        from . import render
        shell.console.print(render.doctor(report, shell.g))
    return 0 if report["ready"] else 1


def _ingest(shell, args: argparse.Namespace) -> int:
    from ..core import repos as core_repos
    from . import live, render

    with shell.ctx.db() as db:
        repo = core_repos.ingest(db, args.git_url, args.branch)
        shell.ctx.select_repo(repo)
        repo_id, label = repo.id, f"{repo.org}/{repo.name}"

    if not args.wait:
        shell.console.print(render.success(f"Indexing {label}", shell.g))
        shell.console.print(render.note("runs in the background — `codeverdict repos` to check",
                                        shell.g))
        return 0

    # Indexing is a background job that reports through the Repo row; there is
    # no event for it, so --wait polls that row rather than pretending otherwise.
    # Same live view as `/kb add`, so the two surfaces report identically.
    shell.console.print(render.note(f"Indexing {label} — Ctrl-C detaches, the build "
                                    "keeps going", shell.g))
    status, step = live.watch_kb(shell.console, shell.g, shell.ctx, repo_id)

    if status == "failed":
        shell.console.print(render.error(step or "Indexing failed — see the log above.",
                                         shell.g))
        return 1
    if status == "detached":
        shell.console.print(render.note("Detached — indexing continues.", shell.g,
                                        style="warn"))
        return 0
    shell.console.print(render.success(step or "Knowledge base ready.", shell.g))
    return 0


def _draft(shell, args: argparse.Namespace) -> int:
    from ..core import scoping
    from . import render

    with shell.ctx.db() as db:
        shell.ctx.ensure_repo(db)
        scope = shell.ctx.scope(db)
        if scope is None:
            raise CoreError(409, "No scope selected — run `codeverdict scope \"...\"` first.")
        with shell.thinking("PM is drafting engineering tickets"):
            created = scoping.draft_tickets(db, scope.id)

    if args.json:
        print(json.dumps([{
            "key": t.key, "title": t.title, "description": t.description,
            "acceptance_criteria": t.acceptance_criteria,
            "affected_files": t.affected_files, "priority": t.priority,
            "approved": t.approved,
        } for t in created], indent=2))
        return 0

    shell.console.print(render.tickets(created, shell.g, title="Drafted tickets"))
    return 0


def _review(shell, args: argparse.Namespace) -> int:
    from ..core import repos as core_repos
    from ..core import tasks as core_tasks

    if not args.json:
        shell.handle(f"/review {args.ticket}".strip())
        return 0

    with shell.ctx.db() as db:
        repo = shell.ctx.repo(db) or core_repos.listing(db)[0]
        if args.ticket:
            task = core_tasks.resolve(db, args.ticket, repo.id)
        else:
            delivered = [t for t in core_tasks.listing(db, repo_id=repo.id)
                         if t.status in ("review", "pr", "done") or t.review_summary]
            if not delivered:
                raise CoreError(409, "Nothing has been reviewed yet.")
            task = delivered[0]
        print(json.dumps(core_tasks.review(db, task.id), indent=2, default=str))
    return 0


def _settings(shell, args: argparse.Namespace) -> int:
    from ..services import runtime_settings
    from . import render

    action = args.action

    if action == "open":
        shell.handle("/settings")
        return 0

    view = runtime_settings.view()
    if action == "list":
        if args.json:
            print(json.dumps(view, indent=2, default=str))
            return 0
        for group in view["groups"]:
            shell.console.print(f"\n[heading]{group['label']}[/heading]")
            for field in group["fields"]:
                value = field["value"]
                shown = value if value not in ("", None) else "[muted]—[/muted]"
                shell.console.print(f"  [brand]{field['name']}[/brand] = {shown}")
        return 0

    index = {f["name"]: f for g in view["groups"] for f in g["fields"]}

    if action == "get":
        if not args.params:
            raise CoreError(409, "Which setting? codeverdict settings get dev_provider")
        out = {}
        for name in args.params:
            if name not in index:
                raise CoreError(404, f"No such setting '{name}'. Try `codeverdict settings list`.")
            out[name] = index[name]["value"]
        if args.json:
            print(json.dumps(out, indent=2, default=str))
        else:
            for name, value in out.items():
                shell.console.print(f"[brand]{name}[/brand] = {value if value != '' else '—'}")
        return 0

    # set
    values: dict[str, str] = {}
    for param in args.params:
        if "=" not in param:
            raise CoreError(409, f"Use NAME=VALUE (got '{param}').")
        name, _, value = param.partition("=")
        name = name.strip()
        if name not in index:
            raise CoreError(404, f"No such setting '{name}'. Try `codeverdict settings list`.")
        values[name] = value.strip()
    if not values:
        raise CoreError(409, "Nothing to set. Usage: codeverdict settings set dev_provider=anthropic")

    with shell.ctx.db() as db:
        try:
            changed = runtime_settings.update(db, values)
        except ValueError as exc:
            raise CoreError(422, str(exc))
    shell.console.print(render.success(f"Updated: {', '.join(changed)}", shell.g))
    return 0
