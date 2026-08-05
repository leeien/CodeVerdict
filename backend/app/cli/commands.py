"""Slash commands.

Every command is a small function over a ``Shell``, registered by decorator.
The registry is the single source of truth for three things that otherwise drift
apart: dispatch, tab-completion, and ``/help``.

Commands do no work of their own — they resolve arguments, call ``core``, and
hand the result to ``render``. Anything that touches the pipeline, the database
or a provider belongs below this layer.
"""

from __future__ import annotations

import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Group
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from ..core import CoreError, scoping
from ..core import costs as core_costs
from ..core import repos as core_repos
from ..core import tasks as core_tasks
from . import live, render

if TYPE_CHECKING:  # the shell imports this module, so the arrow only goes one way
    from .repl import Shell


@dataclass
class Command:
    name: str
    args: str
    help: str
    run: Callable[[Shell, str], None]
    aliases: tuple[str, ...] = ()
    # Values a completer can offer after the command name. Static list, or a
    # callable given the Shell for things only knowable at runtime (repo names,
    # ticket keys) — the completer is worth nothing if it can't offer those.
    completions: Callable[[Shell], list[str]] | list[str] = field(default_factory=list)


REGISTRY: dict[str, Command] = {}
ORDER: list[str] = []


def command(name: str, args: str = "", help: str = "", aliases: tuple[str, ...] = (),
            completions: Callable | list | None = None):
    def decorate(fn: Callable[[Shell, str], None]) -> Callable:
        cmd = Command(name=name, args=args, help=help, run=fn, aliases=aliases,
                      completions=completions or [])
        REGISTRY[name] = cmd
        ORDER.append(name)
        for alias in aliases:
            REGISTRY[alias] = cmd
        return fn
    return decorate


def resolve(name: str) -> Command | None:
    """Exact match, then unambiguous prefix — ``/appr`` should just work.

    Registry keys carry the leading slash, so a bare name is normalised up to
    one rather than the keys being stripped down.
    """
    name = name.strip()
    if not name.startswith("/"):
        name = f"/{name}"
    if name in REGISTRY:
        return REGISTRY[name]
    hits = {REGISTRY[k].name for k in REGISTRY if k.startswith(name)}
    if len(hits) == 1:
        return REGISTRY[hits.pop()]
    return None


# ── completion sources ────────────────────────────────────────────────────────
def _repo_names(shell: Shell) -> list[str]:
    with shell.ctx.db() as db:
        return [f"{r.org}/{r.name}" for r in core_repos.listing(db)]


def _ticket_keys(shell: Shell) -> list[str]:
    with shell.ctx.db() as db:
        repo = shell.ctx.repo(db)
        return [t.key for t in core_tasks.listing(db, repo_id=repo.id if repo else None)]


def _stage_names(shell: Shell) -> list[str]:
    return [key for key, _ in render.STAGES]


# ── conversation ──────────────────────────────────────────────────────────────
def talk(shell: Shell, text: str) -> None:
    """Plain text: one turn with the PM agent.

    A scope session is created on demand. Requiring the user to open one first
    would put a piece of our data model between them and the first thing they
    want to do, which is describe a change.
    """
    with shell.ctx.db() as db:
        repo = shell.ctx.ensure_repo(db)
        scope = shell.ctx.scope(db)
        if scope is None:
            scope = scoping.create(db, repo.id, kind="pm")
            shell.ctx.select_scope(scope)
            shell.note(f"New scope opened against {repo.org}/{repo.name}.")

        with shell.thinking("PM is working the knowledge base"):
            turn = scoping.scope_turn(db, scope.id, text)

    shell.print(render.agent_message("PM", "pm", turn["message"], shell.g))

    if turn["retrieved"]:
        looked_up = "; ".join(str(r) for r in turn["retrieved"][:4])
        shell.print(render.note(f"looked up: {looked_up}", shell.g))

    if turn["locked"]:
        scope_obj = turn["scope"] or {}
        criteria = scope_obj.get("acceptance_criteria") or []
        shell.print(render.success(
            f"Scope locked — {len(criteria)} acceptance criteria.", shell.g))
        if turn["cleared_drafts"]:
            shell.print(render.note(
                f"cleared {turn['cleared_drafts']} stale draft ticket(s) from the previous scope",
                shell.g, style="warn"))
        shell.print(render.note("/tickets to draft engineering tickets from this scope", shell.g))


# ── repositories & knowledge ──────────────────────────────────────────────────
@command("/repo", "[name]", "Show or switch the active repository", completions=_repo_names)
def cmd_repo(shell: Shell, args: str) -> None:
    with shell.ctx.db() as db:
        if not args.strip():
            shell.print(render.repos(core_repos.listing(db), shell.ctx.repo_id, shell.g))
            return
        repo = core_repos.resolve(db, args.strip())
        shell.ctx.select_repo(repo)
    shell.print(render.success(f"Active repository: {repo.org}/{repo.name}", shell.g))


@command("/kb", "add <url> | status | reindex | views",
         "Manage the repository knowledge base",
         completions=["add", "status", "reindex", "views"])
def cmd_kb(shell: Shell, args: str) -> None:
    parts = args.split(maxsplit=1)
    action = (parts[0] if parts else "status").lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    # `add`/`reindex` start a background job and then watch it. The watcher polls
    # the Repo row on its own short-lived sessions, so this session is closed
    # first — holding one open across a multi-minute index would pin a SQLite
    # connection for the whole build.
    if action in ("add", "reindex"):
        with shell.ctx.db() as db:
            if action == "add":
                if not rest:
                    raise CoreError(409, "Give a git URL: /kb add https://github.com/org/repo")
                repo = core_repos.ingest(db, rest)
                shell.ctx.select_repo(repo)
                verb = "Indexing"
            else:
                repo = shell.ctx.ensure_repo(db)
                core_repos.reindex(db, repo.id)
                verb = "Reindexing"
            label, repo_id = f"{repo.org}/{repo.name}", repo.id
        _watch_kb(shell, repo_id, f"{verb} {label}")
        return

    with shell.ctx.db() as db:
        if action == "views":
            repo = shell.ctx.ensure_repo(db)
            views = core_repos.knowledge(db, repo.id)
            # knowledge() returns the API envelope — {slug, total, labels, order,
            # domains} — not a bare {domain: items} mapping. Walk `order` so the
            # CLI groups them the same way the web Analysis screen does.
            domains = (views or {}).get("domains") or {}
            if not any(domains.values()):
                shell.note("No structured knowledge generated for this repo yet.")
                return
            labels = views.get("labels") or {}
            for domain in views.get("order") or sorted(domains):
                items = domains.get(domain) or []
                shell.print(render.note(
                    f"{labels.get(domain, domain)}: {len(items)} entries", shell.g))
            return

        # status
        shell.print(render.repos(core_repos.listing(db), shell.ctx.repo_id, shell.g))


def _watch_kb(shell: Shell, repo_id: int, headline: str) -> None:
    """Render a KB build live, then say plainly how it ended.

    Indexing is minutes of work (clone → code graph → symbol map → embeddings),
    and it used to print "runs in the background" and hand back the prompt —
    leaving `/kb status` as the only way to find out anything, which reported a
    bare word with no percentage and no step.
    """
    shell.print(render.note(f"{headline} — Ctrl-C detaches, the build keeps going",
                            shell.g))
    status, step = live.watch_kb(shell.console, shell.g, shell.ctx, repo_id)

    if status == "ready":
        shell.print(render.success(step or "Knowledge base ready.", shell.g))
    elif status == "failed":
        shell.print(render.error(step or "Indexing failed — see the log above.", shell.g))
    else:
        shell.print(render.note("Detached — indexing continues. /kb to check on it.",
                                shell.g, style="warn"))


# ── scopes & tickets ──────────────────────────────────────────────────────────
@command("/scope", "[new | list | <id>]", "Switch, list, or start a scope session",
         completions=["new", "list"])
def cmd_scope(shell: Shell, args: str) -> None:
    arg = args.strip().lower()
    with shell.ctx.db() as db:
        repo = shell.ctx.ensure_repo(db)

        if arg == "new":
            scope = scoping.create(db, repo.id, kind="pm")
            shell.ctx.select_scope(scope)
            shell.print(render.success("New scope opened — describe the change you want.", shell.g))
            return

        if not arg or arg == "list":
            sessions = scoping.listing(db, repo_id=repo.id, kind="pm")
            if not sessions:
                shell.note("No scopes yet — just describe a change to start one.")
                return
            for s in sessions[:15]:
                marker = shell.g.step if s.id == shell.ctx.session_id else " "
                status = s.status or "open"
                shell.print(render.note(
                    f"{marker} #{s.id}  {(s.title or 'untitled')[:50]}  [{status}]", shell.g))
            return

        if not arg.lstrip("#").isdigit():
            raise CoreError(409, "Use /scope new, /scope list, or /scope <id>.")
        scope = scoping.require(db, int(arg.lstrip("#")))
        shell.ctx.select_scope(scope)
        shell.print(render.success(f"Scope #{scope.id}: {scope.title or 'untitled'}", shell.g))


@command("/tickets", "[draft]", "List this scope's tickets, or draft them from the locked scope",
         aliases=("/t",), completions=["draft"])
def cmd_tickets(shell: Shell, args: str) -> None:
    with shell.ctx.db() as db:
        repo = shell.ctx.ensure_repo(db)
        scope = shell.ctx.scope(db)
        if scope is None:
            shell.print(render.tickets(core_tasks.listing(db, repo_id=repo.id), shell.g,
                                       title=f"All tickets — {repo.org}/{repo.name}"))
            return

        existing = scoping.tasks_for(db, scope.id)
        wants_draft = args.strip().lower() == "draft" or not existing
        if wants_draft:
            with shell.thinking("PM is drafting engineering tickets"):
                created = scoping.draft_tickets(db, scope.id)
            shell.print(render.success(f"Drafted {len(created)} ticket(s).", shell.g))
            existing = scoping.tasks_for(db, scope.id)

        shell.print(render.tickets(existing, shell.g,
                                   title=f"Scope #{scope.id} — {scope.title or 'untitled'}"))


@command("/show", "<KEY>", "Show one ticket in full", completions=_ticket_keys)
def cmd_show(shell: Shell, args: str) -> None:
    if not args.strip():
        raise CoreError(409, "Which ticket? /show TASK-101")
    with shell.ctx.db() as db:
        repo = shell.ctx.repo(db)
        task = core_tasks.resolve(db, args.strip(), repo.id if repo else None)
        shell.print(render.ticket_detail(task, shell.g))


@command("/approve", "<KEY|all>", "Approve tickets — the gate before any agent writes code",
         completions=_ticket_keys)
def cmd_approve(shell: Shell, args: str) -> None:
    target = args.strip()
    if not target:
        raise CoreError(409, "Which ticket? /approve TASK-101, or /approve all")

    with shell.ctx.db() as db:
        repo = shell.ctx.ensure_repo(db)
        scope = shell.ctx.scope(db)
        if target.lower() == "all":
            pool = scoping.tasks_for(db, scope.id) if scope else \
                core_tasks.listing(db, repo_id=repo.id)
            pending = [t for t in pool if not t.approved]
            if not pending:
                shell.note("Every ticket here is already approved.")
                return
            for t in pending:
                core_tasks.approve(db, t.id)
            shell.print(render.success(
                f"Approved {len(pending)} ticket(s): {', '.join(t.key for t in pending)}", shell.g))
        else:
            task = core_tasks.resolve(db, target, repo.id)
            core_tasks.approve(db, task.id)
            shell.print(render.success(f"Approved {task.key}.", shell.g))
            if task.jira_key:
                shell.print(render.note(f"Jira: {task.jira_url or task.jira_key}", shell.g))
        shell.print(render.note("/run to deliver the scope", shell.g))


# ── the pipeline ──────────────────────────────────────────────────────────────
@command("/run", "", "Run the scope: Planner → Dev → QA → jury review → PR")
def cmd_run(shell: Shell, args: str) -> None:
    from ..services import orchestrator

    with shell.ctx.db() as db:
        shell.ctx.ensure_repo(db)      # refuses early, with a useful message
        scope = shell.ctx.scope(db)
        if scope is None:
            raise CoreError(409, "No scope selected — describe a change first, or /scope list.")
        approved = scoping.approved_subtasks(db, scope.id)
        if not approved:
            raise CoreError(409, "No approved tickets to run — /approve all first.")
        session_id, keys = scope.id, [t.key for t in approved]
        demo = shell.demo_mode()

    shell.print(render.note(
        f"Delivering {len(keys)} ticket(s) on one branch: {', '.join(keys)}", shell.g))
    if demo:
        shell.print(render.note("demo mode is on — the PR stage is a dry run", shell.g, style="warn"))

    worker = threading.Thread(target=orchestrator.run_scope, args=(session_id,),
                              name="codeverdict-pipeline", daemon=True)
    view, completed = live.watch(shell.console, shell.g, shell.unicode, worker)

    if not completed:
        shell.print(render.note(
            "Detached — the run continues in the background. /board to check on it.",
            shell.g, style="warn"))
        return

    failed = [s for s in view.stages.values() if s.state == "failed"]
    if failed:
        shell.print(render.error(
            f"{', '.join(live.STAGE_LABEL.get(s.key, s.key) for s in failed)} failed — "
            "see the log above.", shell.g))
    else:
        cost = f" for ${view.cost:.4f}" if view.cost else ""
        shell.print(render.success(f"Scope delivered{cost}.", shell.g))
    shell.print(render.note(f"/review {keys[0]} to open the jury panel", shell.g))


@command("/board", "", "The delivery board for this repository", aliases=("/b",))
def cmd_board(shell: Shell, args: str) -> None:
    with shell.ctx.db() as db:
        repo = shell.ctx.ensure_repo(db)
        shell.print(render.board(core_tasks.board(db, repo.id), shell.g))


@command("/plan", "", "The Planner's verified implementation plan for this scope")
def cmd_plan(shell: Shell, args: str) -> None:
    with shell.ctx.db() as db:
        scope = shell.ctx.scope(db)
        if scope is None:
            raise CoreError(409, "No scope selected.")
        shell.print(render.plan(scope.plan or {}, shell.g))


# ── review & delivery ─────────────────────────────────────────────────────────
@command("/review", "[KEY]", "Open the jury review panel for a delivered ticket",
         aliases=("/r",), completions=_ticket_keys)
def cmd_review(shell: Shell, args: str) -> None:
    from .panels import review as review_panel

    with shell.ctx.db() as db:
        repo = shell.ctx.ensure_repo(db)
        if args.strip():
            task = core_tasks.resolve(db, args.strip(), repo.id)
        else:
            # No argument: the most recently delivered ticket is almost always
            # the one they mean.
            delivered = [t for t in core_tasks.listing(db, repo_id=repo.id)
                         if t.status in ("review", "pr", "done") or t.review_summary]
            if not delivered:
                raise CoreError(409, "Nothing has been reviewed yet — /run a scope first.")
            task = delivered[0]
        payload = core_tasks.review(db, task.id)
        task_id = task.id

    outcome = review_panel.open_panel(payload, unicode=shell.unicode)
    _apply_review_outcome(shell, task_id, outcome)


def _apply_review_outcome(shell: Shell, task_id: int, outcome: dict | None) -> None:
    """The review panel decides; the mutation happens here, in one place."""
    if not outcome or not outcome.get("action"):
        return
    action = outcome["action"]
    with shell.ctx.db() as db:
        if action == "pr":
            result = core_tasks.create_pr(db, task_id, shell.ctx.user(db))
            if result.get("created"):
                shell.print(render.success(f"Pull request opened: {result['url']}", shell.g))
                if result.get("forked"):
                    shell.print(render.note(f"via your fork {result['fork_repo']}", shell.g))
            else:
                shell.print(render.note(f"Already open: {result['url']}", shell.g))
        elif action == "changes":
            note = outcome.get("note", "")
            if outcome.get("ask_note"):
                # Asked at the prompt rather than in a modal: this note goes
                # onto the ticket and back into the Dev agent's next round, so
                # it deserves real line editing and history.
                note = shell.ask("What should change? (blank to skip)")
            result = core_tasks.request_changes(db, task_id, note)
            shell.print(render.success(
                f"Sent back for changes: {', '.join(result['tasks'])}. /run to re-deliver.",
                shell.g))
        elif action == "merge":
            result = core_tasks.merge(db, task_id)
            shell.print(render.success(f"Marked done: {', '.join(result['tasks'])}", shell.g))
        elif action == "open":
            url = outcome.get("url")
            if url:
                webbrowser.open(url)


@command("/pr", "<KEY>", "Push the delivery branch and open a real pull request",
         completions=_ticket_keys)
def cmd_pr(shell: Shell, args: str) -> None:
    if not args.strip():
        raise CoreError(409, "Which ticket? /pr TASK-101")
    with shell.ctx.db() as db:
        repo = shell.ctx.ensure_repo(db)
        task = core_tasks.resolve(db, args.strip(), repo.id)
        result = core_tasks.create_pr(db, task.id, shell.ctx.user(db))
    if result.get("created"):
        shell.print(render.success(f"Pull request opened: {result['url']}", shell.g))
    else:
        shell.print(render.note(f"Already open: {result['url']}", shell.g))


@command("/changes", "<KEY> [note]", "Send the scope back for changes", completions=_ticket_keys)
def cmd_changes(shell: Shell, args: str) -> None:
    parts = args.strip().split(maxsplit=1)
    if not parts:
        raise CoreError(409, "Which ticket? /changes TASK-101 the retry should be bounded")
    with shell.ctx.db() as db:
        repo = shell.ctx.ensure_repo(db)
        task = core_tasks.resolve(db, parts[0], repo.id)
        result = core_tasks.request_changes(db, task.id, parts[1] if len(parts) > 1 else "")
    shell.print(render.success(
        f"Sent back for changes: {', '.join(result['tasks'])}. /run to re-deliver.", shell.g))


# ── configuration ─────────────────────────────────────────────────────────────
@command("/settings", "", "Open Settings — providers, per-stage models, limits, delivery, Jira",
         aliases=("/config",))
def cmd_settings(shell: Shell, args: str) -> None:
    from .panels import settings as settings_panel

    # Probing the coding CLIs is seconds of subprocess work; do it out here
    # under a spinner so the panel itself opens instantly (see warm_up).
    with shell.thinking("checking which coding CLIs are installed"):
        settings_panel.warm_up()
    changed = settings_panel.open_panel(shell.ctx, unicode=shell.unicode)
    if changed:
        shell.print(render.success(f"Saved {changed} setting(s).", shell.g))


@command("/doctor", "", "Preflight — what's installed, what's degraded, what blocks a run",
         aliases=("/check",))
def cmd_doctor(shell: Shell, args: str) -> None:
    from ..core import doctor as core_doctor

    # Probing every coding CLI forks a subprocess each; keep the user informed
    # rather than looking hung for a few seconds.
    with shell.thinking("probing tools, providers and coding CLIs"):
        report = core_doctor.check()
    shell.print(render.doctor(report, shell.g))


@command("/models", "", "Which provider and model owns each pipeline stage", aliases=("/m",))
def cmd_models(shell: Shell, args: str) -> None:
    from ..config import settings as cfg
    from ..services import agent_backends, providers

    values = {f"{k}_{suffix}": getattr(cfg, f"{k}_{suffix}", "")
              for k, _ in render.STAGES for suffix in ("provider", "model")}
    avail = agent_backends.availability()
    backends = {}
    for pid, p in providers.PROVIDERS.items():
        if p.backend:
            backends[pid] = avail.get(p.backend, {})
    shell.print(render.stage_models(values, backends, shell.g))


@command("/model", "<stage> <provider> [model]", "Point one stage at a provider and model",
         completions=_stage_names)
def cmd_model(shell: Shell, args: str) -> None:
    from ..services import runtime_settings

    parts = args.split()
    if len(parts) < 2:
        raise CoreError(409, "Usage: /model dev anthropic claude-sonnet-5  "
                             "(stages: " + ", ".join(k for k, _ in render.STAGES) + ")")
    stage, provider = parts[0].lower(), parts[1]
    valid = [k for k, _ in render.STAGES]
    if stage not in valid:
        raise CoreError(409, f"Unknown stage '{stage}'. One of: {', '.join(valid)}")

    values = {f"{stage}_provider": provider}
    if len(parts) > 2:
        # Everything after the provider is the model id, because some are
        # slash-separated ('openai/gpt-oss-120b'). That makes a trailing typo
        # invisible: `/model planner claude-cli sonnet /models` silently stored
        # the model as "sonnet /models", and the only symptom was the stage
        # failing mid-run with the provider's own complaint about it.
        model = " ".join(parts[2:])
        if any(p.startswith("/") for p in parts[3:]):
            raise CoreError(409,
                            f"'{model}' doesn't look like a model id — it contains a "
                            f"command. Did you mean two lines?\n"
                            f"  /model {stage} {provider} {parts[2]}\n"
                            f"  {parts[3]}")
        values[f"{stage}_model"] = model
    with shell.ctx.db() as db:
        runtime_settings.update(db, values)
    shell.print(render.success(
        f"{stage} → {provider}" + (f" / {values.get(f'{stage}_model')}" if len(parts) > 2 else ""),
        shell.g))


@command("/jury", "", "The review panel's roster — seat judges and give each its own model")
def cmd_jury(shell: Shell, args: str) -> None:
    from .panels import jury as jury_panel

    changed = jury_panel.open_panel(shell.ctx, unicode=shell.unicode)
    if changed:
        shell.print(render.success("Jury roster updated.", shell.g))


@command("/costs", "", "Token and dollar breakdown per scope, ticket and agent")
def cmd_costs(shell: Shell, args: str) -> None:
    with shell.ctx.db() as db:
        repo = shell.ctx.repo(db)
        data = core_costs.breakdown(db, repo.id if repo else None)

    totals = data["totals"]
    if not totals["tickets"]:
        shell.note("Nothing has run yet, so there is nothing to account for.")
        return

    table = Table(box=None, pad_edge=False, show_header=True, header_style="muted",
                  padding=(0, 2, 0, 0))
    table.add_column("SCOPE", overflow="ellipsis")
    table.add_column("TICKETS", justify="right", no_wrap=True)
    table.add_column("TOKENS", justify="right", no_wrap=True)
    table.add_column("COST", justify="right", no_wrap=True)
    for scope in data["scopes"][:12]:
        table.add_row(scope["title"][:44], str(len(scope["tickets"])),
                      f"{scope['tokens']:,}", f"${scope['cost']:.4f}")

    footer = Text(f"{totals['tokens']:,} tokens   {shell.g.bullet}   "
                  f"${totals['cost']:.4f} total", style="brand")
    if totals.get("cost_unknown"):
        footer += Text("   (some backends report tokens but not cost)", style="muted")
    shell.print(Padding(Group(Text("Costs", style="heading"), Text(""), table, Text(""), footer),
                        (1, 0, 0, 2)))


# ── shell meta ────────────────────────────────────────────────────────────────
@command("/clear", "", "Clear the screen", aliases=("/cls",))
def cmd_clear(shell: Shell, args: str) -> None:
    shell.console.clear()
    shell.banner()


@command("/help", "", "This list", aliases=("/?", "/h"))
def cmd_help(shell: Shell, args: str) -> None:
    rows = [(REGISTRY[name].name, REGISTRY[name].args, REGISTRY[name].help) for name in ORDER]
    shell.print(render.help_panel(rows, shell.g))


@command("/quit", "", "Leave (a running pipeline is unaffected)", aliases=("/exit", "/q"))
def cmd_quit(shell: Shell, args: str) -> None:
    raise SystemExit(0)
