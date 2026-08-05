"""Real agent pipeline: Dev → QA → Review → PR (gh).

Runs on a cloned working copy branch. Dev edits (via whichever agent backend the
stage is pointed at — Claude Code, Codex, Cursor, Aider, Gemini CLI, or an HTTP
coding loop); QA runs tests and an unbiased chat review; Review is a separate
agent pass over the diff (deliberately selectable on a different backend/model
family than Dev); PR pushes the branch and opens a real PR via gh. The task is
left in the `pr` column for a human to merge.
"""

import logging
import re
import time
from pathlib import Path

from sqlmodel import Session, select

from ..config import settings
from ..database import engine
from ..models import (
    ChatMessage,
    MessageRole,
    Repo,
    ScopeSession,
    Task,
    TaskStatus,
    utcnow,
)
from . import (
    agent_backends,
    agent_runner,
    events,
    git_ops,
    jury,
    lang,
    llm,
    openai_agent,
    planner,
    precision,
    prompts,
    providers,
    rag,
    search,
)
from .knowledge import freshness, graph, symbol_map, write_back
from .knowledge import tools as kb_tools

logger = logging.getLogger(__name__)

# Model aliases the Claude CLI understands (vs an OpenAI-compatible model id).
_CLI_ALIASES = {"sonnet", "opus", "haiku", "default"}


def _review_label(provider: str) -> str:
    if providers.agent_backend(provider) == "claude-code":
        m = settings.review_model if settings.review_model in _CLI_ALIASES else settings.claude_model
        return f"claude-cli {m}"
    return providers.label(provider, settings.review_model)


def _agent_label(provider: str, model: str | None) -> str:
    """Run-row label for an agent-backend provider ('auto'/'anthropic' show as
    the claude-cli they resolve to)."""
    if providers.agent_backend(provider) == "claude-code":
        return f"claude-cli {model or settings.claude_model}"
    return providers.label(provider, model or "")


def _agent_model(provider: str, backend: str, configured: str) -> str | None:
    """Model to hand an agent backend: for the Claude CLI only its aliases are
    valid; other backends take the configured model id (or their default)."""
    if backend == "claude-code":
        return configured if configured in _CLI_ALIASES else None
    p = providers.PROVIDERS.get(provider)
    return (configured or "").strip() or (p.default_model if p else "") or None


def _qa_label() -> str:
    return providers.label(settings.qa_provider, settings.qa_model)


def _prs_enabled() -> bool:
    """Real pushes/PRs only when explicitly enabled AND not in demo mode."""
    return settings.open_real_pr and not settings.demo_mode


def _kb_context(repo_url: str, info: dict, on_event=None) -> str:
    """Feed the Dev agent the RIGHT knowledge for this task: a use-case-scoped,
    token-budgeted slice from Deep Analysis (with exact source files), so Claude
    goes straight to the right files instead of grepping the whole repo. Falls
    back to a focused knowledge-base ask when the analysis isn't available."""
    if not repo_url:
        return ""
    query = f"{info.get('title', '')}: {info.get('description', '')}"
    # Precision retrieval first — right knowledge, not more. The plan's verified
    # symbols steer the reranker: a location the Planner already confirmed is the
    # target, not another candidate to weigh.
    try:
        ctx = precision.retrieve(query, use_case="task-breakdown", repo_url=repo_url,
                                 plan_symbols=info.get("target_symbols"))
        if ctx:
            if on_event:
                on_event("info", f"Precision KB: task-scoped slice ({len(ctx)} chars)")
            return ctx
    except Exception:  # noqa: BLE001
        pass
    # Fallback: focused knowledge-base ask.
    try:
        ctx = rag.ask(repo_url, [{"role": "user", "content": (
            f"For implementing '{info['title']}: {info.get('description', '')}', list the specific "
            "repository files and functions that must change and how they currently work. "
            "Be concise and cite exact file paths.")}], timeout=150)
        if on_event and ctx:
            on_event("info", f"KB context ({len(ctx)} chars, knowledge-base fallback)")
        return ctx
    except Exception:  # noqa: BLE001
        return ""


def _pick_dev_model(info: dict) -> str:
    """Cheaper Haiku for pure test-writing subtasks, Sonnet for real code. A
    whole-scope work order always gets the full model (it mixes code + tests)."""
    if info.get("is_scope"):
        return settings.claude_model
    text = f"{info.get('title', '')} {info.get('description', '')}".lower()
    is_test = any(k in text for k in ("test", "coverage", "pytest", "unit test"))
    return settings.claude_test_model if is_test else settings.claude_model


def _cli_dev_model(info: dict) -> str:
    """CLI model for the Dev stage: the operator's ``dev_model`` when it names a CLI
    alias (sonnet/opus/haiku), else the size-picked default (which also honors the
    cheaper test model for test-writing subtasks)."""
    m = (settings.dev_model or "").strip()
    return m if m in _CLI_ALIASES else _pick_dev_model(info)


def _auto_http_target() -> tuple[str, str]:
    """(provider, model) for the 'auto' Dev fallback when the Claude CLI can't run.
    Routes dev_model by name so a legacy gemini-* dev model still reaches Gemini
    (matching the pre-registry auto-fallback behavior)."""
    m = settings.dev_model
    return ("gemini" if (m or "").startswith("gemini") else "openai"), m


_SYM_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Injected file-content budget for the Dev prompt: combined char cap across all
# pinned files, per-file cap, and the overall verified-locations block cap.
# The budget is spent on what the PLAN names, not on everything anyone guessed —
# injection is what beat a cold `claude -p` on the rich benchmark, but the old
# version spent it on the PM's unverified hypothesis.
_FILE_CONTENT_BUDGET = 9000
_FILE_CONTENT_MAX_PER_FILE = 4000
_VERIFIED_BLOCK_MAX = 16000


def _verified_locations(path: str, files: list[str] | None, symbols: list[str] | None) -> str:
    """Harness-computed localization brief for the Dev prompt: exact file:line
    pins for every known symbol plus a symbol outline of every affected file.
    Live data shows the Dev agent otherwise spends ~70% of its tool calls (and
    most of the run's input tokens) on find/ls/whole-file reads rediscovering
    exactly this.

    Three free sources, cross-checked: the code graph (AST-verified, kept
    current by knowledge/freshness.py at run entry) names the DEFINITION site,
    powers the outlines and adds CALLER pins (the blast radius a flat map
    can't see); the symbol map is the no-binary fallback; live ripgrep
    verifies against this exact working copy and adds usage pins. Symbols
    found nowhere get 'did you mean' suggestions instead of sending Dev
    hunting."""
    # The graph is keyed by repo slug == workdir basename (git_ops.slug is
    # idempotent on slugs, so the wrapper resolves it to the same project).
    slug = Path(path).name
    use_graph = graph.available()
    smap = symbol_map.load_slug(slug)
    lines: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    test_files: dict[str, None] = {}  # ordered set of existing tests to surface
    noise = (".md", ".rst", ".txt", ".yml", ".yaml", ".lock", ".cfg", ".toml")
    for sym in (symbols or [])[:12]:
        s = str(sym)
        if "(new" in s:  # tagged by ground_tickets: not found anywhere in the repo
            unknown.append(s.split("(")[0].strip())
            continue
        pinned_file = s.split("::")[0].strip() if "::" in s else ""
        name = s.split("::")[-1].strip()
        idents = _SYM_NAME_RE.findall(name)
        if not idents:
            continue
        ident = max(idents, key=len)
        if ident in seen:
            continue
        seen.add(ident)
        entry: list[str] = []
        # Definition site(s) — graph first (AST-verified, cross-language),
        # symbol map as the no-binary fallback. The highest-value pin.
        graph_hits = graph.lookup(slug, ident, limit=2) if use_graph else []
        map_hits = [] if graph_hits else (smap.lookup(ident) if smap else [])
        if graph_hits:
            defs = "; ".join(
                f"{g['file_path']}:{g['start_line'] or '?'} ({g['label'].lower()})"
                for g in graph_hits)
            entry.append(f"    defined at {defs}")
            if not pinned_file:
                pinned_file = graph_hits[0]["file_path"]
            # Callers from the graph: where behavior changes propagate — the
            # pins that keep Dev from breaking call sites it never opened.
            calls = graph.callers(slug, ident, limit=3)
            if calls:
                entry.append("    called from " + "; ".join(
                    f"{c['file_path']}:{c['start_line'] or '?'} ({c['name']})"
                    for c in calls))
        elif map_hits:
            defs = "; ".join(
                f"{f}:{m['l']} ({m['k']}" + (f" of {m['p']}" if m.get("p") else "") + ")"
                for f, m in map_hits[:2])
            entry.append(f"    defined at {defs}")
            if not pinned_file:  # grep the definition file for precise usage rows
                pinned_file = map_hits[0][0]
        # Search the symbol's own pinned file first (precise), repo-wide only as
        # fallback — and drop docs/config hits, which just distract the agent.
        pat = rf"\b{re.escape(ident)}\b"
        hits = search.lines(path, pat, max_lines=3, pathspec=pinned_file) if pinned_file else ""
        if not hits or hits == "(no matches)":
            hits = search.lines(path, pat, max_lines=10)
        rows = [h for h in hits.splitlines()
                if ":" in h and not h.split(":", 1)[0].endswith(noise)][:3]
        entry.extend(f"    {h[:140]}" for h in rows)
        if entry:
            lines.append(f"  {ident}:\n" + "\n".join(entry))
            # Localize the EXISTING tests exercising this symbol too — Dev should
            # extend the real suite (and run it), not write a parallel one.
            for tf in search.files(path, pat, pathspec="*test*", max_files=3):
                if not tf.endswith(noise):
                    test_files.setdefault(tf)
        elif smap or use_graph:  # nowhere in graph/map or grep — likely a PM invention
            unknown.append(ident)
    # Full contents of the pinned files, budgeted — this is what actually saves
    # Dev a Read tool call. Live runs showed Dev re-reading whole files it was
    # already pointed at (~70% of its input tokens on rediscovery); an outline
    # alone still leaves Dev needing the real text before it can edit.
    content_budget = _FILE_CONTENT_BUDGET if settings.dev_inject_file_contents else 0
    contents: list[str] = []
    for f in (files or [])[:8]:
        text = git_ops.read_file(path, f, max_chars=_FILE_CONTENT_MAX_PER_FILE + 1)
        if text is None:
            lines.append(f"  file {f}: DOES NOT EXIST (create it if needed)")
            continue
        outline = ""
        if use_graph:
            outline = ", ".join(
                f"{o['name']}:{o['start_line'] or '?'}"
                for o in graph.outline(slug, f, limit=25) if o.get("name"))
        if not outline and smap:
            outline = smap.outline(f, max_symbols=25)
        lines.append(f"  file {f}: exists" + (f" — contains: {outline}" if outline else ""))
        if content_budget > 500 and len(contents) < 3:
            snippet = text[:min(_FILE_CONTENT_MAX_PER_FILE, content_budget)]
            note = "\n... (truncated — read the rest yourself if you need it)" if len(text) > len(snippet) else ""
            contents.append(f"--- {f} ---\n{snippet}{note}")
            content_budget -= len(snippet)
    out = ""
    if lines:
        out += ("Verified code locations (code graph + ripgrep against this working copy "
                "just now — these pins are real, go straight to them):\n" + "\n".join(lines))
    if test_files:
        out += ("\n  existing tests exercising these symbols (extend/update THESE and run "
                "them — don't create a parallel suite): " + ", ".join(list(test_files)[:6]))
    if unknown:
        notes = []
        for u in dict.fromkeys(unknown):
            close = smap.suggest(u) if smap else []
            notes.append(u + (f" (similar existing: {', '.join(close)})" if close else ""))
        out += ("\nNot found anywhere in the repo (either new things to create, or PM "
                "inventions to ignore — use judgment): " + ", ".join(notes))
    if contents:
        out += ("\n\nFull current contents of the pinned files (already read for you — do NOT "
                "spend a Read tool call on these; only read a file yourself if it's not shown "
                "here, or a shown file was truncated and you need the rest):\n\n"
                + "\n\n".join(contents))
    return out[:_VERIFIED_BLOCK_MAX]


def _report_dev_efficiency(res: dict, verified: str, on_event) -> None:
    """Log WHY the Dev stage cost what it cost.

    Dev is ~84% of a delivery's tokens, and that spend is not one big prompt —
    it is turns. Each turn re-sends the conversation, so a re-read of a file we
    already injected is paid twice: once to hand it over and again to fetch it.
    The stream reports every tool call, so the re-read count is a fact we can
    log rather than a theory to argue about, and it is the number to watch when
    tuning the injection budget.
    """
    turns = res.get("turns") or 0
    tools = res.get("tools") or {}
    reads = res.get("read_paths") or []
    if not turns and not tools:
        return  # a backend that doesn't report this (fine — no fake numbers)
    # Which reads were of files whose contents the prompt already carried?
    injected = {line[4:-4].strip() for line in verified.splitlines()
                if line.startswith("--- ") and line.endswith(" ---")}
    redundant = [p for p in reads
                 if injected and any(p.endswith(f) or f.endswith(p) for f in injected)]
    summary = ", ".join(f"{n}×{c}" for n, c in sorted(tools.items(), key=lambda kv: -kv[1])[:6])
    on_event("info", f"Dev efficiency: {turns} turn(s) · {summary or 'no tool calls'}")
    if injected:
        on_event(
            "warn" if redundant else "info",
            f"Dev re-read {len(redundant)} of {len(injected)} injected file(s)"
            + (f" — {', '.join(sorted(set(redundant))[:4])}" if redundant else
               " — injection held; no wasted reads"))


def _test_cmd(path: str) -> str:
    """Test command the Dev agent should verify with — '' when tests can't
    actually run here (advertising a broken command just misleads the agent).
    Ecosystem-aware: pytest in the per-repo venv, npm test, go test, cargo test,
    mvn/gradlew test, rspec (see lang.detect_runner); building the env is a
    side effect."""
    try:
        return git_ops.test_command(path)
    except Exception:  # noqa: BLE001
        return ""


def _is_transient(err: str) -> bool:
    """Errors worth retrying on the SAME (strong) model: overload, rate limit,
    timeout, flaky network. Distinct from 'Claude genuinely can't run here'."""
    low = (err or "").lower()
    return any(k in low for k in ("429", "rate limit", "overload", "timed out",
                                  "timeout", "529", "500", "connection", "network"))


def _backend_unavailable(backend: str, err: str) -> bool:
    """True only when the agent backend can't run AT ALL on this machine (missing
    binary, no headless mode, or unauthenticated) — the only case where degrading
    to the HTTP coding loop beats failing visibly."""
    if not agent_backends.is_available(backend):
        return True
    low = (err or "").lower()
    return any(k in low for k in ("not runnable", "login", "authent", "api key",
                                  "credit balance", "billing", "unknown agent backend",
                                  "no scriptable", "headless"))


def _install_dev_tools(path: str, on_event) -> str:
    """Give a CLI-backed Dev agent real, callable retrieval tools: drop the
    `.codeverdict/kb` shim into the working copy (git-ignored) so it can query the
    code graph + semantic index through the shell instead of grepping around.

    Backend-agnostic on purpose — every headless coding CLI has a shell, so the
    tools don't depend on one vendor's MCP support. Returns the command to
    advertise, or "" (fail open: the agent works as before)."""
    cmd = kb_tools.install(path, Path(path).name)
    if cmd:
        on_event("info", f"Dev index tools installed: {cmd} "
                         f"({', '.join(kb_tools.TOOL_NAMES)})")
    return cmd


def _refresh_and_plan(session_id: int, rep: int, repo_url: str, path: str, scope: dict,
                      subs: list[dict]) -> dict:
    """Prepare the run, then decide HOW this scope gets built.

    Both halves happen here, before the Dev run row opens, and both for the same
    reason: the tree has just been reset to origin's default branch, so this is
    the one moment when the working copy is clean, the code graph can be synced
    to exactly what Dev will edit, and nothing has been changed yet. Keeping them
    out of the Dev run row also keeps Dev's duration and cost measuring Dev
    rather than the preparation done on its behalf.

    Returns the verified plan ({} when planning is off or could not run — the
    pipeline then works from the scope alone, which is exactly the no-planner
    ablation).
    """
    if not settings.planner_enabled:
        freshness.refresh_if_stale(repo_url)
        return {}
    rid, t0 = agent_runner.start_run(rep, "plan")
    agent_runner.set_model(rid, providers.label(
        settings.planner_provider, settings.planner_model))
    freshness.refresh_if_stale(repo_url, on_event=agent_runner.logger_for(rid))
    agent_runner.log(rid, "info", "Planner: reading the repository to decide the approach")
    res = planner.plan(repo_url, path, scope, subs, on_event=agent_runner.logger_for(rid))
    plan_obj = res.get("plan") or {}
    if plan_obj:
        agent_runner.log(rid, "info", planner.as_prompt(plan_obj))
    agent_runner.finish_run(rid, rep, t0, tokens_in=res.get("tokens_in", 0),
                            tokens_out=res.get("tokens_out", 0), cost=res.get("cost", 0.0),
                            error=res.get("error") if not plan_obj else None)
    with Session(engine) as db:
        session = db.get(ScopeSession, session_id)
        if session is not None:
            session.plan = plan_obj
            db.add(session)
            db.commit()
    return plan_obj


def _dev_agent(path: str, key: str, info: dict, on_event, context: str = "",
               test_cmd: str = "") -> tuple[dict, str]:
    """Run the Dev coding agent. When dev_provider maps to an agent backend
    (Claude CLI, Codex, Cursor, Aider, Gemini CLI, or 'auto'), run that headless
    CLI; a backend that can't run at all on this machine falls back to the HTTP
    coding loop instead of failing the scope. A concrete OpenAI-compatible
    provider runs the SEARCH/REPLACE loop on that provider.
    Returns (result, model_label)."""
    provider = settings.dev_provider
    backend = providers.agent_backend(provider)
    if backend:
        model = _cli_dev_model(info) if backend == "claude-code" \
            else _agent_model(provider, backend, settings.dev_model)
        label = _agent_label(provider, model)
        on_event("info", f"Dev model: {label}")
        tools_cmd = _install_dev_tools(path, on_event)
        verified = _verified_locations(path, info.get("affected_files"),
                                       info.get("target_symbols"))
        prompt = prompts.dev(key, info["title"], info["description"], info["criteria"], context,
                             affected_files=info.get("affected_files"),
                             target_symbols=info.get("target_symbols"),
                             test_cmd=test_cmd, tools_cmd=tools_cmd,
                             plan=planner.as_prompt(info.get("plan") or {}),
                             verified=verified)
        res = agent_backends.run(backend, path, prompt, on_event, model=model)
        _report_dev_efficiency(res, verified, on_event)
        if res["error"] and _is_transient(res["error"]):
            # Overload/rate-limit/timeout: retry the same backend once. NEVER hand
            # transient failures to the free-tier coder — it produces broken edits
            # (proven in the run logs), which is worse than failing loudly.
            on_event("warn", f"{backend} transient error ({res['error'][:80]}) — retrying once")
            time.sleep(15)
            failed = res
            res = llm.carry_usage(
                agent_backends.run(backend, path, prompt, on_event, model=model), failed)
        # Fail open: degrade to the HTTP coder only when the backend can't run at
        # all (not installed / no headless mode / unauthenticated).
        if not res["error"] or not _backend_unavailable(backend, res["error"]):
            return res, label
        on_event("warn", f"{backend} unavailable ({(res['error'] or '')[:50]}…) — "
                         "falling back to the HTTP coding agent")
        fb_provider, fb_model = _auto_http_target()
    else:
        fb_provider, fb_model = provider, settings.dev_model
    cr = openai_agent.code(path, key, info["title"], info["description"], info["criteria"],
                           on_event, provider=fb_provider, model=fb_model, context=context,
                           affected_files=info.get("affected_files"),
                           target_symbols=info.get("target_symbols"))
    return ({"text": cr["summary"], "tokens_in": cr["tokens_in"], "tokens_out": cr["tokens_out"],
             "cost": cr["cost"], "error": cr["error"]}, providers.label(fb_provider, fb_model))


def _scoped_test_paths(path: str, branch: str) -> list[str] | None:
    """Files to narrow the suite to, or None to run all of it.

    Only for Go, and only because `go test ./...` on a monorepo does not
    finish: on gitea it compiles 3,024 files, blew the 600s timeout, and QA got
    NO signal at all. `go test` takes package directories, so the branch's
    changed files map onto exactly the packages the change can break — which is
    what gitea's own Makefile does via GO_TEST_PACKAGES.

    Deliberately not extended to other ecosystems. `test_paths` means *test
    files* to pytest, so handing it changed source files would run pytest on
    non-test modules; and pytest on a normal repo completes anyway, so there is
    nothing to buy. The trade is real and one-directional: a scoped run can
    miss a regression in a package the diff did not touch. Partial signal beats
    the timeout's none, and the full suite still runs in CI.
    """
    try:
        runner = lang.detect_runner(Path(path))
        if runner is None or runner.kind != "go":
            return None
        changed = [f for f in git_ops.diff_names(path, ref=branch) if f.endswith(".go")]
        return changed or None
    except Exception:  # noqa: BLE001 — scoping is an optimisation, never a blocker
        return None


def _test_outcome(passed: bool | None, output: str) -> str:
    """How the suite ended, in the log line a human actually reads.

    `passed is None` covers three very different situations that all used to
    print "no suite/deps": the repo genuinely has no tests, the runner could not
    start, and the suite ran but exceeded its timeout. On gitea the third is the
    common one — `go test ./...` compiles 3,024 files — and calling that "no
    suite" invites the exact wrong conclusion, that QA had nothing to check.
    """
    if passed is True:
        return "passed"
    if passed is False:
        return "failures"
    low = (output or "").lower()
    if "timeoutexpired" in low or "timed out" in low:
        return "TIMED OUT (suite exists — QA has no signal from it)"
    if "could not run tests" in low:
        return "could not run (see log — QA has no signal from it)"
    return "no suite/deps"


def _should_stop_after_dev(res: dict, committed: bool) -> bool:
    """Whether a Dev result must halt the scope before QA.

    Any error does, whether or not files were written. `committed` stays in the
    signature because the caller reports it (partial work is discarded on the
    next run) — it is deliberately NOT part of the decision. It used to be, as
    `error and not committed`, which let a quota-killed agent that had already
    edited files hand a truncated change to the jury.
    """
    return bool(res.get("error"))


def _inconclusive(res: dict) -> bool:
    """The agent call produced NO verdict at all (errored with empty text) —
    fundamentally different from a verdict of PASS/FAIL/APPROVED."""
    return bool(res.get("error")) and not (res.get("text") or "").strip()


def _review_once(path: str, key: str, criteria: list, diff: str, on_event,
                 impact: str = "") -> tuple[dict, str]:
    provider = settings.review_provider  # separate from Dev — unbiased reviewer
    prompt = prompts.review(key, criteria, diff) + (f"\n\n{impact}" if impact else "")
    backend = providers.agent_backend(provider)
    if backend:  # any agentic CLI can review — cross-provider vs Dev by config
        model = _agent_model(provider, backend, settings.review_model)
        res = agent_backends.run(backend, path, prompt, on_event,
                                 model=model)
        return res, provider
    r = llm.chat(prompts.REVIEW_SYSTEM, prompt,
                 provider=provider, model=settings.review_model)
    if r.get("text"):
        on_event("info", r["text"])  # full review — the log panel shows the whole verdict
    return ({"text": r.get("text", ""), "tokens_in": r.get("tokens_in", 0),
             "tokens_out": r.get("tokens_out", 0), "cost": r.get("cost", 0.0),
             "error": r.get("error")}, provider)


def _original_request(session_id: int) -> str:
    """What the human actually typed, before the PM restated it as criteria.

    The jury needs this: acceptance criteria are a lossy paraphrase written
    before anyone read the code, so a panel holding only the criteria can verify
    the change against the paraphrase but cannot notice that the paraphrase
    missed the point. Joined across turns because the clarifying answers are
    often where the real requirement lives."""
    with Session(engine) as db:
        msgs = db.exec(
            select(ChatMessage).where(
                ChatMessage.session_id == session_id,
                ChatMessage.role == MessageRole.user.value,
            ).order_by(ChatMessage.id)
        ).all()
    return "\n\n---\n\n".join((m.content or "").strip() for m in msgs if (m.content or "").strip())


def _localization_brief(files: list, symbols: list, plan_obj: dict | None = None) -> str:
    """What the change was SUPPOSED to do and where, handed to the jury to check
    the diff against.

    With a plan, this is the strongest evidence the panel gets: the steps were
    decided against the code graph and every symbol was verified, so a diff that
    landed somewhere else is a real signal — either the Dev agent found something
    the Planner missed (it is told to, and to say so), or it went wrong. Without
    a plan the same block carries the PM's unverified hypothesis, and says so —
    a juror weighing a guess as a fact is how a correct change gets rejected.
    """
    if plan_obj and plan_obj.get("steps"):
        lines = ["The plan this change was supposed to implement (decided against the "
                 "code graph before any code was written; every file::symbol below was "
                 "VERIFIED to exist). Judge the diff against it: work that landed "
                 "elsewhere is worth questioning, though the Dev agent is explicitly "
                 "allowed to overrule a step it found to be wrong — if it did, it should "
                 "have said so in its summary."]
        lines.append(planner.as_prompt(plan_obj))
        if plan_obj.get("open_questions"):
            lines.append("The Planner could NOT confirm these, so do not treat them as "
                         "settled: " + "; ".join(plan_obj["open_questions"]))
        return "\n".join(lines)
    if not files and not symbols:
        return ""
    lines = ["Where the change was expected to land (an UNVERIFIED hypothesis — no "
             "planner ran on this delivery — judge whether the diff's actual location "
             "is the RIGHT one):"]
    if files:
        lines.append("  files: " + ", ".join(str(f) for f in files[:10]))
    if symbols:
        lines.append("  symbols: " + ", ".join(str(s) for s in symbols[:12]))
    return "\n".join(lines)


def _jury_review(task_id: int, path: str, key: str, title: str, criteria: list, diff: str,
                 on_event, impact: str = "", context: str = "", dev_summary: str = "",
                 test_output: str = "", request: str = "", description: str = "",
                 localization: str = "") -> tuple[dict, str, dict]:
    """Review by the jury: N specialized judges in parallel, then a foreperson.

    Each juror is billed to its OWN AgentRun so the Costs page shows what the
    panel actually costs per judge (a jury silently multiplies the review stage's
    bill; hiding that in one row would be dishonest). The run this is called
    from carries the foreperson's usage and the final verdict.

    Returns (result, label, decision)."""
    def _bill(op) -> None:
        rid, t0 = agent_runner.start_run(task_id, "review")
        agent_runner.set_model(rid, f"{op.name} · {providers.label(op.provider, op.model)}")
        agent_runner.log(rid, "warn" if op.error else "info",
                         op.error or (op.summary or f"{op.name}: {op.verdict}"))
        for f in op.findings:
            agent_runner.log(rid, "info",
                             f"[{f['severity']} · {f['confidence']:.0%}] {f['title']}"
                             + (f" — {f['location']}" if f["location"] else ""))
        agent_runner.finish_run(rid, task_id, t0, tokens_in=op.tokens_in,
                                tokens_out=op.tokens_out, cost=op.cost,
                                error=op.error or None)

    res = jury.review(key, title, criteria, diff, workdir=path, context=context, impact=impact,
                      dev_summary=dev_summary, test_output=test_output, request=request,
                      description=description, localization=localization,
                      on_event=on_event, on_opinion=_bill)
    decision = res["decision"]
    n = len(res["opinions"])
    label = f"jury ({n} judge{'s' if n != 1 else ''}) → " + (
        decision.get("foreperson") or providers.label(
            settings.jury_synthesis_provider or settings.review_provider,
            settings.jury_synthesis_model or settings.review_model))
    return res, label, decision


def _review_agent(path: str, key: str, criteria: list, diff: str, on_event,
                  impact: str = "") -> tuple[dict, str]:
    """Single-reviewer path (jury disabled), with an explicit INCONCLUSIVE
    outcome: when the reviewer can't run at all (network/provider failure, even
    after in-call retries), retry once, then stamp a loud INCONCLUSIVE verdict
    instead of an empty summary. A blank review_summary reads as 'no issues'
    downstream (board, PR body, write-back) — which is how an unreviewed
    delivery once shipped looking clean."""
    res, provider = _review_once(path, key, criteria, diff, on_event, impact=impact)
    if _inconclusive(res):
        on_event("warn", f"Review agent could not run ({(res.get('error') or '')[:80]}) — retrying once")
        time.sleep(20)
        failed = res
        res, provider = _review_once(path, key, criteria, diff, on_event, impact=impact)
        llm.carry_usage(res, failed)
    if _inconclusive(res):
        res["text"] = (f"VERDICT: INCONCLUSIVE — the review agent could not run "
                       f"({(res.get('error') or 'unknown error')[:160]}). "
                       "This change has NOT been code-reviewed.")
        on_event("error", "Review INCONCLUSIVE — this delivery has NOT been code-reviewed")
    return res, provider


def _impact_brief(repo_url: str, path: str) -> str:
    """Call-graph impact of this branch's committed changes (graph
    detect_changes since origin's default branch): the symbols the change can
    actually affect. Focuses QA/Review on the real blast radius instead of
    re-auditing the whole repo — and away from pre-existing issues elsewhere."""
    try:
        imp = graph.impact(repo_url, since=f"origin/{git_ops.default_branch(path)}")
    except Exception:  # noqa: BLE001
        return ""
    syms = (imp or {}).get("impacted_symbols") or []
    if not syms:
        return ""
    # Name each impacted symbol's CALL SITES, not just the symbol. A reviewer
    # reasoning about "does this edit reach the render path or only the
    # measurement path?" is guessing without them — and a juror guessing wrong
    # costs a full Dev+QA+Review round (observed live on rich: a blocking
    # finding claimed a helper was measurement-only when the graph shows both
    # _measure_column and _render calling it).
    rows: list[str] = []
    slug = Path(path).name
    use_graph = graph.available()
    for s in syms[:20]:
        name = s.get("name")
        row = f"  {name} ({str(s.get('label', '')).lower()}, {s.get('file')})"
        calls = graph.callers(slug, str(name), limit=4) if (use_graph and name) else []
        if calls:
            row += "\n      called from: " + "; ".join(
                f"{c['file_path']}:{c['start_line'] or '?'} ({c['name']})" for c in calls)
        rows.append(row)
    return ("Impact analysis (from the repo's call graph — symbols this change "
            "affects directly or through callers; verification belongs HERE, "
            "code outside this list is untouched by the change). The 'called "
            "from' lines are AST-verified: trust them over your own reading of "
            "which paths reach an edited function:\n"
            + "\n".join(rows))


def _qa_agent(key: str, title: str, criteria: list, diff: str, test_out: str, on_event,
              impact: str = "", path: str = "") -> dict:
    """QA with the same explicit INCONCLUSIVE outcome as _review_agent, and the
    same one-shot reachback the jurors get: QA judges correctness against code it
    can only see a diff of, so it may ask the index before deciding."""
    user = prompts.qa_user(key, title, criteria, diff, test_out) \
        + (f"\n\n{impact}" if impact else "")
    if settings.jury_tool_calls > 0 and path:
        user += "\n" + kb_tools.evidence_block()
    # The case is the cacheable part: the reachback follow-up below re-sends it
    # verbatim, and a revise round re-sends most of it again next attempt.
    qa = llm.chat(prompts.QA_SYSTEM, "", provider=settings.qa_provider,
                  model=settings.qa_model, cache_prefix=user)
    requests = kb_tools.parse_requests(qa.get("text") or "")
    # An answer that is ONLY lookups is a question, not a verdict — answer it and
    # ask once more. A reply that already reached a verdict stands: paying for a
    # second opinion from the same model on the same evidence buys nothing.
    if requests and settings.jury_tool_calls > 0 and path and "VERDICT" not in (qa.get("text") or "").upper():
        answers = kb_tools.run_requests(Path(path).name, path, requests,
                                        limit=settings.jury_tool_calls)
        if answers:
            on_event("info", "QA queried the index: "
                             + ", ".join(f"{n} {a[:40]}" for n, a in requests[:3]))
            # Same prefix, new tail — so this call reads the case back out of the
            # cache the first one just paid to write instead of buying it twice.
            follow = llm.chat(prompts.QA_SYSTEM,
                              f"\n\n{answers}\n\nNow give your verdict. Do not "
                              "request more lookups.",
                              provider=settings.qa_provider, model=settings.qa_model,
                              cache_prefix=user)
            for k in ("tokens_in", "tokens_out", "cost"):
                follow[k] = (follow.get(k) or 0) + (qa.get(k) or 0)
            qa = follow
    if _inconclusive(qa):
        on_event("warn", f"QA agent could not run ({(qa.get('error') or '')[:80]}) — retrying once")
        time.sleep(20)
        failed = qa
        qa = llm.carry_usage(
            llm.chat(prompts.QA_SYSTEM, "", provider=settings.qa_provider,
                     model=settings.qa_model, cache_prefix=user), failed)
    if _inconclusive(qa):
        qa["text"] = (f"VERDICT: INCONCLUSIVE — the QA agent could not run "
                      f"({(qa.get('error') or 'unknown error')[:160]}). "
                      "This change has NOT been verified by QA.")
        on_event("error", "QA INCONCLUSIVE — this delivery has NOT been verified")
    return qa


# Bounded automated back-and-forth: if Review requests changes (or QA fails),
# feed the feedback back to Dev and re-run QA+Review, up to this many rounds.
# Runtime-configurable (Settings → Pipeline); read at loop entry so an edit
# mid-flight applies to the next run, not the one in progress.
def _max_revision_rounds() -> int:
    return max(0, int(settings.max_revision_rounds))


# --- Trivial-task fast path ---------------------------------------------------
# The fixed PM+QA+Review floor (~$0.14 in the benchmarks) makes very cheap
# greppable edits lose to a cold `claude -p`; on trivial tasks LLM QA is also
# the false-fail risk that burns paid revision rounds. Triage is deterministic
# on BOTH sides of Dev (no LLM self-estimate — same rationale as
# pm_agent.ground_tickets: cheap and can't hallucinate): pre-Dev the scope must
# be one ticket with small, grep-pinned localization; post-Dev the actual diff
# must be small AND the deterministic test gate green (runnable suite, zero new
# failures). Only then are the LLM QA + Review passes skipped, with explicit
# fast-path verdicts stamped — never blank summaries. No suite, new failures,
# or a bigger change than triaged all fall through to the full loop.

_FAST_PATH_MAX_FILES = 2      # scoped AND actually-touched non-test source files
_FAST_PATH_MAX_LINES = 120    # changed non-test lines in the final diff


def _fast_path_eligible(subs: list[dict]) -> bool:
    """Pre-Dev triage: one ticket, ≤2 affected files, few criteria, and every
    target symbol grep-pinned to a real definition ('file::name' — the form
    ground_tickets emits only after verifying the definition site). Unpinned or
    '(new — not in repo yet)' symbols mean unverified localization → full loop."""
    if not settings.fast_path_enabled or len(subs) != 1:
        return False
    s = subs[0]
    files = s.get("affected_files") or []
    symbols = s.get("target_symbols") or []
    if not files or len(files) > _FAST_PATH_MAX_FILES:
        return False
    if len(s.get("criteria") or []) > 4:
        return False
    return bool(symbols) and all(
        "::" in str(y) and "(new" not in str(y) for y in symbols)


def _diff_stats(diff_text: str) -> tuple[int, int]:
    """(non-test source files touched, changed non-test lines) of a unified
    diff — the post-Dev reality check that the change is as small as triaged.
    Test files don't count against the budget: a one-line fix plus real
    regression tests is exactly the shape the fast path is for."""
    files: set[str] = set()
    in_test = False
    changed = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            rel = line.split(" b/", 1)[-1]
            in_test = lang.is_test_file(rel)
            if not in_test:
                files.add(rel)
        elif not in_test and (
                (line.startswith("+") and not line.startswith("+++"))
                or (line.startswith("-") and not line.startswith("---"))):
            changed += 1
    return len(files), changed


def _review_changes_requested(text: str) -> bool:
    return "CHANGES REQUESTED" in (text or "").upper()


def _inconclusive_verdict(text: str) -> bool:
    return "INCONCLUSIVE" in (text or "").upper()


def _qa_failed(text: str) -> bool:
    """True only on an explicit VERDICT: FAIL (CONCERNS does not trigger a revise)."""
    up = (text or "").upper()
    i = up.find("VERDICT")
    return i != -1 and "FAIL" in up[i:i + 40]


def _dev_revise(path: str, key: str, title: str, criteria: list, review_text: str,
                qa_text: str, on_event, context: str = "",
                affected_files: list | None = None, target_symbols: list | None = None,
                diff: str = "", test_cmd: str = "", plan_obj: dict | None = None) -> tuple[dict, str]:
    """Revision Dev pass: edit the already-committed change to address feedback.
    Inherits the verified localization (affected_files/target_symbols), the plan,
    and the current branch diff, so the reviser edits the right files instead of
    scraping paths out of review prose (which produced zero-change revisions)."""
    provider = settings.dev_provider
    backend = providers.agent_backend(provider)
    prompt = prompts.revise(key, title, criteria, review_text, qa_text, context, test_cmd=test_cmd,
                            tools_cmd=_install_dev_tools(path, on_event) if backend else "",
                            plan=planner.as_prompt(plan_obj or {}),
                            verified=_verified_locations(path, affected_files, target_symbols))
    if backend:
        if backend == "claude-code":
            model = settings.dev_model if settings.dev_model in _CLI_ALIASES else settings.claude_model
        else:
            model = _agent_model(provider, backend, settings.dev_model)
        label = _agent_label(provider, model)
        on_event("info", f"Dev model: {label} (revision)")
        res = agent_backends.run(backend, path, prompt, on_event, model=model)
        if res["error"] and _is_transient(res["error"]):
            on_event("warn", f"{backend} transient error ({res['error'][:80]}) — retrying once")
            time.sleep(15)
            failed = res
            res = llm.carry_usage(
                agent_backends.run(backend, path, prompt, on_event, model=model), failed)
        if not res["error"] or not _backend_unavailable(backend, res["error"]):
            return res, label
        on_event("warn", f"{backend} unavailable — HTTP coding agent for revision")
        fb_provider, fb_model = _auto_http_target()
    else:
        fb_provider, fb_model = provider, settings.dev_model
    # OpenAI-compatible path: feedback becomes the task; the current diff rides in the context.
    desc = (f"REVISION of an already-committed change (see the current branch diff in the "
            f"repository knowledge below). Address this reviewer feedback:\n{review_text[:4000]}\n\n"
            f"QA findings:\n{qa_text[:2000]}")
    ctx = f"Current branch diff (the change being revised):\n```diff\n{diff[:8000]}\n```\n\n{context}"
    cr = openai_agent.code(path, key, title, desc, criteria, on_event, provider=fb_provider,
                           model=fb_model, context=ctx,
                           affected_files=affected_files, target_symbols=target_symbols)
    return ({"text": cr["summary"], "tokens_in": cr["tokens_in"], "tokens_out": cr["tokens_out"],
             "cost": cr["cost"], "error": cr["error"]}, providers.label(fb_provider, fb_model))


def _update(task_id: int, **fields) -> None:
    with Session(engine) as db:
        task = db.get(Task, task_id)
        if task is None:
            return
        for k, v in fields.items():
            setattr(task, k, v)
        task.updated_at = utcnow()
        db.add(task)
        db.commit()
    # Lets an attached terminal client redraw the stage timeline the moment a
    # task moves, rather than noticing on its next poll.
    events.publish("task.updated", task_id=task_id, fields=fields)


def _update_all(ids: list[int], **fields) -> None:
    for i in ids:
        _update(i, **fields)


def _load(task_id: int) -> tuple[dict, str] | None:
    with Session(engine) as db:
        task = db.get(Task, task_id)
        if task is None:
            return None
        repo = db.get(Repo, task.repo_id)
        info = {
            "key": task.key, "title": task.title, "description": task.description or "",
            "criteria": task.acceptance_criteria or [],
            "affected_files": task.affected_files or [],
            "target_symbols": task.target_symbols or [],
        }
        return info, (repo.git_url if repo else "")




def run_scope(session_id: int) -> None:
    """Public entry point. Serializes all work on a repo's SHARED working copy so
    two scopes never run it at once — concurrent scopes race on branch/commit
    state (a `reset --hard` under a live run destroys the other's work), which is
    a real corruption we hit, not a theoretical one. Different repos still run in
    parallel; only same-repo scopes queue behind this lock."""
    with Session(engine) as db:
        session = db.get(ScopeSession, session_id)
        repo = db.get(Repo, session.repo_id) if session else None
        repo_url = repo.git_url if repo else ""
    if not repo_url:
        _run_scope_locked(session_id)
        return
    with git_ops.repo_lock(repo_url):
        _run_scope_locked(session_id)


def _mark_scope_blocked(session_id: int, reason: str) -> None:
    """Persist an interrupted/unsafe run so it cannot look runnable or disappear."""
    with Session(engine) as db:
        tasks = db.exec(select(Task).where(Task.session_id == session_id)).all()
        message = f"PIPELINE BLOCKED — {reason[:1000]}"
        for task in tasks:
            if task.status != TaskStatus.done.value:
                task.status = TaskStatus.blocked.value
                task.current_agent = None
                task.review_summary = message
                task.updated_at = utcnow()
                db.add(task)
        session = db.get(ScopeSession, session_id)
        if session is not None:
            session.status = "failed"
            db.add(session)
        db.commit()


def _run_scope_locked(session_id: int) -> None:
    try:
        _run_scope_locked_impl(session_id)
    except Exception as exc:  # noqa: BLE001 — persist failure before the worker exits
        logger.exception("run_scope %s failed unexpectedly", session_id)
        _mark_scope_blocked(session_id, f"unexpected pipeline error: {exc}")


def _run_scope_locked_impl(session_id: int) -> None:
    """Run a whole scope as ONE deliverable: every approved subtask is implemented
    on a SINGLE branch (accumulating), then one QA + Review + a single PR for the
    scope. All subtasks share that PR — so the PR is a complete, valid feature.
    Always called under `git_ops.repo_lock` via `run_scope`."""
    with Session(engine) as db:
        session = db.get(ScopeSession, session_id)
        if session is None:
            return
        repo = db.get(Repo, session.repo_id)
        repo_url = repo.git_url if repo else ""
        subtasks = db.exec(
            select(Task).where(
                Task.session_id == session_id, Task.approved == True,  # noqa: E712
                Task.status.in_([TaskStatus.scoped.value, TaskStatus.backlog.value]),
            ).order_by(Task.key)
        ).all()
        subs = [{"id": t.id, "key": t.key, "title": t.title, "description": t.description or "",
                 "criteria": t.acceptance_criteria or [],
                 "affected_files": t.affected_files or [], "target_symbols": t.target_symbols or []}
                for t in subtasks]
        scope_title = session.title or f"Scope #{session_id}"
    if not subs:
        logger.info("run_scope %s: no approved subtasks", session_id)
        return
    ids = [s["id"] for s in subs]
    all_criteria = [c for s in subs for c in s["criteria"]]

    try:
        path = git_ops.ensure_clone(repo_url)
        branch = f"agent/scope-{session_id}"
        git_ops.checkout_branch(path, branch)
    except Exception as exc:  # noqa: BLE001
        logger.error("run_scope %s workspace prep failed: %s", session_id, exc)
        _mark_scope_blocked(session_id, f"workspace preparation failed: {exc}")
        return

    # --- Dev: ONE work order for the whole scope ---
    # Fragmenting a feature into per-ticket Dev runs is what produced "flag
    # registered but never threaded through": no single run ever saw the whole
    # feature. One Dev session gets every subtask + criteria and can wire the
    # change end-to-end; tickets stay as tracking/approval artifacts.
    rep = subs[0]["id"]  # scope-level Dev/QA/Review/PR runs hang off the first subtask
    all_files = list(dict.fromkeys(f for s in subs for f in s["affected_files"]))
    all_symbols = list(dict.fromkeys(y for s in subs for y in s["target_symbols"]))
    desc = "\n\n".join(
        f"Subtask {s['key']}: {s['title']}\n{s['description']}\nAcceptance criteria:\n"
        + ("\n".join(f"- {c}" for c in s["criteria"]) or "- (use your judgment)")
        for s in subs)
    work = {"key": f"scope-{session_id}", "title": scope_title, "description": desc,
            "criteria": all_criteria, "affected_files": all_files,
            "target_symbols": all_symbols, "is_scope": True}

    _update_all(ids, status=TaskStatus.in_dev.value, branch=branch)

    # --- Refresh the knowledge layers, then plan — both before Dev opens ---
    # The plan's localization REPLACES whatever the tickets carried, rather than
    # supplementing it: the PM works from summaries and never opens the code,
    # while the Planner queried the graph and had every symbol it named verified
    # against the real repo. Two sources of "which file" is how a Dev agent ends
    # up editing three.
    plan_obj = _refresh_and_plan(session_id, rep, repo_url, path, {
        "summary": scope_title,
        "acceptance_criteria": all_criteria,
    }, subs)
    if plan_obj:
        plan_files, plan_symbols = planner.targets(plan_obj)
        if plan_files or plan_symbols:
            all_files, all_symbols = plan_files, plan_symbols
            work["affected_files"], work["target_symbols"] = plan_files, plan_symbols
            # Write the verified localization onto the tickets so the board shows
            # where the work actually goes, not where anyone guessed.
            _update_all(ids, affected_files=plan_files, target_symbols=plan_symbols)
    work["plan"] = plan_obj

    rid, t0 = agent_runner.start_run(rep, "dev")
    agent_runner.log(rid, "info",
                     f"Dev agent implementing {len(subs)} subtask(s) as one work order on {branch}")
    kb = _kb_context(repo_url, work, agent_runner.logger_for(rid))
    test_cmd = _test_cmd(path)  # builds the per-repo test env so Dev can verify
    if test_cmd:
        agent_runner.log(rid, "info", f"Repo test env ready: {test_cmd}")
    # Baseline the suite on the CLEAN tree (before Dev edits) so QA later judges
    # only NEW failures. A repo that ships failing tests otherwise makes QA read
    # every stale failure as a regression and burn revision rounds fixing nothing.
    baseline_fail = git_ops.failing_tests(path)
    if baseline_fail:
        agent_runner.log(rid, "warn",
                         f"{len(baseline_fail)} test(s) already fail on clean "
                         f"{git_ops.default_branch(path)} — QA will discount these as pre-existing")
    res, dev_label = _dev_agent(path, work["key"], work, agent_runner.logger_for(rid), kb, test_cmd)
    agent_runner.set_model(rid, dev_label)
    committed = git_ops.add_commit(path, f"scope-{session_id}: {scope_title[:60]}")
    agent_runner.log(rid, "success" if committed else "warn",
                     "Committed changes" if committed else "No file changes produced")
    agent_runner.finish_run(rid, rep, t0, tokens_in=res["tokens_in"], tokens_out=res["tokens_out"],
                            cost=res["cost"], error=res["error"])
    if _should_stop_after_dev(res, committed):
        # ANY Dev failure stops the scope — not just one that wrote nothing.
        # This used to require `and not committed`, so an agent killed part-way
        # through (a quota cutoff after 865s and 1.2M tokens, observed live) had
        # already written files, `committed` was True, and the `and` let the run
        # walk into QA to judge a half-finished change. A partial edit set from
        # a killed agent is debris, not a deliverable, and the jury cannot tell
        # the difference from the diff alone.
        partial = " (partial changes were committed and will be discarded)" if committed else ""
        message = f"Dev failed: {res['error'][:400]}"
        logger.error("run_scope %s: %s%s — stopping before QA", session_id, message, partial)
        agent_runner.log(rid, "error", f"{message}{partial} — stopping before QA")
        # Recovery: undo the in_dev marker on every subtask so the scope is
        # immediately re-runnable (no manual DB reset needed). checkout_branch
        # resets the branch from origin's default on the next run, so the
        # partial commit above cannot leak into it.
        _update_all(ids, status=TaskStatus.scoped.value, review_summary=message)
        return

    diff = git_ops.diff(path, ref=branch)

    # --- QA + Review with a bounded revise loop: on CHANGES REQUESTED or QA FAIL,
    #     feed the feedback back to Dev, re-commit, and re-run — up to N rounds. ---
    qa_text = ""
    rev_text = ""
    fast_candidate = _fast_path_eligible(subs)
    max_rounds = _max_revision_rounds()
    for attempt in range(max_rounds + 1):
        # QA (OpenAI) over the whole scope diff
        _update_all(ids, status=TaskStatus.qa.value)
        rid, t0 = agent_runner.start_run(rep, "qa")
        passed, test_out = git_ops.run_tests(path, _scoped_test_paths(path, branch))
        # Only re-diff failures when the suite failed AND the repo had pre-existing
        # failures — the exact case where QA would otherwise misread stale failures
        # as regressions. Clean repos / clean runs pay no extra suite run.
        if passed is False and baseline_fail:
            now_fail = git_ops.failing_tests(path) or set()
            new_fail = sorted(now_fail - baseline_fail)
            if new_fail:
                banner = ("REGRESSION CHECK: this change introduced " f"{len(new_fail)} NEW "
                          "test failure(s) not present before it — judge ONLY these, the rest are "
                          "pre-existing:\n  " + "\n  ".join(new_fail[:25]) + "\n\n")
            else:
                banner = ("REGRESSION CHECK: every failing test below ALSO failed on the clean tree "
                          "BEFORE this change (pre-existing, unrelated to it). This change introduced "
                          "ZERO new failures — treat the suite as GREEN for acceptance purposes.\n\n")
            test_out = banner + test_out
            passed = not new_fail  # no NEW failures ⇒ green for this change
        agent_runner.log(rid, "success" if passed else ("warn" if passed is None else "error"),
                         f"Tests: {_test_outcome(passed, test_out)}")
        # Fast path: the triaged-trivial change verified through the deterministic
        # gate (suite ran, zero new failures) and the diff is as small as triaged
        # — skip the paid LLM QA + Review passes with explicit verdicts stamped.
        if attempt == 0 and fast_candidate and passed is True and not res["error"]:
            n_files, n_lines = _diff_stats(diff)
            if n_files <= _FAST_PATH_MAX_FILES and n_lines <= _FAST_PATH_MAX_LINES:
                qa_text = (
                    "VERDICT: PASS — FAST PATH (task triaged trivial: one ticket, grep-pinned "
                    f"localization). Deterministic test gate green: suite ran with zero new "
                    f"failures; diff touches {n_files} source file(s), ~{n_lines} changed "
                    "line(s). LLM QA was skipped for this delivery.")
                rev_text = (
                    "FAST PATH: code review skipped — trivial, fully-localized change verified "
                    "by the deterministic test gate (see QA verdict). This diff was NOT "
                    "LLM-reviewed.")
                _update_all(ids, qa_summary=qa_text, review_summary=rev_text)
                agent_runner.set_model(rid, "deterministic-gate")
                agent_runner.log(rid, "success",
                                 f"Fast path: skipping LLM QA + Review (trivial task, suite green, "
                                 f"{n_files} file(s)/{n_lines} line(s)) — saved the QA+Review gate cost")
                agent_runner.finish_run(rid, rep, t0)
                break
        # A suite that never ran must not read as a quiet green. QA sees only a
        # diff and the test output, so "no failures in the output" is exactly
        # what a passing run looks like — on gitea the suite timed out and QA
        # returned PASS having verified nothing.
        if passed is None:
            test_out = ("NO TEST SIGNAL: the suite did NOT run for this change "
                        f"({_test_outcome(passed, test_out)}). The absence of failures below "
                        "is NOT evidence of correctness. Judge the diff by reading it, and "
                        "state explicitly in your verdict that no tests were executed — do "
                        "not claim the change is verified.\n\n") + test_out
        impact_text = _impact_brief(repo_url, path)
        qa = _qa_agent(f"scope-{session_id}", scope_title, all_criteria, diff, test_out,
                       agent_runner.logger_for(rid), impact=impact_text, path=path)
        agent_runner.set_model(rid, _qa_label())
        if qa.get("text"):
            agent_runner.log(rid, "info", qa["text"])  # full QA verdict for the log panel
        qa_text = qa.get("text", "")
        # Stamp the record too, not just the prompt: qa_summary is what the board,
        # the PR body and the write-back all read, and a bare "PASS" there claims
        # a verification that never happened.
        if qa_text and passed is None:
            qa_text += ("\n\n[NO TEST SIGNAL — the suite did not run for this change; "
                        "this verdict is a reading of the diff, not a verified result.]")
            qa["text"] = qa_text
        if qa_text:
            _update_all(ids, qa_summary=qa_text)  # errored calls keep the last good verdict
        agent_runner.finish_run(rid, rep, t0, tokens_in=qa.get("tokens_in", 0), tokens_out=qa.get("tokens_out", 0),
                                cost=qa.get("cost", 0.0), error=qa.get("error"))

        # Review over the whole scope diff — the jury (several specialized judges
        # + a foreperson) unless the operator turned the ensemble off.
        _update_all(ids, status=TaskStatus.review.value)
        verdict = ""
        rid, t0 = agent_runner.start_run(rep, "review")
        if jury.enabled():
            rev, rlabel, decision = _jury_review(
                rep, path, f"scope-{session_id}", scope_title, all_criteria, diff,
                agent_runner.logger_for(rid), impact=impact_text, context=kb,
                dev_summary=res.get("text", ""), test_output=test_out,
                request=_original_request(session_id), description=desc,
                localization=_localization_brief(all_files, all_symbols, plan_obj))
            verdict = decision.get("verdict", "")
            _update_all(ids, review_findings=decision)
        else:
            rev, rprov = _review_agent(path, f"scope-{session_id}", all_criteria, diff,
                                       agent_runner.logger_for(rid), impact=impact_text)
            rlabel = _review_label(rprov)
        agent_runner.set_model(rid, rlabel)
        rev_text = rev.get("text", "")
        if rev_text:
            _update_all(ids, review_summary=rev_text)  # errored calls keep the last good verdict
        agent_runner.finish_run(rid, rep, t0, tokens_in=rev["tokens_in"], tokens_out=rev["tokens_out"],
                                cost=rev["cost"], error=rev["error"])

        # Decide: ship, or send back to Dev for another round? The jury reports a
        # machine-readable verdict; the single reviewer only ever produced prose.
        needs_fix = (verdict == "CHANGES REQUESTED" if verdict
                     else _review_changes_requested(rev_text)) or _qa_failed(qa_text)
        if (_inconclusive_verdict(qa_text) or _inconclusive_verdict(rev_text)
                or verdict == "INCONCLUSIVE"):
            reason = ("QA or Review could not produce a trustworthy verdict; no PR may be "
                      "opened until the pipeline is rerun successfully.")
            _mark_scope_blocked(session_id, reason)
            if jury.enabled() and decision:
                decision["blocked"] = True
                _update_all(ids, review_findings=decision)
            agent_runner.log(rid, "error", reason)
            return
        if not needs_fix:
            break
        if attempt >= max_rounds:
            # Rounds exhausted with the review still blocking. This delivery is
            # NOT approved, and it must not reach the PR lane looking like one
            # that was: same status, same green board card, an unresolved defect
            # inside. (Live run on rich: shipped with a jury-confirmed early
            # `return` that skipped the table's bottom border.) Stamp it on the
            # summary the UI and the PR body both render, and flag it on the
            # decision so the board can badge it.
            blocking = (decision.get("blocking") or []) if jury.enabled() else []
            titles = "; ".join(str(b.get("title") or "")[:120] for b in blocking[:3])
            banner = (
                f"⚠️ DELIVERED WITHOUT APPROVAL — the review still requested changes after "
                f"{max_rounds} revision round(s), so the pipeline stopped paying for more. "
                f"{len(blocking) or 'Outstanding'} finding(s) remain UNRESOLVED"
                + (f": {titles}" if titles else "")
                + ". Do NOT merge without addressing them or deciding they are wrong.\n\n"
            )
            rev_text = banner + rev_text
            _update_all(ids, review_summary=rev_text)
            if jury.enabled():
                decision["unresolved_blocking"] = len(blocking) or 1
                decision["rounds_exhausted"] = max_rounds
                _update_all(ids, review_findings=decision)
            agent_runner.log(rid, "error",
                             f"Still CHANGES REQUESTED after {max_rounds} revision round(s) — "
                             "opening the PR flagged as UNAPPROVED with the findings outstanding")
            break

        # --- Revision Dev pass over the whole scope: fix, re-commit, re-diff ---
        round_no = attempt + 1
        _update_all(ids, status=TaskStatus.in_dev.value)
        rid, t0 = agent_runner.start_run(rep, "dev")
        agent_runner.log(rid, "info",
                         f"Revision round {round_no}/{max_rounds}: addressing review + QA feedback")
        kb = _kb_context(repo_url, {"title": scope_title, "description": ""}, agent_runner.logger_for(rid))
        # The reviser inherits the scope's localization plus the current branch diff.
        res, dev_label = _dev_revise(path, f"scope-{session_id}", scope_title, all_criteria,
                                     rev_text, qa_text, agent_runner.logger_for(rid), kb,
                                     affected_files=all_files, target_symbols=all_symbols,
                                     diff=diff, test_cmd=test_cmd, plan_obj=plan_obj)
        agent_runner.set_model(rid, dev_label)
        committed = git_ops.add_commit(path, f"scope-{session_id}: address review feedback (round {round_no})")
        agent_runner.log(rid, "success" if committed else "warn",
                         "Committed revision" if committed else "No changes produced by revision")
        agent_runner.finish_run(rid, rep, t0, tokens_in=res["tokens_in"], tokens_out=res["tokens_out"],
                                cost=res["cost"], error=res["error"])
        if not committed:
            # Dev disputed the feedback (or had nothing to change). Re-running QA+Review
            # on a byte-identical diff buys the same verdict twice — stop the loop and
            # ship with the outstanding feedback noted instead of burning rounds.
            agent_runner.log(rid, "warn",
                             "Revision produced no changes — skipping re-QA/re-review of an "
                             "identical diff; opening the PR with the outstanding feedback noted")
            break
        diff = git_ops.diff(path, ref=branch)

    # --- ONE PR for the whole scope ---
    _update_all(ids, status=TaskStatus.pr.value)
    rid, t0 = agent_runner.start_run(rep, "pr")
    pr_title = f"Scope: {scope_title[:60]}"
    pr_body = prompts.scope_pr_body(scope_title, subs, qa_text, review_summary=rev_text)
    pr_link = ""
    if not _prs_enabled():
        agent_runner.log(rid, "info",
                         f"[DEMO_MODE] Dry-run — would open one scope PR for {len(subs)} subtasks: {pr_title}")
        agent_runner.log(rid, "info", pr_body)
        agent_runner.finish_run(rid, rep, t0)
    else:
        try:
            git_ops.push(path, branch)
            agent_runner.log(rid, "info", f"Pushed {branch} to origin")
            pr_link = git_ops.gh_pr_create(path, pr_title, pr_body)
            _update_all(ids, pr_url=pr_link)
            agent_runner.log(rid, "success", f"Opened one scope PR for {len(subs)} subtasks: {pr_link}")
            agent_runner.finish_run(rid, rep, t0)
        except Exception as exc:  # noqa: BLE001
            agent_runner.finish_run(rid, rep, t0, error=f"PR failed: {exc}")

    # --- Write-back: persist what this delivery learned (files, symbols,
    # gotchas) into the knowledge base so future runs on this repo start from
    # it instead of rediscovering everything. ---
    write_back.record_delivery(
        repo_url, f"scope-{session_id}", scope_title, path, branch,
        dev_summary=res.get("text", ""), qa_text=qa_text, review_text=rev_text,
        pr_url=pr_link, on_event=lambda lvl, msg: agent_runner.log(rid, lvl, msg))

    with Session(engine) as db:
        session = db.get(ScopeSession, session_id)
        if session is not None:
            session.status = "delivered"
            db.add(session)
            db.commit()
    logger.info("run_scope %s finished (%d subtasks)", session_id, len(subs))
