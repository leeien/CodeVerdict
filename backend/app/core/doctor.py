"""Preflight: can this machine actually run a delivery, and what is degraded?

Every dependency in this product is optional in a different way. Some are hard
requirements (git — the agents clone and branch). Some silently downgrade
quality (no ripgrep → coarser localization; no code graph → the symbol-map
tier). And the LLM side has no single answer at all: a stage is runnable if its
provider has a key *or* if the coding CLI it points at is installed and logged
in. So "is it set up?" is genuinely hard to answer by reading the docs, which is
exactly why it should be a command.

Returns plain dicts rather than raising: a preflight that stops at the first
problem makes the operator fix things one round-trip at a time. The caller
renders every row and reads ``ready`` for the verdict.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from importlib.util import find_spec

OK = "ok"
WARN = "warn"
FAIL = "fail"


def _check(name: str, status: str, detail: str = "", hint: str = "") -> dict:
    return {"name": name, "status": status, "detail": detail, "hint": hint}


def _version(exe: str, *args: str) -> str:
    try:
        p = subprocess.run([exe, *(args or ("--version",))],
                           capture_output=True, text=True, timeout=10)
        line = (p.stdout or p.stderr or "").strip().splitlines()
        return line[0][:60] if line else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


# ── groups ────────────────────────────────────────────────────────────────────
def _environment() -> list[dict]:
    checks = [_check("python", OK, f"{sys.version_info.major}.{sys.version_info.minor}"
                     f".{sys.version_info.micro}")]

    git = shutil.which("git")
    checks.append(_check("git", OK, _version(git) or git) if git else _check(
        "git", FAIL, "not on PATH",
        "Required — the agents clone the repo and work on a branch of it."))

    rg = shutil.which("rg")
    checks.append(_check("ripgrep", OK, _version(rg)) if rg else _check(
        "ripgrep", WARN, "not on PATH — searches fall back to `git grep`",
        "apt install ripgrep / brew install ripgrep. Without it, search covers "
        "tracked files only and has no .gitignore-aware walk; definition searches "
        "still retain language filters, so localization is measurably coarser."))

    gh = shutil.which("gh")
    checks.append(_check("gh (GitHub CLI)", OK, _version(gh)) if gh else _check(
        "gh (GitHub CLI)", WARN, "not on PATH — the PR stage cannot open a real PR",
        "Only needed to open pull requests. Demo mode dry-runs that stage anyway."))
    return checks


def _knowledge() -> list[dict]:
    from ..services.knowledge import graph

    checks = []
    probe = graph.probe()
    if probe["ok"]:
        checks.append(_check("code graph (codebase-memory-mcp)", OK,
                             probe["output"].splitlines()[-1][:60]))
    else:
        checks.append(_check(
            "code graph (codebase-memory-mcp)", WARN,
            "unavailable — the KB falls back to its symbol-map + grep tier",
            "npm i -g codebase-memory-mcp (or brew/scoop/a release binary). This "
            "is the primary localization engine; the fallback works but costs "
            "more tokens per task."))

    if find_spec("fastembed") and find_spec("qdrant_client"):
        checks.append(_check("semantic search (fastembed + qdrant)", OK, "installed"))
    else:
        checks.append(_check(
            "semantic search (fastembed + qdrant)", WARN, "not installed — retrieval is keyword-only",
            'pip install -e ".[semantic]" — adds the local dense-embedding '
            "channel that finds code by intent rather than by token match."))

    if find_spec("tree_sitter"):
        checks.append(_check("tree-sitter extractors", OK, "installed"))
    else:
        checks.append(_check(
            "tree-sitter extractors", WARN,
            "not installed — non-Python symbol extraction uses regex",
            'pip install -e ".[treesitter]" — Python is exact either way (stdlib ast).'))
    return checks


def _agents() -> list[dict]:
    """One row per pipeline stage: is the thing it points at actually usable?

    This is the check that matters most and the one a README cannot make for
    you, because "configured" means a key for an API provider and an installed
    login for a coding CLI.
    """
    from ..services import providers

    checks = []
    for stage in providers.STAGES:
        pid = providers.stage_provider(stage)
        model = providers.stage_model(stage)
        label = f"{pid}" + (f" / {model}" if model else "")
        if providers.can_chat(pid):
            checks.append(_check(f"stage: {stage}", OK, label))
            continue
        if providers.kind(pid) in ("claude-cli", "agent"):
            hint = (f"`{pid}` is selected but its CLI isn't installed or logged in. "
                    f"Install it from /settings, or repoint the stage: "
                    f"`/model {stage} groq`.")
        else:
            hint = (f"`{pid}` has no API key set. Add one in /settings, or repoint "
                    f"the stage at an installed CLI: `/model {stage} claude-cli`.")
        checks.append(_check(f"stage: {stage}", FAIL, f"{label} — not runnable", hint))
    return checks


def _backends() -> list[dict]:
    """Which coding CLIs are on this machine. Informational, never fatal — you
    need exactly one working path to the LLM, not all of them."""
    from ..services import agent_backends

    checks = []
    for bid, det in agent_backends.availability().items():
        if det.get("available"):
            checks.append(_check(bid, OK, det.get("version", "") or "installed"))
        else:
            checks.append(_check(bid, WARN, det.get("reason", "not available"),
                                 det.get("connect_hint", "")))
    return checks


def _delivery() -> list[dict]:
    from ..config import settings

    checks = []
    if settings.demo_mode:
        checks.append(_check("demo mode", OK, "on — the PR stage is a dry run",
                             "Safe default. Turn it off in /settings only for a repo you own."))
    else:
        checks.append(_check("demo mode", WARN, "OFF — the PR stage will push and open real PRs",
                             "Make sure the active repo is one you own and `gh` is authenticated."))

    if any((settings.jira_base_url, settings.jira_email, settings.jira_api_token)):
        complete = all((settings.jira_base_url, settings.jira_email,
                        settings.jira_api_token, settings.jira_project_key))
        checks.append(_check("jira", OK, "configured") if complete else _check(
            "jira", WARN, "partially configured — ticket push will no-op",
            "Set jira_base_url, jira_email, jira_api_token and jira_project_key, "
            "or clear them all."))
    else:
        checks.append(_check("jira", OK, "not configured (optional — pushes are a no-op)"))
    return checks


# ── entry point ───────────────────────────────────────────────────────────────
def check() -> dict:
    """``{groups: [{name, checks: [...]}, ...], ready, failures, warnings}``.

    ``ready`` is the only thing a caller needs to decide whether to tell the
    operator to fix something before running a delivery.
    """
    groups = [
        {"name": "Environment", "checks": _environment()},
        {"name": "Knowledge base", "checks": _knowledge()},
        {"name": "Pipeline stages", "checks": _agents()},
        {"name": "Coding CLIs detected", "checks": _backends()},
        {"name": "Delivery", "checks": _delivery()},
    ]
    every = [c for g in groups for c in g["checks"]]
    failures = [c for c in every if c["status"] == FAIL]
    warnings = [c for c in every if c["status"] == WARN]
    return {
        "groups": groups,
        "ready": not failures,
        "failures": len(failures),
        "warnings": len(warnings),
    }
