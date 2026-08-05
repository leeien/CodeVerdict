"""Runs the Claude Code CLI headlessly as an agent (PM / Dev / Review / PR).

Streams `--output-format stream-json` events into a callback so each tool call
and assistant message becomes a log line, and pulls real token usage + cost from
the final `result` event. The prompt is sent on stdin to avoid arg-length limits.
"""

import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable

from ..config import settings

Event = Callable[[str, str], None]  # (severity, message)


def _resolve() -> str:
    return shutil.which(settings.claude_cli_path) or settings.claude_cli_path


def _child_env() -> dict:
    """Environment for the spawned CLI.

    Strips nested-session / gateway overrides that Claude Code injects into child
    processes (they carry session-scoped auth a fresh CLI can't use), so the agent
    authenticates via the host's own Claude login (or an explicit API key). In a
    normal user terminal these vars aren't set, so this is a no-op there.
    """
    env = os.environ.copy()
    for k in list(env):
        if k.startswith("CLAUDE_CODE_") or k in (
            "CLAUDECODE", "ANTHROPIC_BASE_URL", "CLAUDE_AGENT_SDK_VERSION", "CLAUDE_EFFORT",
        ):
            env.pop(k, None)
    if settings.anthropic_api_key:
        env["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    return env


# Tools whose argument is a path we may already have injected into the prompt.
# A Read of a file whose full contents were handed over is a wasted turn, and
# turns — not prompt size — are what make the Dev stage 84% of a run's tokens.
_PATH_TOOLS = ("Read", "NotebookRead")


def _record_tool(result: dict, name: str | None, args) -> None:
    """Tally one tool call so a run can be asked *why* it cost what it did.

    The stream already carries this; not counting it was the reason the token
    bill could only ever be described, never explained.
    """
    if not name:
        return
    result["tools"][name] = result["tools"].get(name, 0) + 1
    if name in _PATH_TOOLS and isinstance(args, dict):
        path = args.get("file_path") or args.get("path") or ""
        if path and len(result["read_paths"]) < 200:
            result["read_paths"].append(str(path))


def _short(x, n: int = 160) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = " ".join(s.split())
    return s[:n]


# The CLI reports an exhausted plan by *answering* with it: exit 0, is_error
# unset, and the quota notice as the result text. Nothing downstream can tell
# that apart from a real reply, so it was stored as the agent's own words —
# observed live, where "You've hit your session limit · resets 2:50am" became a
# PM turn in the scope history and 34,466 tokens were billed for it.
_QUOTA_MARKERS = (
    "hit your session limit",
    "usage limit reached",
    "session limit reached",
    "out of credits",
    "insufficient credits",
    "upgrade to increase your usage limit",
)


def quota_error(text: str) -> str:
    """The plan-exhausted message hiding in a reply, or '' if there isn't one.

    Deliberately narrow: it only fires on a SHORT reply, because a long answer
    that happens to discuss rate limits is a real answer. A quota notice is
    always terse — the CLI has nothing else to say.
    """
    body = (text or "").strip()
    if not body or len(body) > 300:
        return ""
    low = body.lower()
    return body if any(m in low for m in _QUOTA_MARKERS) else ""


def chat(system: str, user: str = "", *, model: str | None = None, timeout: int = 180,
         json_mode: bool = False, messages: list[dict] | None = None) -> dict:
    """One pure-chat completion through the Claude Code CLI (host login — no API
    key required). Mirrors anthropic_api.chat's shape so llm.chat can dispatch to
    it. Runs in an empty scratch dir with a no-tools instruction so the agentic
    CLI behaves like a plain completion; cost comes from the CLI's own meter."""
    from pathlib import Path

    scratch = Path(settings.repos_dir).expanduser() / "_chat"
    scratch.mkdir(parents=True, exist_ok=True)

    parts = [system]
    if json_mode:
        parts.append("Respond with a single valid JSON object and NOTHING else — "
                     "no prose, no markdown fences.")
    parts.append("Do not use any tools (no file reads, searches, or commands). "
                 "Answer directly from the information below.")
    if messages is not None:
        for m in messages:
            parts.append(f"[{m.get('role', 'user')}]\n{m.get('content', '')}")
    else:
        parts.append(user)
    prompt = "\n\n".join(p for p in parts if p)

    events: list[str] = []
    res = run_claude(str(scratch), prompt, lambda sev, msg: events.append(f"{sev}: {msg}"),
                     model=model, timeout=timeout)
    return {"text": res["text"], "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"],
            "cost": res["cost"], "model": model or settings.claude_model, "error": res["error"]}


def run_claude(cwd: str, prompt: str, on_event: Event, *, model: str | None = None, timeout: int = 1800) -> dict:
    """Run one Claude agent turn in `cwd`. Returns {text, tokens_in, tokens_out, cost, error}."""
    exe = _resolve()
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose",
           "--no-session-persistence"]
    if settings.claude_skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if settings.claude_max_budget_usd > 0:
        # Hard dollar ceiling per agent run — a runaway session can't overspend.
        cmd += ["--max-budget-usd", str(settings.claude_max_budget_usd)]
    m = model or settings.claude_model
    if m:
        cmd += ["--model", m]

    result = {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "error": None,
              "turns": 0, "tools": {}, "read_paths": []}
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=_child_env(),
        )
    except (FileNotFoundError, OSError) as exc:
        result["error"] = f"claude CLI not runnable: {exc}"
        on_event("error", result["error"])
        return result

    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"failed to send prompt: {exc}"
        on_event("error", result["error"])
        proc.kill()
        return result

    # Watchdog: `for line in proc.stdout` blocks forever if the CLI hangs
    # mid-stream (the old wait-timeout only fired AFTER stdout EOF). A timer
    # hard-kills the process at the deadline so a stuck agent can't wedge the
    # whole pipeline thread.
    watchdog_fired = threading.Event()

    def _kill() -> None:
        watchdog_fired.set()
        proc.kill()

    watchdog = threading.Timer(timeout, _kill)
    watchdog.start()
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                on_event("info", line[:200])
                continue
            t = ev.get("type")
            if t == "assistant":
                for b in ev.get("message", {}).get("content", []):
                    if b.get("type") == "text" and b.get("text", "").strip():
                        on_event("info", _short(b["text"].strip(), 400))
                    elif b.get("type") == "tool_use":
                        on_event("info", f"→ {b.get('name')}: {_short(b.get('input'))}")
                        _record_tool(result, b.get("name"), b.get("input"))
            elif t == "result":
                result["text"] = ev.get("result", "") or ""
                u = ev.get("usage", {}) or {}
                result["tokens_in"] = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                result["tokens_out"] = u.get("output_tokens") or 0
                result["cost"] = ev.get("total_cost_usd") or 0.0
                result["turns"] = ev.get("num_turns") or 0
                if ev.get("is_error"):
                    result["error"] = result["text"] or "agent reported an error"
                elif quota_error(result["text"]):
                    # Not an answer — the plan is exhausted. Surface it as an
                    # error so the caller abstains loudly instead of recording a
                    # billing notice as the agent's opinion.
                    result["error"] = f"Claude plan limit reached: {result['text'].strip()}"
                    result["text"] = ""
                    on_event("error", result["error"])
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        result["error"] = "agent timed out"
        on_event("error", result["error"])
    finally:
        watchdog.cancel()
    if watchdog_fired.is_set() and not result["error"]:
        result["error"] = f"agent timed out after {timeout}s (watchdog)"
        on_event("error", result["error"])
    if proc.returncode not in (0, None) and not result["error"]:
        result["error"] = f"claude exited with code {proc.returncode}"
    return result
