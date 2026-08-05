"""Cursor CLI (``cursor-agent``) backend.

Headless mode: ``cursor-agent -p --output-format stream-json`` emits
Claude-Code-style JSONL (``assistant`` message events + a final ``result``).
``--force`` allows non-interactive file edits. Cursor's stream does not report
token usage or dollar cost, so both stay None (honest unknown).
"""

from __future__ import annotations

import json
import os

from ...config import settings
from .base import AgentBackend, Event, new_result, short


class CursorBackend(AgentBackend):
    id = "cursor"
    name = "Cursor CLI"
    connect_hint = ("Install the Cursor CLI (cursor-agent), then run "
                    "`cursor-agent login` — or set a Cursor API key on the Providers tab.")
    # Cursor's official installer script (per cursor.com/docs/cli).
    install_cmd = ("bash", "-lc", "curl -fsS https://cursor.com/install | bash")
    install_requires = "curl"

    def executable(self) -> str:
        return settings.cursor_cli_path

    def _env(self) -> dict:
        env = os.environ.copy()
        if settings.cursor_api_key:
            env.setdefault("CURSOR_API_KEY", settings.cursor_api_key)
        return env

    def run(self, cwd: str, prompt: str, on_event: Event, *,
            model: str | None = None, timeout: int = 1800) -> dict:
        result = new_result()
        exe = self.resolve() or self.executable()
        cmd = [exe, "-p", "--output-format", "stream-json", "--force"]
        if model:
            cmd += ["--model", model]

        def on_line(line: str) -> None:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                on_event("info", line[:200])
                return
            t = ev.get("type")
            if t == "assistant":
                for b in (ev.get("message", {}) or {}).get("content", []) or []:
                    if b.get("type") == "text" and (b.get("text") or "").strip():
                        on_event("info", short(b["text"].strip(), 400))
                    elif b.get("type") == "tool_use":
                        on_event("info", f"→ {b.get('name')}: {short(b.get('input'))}")
            elif t == "tool_call":
                on_event("info", f"→ {short(ev.get('subtype') or ev.get('name') or 'tool', 60)}")
            elif t == "result":
                result["text"] = (ev.get("result") or "").strip()
                u = ev.get("usage") or {}
                if u.get("input_tokens") is not None:
                    result["tokens_in"] = (u.get("input_tokens") or 0) + \
                                          (u.get("cache_read_input_tokens") or 0)
                    result["tokens_out"] = u.get("output_tokens") or 0
                if ev.get("total_cost_usd") is not None:
                    result["cost"] = ev.get("total_cost_usd")
                if ev.get("is_error") or ev.get("subtype") == "error":
                    result["error"] = result["text"] or "cursor-agent reported an error"

        self.stream(cmd, cwd, on_line, on_event, result,
                    stdin_text=prompt, timeout=timeout, env=self._env())
        if result["cost"] is None:
            on_event("info", "cursor-agent did not report cost — recorded as unknown")
        return result
