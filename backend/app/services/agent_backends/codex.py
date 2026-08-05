"""OpenAI Codex CLI backend.

Headless mode: ``codex exec --json`` emits JSONL events; the prompt is passed on
stdin (``codex exec -``). File edits are allowed via the workspace-write sandbox
(``--full-auto``). Token usage comes from the ``turn.completed`` event's
``usage`` block; Codex does not report a dollar cost, so ``cost`` stays None.
"""

from __future__ import annotations

import json
import os

from ...config import settings
from .base import AgentBackend, Event, new_result, short


class CodexBackend(AgentBackend):
    id = "codex"
    name = "Codex CLI"
    connect_hint = ("Install: npm i -g @openai/codex — then run `codex login` "
                    "(ChatGPT), or set an OpenAI API key on the Providers tab.")
    install_cmd = ("npm", "install", "-g", "@openai/codex")
    install_requires = "npm"

    def executable(self) -> str:
        return settings.codex_cli_path

    def _env(self) -> dict:
        env = os.environ.copy()
        if settings.openai_api_key:
            env.setdefault("OPENAI_API_KEY", settings.openai_api_key)
        return env

    def run(self, cwd: str, prompt: str, on_event: Event, *,
            model: str | None = None, timeout: int = 1800) -> dict:
        result = new_result()
        exe = self.resolve() or self.executable()
        # --sandbox workspace-write replaces the deprecated --full-auto (edits in
        # the workspace allowed, network/other paths blocked); exec is already
        # non-interactive so no approval prompts.
        cmd = [exe, "exec", "--json", "--sandbox", "workspace-write",
               "--skip-git-repo-check", "--cd", cwd]
        # A ChatGPT-account login rejects an explicit --model (400: "not supported
        # when using Codex with a ChatGPT account") — only the account's own default
        # is allowed. So only pass --model when authenticating with an API key;
        # otherwise let Codex pick the model its plan supports.
        if model and settings.openai_api_key:
            cmd += ["--model", model]
        elif model:
            on_event("info", f"ChatGPT-account Codex: ignoring model '{model}', using the plan default")
        cmd.append("-")  # prompt on stdin (avoids arg-length limits)

        def on_line(line: str) -> None:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                on_event("info", line[:200])
                return
            # New JSONL shape: {"type": "item.completed", "item": {...}} /
            # {"type": "turn.completed", "usage": {...}}.
            t = ev.get("type")
            if t == "item.completed":
                item = ev.get("item", {}) or {}
                it = item.get("item_type") or item.get("type")
                if it == "agent_message":
                    txt = (item.get("text") or "").strip()
                    if txt:
                        result["text"] = txt
                        on_event("info", short(txt, 400))
                elif it == "command_execution":
                    on_event("info", f"→ exec: {short(item.get('command', ''))}")
                elif it in ("file_change", "patch_apply"):
                    on_event("info", f"→ edit: {short(item.get('changes') or item, 200)}")
                elif it == "reasoning":
                    pass  # too chatty for the run log
                else:
                    on_event("info", short(item, 200))
            elif t == "turn.completed":
                u = ev.get("usage", {}) or {}
                tin = (u.get("input_tokens") or 0) + (u.get("cached_input_tokens") or 0)
                result["tokens_in"] = (result["tokens_in"] or 0) + tin
                result["tokens_out"] = (result["tokens_out"] or 0) + (u.get("output_tokens") or 0)
            elif t in ("turn.failed", "error"):
                result["error"] = short(ev.get("error") or ev.get("message") or "codex error", 300)
                on_event("error", result["error"])
            # Legacy protocol shape: {"msg": {"type": "agent_message", ...}}.
            elif "msg" in ev:
                msg = ev["msg"] or {}
                mt = msg.get("type")
                if mt == "agent_message" and (msg.get("message") or "").strip():
                    result["text"] = msg["message"].strip()
                    on_event("info", short(result["text"], 400))
                elif mt == "token_count":
                    info = msg.get("info") or msg
                    tu = (info.get("total_token_usage") or info) if isinstance(info, dict) else {}
                    if tu.get("input_tokens") is not None:
                        result["tokens_in"] = (tu.get("input_tokens") or 0) + \
                                              (tu.get("cached_input_tokens") or 0)
                        result["tokens_out"] = tu.get("output_tokens") or 0
                elif mt == "error":
                    result["error"] = short(msg.get("message") or "codex error", 300)
                    on_event("error", result["error"])

        self.stream(cmd, cwd, on_line, on_event, result,
                    stdin_text=prompt, timeout=timeout, env=self._env())
        if result["tokens_in"] is None:
            on_event("info", "codex did not report token usage — recorded as unknown")
        return result
