"""Google Gemini CLI backend.

Headless mode: ``gemini --output-format json`` with the prompt on stdin runs one
non-interactive turn; ``--yolo`` auto-approves file edits and commands. The JSON
envelope's ``stats`` block carries real token counts. Gemini CLI does not report
a dollar cost, so ``cost`` stays None.
"""

from __future__ import annotations

import json
import os

from ...config import settings
from .base import AgentBackend, Event, new_result, short


class GeminiCliBackend(AgentBackend):
    id = "gemini-cli"
    name = "Gemini CLI"
    connect_hint = ("Install: npm i -g @google/gemini-cli — then run `gemini` once to "
                    "log in, or set a Gemini API key on the Providers tab.")
    install_cmd = ("npm", "install", "-g", "@google/gemini-cli")
    install_requires = "npm"

    def executable(self) -> str:
        return settings.gemini_cli_path

    def _env(self) -> dict:
        env = os.environ.copy()
        if settings.gemini_api_key:
            env.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
        # Gemini CLI 0.52+ refuses to run in an "untrusted" folder and exits 55
        # in headless mode. We drive it in isolated agent workspaces, so opt in.
        env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
        return env

    def run(self, cwd: str, prompt: str, on_event: Event, *,
            model: str | None = None, timeout: int = 1800) -> dict:
        result = new_result()
        exe = self.resolve() or self.executable()
        # --skip-trust: same trust bypass as the env var above (belt-and-suspenders
        # across Gemini CLI versions) — without it a non-interactive run aborts 55.
        cmd = [exe, "--output-format", "json", "--yolo", "--skip-trust"]
        if model:
            cmd += ["--model", model]

        # JSON mode prints ONE envelope at the end (no streaming events), so
        # buffer stdout and parse after the process exits.
        lines: list[str] = []
        on_event("info", "gemini-cli running (single JSON envelope — no live stream)")
        self.stream(cmd, cwd, lines.append, on_event, result,
                    stdin_text=prompt, timeout=timeout, env=self._env())
        raw = "\n".join(lines).strip()
        if not raw:
            if not result["error"]:
                result["error"] = "gemini-cli produced no output"
                on_event("error", result["error"])
            return result
        # The envelope is the last JSON object in stdout (startup noise may precede it).
        start = raw.find("{")
        try:
            env_obj = json.loads(raw[start:]) if start != -1 else {}
        except json.JSONDecodeError:
            # Not JSON after all — treat the raw text as the answer.
            result["text"] = raw
            on_event("info", short(raw, 400))
            return result
        result["text"] = (env_obj.get("response") or "").strip()
        if result["text"]:
            on_event("info", short(result["text"], 400))
        err = env_obj.get("error")
        if err and not result["error"]:
            result["error"] = short(err.get("message") if isinstance(err, dict) else err, 300)
            on_event("error", result["error"])
        # stats.models.<model-id>.tokens.{prompt,candidates,cached,...}
        tin = tout = 0
        seen = False
        for m in ((env_obj.get("stats") or {}).get("models") or {}).values():
            tok = (m or {}).get("tokens") or {}
            if tok:
                seen = True
                tin += (tok.get("prompt") or 0) + (tok.get("cached") or 0)
                tout += (tok.get("candidates") or 0) + (tok.get("thoughts") or 0)
        if seen:
            result["tokens_in"], result["tokens_out"] = tin, tout
        else:
            on_event("info", "gemini-cli did not report token usage — recorded as unknown")
        return result
