"""Aider backend.

Headless mode: ``aider --message-file <f> --yes-always`` runs one non-interactive
turn and edits files in the repo. Auto-commits are disabled — this pipeline owns
the git history (Dev output is committed by git_ops, same as Claude Code).

Aider prints its own usage meter as plain text ("Tokens: 4.2k sent, 291
received. Cost: $0.0038 message, $0.012 session."); we parse that. When the
lines never appear (e.g. a free/local model), usage stays None.
"""

from __future__ import annotations

import os
import re
import tempfile
from urllib.parse import urlsplit

from ...config import settings
from .base import AgentBackend, Event, new_result, short

_NUM = r"([\d.,]+)\s*(k|m)?"
_TOKENS_RE = re.compile(rf"Tokens:\s*{_NUM}\s*sent.*?{_NUM}\s*received", re.IGNORECASE)
_COST_RE = re.compile(r"Cost:\s*\$([\d.,]+)\s*message,\s*\$([\d.,]+)\s*session", re.IGNORECASE)


def _is_local_url(value: str) -> bool:
    try:
        host = (urlsplit(value).hostname or "").lower()
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1", "host.docker.internal"} \
        or host.endswith(".localhost")


def _num(value: str, suffix: str | None) -> int:
    n = float(value.replace(",", ""))
    if (suffix or "").lower() == "k":
        n *= 1_000
    elif (suffix or "").lower() == "m":
        n *= 1_000_000
    return int(n)


class AiderBackend(AgentBackend):
    id = "aider"
    name = "Aider"
    connect_hint = ("Install: pip install aider-chat — then set the API key for its "
                    "model (e.g. OpenAI or Anthropic) in the environment.")
    # --user keeps it out of this app's venv and on the host PATH (~/.local/bin).
    install_cmd = ("python3", "-m", "pip", "install", "--user", "--upgrade", "aider-chat")
    install_requires = "python3"

    def executable(self) -> str:
        return settings.aider_cli_path

    def _env(self, model: str = "") -> dict:
        """Pass through whichever provider keys are configured — aider picks the
        right one from the model name (gpt-*→OpenAI, claude-*→Anthropic, …)."""
        env = os.environ.copy()
        for var, key in (("OPENAI_API_KEY", settings.openai_api_key),
                         ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
                         ("GEMINI_API_KEY", settings.gemini_api_key),
                         ("XAI_API_KEY", settings.xai_api_key),
                         ("GROQ_API_KEY", settings.groq_api_key),
                         ("OPENAI_API_KEY", settings.custom_api_key)):
            if key:
                env.setdefault(var, key)
        base = (settings.custom_base_url or settings.openai_base_url or "").strip()
        local = _is_local_url(base)
        lower = (model or "").lower()
        if lower.startswith(("ollama/", "ollama_chat/")):
            ollama_base = base if "11434" in base else "http://127.0.0.1:11434"
            env["OLLAMA_API_BASE"] = ollama_base.split("/v1", 1)[0].rstrip("/")
        elif lower.startswith("lm_studio/"):
            env["LM_STUDIO_API_BASE"] = base if local and base else "http://127.0.0.1:1234/v1"
            env.setdefault("LM_STUDIO_API_KEY", "dummy-api-key")
        elif lower.startswith("openai/") and settings.custom_base_url:
            # Aider's OpenAI-compatible adapter reads this variable.  This must
            # also work for remote custom gateways; local endpoints are merely
            # the special case where a dummy key is acceptable.
            env["AIDER_OPENAI_API_BASE"] = base
            if local:
                env.setdefault("OPENAI_API_KEY", "dummy-api-key")
        return env

    _KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                 "XAI_API_KEY", "GROQ_API_KEY")

    def run(self, cwd: str, prompt: str, on_event: Event, *,
            model: str | None = None, timeout: int = 1800) -> dict:
        result = new_result()
        exe = self.resolve() or self.executable()
        env = self._env(model or "")
        # Aider has no login of its own — with no provider key AND no model it
        # drops into an interactive OpenRouter OAuth browser flow and blocks for
        # minutes. Fail fast with a clear message instead of hanging the stage.
        local_model = (model or "").lower().startswith(("ollama/", "ollama_chat/", "lm_studio/"))
        if not model and not any(v in env for v in self._KEY_VARS):
            result["error"] = ("aider needs a model + a provider API key — set one on "
                               "the Providers tab (aider has no login of its own).")
            on_event("error", result["error"])
            return result
        if model and not any(v in env for v in self._KEY_VARS) and not local_model:
            result["error"] = ("aider needs a configured provider key for this model, or "
                                "a local model prefix such as ollama_chat/ or lm_studio/.")
            on_event("error", result["error"])
            return result
        # --message-file avoids arg-length limits; --yes-always answers every
        # "add file to the chat?" prompt; the pipeline owns git, so aider must
        # not commit or manage .gitignore.
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(prompt)
            msg_file = f.name
        cmd = [exe, "--message-file", msg_file, "--yes-always", "--no-gitignore",
               "--no-auto-commits", "--no-dirty-commits", "--no-attribute-author",
               "--no-check-update", "--no-show-model-warnings", "--no-pretty",
               "--no-stream"]
        if model:
            cmd += ["--model", model]

        tail: list[str] = []

        def on_line(line: str) -> None:
            tail.append(line)
            m = _TOKENS_RE.search(line)
            if m:
                result["tokens_in"] = (result["tokens_in"] or 0) + _num(m.group(1), m.group(2))
                result["tokens_out"] = (result["tokens_out"] or 0) + _num(m.group(3), m.group(4))
            c = _COST_RE.search(line)
            if c:
                # The session figure is cumulative — keep the latest, not a sum.
                result["cost"] = float(c.group(2).replace(",", ""))
            if line.strip():
                on_event("info", short(line, 300))

        try:
            self.stream(cmd, cwd, on_line, on_event, result,
                        stdin_text="", timeout=timeout, env=env)
        finally:
            try:
                os.unlink(msg_file)
            except OSError:
                pass
        # Aider has no single "final answer" event — the last chunk of output is
        # the closest thing to a summary of what it did.
        result["text"] = "\n".join(tail[-40:]).strip()
        if result["cost"] is None:
            on_event("info", "aider did not report cost — recorded as unknown")
        return result
