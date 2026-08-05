"""Claude Code CLI backend — thin wrapper over the battle-tested
``services/claude_agent.py`` runner (the reference implementation of the
adapter contract)."""

from __future__ import annotations

from ...config import settings
from .. import claude_agent
from .base import AgentBackend, Event


class ClaudeCodeBackend(AgentBackend):
    id = "claude-code"
    name = "Claude Code"
    connect_hint = ("Install: npm i -g @anthropic-ai/claude-code — then run `claude` "
                    "once to log in, or set an Anthropic API key on the Providers tab.")
    install_cmd = ("npm", "install", "-g", "@anthropic-ai/claude-code")
    install_requires = "npm"

    def executable(self) -> str:
        return settings.claude_cli_path

    def run(self, cwd: str, prompt: str, on_event: Event, *,
            model: str | None = None, timeout: int = 1800) -> dict:
        return claude_agent.run_claude(cwd, prompt, on_event, model=model, timeout=timeout)

    def chat(self, system: str, user: str = "", *, model: str | None = None,
             timeout: int = 180, json_mode: bool = False,
             messages: list[dict] | None = None) -> dict:
        return claude_agent.chat(system, user, model=model, timeout=timeout,
                                 json_mode=json_mode, messages=messages)
