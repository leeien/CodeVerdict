"""Google Antigravity — registered so it shows up in the picker, but marked
unavailable: Antigravity is an IDE (agent-first VS Code fork) with no
scriptable/headless CLI to drive from a pipeline. If Google ships one, implement
``run()`` here and flip ``headless``; nothing else in the system changes."""

from __future__ import annotations

from .base import AgentBackend, Event, new_result


class AntigravityBackend(AgentBackend):
    id = "antigravity"
    name = "Antigravity"
    headless = False
    no_headless_reason = ("Antigravity is an IDE with no scriptable/headless "
                          "interface — it cannot be driven by the pipeline")

    def executable(self) -> str:
        return ""

    def run(self, cwd: str, prompt: str, on_event: Event, *,
            model: str | None = None, timeout: int = 1800) -> dict:
        result = new_result()
        result["error"] = self.no_headless_reason
        on_event("error", result["error"])
        return result
