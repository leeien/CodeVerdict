"""Registry + dispatch for headless agentic coding backends.

Adding a new backend = one adapter module here + one Provider entry in
``services/providers.py``. The orchestrator never names a tool — it calls
``agent_backends.run(backend_id, ...)`` and gets the common result dict.

Fail open: an unknown or unavailable backend returns an error result (which the
orchestrator's existing fallback/INCONCLUSIVE machinery absorbs) — it never
raises into a scope run.
"""

from __future__ import annotations

from .aider import AiderBackend
from .antigravity import AntigravityBackend
from .base import AgentBackend, Event, new_result
from .claude_code import ClaudeCodeBackend
from .codex import CodexBackend
from .cursor import CursorBackend
from .gemini_cli import GeminiCliBackend

# id → adapter instance. Order is cosmetic (availability listing).
BACKENDS: dict[str, AgentBackend] = {
    b.id: b for b in (
        ClaudeCodeBackend(), CodexBackend(), CursorBackend(),
        AiderBackend(), GeminiCliBackend(), AntigravityBackend(),
    )
}


def get(backend_id: str) -> AgentBackend | None:
    return BACKENDS.get(backend_id)


def run(backend_id: str, cwd: str, prompt: str, on_event: Event, *,
        model: str | None = None, timeout: int = 1800) -> dict:
    """Dispatch one agent turn. Never raises — unknown/broken backends come back
    as an error result the pipeline already knows how to degrade around."""
    backend = BACKENDS.get(backend_id)
    if backend is None:
        res = new_result()
        res["error"] = f"unknown agent backend '{backend_id}'"
        on_event("error", res["error"])
        return res
    try:
        return backend.run(cwd, prompt, on_event, model=model, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — adapter bug must not kill a scope run
        res = new_result()
        res["error"] = f"{backend_id} backend failed: {exc}"
        on_event("error", res["error"])
        return res


def chat(backend_id: str, system: str, user: str = "", *, model: str | None = None,
         timeout: int = 180, json_mode: bool = False,
         messages: list[dict] | None = None) -> dict:
    """Pure-chat completion through an agentic CLI (for the chat stages)."""
    backend = BACKENDS.get(backend_id)
    if backend is None:
        return {"text": "", "tokens_in": None, "tokens_out": None, "cost": None,
                "model": model or "", "error": f"unknown agent backend '{backend_id}'"}
    try:
        return backend.chat(system, user, model=model, timeout=timeout,
                            json_mode=json_mode, messages=messages)
    except Exception as exc:  # noqa: BLE001
        return {"text": "", "tokens_in": None, "tokens_out": None, "cost": None,
                "model": model or "", "error": f"{backend_id} backend failed: {exc}"}


def detect(backend_id: str) -> dict:
    backend = BACKENDS.get(backend_id)
    if backend is None:
        return {"available": False, "version": "", "reason": f"unknown backend '{backend_id}'"}
    return backend.detect()


def is_available(backend_id: str) -> bool:
    return bool(detect(backend_id).get("available"))


def availability() -> dict[str, dict]:
    """id → {available, version, reason, connect_hint, installable} for every
    registered backend (cached per-backend for a minute — safe to call from the
    settings view)."""
    return {bid: {**b.detect(), "installable": b.installable()}
            for bid, b in BACKENDS.items()}


def refresh() -> dict[str, dict]:
    """Drop every detection cache and re-probe — the 'Re-check' button, for when
    the operator just installed or logged into a tool in their own terminal."""
    for b in BACKENDS.values():
        b.reset_detection()
    return availability()


def install(backend_id: str) -> dict:
    """Run a backend's official installer (admin-triggered from Settings).
    Returns {ok, output, detect}; unknown ids fail soft like everything here."""
    backend = BACKENDS.get(backend_id)
    if backend is None:
        return {"ok": False, "output": f"unknown agent backend '{backend_id}'",
                "detect": {"available": False, "version": "", "reason": "unknown backend"}}
    return backend.install()
