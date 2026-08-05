"""The pluggable agent-backend layer: adapter contract, registry dispatch,
availability detection, output parsing per tool, and fail-open behavior."""

import json
import os
import stat

import pytest
from app.services import agent_backends, providers

REQUIRED_KEYS = {"text", "tokens_in", "tokens_out", "cost", "error"}


def _fake_cli(tmp_path, name: str, script: str) -> str:
    """Drop an executable fake CLI on disk and return its path.

    Shebang dispatch is POSIX-only, so the handful of tests that stand up a fake
    binary skip on Windows rather than being rewritten around .bat shims. The
    adapters themselves are portable — they only ever ``subprocess.run`` a
    resolved path — and every other test in this module covers them without
    executing anything.
    """
    if os.name == "nt":
        pytest.skip("fake CLIs rely on shebang dispatch (POSIX only)")
    p = tmp_path / name
    # NB: not sys.executable — a shebang path with spaces (this repo's venv) is invalid.
    p.write_text(f"#!/usr/bin/env python3\n{script}")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


# --- registry / contract -------------------------------------------------------

def test_all_six_backends_registered():
    assert set(agent_backends.BACKENDS) == {
        "claude-code", "codex", "cursor", "aider", "gemini-cli", "antigravity"}


def test_every_agent_provider_maps_to_a_registered_backend():
    for pid, p in providers.PROVIDERS.items():
        if p.kind in ("agent", "claude-cli"):
            assert providers.agent_backend(pid) in agent_backends.BACKENDS, pid


def test_auto_and_anthropic_resolve_to_claude_code():
    assert providers.agent_backend("auto") == "claude-code"
    assert providers.agent_backend("anthropic") == "claude-code"
    assert providers.agent_backend("groq") == ""  # HTTP providers are not backends


def test_unknown_backend_fails_open():
    events = []
    res = agent_backends.run("no-such-tool", ".", "hi", lambda s, m: events.append((s, m)))
    assert set(res) >= REQUIRED_KEYS
    assert "unknown agent backend" in res["error"]
    assert events  # error surfaced to the run log, not raised


def test_adapter_exception_fails_open(monkeypatch):
    b = agent_backends.BACKENDS["codex"]
    monkeypatch.setattr(type(b), "run", lambda *a, **k: 1 / 0)
    res = agent_backends.run("codex", ".", "hi", lambda s, m: None)
    assert res["error"] and "codex backend failed" in res["error"]


def test_antigravity_unavailable_with_reason():
    det = agent_backends.detect("antigravity")
    assert det["available"] is False
    assert "headless" in det["reason"]
    res = agent_backends.run("antigravity", ".", "hi", lambda s, m: None)
    assert res["error"]


def test_missing_binary_detected(monkeypatch):
    monkeypatch.setattr("app.config.settings.codex_cli_path", "definitely-not-a-real-cli")
    b = agent_backends.BACKENDS["codex"]
    b._detect_cache = None  # drop the TTL cache
    det = b.detect()
    assert det["available"] is False and "not found" in det["reason"]
    b._detect_cache = None


# --- per-tool output parsing (fake CLIs) ---------------------------------------

def test_codex_parses_jsonl_and_usage(tmp_path, monkeypatch):
    events = [
        {"type": "item.completed", "item": {"item_type": "command_execution", "command": "ls"}},
        {"type": "item.completed", "item": {"item_type": "agent_message", "text": "did the thing"}},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 20,
                                             "output_tokens": 30}},
    ]
    exe = _fake_cli(tmp_path, "codex",
                    "import sys; sys.stdin.read()\n" +
                    "\n".join(f"print({json.dumps(json.dumps(e))})" for e in events))
    monkeypatch.setattr("app.config.settings.codex_cli_path", exe)
    log = []
    res = agent_backends.run("codex", str(tmp_path), "prompt", lambda s, m: log.append(m))
    assert res["error"] is None
    assert res["text"] == "did the thing"
    assert res["tokens_in"] == 120 and res["tokens_out"] == 30
    assert res["cost"] is None  # codex reports no dollars — honest unknown


def test_cursor_parses_stream_json_without_usage(tmp_path, monkeypatch):
    events = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "working"}]}},
        {"type": "result", "result": "all done"},
    ]
    exe = _fake_cli(tmp_path, "cursor-agent",
                    "import sys; sys.stdin.read()\n" +
                    "\n".join(f"print({json.dumps(json.dumps(e))})" for e in events))
    monkeypatch.setattr("app.config.settings.cursor_cli_path", exe)
    res = agent_backends.run("cursor", str(tmp_path), "prompt", lambda s, m: None)
    assert res["error"] is None and res["text"] == "all done"
    assert res["tokens_in"] is None and res["cost"] is None  # never a fake zero


def test_aider_parses_text_meter(tmp_path, monkeypatch):
    out = ["Applied edit to foo.py",
           "Tokens: 4.2k sent, 291 received. Cost: $0.0038 message, $0.0120 session."]
    exe = _fake_cli(tmp_path, "aider",
                    "\n".join(f"print({json.dumps(l)})" for l in out))
    monkeypatch.setattr("app.config.settings.aider_cli_path", exe)
    # aider has no login of its own, so the adapter requires a provider key (or a
    # model) before it will run — set one so this exercises the meter parsing and
    # doesn't depend on the ambient env having a key (CI runs with none).
    monkeypatch.setattr("app.config.settings.openai_api_key", "sk-test")
    res = agent_backends.run("aider", str(tmp_path), "prompt", lambda s, m: None)
    assert res["error"] is None
    assert res["tokens_in"] == 4200 and res["tokens_out"] == 291
    assert res["cost"] == pytest.approx(0.012)
    assert "Applied edit" in res["text"]


def test_aider_supports_keyless_ollama_and_passes_its_endpoint(tmp_path, monkeypatch):
    exe = _fake_cli(tmp_path, "aider",
                    "import os, sys; sys.stdin.read(); print(os.environ.get('OLLAMA_API_BASE'))")
    monkeypatch.setattr("app.config.settings.aider_cli_path", exe)
    monkeypatch.setattr("app.config.settings.custom_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr("app.config.settings.custom_api_key", "")
    for name in ("openai_api_key", "anthropic_api_key", "gemini_api_key",
                 "xai_api_key", "groq_api_key"):
        monkeypatch.setattr("app.config.settings." + name, "")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                 "XAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    res = agent_backends.run("aider", str(tmp_path), "prompt", lambda s, m: None,
                             model="ollama_chat/qwen2.5-coder:7b")
    assert res["error"] is None


def test_aider_passes_remote_custom_openai_endpoint(tmp_path, monkeypatch):
    exe = _fake_cli(tmp_path, "aider",
                    "import os, sys; sys.stdin.read(); "
                    "print(os.environ.get('AIDER_OPENAI_API_BASE'))")
    monkeypatch.setattr("app.config.settings.aider_cli_path", exe)
    monkeypatch.setattr("app.config.settings.custom_base_url", "https://gateway.example/v1")
    monkeypatch.setattr("app.config.settings.custom_api_key", "custom-test-key")

    res = agent_backends.run("aider", str(tmp_path), "prompt", lambda s, m: None,
                             model="openai/provider-model")
    assert res["error"] is None
    assert "https://gateway.example/v1" in res["text"]


def test_gemini_parses_json_envelope(tmp_path, monkeypatch):
    envelope = {"response": "done here",
                "stats": {"models": {"gemini-2.5-pro": {
                    "tokens": {"prompt": 500, "cached": 100, "candidates": 60, "thoughts": 40}}}}}
    exe = _fake_cli(tmp_path, "gemini",
                    f"import sys; sys.stdin.read()\nprint({json.dumps(json.dumps(envelope))})")
    monkeypatch.setattr("app.config.settings.gemini_cli_path", exe)
    res = agent_backends.run("gemini-cli", str(tmp_path), "prompt", lambda s, m: None)
    assert res["error"] is None and res["text"] == "done here"
    assert res["tokens_in"] == 600 and res["tokens_out"] == 100
    assert res["cost"] is None


def test_nonzero_exit_becomes_error(tmp_path, monkeypatch):
    exe = _fake_cli(tmp_path, "codex", "import sys; sys.stdin.read(); sys.exit(3)")
    monkeypatch.setattr("app.config.settings.codex_cli_path", exe)
    res = agent_backends.run("codex", str(tmp_path), "prompt", lambda s, m: None)
    assert res["error"] and "exited with code 3" in res["error"]


# --- registry surface for the settings UI --------------------------------------

def test_availability_lists_every_backend():
    avail = agent_backends.availability()
    assert set(avail) == set(agent_backends.BACKENDS)
    for det in avail.values():
        assert {"available", "version", "reason"} <= set(det)


def test_stage_lists_offer_agent_backends():
    for stage in providers.STAGES:
        ids = providers.stage_provider_ids(stage)
        assert "codex" in ids and "gemini-cli" in ids and "claude-cli" in ids
    assert providers.stage_provider_ids("dev")[0] == "auto"


def test_review_can_run_a_different_backend_than_dev():
    """Cross-provider review guarantee: the review stage's selectable set is not
    tied to dev's choice — every agent backend is independently selectable."""
    dev = set(providers.stage_provider_ids("dev")) - {"auto"}
    review = set(providers.stage_provider_ids("review"))
    assert review == dev  # same catalog, independent settings fields
