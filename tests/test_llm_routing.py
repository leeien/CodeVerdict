"""The unified chat entry point must route each provider to the correct transport.

This is what makes the pipeline provider-agnostic: a stage configured for `anthropic`
hits the native Messages API, one for `claude-cli` drives the Claude Code CLI, and any
OpenAI-compatible provider (groq/openai/gemini/xai/custom) goes through the shared
OpenAI-compatible client. The routing is by provider *kind*, not by guessing from the
model name.
"""

import pytest
from app.services import llm, providers


@pytest.fixture
def spy_transports(monkeypatch):
    """Replace the three transports with markers so we can see which one was called."""
    calls = {}

    def make(name):
        def _fn(*args, **kwargs):
            calls["hit"] = name
            calls["provider"] = kwargs.get("provider")
            calls["model"] = kwargs.get("model")
            return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "error": None}
        return _fn

    def agent_chat(backend_id, *args, **kwargs):
        calls["hit"] = "agent"
        calls["backend"] = backend_id
        calls["model"] = kwargs.get("model")
        return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "error": None}

    monkeypatch.setattr(llm.anthropic_api, "chat", make("anthropic"))
    monkeypatch.setattr(llm.agent_backends, "chat", agent_chat)
    monkeypatch.setattr(llm.openai_agent, "chat", make("openai"))
    return calls


@pytest.mark.parametrize(
    "provider, expected_transport",
    [
        ("groq", "openai"),
        ("openai", "openai"),
        ("gemini", "openai"),
        ("xai", "openai"),
        ("custom", "openai"),
        ("anthropic", "anthropic"),
        ("claude-cli", "agent"),
        ("codex", "agent"),
        ("cursor-cli", "agent"),
        ("aider", "agent"),
        ("gemini-cli", "agent"),
    ],
)
def test_chat_routes_by_provider_kind(spy_transports, provider, expected_transport):
    llm.chat("sys", "user", provider=provider, model="some-model")
    assert spy_transports["hit"] == expected_transport


def test_agent_chat_receives_the_adapter_id(spy_transports):
    llm.chat("sys", "user", provider="claude-cli", model="sonnet")
    assert spy_transports["backend"] == "claude-code"
    llm.chat("sys", "user", provider="codex", model="gpt-5.1-codex")
    assert spy_transports["backend"] == "codex"


def test_openai_kind_forwards_provider_and_model(spy_transports):
    # The OpenAI-compatible transport needs the provider id to resolve the endpoint.
    llm.chat("sys", "user", provider="gemini", model="gemini-2.5-flash")
    assert spy_transports["provider"] == "gemini"
    assert spy_transports["model"] == "gemini-2.5-flash"


def test_every_registered_openai_provider_has_an_endpoint():
    # A provider is only usable if it resolves to a base URL — guard against a
    # registry entry that would silently route nowhere.
    for pid, p in providers.PROVIDERS.items():
        if p.kind != "openai":
            continue
        base, _key = providers.endpoint(pid)
        # `custom` is legitimately blank until the operator sets its base URL.
        if pid == "custom":
            continue
        assert base, f"{pid} has no base URL"


def test_stages_expose_multiple_providers():
    # Provider-agnostic per stage: every stage offers more than one choice.
    for stage in providers.STAGES:
        assert len(providers.stage_provider_ids(stage)) >= 2
