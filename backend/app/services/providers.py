"""Provider registry — the single source of truth for which LLM providers exist,
where they live, which key they use, and which models they offer.

Each pipeline stage (knowledge, pm, planner, dev, qa, review) stores a
``<stage>_provider`` + ``<stage>_model`` in config; routing resolves the endpoint
from the provider id here instead of guessing from the model name. A provider has
one of three "kinds":

  * ``openai``     — an OpenAI-compatible HTTP endpoint (groq, openai, gemini, xai,
                     custom). Served by ``openai_agent.chat``/``code``.
  * ``anthropic``  — Anthropic's native Messages API. Served by ``anthropic_api.chat``.
                     Used for the pure-chat stages (knowledge, pm, qa, review).
  * ``claude-cli`` — the Claude Code CLI (``claude_agent.run_claude``). The strongest
                     coding path; offered for the dev/review stages only.
  * ``agent``      — any other headless agentic coding tool (Codex CLI, Cursor CLI,
                     Aider, Gemini CLI, …). Served by the adapter registry in
                     ``services/agent_backends`` — the ``backend`` field names the
                     adapter. Adding a tool = one adapter module + one entry here;
                     the orchestrator dispatches on kind and never names a tool.

Anthropic *coding* (dev/review) intentionally runs through the CLI, so those two
stages expose ``claude-cli`` rather than the ``anthropic`` API provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from ..config import settings

STAGES: tuple[str, ...] = ("knowledge", "pm", "planner", "dev", "qa", "review")


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    kind: str                       # "openai" | "anthropic" | "claude-cli" | "agent"
    key_field: str                  # settings attr holding the API key
    base_url: str = ""              # static base URL (openai-kind, when no field)
    base_url_field: str = ""        # settings attr holding the base URL (overrides base_url)
    models: tuple[str, ...] = ()    # catalog offered in the UI (free text still allowed)
    default_model: str = ""         # used by the "apply to all stages" preset
    note: str = ""                  # short UI hint
    backend: str = ""               # agent_backends adapter id (kind "agent"/"claude-cli")
    path_field: str = ""            # runtime-editable settings attr for the CLI binary path


# The registry. Order here is the UI order.
PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        "groq", "Groq (free tier)", "openai", "groq_api_key",
        base_url="https://api.groq.com/openai/v1",
        models=(
            "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ),
        default_model="openai/gpt-oss-120b",
        note="Fast, free-tier OpenAI-compatible endpoint. Tight per-minute token budgets.",
    ),
    "openai": Provider(
        "openai", "OpenAI (or primary endpoint)", "openai", "openai_api_key",
        base_url_field="openai_base_url",
        # Fallback list only — OpenAI's ids move fast. The picker's "Refresh
        # model lists" pulls the live set once a key is set.
        models=("gpt-5.1", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4o-mini", "o3", "o4-mini"),
        default_model="gpt-5.1",
        note="The 'primary' endpoint — base URL configurable (points at Groq out of the "
             "box). Set it to https://api.openai.com/v1 for real OpenAI. Models refresh live.",
    ),
    "gemini": Provider(
        "gemini", "Google Gemini", "openai", "gemini_api_key",
        base_url_field="gemini_base_url",
        # Prefer the never-stale "-latest" aliases as defaults; the exact-version
        # ids are just a starting list — the picker refreshes them live from the
        # provider (fetch_models). Verified present in the live /models response.
        models=(
            "gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash",
            "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
        ),
        default_model="gemini-flash-latest",
        note="Google AI Studio via its OpenAI-compatible endpoint. Models refresh live; "
             "the '-latest' aliases never go stale.",
    ),
    "xai": Provider(
        "xai", "xAI (Grok)", "openai", "xai_api_key",
        base_url="https://api.x.ai/v1",
        models=("grok-4.5",),
        default_model="grok-4.5",
        note="xAI Grok, OpenAI-compatible. Model ids change over time — models refresh live.",
    ),
    "anthropic": Provider(
        "anthropic", "Anthropic (API)", "anthropic", "anthropic_api_key",
        models=("claude-fable-5", "claude-opus-5", "claude-sonnet-5",
                "claude-haiku-4-5"),
        default_model="claude-sonnet-5",
        note="Anthropic's native Messages API (used for the chat stages). For coding, pick "
             "'Claude Code (CLI)' on the Dev/Review stages instead.",
    ),
    "claude-cli": Provider(
        "claude-cli", "Claude Code (agentic CLI)", "claude-cli", "anthropic_api_key",
        models=("sonnet", "opus", "haiku"),
        default_model="sonnet",
        note="The Claude Code CLI — real agentic file editing. The strongest coding path; "
             "Dev/Review only. Uses the host's Claude login when no API key is set.",
        backend="claude-code",
    ),
    "codex": Provider(
        "codex", "Codex CLI (agentic)", "agent", "openai_api_key",
        models=("gpt-5.1-codex-max", "gpt-5.1-codex", "gpt-5-codex", "gpt-5.1"),
        # No forced default: the explicit '-codex' models are only valid with an
        # OpenAI API key — a ChatGPT-account login rejects them (400). Leaving the
        # model unset lets Codex auto-pick one its current auth supports.
        default_model="",
        note="OpenAI's Codex CLI in headless mode (codex exec). Uses the host's ChatGPT "
             "login when no OpenAI API key is set — leave the model blank on a ChatGPT "
             "login; the '-codex' models need an API key. Reports tokens but not cost.",
        backend="codex", path_field="codex_cli_path",
    ),
    "cursor-cli": Provider(
        "cursor-cli", "Cursor CLI (agentic)", "agent", "cursor_api_key",
        models=("sonnet-4.5", "opus-4.5", "gpt-5.1", "composer-1"),
        default_model="sonnet-4.5",
        note="Cursor's cursor-agent in headless print mode. Uses the host's cursor-agent "
             "login when no CURSOR_API_KEY is set. Does not report usage (shown as unknown).",
        backend="cursor", path_field="cursor_cli_path",
    ),
    "aider": Provider(
        "aider", "Aider (agentic)", "agent", "",
        models=("sonnet", "opus", "haiku", "gpt-5.1", "gpt-4.1", "o3",
                "gemini/gemini-2.5-pro", "groq/llama-3.3-70b-versatile",
                "deepseek/deepseek-chat"),
        default_model="sonnet",
        note="Aider in one-shot --message mode. Supports API models plus local Ollama "
             "(`ollama_chat/<model>`) and LM Studio (`lm_studio/<model>`) models; enter "
             "local ids under Custom model. "
             "Auto-commits are disabled — the pipeline owns git history.",
        backend="aider", path_field="aider_cli_path",
    ),
    "gemini-cli": Provider(
        "gemini-cli", "Gemini CLI (agentic)", "agent", "gemini_api_key",
        models=("gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash",
                "gemini-3.5-flash-lite", "gemini-2.5-pro"),
        default_model="gemini-flash-latest",
        note="Google's Gemini CLI in non-interactive --yolo mode. Uses the Gemini key (or "
             "the host's Google login). Reports tokens but not cost.",
        backend="gemini-cli", path_field="gemini_cli_path",
    ),
    "antigravity": Provider(
        "antigravity", "Antigravity (no headless mode)", "agent", "",
        models=(),
        default_model="",
        note="Google Antigravity is an IDE without a scriptable/headless interface — "
             "listed for completeness, always unavailable for pipeline runs.",
        backend="antigravity",
    ),
    "custom": Provider(
        "custom", "Custom (OpenAI-compatible)", "openai", "custom_api_key",
        base_url_field="custom_base_url",
        models=(),
        default_model="",
        note="Any OpenAI-compatible endpoint (OpenRouter, Together, DeepSeek, Ollama, …). "
             "Set the base URL and type the model id; API keys are optional for local hosts.",
    ),
}

# Which providers each stage may select. Dev/Review get the agentic CLIs (+ dev
# gets 'auto'); the pure-chat stages get the API providers plus the agentic CLIs
# (pure-chat invocation in a scratch dir).
_AGENT_PROVIDERS = ["claude-cli", "codex", "cursor-cli", "aider", "gemini-cli", "antigravity"]
_CHAT_PROVIDERS = ["groq", "openai", "gemini", "xai", "anthropic", "custom"]
_CODE_PROVIDERS = _AGENT_PROVIDERS + ["groq", "openai", "gemini", "xai", "custom"]


def provider_ids() -> list[str]:
    return list(PROVIDERS.keys())


def stage_provider_ids(stage: str) -> list[str]:
    if stage == "dev":
        return ["auto"] + _CODE_PROVIDERS
    if stage == "review":
        return list(_CODE_PROVIDERS)
    # Chat stages (knowledge/pm/planner/qa) may also run through any agentic CLI
    # (pure-chat invocation via agent_backends.chat) — for Claude, the only path
    # when no ANTHROPIC_API_KEY is set. The Planner in particular is a natural fit:
    # it is read-only and reaches the same index tools through the shim.
    return list(_CHAT_PROVIDERS) + _AGENT_PROVIDERS


def kind(provider_id: str) -> str:
    p = PROVIDERS.get(provider_id)
    return p.kind if p else "openai"


def is_cli(provider_id: str) -> bool:
    # 'auto' and 'anthropic' both resolve to the Claude CLI on the coding stages.
    return provider_id in ("claude-cli", "auto", "anthropic")


def agent_backend(provider_id: str) -> str:
    """agent_backends adapter id for a coding-stage provider, or '' for the plain
    HTTP providers. 'auto' and 'anthropic' resolve to the Claude CLI (the coding
    stages' Claude path), matching the old is_cli() behavior."""
    if is_cli(provider_id):
        return "claude-code"
    p = PROVIDERS.get(provider_id)
    return p.backend if p and p.kind == "agent" else ""


def endpoint(provider_id: str) -> tuple[str, str]:
    """(base_url, api_key) for an OpenAI-compatible provider, read live from settings."""
    p = PROVIDERS.get(provider_id)
    if p is None:
        return "", ""
    base = getattr(settings, p.base_url_field, "") if p.base_url_field else p.base_url
    key = getattr(settings, p.key_field, "") if p.key_field else ""
    base = base or p.base_url
    # Local OpenAI-compatible servers commonly do not require authentication.
    # Return a harmless placeholder so the shared client can still send its
    # normal Authorization header without ever treating a remote keyless endpoint
    # as configured.
    return base, key or ("local" if is_local_endpoint(base) else "")


def is_local_endpoint(base_url: str) -> bool:
    """Whether an endpoint is local enough to permit keyless operation."""
    try:
        host = (urlsplit(base_url).hostname or "").lower()
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1", "host.docker.internal"} \
        or host.endswith(".localhost")


def has_key(provider_id: str) -> bool:
    p = PROVIDERS.get(provider_id)
    if p is None or not p.key_field:
        return False
    key = bool(getattr(settings, p.key_field, ""))
    if key or p.kind != "openai":
        return key
    base, _ = endpoint(provider_id)
    return bool(base and is_local_endpoint(base))


def can_chat(provider_id: str) -> bool:
    """Whether a chat stage pointed at `provider_id` can actually run: an
    API-kind provider with its key set, or an agentic CLI that is installed
    (CLIs may run on the host's login with no key at all). This — not
    ``settings.openai_api_key`` — is the correct gate for stage-routed LLM
    work; checking one hardcoded key contradicts the per-stage registry."""
    k = kind(provider_id)
    if k in ("claude-cli", "agent"):
        from . import agent_backends  # local import — avoids a module cycle

        return agent_backends.is_available(agent_backend(provider_id))
    return has_key(provider_id)


def fetch_models(provider_id: str) -> list[str]:
    """Live model ids from the provider's own API, so the picker never shows a
    model that no longer exists. Never raises — returns [] on any failure.

    OpenAI-compatible providers (openai/groq/gemini/xai/custom) expose
    ``GET {base}/models``; Anthropic its own ``/v1/models``. Agent-CLI backends
    don't have a model API, so they fall back to their static aliases.
    """
    import httpx

    p = PROVIDERS.get(provider_id)
    if p is None:
        return []
    try:
        if p.kind == "anthropic":
            key = getattr(settings, p.key_field, "") if p.key_field else ""
            if not key:
                return []
            r = httpx.get("https://api.anthropic.com/v1/models",
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                          timeout=15)
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", []) if m.get("id")]
        if p.kind == "openai":
            base, key = endpoint(provider_id)
            if not base or (not key and not is_local_endpoint(base)):
                return []
            r = httpx.get(f"{base.rstrip('/')}/models",
                          headers={"Authorization": f"Bearer {key}"}, timeout=15)
            if r.status_code == 404 and "11434" in base:
                # Ollama's native catalog is useful when its OpenAI-compatible
                # /v1/models route is unavailable.
                root = base.split("/v1", 1)[0].rstrip("/")
                r = httpx.get(f"{root}/api/tags", timeout=15)
                r.raise_for_status()
                return sorted({m["name"] for m in r.json().get("models", []) if m.get("name")})
            r.raise_for_status()
            # Gemini's OpenAI-compat endpoint prefixes ids with "models/".
            ids = [(m.get("id") or "").split("/")[-1] for m in r.json().get("data", [])]
            return sorted({i for i in ids if i})
    except Exception:  # noqa: BLE001 — a live fetch failure must never break Settings
        return []
    return list(p.models)  # agent CLIs: their static aliases


def label(provider_id: str, model: str = "") -> str:
    """Human label for run rows, e.g. 'xai grok-4' or 'claude-cli sonnet'."""
    name = provider_id or "?"
    return f"{name} {model}".strip()


def stage_provider(stage: str) -> str:
    return getattr(settings, f"{stage}_provider", "openai")


def stage_model(stage: str) -> str:
    return getattr(settings, f"{stage}_model", "")


# Recommended per-stage model when the operator applies a whole-provider preset.
# For anthropic, the chat stages use the API and dev/review flip to the CLI.
_ANTHROPIC_PRESET = {
    "knowledge": ("anthropic", "claude-haiku-4-5"),
    "pm": ("anthropic", "claude-opus-4-8"),
    # The Planner decides where every edit lands; a weak model here is paid for
    # by every downstream stage, so it gets the preset's strongest reasoner.
    "planner": ("anthropic", "claude-opus-4-8"),
    "dev": ("claude-cli", "sonnet"),
    "qa": ("anthropic", "claude-sonnet-5"),
    "review": ("claude-cli", "opus"),
}


def preset_values(provider_id: str) -> dict[str, str]:
    """All ``<stage>_provider`` / ``<stage>_model`` values for 'apply this provider to
    every stage'. Raises ValueError for an unknown provider."""
    if provider_id == "anthropic":
        out: dict[str, str] = {}
        for stage, (pid, model) in _ANTHROPIC_PRESET.items():
            out[f"{stage}_provider"] = pid
            out[f"{stage}_model"] = model
        # The jury's foreperson decides; give it the preset's strongest model.
        out["jury_synthesis_provider"] = "anthropic"
        out["jury_synthesis_model"] = "claude-opus-4-8"
        return out
    p = PROVIDERS.get(provider_id)
    if p is None or provider_id == "claude-cli":
        # claude-cli can't drive the chat stages (knowledge/pm/qa), so it isn't a
        # valid whole-system preset; use 'anthropic' for an all-Claude setup.
        raise ValueError(f"'{provider_id}' is not a valid apply-to-all provider")
    out = {}
    for stage in STAGES:
        out[f"{stage}_provider"] = provider_id
        out[f"{stage}_model"] = p.default_model
    out["jury_synthesis_provider"] = provider_id
    out["jury_synthesis_model"] = p.default_model
    return out


def preset_provider_ids() -> list[str]:
    """Providers offerable in the 'apply to all stages' dropdown."""
    return ["anthropic", "groq", "openai", "gemini", "xai", "custom"]
