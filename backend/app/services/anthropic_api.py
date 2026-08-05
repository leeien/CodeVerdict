"""Anthropic native Messages API adapter for the pure-chat pipeline stages
(knowledge, pm, qa, review) when their provider is set to ``anthropic``.

Mirrors ``openai_agent.chat``'s call shape and return dict so ``llm.chat`` can
dispatch to either transport transparently. Anthropic *coding* (dev/review) runs
through the Claude Code CLI instead — see services/claude_agent.py.
"""

from __future__ import annotations

from ..config import settings

# List prices ($ per 1M tokens: input, output) for the cost meter. Unknown models
# fall back to Sonnet-tier pricing.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
_DEFAULT_PRICE = (3.0, 15.0)

# Prompt-cache multipliers on the input price. Writing a cache entry costs a
# premium; reading one is the whole point.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.10

# Below this, a cache breakpoint is not worth the write premium — Anthropic will
# not cache a short prefix anyway (1024 tokens for Sonnet/Opus, 2048 for Haiku),
# so the write is pure overhead. Chars, not tokens, because that is what the
# callers budget in; ~4 chars/token puts this comfortably above the larger floor.
_CACHE_MIN_CHARS = 9000

_JSON_NUDGE = (
    "\n\nRespond with a single valid JSON object and NOTHING else — no prose, no "
    "markdown fences."
)


def _system_blocks(sys_prompt: str, cache: bool) -> list[dict] | str:
    """The system prompt, marked as a cache breakpoint when it is worth caching.

    Every stage sends the same system prompt on every call, and the jury sends
    the same one once per juror per round, so this is the cheapest breakpoint
    available: it is stable by construction.
    """
    if not cache or len(sys_prompt) < _CACHE_MIN_CHARS:
        return sys_prompt
    return [{"type": "text", "text": sys_prompt,
             "cache_control": {"type": "ephemeral"}}]


def chat(system: str, user: str = "", *, model: str | None = None, timeout: int = 180,
         json_mode: bool = False, messages: list[dict] | None = None,
         cache_prefix: str = "") -> dict:
    """One Anthropic Messages API completion. Returns
    {text, tokens_in, tokens_out, cost, model, error} — same shape as openai_agent.chat.
    `messages` (multi-turn) overrides `user`. The Anthropic SDK is imported lazily so
    the module loads even if the SDK isn't installed and no Anthropic stage is used.

    ``cache_prefix`` is content the caller knows is stable across a burst of
    calls — the shared case file every juror reviews, the work order a revise
    round re-sends. It is sent as its own content block with a cache breakpoint,
    so the second and later calls read it at a tenth of the input price instead
    of paying full freight for text the provider has already seen. Callers that
    pass nothing behave exactly as before.
    """
    m = model or "claude-sonnet-5"
    if not settings.anthropic_api_key:
        return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "model": m,
                "error": "ANTHROPIC_API_KEY not configured — skipping " + m}
    try:
        import anthropic  # lazy: only needed when an Anthropic stage runs
    except ImportError:
        return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "model": m,
                "error": "anthropic SDK not installed (pip install anthropic)"}

    sys_prompt = system + (_JSON_NUDGE if json_mode else "")
    cache = bool(cache_prefix.strip()) and len(cache_prefix) >= _CACHE_MIN_CHARS
    if messages is not None:
        msgs = messages
    elif cache:
        # Two blocks: the stable prefix (cached) then this call's own tail. The
        # breakpoint has to sit at the END of the shared text — Anthropic caches
        # a prefix, so anything that varies must come after it.
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": cache_prefix,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": user or "(see above)"},
        ]}]
    else:
        msgs = [{"role": "user", "content": (cache_prefix + user) if cache_prefix else user}]
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=float(timeout))
        resp = client.messages.create(
            model=m, max_tokens=8000, system=_system_blocks(sys_prompt, cache), messages=msgs,
        )
    except anthropic.APIStatusError as exc:  # 4xx/5xx with a response
        return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "model": m,
                "error": f"Anthropic {exc.status_code}: {str(exc)[:200]}"}
    except Exception as exc:  # noqa: BLE001 — connection/timeout/etc.
        return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "model": m,
                "error": str(exc)}

    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    usage = resp.usage
    fresh = getattr(usage, "input_tokens", 0) or 0
    cached_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cached_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    tin = fresh + cached_read + cached_write
    tout = getattr(usage, "output_tokens", 0) or 0
    pin, pout = _PRICES.get(m, _DEFAULT_PRICE)
    # Cached input is not priced like fresh input, and reporting it as if it were
    # would hide the saving this feature exists to produce — a cache that works
    # would look identical to one that never hit.
    cost = round(
        (fresh
         + cached_read * _CACHE_READ_MULTIPLIER
         + cached_write * _CACHE_WRITE_MULTIPLIER) / 1_000_000 * pin
        + tout / 1_000_000 * pout, 4)
    return {"text": text, "tokens_in": tin, "tokens_out": tout, "cost": cost,
            "model": m, "error": None,
            "cached_read": cached_read, "cached_write": cached_write}
