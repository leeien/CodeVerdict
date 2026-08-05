"""Unified chat entry point for the pipeline stages.

Dispatches a (provider, model) chat call to the right transport:
  * ``anthropic`` provider  → the native Anthropic Messages API (anthropic_api.chat)
  * ``claude-cli``/``agent`` kinds → the matching headless agentic CLI, in
    pure-chat mode (agent_backends.chat)
  * everything else (OpenAI-compatible: groq/openai/gemini/xai/custom) → openai_agent.chat

Callers pass their stage's ``settings.<stage>_provider`` + ``settings.<stage>_model``.
All transports return the same dict: {text, tokens_in, tokens_out, cost, model, error}.
"""

from __future__ import annotations

from . import agent_backends, anthropic_api, openai_agent, providers

_USAGE_KEYS = ("tokens_in", "tokens_out", "cost")


def carry_usage(result: dict, *previous: dict | None) -> dict:
    """Fold earlier attempts' usage into the attempt that is being kept.

    A call that failed still burned tokens — the provider read the whole prompt
    before it timed out, rate-limited, or returned something unparseable. Every
    retry in this pipeline replaces the result dict wholesale, so without this
    the failed attempt's spend silently vanishes from the cost meter and the
    reported cost of a delivery is lower than the bill. Retries are exactly when
    a run gets expensive, which is exactly when the number has to be true.

    Mutates and returns ``result`` so it can wrap a retry call in place.
    """
    for prev in previous:
        if not prev:
            continue
        for key in _USAGE_KEYS:
            result[key] = (result.get(key) or 0) + (prev.get(key) or 0)
    return result


def chat(system: str, user: str = "", *, provider: str, model: str, timeout: int = 180,
         json_mode: bool = False, messages: list[dict] | None = None,
         cache_prefix: str = "") -> dict:
    """Dispatch one chat call to the right transport.

    ``cache_prefix`` is prompt text the caller knows is stable across a burst of
    calls (the case file every juror reviews; the work order a revise round
    re-sends). It is always placed BEFORE ``user`` — that ordering is what earns
    the discount, and it is the same answer for every transport even though they
    reach it differently:

      * Anthropic wants an explicit ``cache_control`` breakpoint, so the prefix
        is handed over as its own content block.
      * OpenAI, Groq, Gemini and DeepSeek cache automatically, with no parameter
        to send — they discount the longest matching *prefix*, so simply keeping
        the stable text first is the entire optimisation.
      * The agentic CLIs build their own requests; their caching is internal and
        not ours to drive. Concatenating is the correct no-op.
    """
    if providers.kind(provider) in ("claude-cli", "agent"):
        return agent_backends.chat(providers.agent_backend(provider), system,
                                   cache_prefix + user, model=model, timeout=timeout,
                                   json_mode=json_mode, messages=messages)
    if providers.kind(provider) == "anthropic":
        return anthropic_api.chat(system, user, model=model, timeout=timeout,
                                  json_mode=json_mode, messages=messages,
                                  cache_prefix=cache_prefix)
    return openai_agent.chat(system, cache_prefix + user, provider=provider, model=model,
                             timeout=timeout, json_mode=json_mode, messages=messages)
