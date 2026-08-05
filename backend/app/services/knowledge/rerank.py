"""Reranking — the last stage, where a merged candidate pool becomes an order.

By the time retrieval reaches here, candidates have arrived from four channels
whose scores mean different things and cannot be compared: BM25 ranks, dense
cosine, graph adjacency, and lexical hits with no score at all. RRF already
fused the two ranked channels; what it cannot do is judge the ones that were
never ranked, and it has no notion that a hit in a test file is usually not the
code you change, or that a symbol nothing calls is rarely the one behaviour
flows through.

Three tiers, in the shape this codebase already uses for optional capability
(see the semantic channel): a free deterministic scorer that always runs, and
two stronger rerankers the operator can switch on.

  1. **deterministic** (default, free, ~µs) — the signals below.
  2. **LLM** (``rerank_provider``/``rerank_model``) — one batched scoring call
     over the whole pool. Off by default: it is a real call on every retrieval,
     and retrieval happens many times per pipeline run.
  3. **cross-encoder** (``pip install '.[rerank]'``) — a local relevance model.
     Strongest and slowest; needs no API key.

A failure in tier 2 or 3 falls back to tier 1 rather than to an unranked pool:
the pipeline must never trade a working order for a missing one.
"""

from __future__ import annotations

import logging
import re

from ...config import settings

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# CamelCase / snake_case / kebab → the identifier's constituent words, so a
# query saying "cell width" matches a symbol named `cell_width` or `cellWidth`.
_SPLIT_RE = re.compile(r"[_\-./]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_STOP = frozenset(["the", "and", "for", "with", "when", "that", "this", "from", "into",
                   "get", "set", "new", "add", "use", "run", "all", "any", "not"])

_DOC_EXT = (".md", ".rst", ".txt", ".yml", ".yaml", ".lock", ".cfg", ".toml", ".ini",
            ".json", ".xml", ".html", ".css")


def _terms(text: str) -> set[str]:
    out: set[str] = set()
    for word in _WORD_RE.findall(text or ""):
        low = word.lower()
        if len(low) > 2 and low not in _STOP:
            out.add(low)
        for part in _SPLIT_RE.split(word):
            p = part.lower()
            if len(p) > 2 and p not in _STOP:
                out.add(p)
    return out


def _identifier_terms(text: str) -> set[str]:
    """Only the query terms that are shaped like CODE — snake_case, CamelCase,
    CONSTANTS, --flags.

    This distinction is the whole difference between a reranker that helps and
    one that actively hurts. Ordinary English words in a prose query ("measure
    printable cell width ignoring ansi escape sequences") are not evidence about
    which symbol to edit: rewarding a name that contains one promoted
    `markup.escape` over the entire `cells.py` cluster on the strength of the
    word "escape", which is precisely the confident-but-wrong lexical hit the
    RRF dense weighting was tuned to suppress. BM25 has already weighed the
    prose, better than a bag-of-words bonus can; re-applying it here only
    double-counts the weakest channel.
    """
    out: set[str] = set()
    for word in _WORD_RE.findall(text or ""):
        deliberate = ("_" in word or word.isupper()
                      or (word[:1].isupper() and any(c.islower() for c in word)))
        if not deliberate or len(word) < 3:
            continue
        out.add(word.lower())
        out.update(p.lower() for p in _SPLIT_RE.split(word) if len(p) > 2)
    return out


def _plan_targets(plan_symbols: list[str] | None) -> tuple[set[str], set[str]]:
    """(symbol names, file paths) a Planner verified, from its `path::Symbol`
    form.

    Matched EXACTLY, never by shared word fragments. Fragment matching looks
    harmless and is not: a plan naming `test_cell_len` shares the tokens "cell"
    and "len" with `cell_len`, so every loosely-related symbol inherits the
    strongest boost in the scorer and the plan stops meaning anything specific.
    """
    names: set[str] = set()
    files: set[str] = set()
    for raw in plan_symbols or []:
        text = str(raw).split("(")[0].strip()      # drop "(new — not in repo yet)"
        path, _, symbol = text.rpartition("::")
        if path:
            files.add(path.strip())
        leaf = (symbol or text).split(".")[-1].strip()
        if leaf:
            names.add(leaf.lower())
    return names, files


def _is_test(hit: dict) -> bool:
    """Test code — kept in the pool, ranked below source. A verbose natural-
    language query ('render the list items…') otherwise lets test-function names
    outrank the real definitions (observed on rich-A). The graph's own `is_test`
    flag wins when it set one; path shape decides otherwise."""
    from .. import lang

    if hit.get("is_test") is not None:
        return bool(hit["is_test"])
    return lang.is_test_file(str(hit.get("file_path") or ""))


# How much one place in the fused order is worth. Every adjustment below is
# expressed in units of this, so the weights read as "moves it about N places":
# a test demotion is worth ~2, an expanded neighbour ~3, a Planner-verified
# symbol ~8. Crucially it is a FIXED step rather than a share of the pool —
# normalizing by pool size made the base gap 1.0 in a two-candidate pool, where
# no structural signal could move anything, and 0.05 in a large one, where every
# signal overwhelmed the fusion. Neither is what "adjust the order" means.
_RANK_STEP = 0.15


def _deterministic_score(hit: dict, query_idents: set[str], plan: tuple[set[str], set[str]],
                         rank: int) -> float:
    """Score one candidate.

    The governing principle: this stage ADJUSTS the fused order, it does not
    re-derive it. Fusion is a measured consensus of two channels with tuned
    weights; what it structurally cannot see is that a doc file is not code, a
    test is not the thing under change, a neighbour arrived by adjacency rather
    than by matching, and a Planner has already verified some of these targets.
    Those are the signals below, and they are deliberately the only ones.

    * **fused rank** (base) — what RRF already decided, one ``_RANK_STEP`` per
      place, so every adjustment below reads as "moves it about N places". Each
      channel supplies its OWN ordinal via ``rank_hint``: candidates arrive
      concatenated, so scoring by position in the merged list would give the
      last channel's first hit the same base as the first channel's twentieth,
      and no adjustment could recover it. That is not hypothetical — it made
      lexical refinement measurably worthless, recovering the one file that
      mentions an env var and then burying it at rank 20.
    * **lexical match** (+0.3) — this candidate exists because an identifier the
      query NAMED appears in it. That is the same evidence as an identifier
      match, and it is the whole point of the refinement pass.
    * **plan match** (+1.2, ~8 places) — the one signal allowed to overturn the
      fusion. A symbol a Planner verified against the graph is the target, not
      a guess.
    * **identifier match** (+0.3 max, ~2 places) — a tiebreaker, and only for
      query terms shaped like code (see ``_identifier_terms``). Prose words are
      excluded on purpose; rewarding them double-counts the weakest channel.
    * **expansion penalty** (−0.5) — a neighbour earns its place below what
      matched.

    The source/test/docs prior is NOT here — it is a band (see ``_band``),
    because it is not a matter of degree.
    """
    ordinal = hit.get("rank_hint")
    score = -_RANK_STEP * (rank if ordinal is None else int(ordinal))
    if hit.get("lexical"):
        score += 0.3
    if query_idents:
        name_terms = _terms(str(hit.get("name") or ""))
        if name_terms & query_idents:
            score += 0.3 * len(name_terms & query_idents) / max(1, len(name_terms))
        elif _terms(str(hit.get("qualified_name") or "")) & query_idents:
            score += 0.1
    if _plan_named(hit, plan):
        score += 1.2                                 # named by the plan: not a guess
    if hit.get("expanded"):
        score -= 0.5 + 0.2 * (int(hit.get("hop") or 1) - 1)
    if hit.get("label") in ("Function", "Method", "Class", "Route"):
        score += 0.1                                 # a symbol beats a bare Variable
    return score


def _plan_named(hit: dict, plan: tuple[set[str], set[str]]) -> bool:
    """Did the Planner name this exact symbol, or the file it lives in?"""
    names, files = plan
    return (str(hit.get("name") or "").lower() in names
            or str(hit.get("file_path") or "") in files)


def _band(hit: dict, plan: tuple[set[str], set[str]]) -> int:
    """0 = source, 1 = test, 2 = documentation. Sorted on before score.

    A hard partition, not a penalty, because this was measured rather than
    guessed: on a verbose natural-language query, test-function names outrank
    the real definitions outright (rich-A), and they do it in numbers — three
    tests in a top five is not something a −0.35 nudge fixes. Tests stay in the
    pool because "which tests already cover this" is real context; they simply
    are not the code being changed.

    Unless the Planner says they are. A step that targets a test file makes that
    file the work, and a prior derived from paths must not overrule a decision
    made against the graph.
    """
    if _plan_named(hit, plan):
        return 0
    if str(hit.get("file_path") or "").endswith(_DOC_EXT):
        return 2
    return 1 if _is_test(hit) else 0


def _llm_rank(hits: list[dict], query: str) -> list[int] | None:
    """Ask the configured rerank model to order the pool. Returns the new index
    order, or None to fall through to the deterministic tier.

    One call for the whole pool, not one per candidate: the point of a reranker
    is to be cheaper than the generation it feeds, and per-candidate scoring
    inverts that.
    """
    from .. import llm, providers

    provider = settings.rerank_provider or settings.knowledge_provider
    model = settings.rerank_model or settings.knowledge_model
    if not providers.can_chat(provider):
        return None
    listing = "\n".join(
        f"{i}. {h.get('label', '')} {h.get('name')} — {h.get('file_path')}"
        f":{h.get('start_line') or '?'}" for i, h in enumerate(hits))
    system = (
        "You rank code locations by how likely an engineer is to EDIT them to satisfy a "
        "request. Judge the code that must CHANGE, not code that merely mentions the "
        "topic: a test that exercises the behaviour, or a caller that passes through, "
        "ranks below the definition that implements it. Reply ONLY as JSON: "
        '{"order": [indices, most relevant first]}. Include every index exactly once.')
    r = llm.chat(system, f"Request: {query}\n\nCandidates:\n{listing}",
                 provider=provider, model=model, json_mode=True, timeout=60)
    if r.get("error") or not r.get("text"):
        return None
    import json

    try:
        order = json.loads(r["text"]).get("order")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(order, list):
        return None
    seen: set[int] = set()
    clean = [i for i in order if isinstance(i, int) and 0 <= i < len(hits)
             and not (i in seen or seen.add(i))]
    return clean or None


def _cross_encoder_rank(hits: list[dict], query: str) -> list[int] | None:
    """Optional local relevance model (`pip install '.[rerank]'`). Absent by
    design on a default install — it is a real model in memory, and this
    pipeline runs on machines where that matters."""
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError:
        return None
    try:
        encoder = _get_cross_encoder(TextCrossEncoder)
        docs = [f"{h.get('name')} {h.get('signature') or ''} {h.get('file_path')}"
                for h in hits]
        scores = list(encoder.rerank(query, docs))
    except Exception as exc:  # noqa: BLE001
        logger.warning("knowledge.rerank: cross-encoder failed: %s", exc)
        return None
    return [i for i, _ in sorted(enumerate(scores), key=lambda p: -p[1])]


_encoder = None


def _get_cross_encoder(cls):
    global _encoder
    if _encoder is None:
        _encoder = cls(model_name=settings.rerank_model or "Xenova/ms-marco-MiniLM-L-6-v2")
    return _encoder


def available_tiers() -> list[str]:
    """Which rerank tiers this install can actually run — for Settings, so the
    operator picks from what exists rather than from what the docs mention."""
    tiers = ["deterministic"]
    from .. import providers

    if providers.can_chat(settings.rerank_provider or settings.knowledge_provider):
        tiers.append("llm")
    try:
        import fastembed.rerank.cross_encoder  # noqa: F401

        tiers.append("cross-encoder")
    except ImportError:
        pass
    return tiers


def rank(hits: list[dict], query: str, *, plan_symbols: list[str] | None = None,
         limit: int = 12) -> list[dict]:
    """Order the merged candidate pool, best first, and cut it to `limit`.

    The deterministic pass always runs — it is what decides the order when a
    stronger tier is off, and what the stronger tiers fall back to when they
    fail. `plan_symbols` are the Planner's verified targets: a candidate it
    already named is the target, not a guess.
    """
    if not hits:
        return []
    mode = (settings.rerank_mode or "deterministic").strip().lower()
    if mode == "off":
        # The ablation arm: the fused order, untouched. It exists so the
        # deterministic scorer's contribution is a measured number rather than an
        # assumption — every other mode builds on that scorer, so there has to be
        # a way to run without it.
        return hits[:limit]
    query_idents = _identifier_terms(query)
    plan = _plan_targets(plan_symbols)
    scored = sorted(
        range(len(hits)),
        key=lambda i: (_band(hits[i], plan),
                       -_deterministic_score(hits[i], query_idents, plan, i)))
    ordered = [hits[i] for i in scored]

    if mode != "deterministic":
        pool, tail = ordered[:30], ordered[30:]   # rerank the plausible head only
        order = (_cross_encoder_rank(pool, query) if mode == "cross-encoder"
                 else _llm_rank(pool, query))
        if order:
            # Complete the order HERE, not inside each tier: a model that named
            # only its top few has ranked those few, but dropping the rest would
            # silently delete candidates from retrieval. Reranking reorders a
            # pool; it never shrinks one.
            named = set(order)
            ordered = ([pool[i] for i in order]
                       + [h for i, h in enumerate(pool) if i not in named] + tail)
        else:
            logger.info("knowledge.rerank: %s tier unavailable — deterministic order kept",
                        mode)
    return ordered[:limit]
