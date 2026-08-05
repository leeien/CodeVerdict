"""Retrieval over the knowledge base — the pipeline every agent calls.

``retrieve_context`` is the primitive. It is five stages, not one lookup,
because no single channel answers the question a coding agent actually has:

  1. **fuse**     — the graph's BM25 (exact identifiers) and a local dense
                    embedding channel (vocabulary bridging), merged by weighted
                    RRF. Dense alone scored 8/10 on vocabulary-mismatch queries,
                    BM25 alone 5/10, equal-weight fusion 6/10 — the weighting is
                    load-bearing, see ``_rrf``.
  2. **expand**   — each top hit's 1-hop call-graph neighbourhood
                    (knowledge/expand.py). Matching finds the symbol; adjacency
                    finds what breaks when you change it.
  3. **refine**   — ripgrep over the query's identifiers (services/search.py),
                    which recovers what an AST index cannot model: strings,
                    comments, config, templates.
  4. **rerank**   — one order out of four incomparable score scales
                    (knowledge/rerank.py).
  5. **snippets** — the winners' real source, or a cached summary when a node is
                    too large to inline (knowledge/snippets.py).

Stages 2-5 are individually switchable, so the pipeline's own contribution is
measurable rather than assumed; with all four off this is the plain hybrid
search it grew out of.

Alongside it, **delivery notes / lessons** (validated cross-run learnings
written by knowledge/write_back.py) are ranked by a pure-Python TF-IDF over the
stored JSON docs. The corpus is small (≤ kb_delivery_notes_max per repo), so
in-process scoring per query is cheap and needs no vector store.

Everything degrades gracefully: without the graph binary, localization falls
back to the symbol map + ripgrep; with no knowledge at all the retrieval
functions return "" and callers degrade.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from ...config import settings
from .. import git_ops, search
from . import embed, expand, graph, rerank, store, symbol_map
from . import snippets as snippets_mod
from .facts import KnowledgeDocument

logger = logging.getLogger(__name__)

# --- Deterministic code-hits channel -----------------------------------------
# Flags (--timeout), env-var-ish UPPER_SNAKE, and identifier-ish words.
_QUERY_TOKEN_RE = re.compile(
    r"--[A-Za-z][A-Za-z0-9-]*|\b[A-Z][A-Z0-9_]{2,}\b|\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
# Only pure filler is stoplisted — generic-but-meaningful words ("output",
# "session") are handled dynamically by the too-many-files guard instead.
_QUERY_STOP = frozenset(
    ["the", "and", "for", "with", "when", "that", "this", "from", "into", "should", "must", "make", "more", "some", "very", "add", "adds", "adding", "new", "fix", "fixes", "fixing", "bug", "feature", "support", "supports", "use", "using", "set", "sets", "allow", "allows", "enable", "enables", "disable", "disables", "implement", "implements", "create", "creates", "update", "updates", "remove", "removes", "after", "before", "where", "which", "what", "user", "users", "all", "any", "able", "want", "need", "needs", "like", "just", "also", "environment", "variable", "variables", "option", "options", "flag", "flags"])
_NOISE_EXT = (".md", ".rst", ".txt", ".yml", ".yaml", ".lock", ".cfg", ".toml", ".ini")


def code_hits(repo_url: str, query: str, max_tokens: int = 5) -> str:
    """Deterministic localization channel alongside ranked retrieval: exact
    definition hits for identifier-flavored query terms ('--timeout',
    'NO_COLOR', 'ColorFormatter'), which ranked search can misplace.
    Definitions come from the code graph when available (AST-verified,
    line-numbered), else the symbol map; ripgrep over a checkout pinned at
    origin's default branch verifies/supplements — the shared clone may sit on a
    stale agent branch at scoping time, and unmerged code must not look real.
    Explicitly reports deliberate-looking terms found NOWHERE, so the caller
    knows a thing is new instead of assuming it exists. No LLM, ~ms per query."""
    root = git_ops.ref_worktree(repo_url)
    pinned = bool(root)
    if not root:
        # No pinned checkout (worktree unsupported or the repo isn't cloned):
        # search the working copy and SAY so, rather than presenting hits from a
        # possibly-unmerged branch as facts about the default branch.
        path = git_ops.workdir(repo_url)
        if not (path / ".git").exists():
            return ""
        root = str(path)
    use_graph = graph.available()
    smap = symbol_map.load(repo_url)
    blocks: list[str] = []
    seen: set[str] = set()
    for m in _QUERY_TOKEN_RE.finditer(query):
        tok = m.group(0)
        name = tok.lstrip("-")
        low = name.lower()
        if low in _QUERY_STOP or low in seen:
            continue
        seen.add(low)
        # A term that LOOKS like a code artifact (flag, ENV_VAR, snake/CamelCase)
        # deserves a "not found" report; plain words just get skipped quietly.
        deliberate = (tok.startswith("--") or name.isupper() or "_" in name
                      or (name[:1].isupper() and any(c.islower() for c in name)))
        entry: list[str] = []
        defs: list[str] = []
        if use_graph:
            defs = [f"{d['file_path']}:{d['start_line'] or '?'} ({d['label'].lower()})"
                    for d in graph.lookup(repo_url, name, limit=2)]
        if not defs and smap:
            defs = [f"{f}:{s['l']} ({s['k']})" for f, s in (smap.lookup(name) or [])[:2]]
        if defs:
            entry.append("defined at " + "; ".join(defs))
            if use_graph:
                calls = graph.callers(repo_url, name, limit=3)
                if calls:
                    entry.append("called from " + "; ".join(
                        f"{c['file_path']}:{c['start_line'] or '?'} ({c['name']})"
                        for c in calls))
        else:
            pat = re.escape(tok if tok.startswith("--") else name)
            hit_files = search.files(root, pat, max_files=31,
                                     ignore_case=not deliberate)
            if not hit_files:
                if deliberate:
                    entry.append("NOT FOUND anywhere in the repo — do not assume "
                                 "it exists; treat as new")
            elif len(hit_files) > 30:
                continue  # too common to be a useful pin
            else:
                rows = search.lines(root, pat, max_lines=8,
                                    ignore_case=not deliberate)
                good = [r for r in rows.splitlines()
                        if ":" in r and not r.split(":", 1)[0].endswith(_NOISE_EXT)][:3]
                entry.extend(g[:140] for g in good)
                if len(hit_files) > 3:
                    entry.append(f"… appears in {len(hit_files)} files total")
        if entry:
            blocks.append(f"  {tok}:\n" + "\n".join(f"    {e}" for e in entry))
        if len(blocks) >= max_tokens:
            break
    if not blocks:
        return ""
    where = ("a checkout pinned at origin's default branch" if pinned else
             "the WORKING COPY, which may sit on an unmerged agent branch")
    return ("Exact code hits for the request's terms (code graph + ripgrep over "
            f"{where} — deterministic, real locations, NOT inferred):\n"
            + "\n".join(blocks))[:1600]


def available(repo_url: str) -> bool:
    return store.has_knowledge(repo_url) or (graph.available() and graph.indexed(repo_url))


# --- Delivery notes / lessons: pure-Python TF-IDF ranking ---------------------

_TFIDF_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")


def _note_text(doc: KnowledgeDocument) -> str:
    parts = [doc.name, doc.summary or "", " ".join(doc.tags or [])]
    for key in ("files", "symbols", "gotchas", "wiring", "lessons"):
        vals = doc.content.get(key) or []
        parts.append(" ".join(str(v) for v in vals))
    return " ".join(parts)


def notes(repo_url: str, query: str, *, k: int = 4) -> list[tuple[KnowledgeDocument, float]]:
    """The most relevant delivery notes / lessons for `query`, best first.
    TF-IDF + cosine over the whole (small) note corpus — no vector store."""
    docs = [d for d in store.load_all(repo_url) if d.type in ("delivery_note", "lesson")]
    if not docs or not query.strip():
        return []

    def toks(text: str) -> list[str]:
        return _TFIDF_TOKEN_RE.findall(text.lower())

    corpus = [toks(_note_text(d)) for d in docs]
    df: Counter = Counter()
    for c in corpus:
        df.update(set(c))
    n = len(corpus)

    def vec(tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        return {t: (1 + math.log(c)) * math.log(1 + n / (df.get(t) or n))
                for t, c in tf.items()}

    def cos(a: dict[str, float], b: dict[str, float]) -> float:
        num = sum(a[t] * b[t] for t in a.keys() & b.keys())
        den = (math.sqrt(sum(v * v for v in a.values()))
               * math.sqrt(sum(v * v for v in b.values())))
        return num / den if den else 0.0

    q = vec(toks(query))
    scored = [(doc, cos(q, vec(c))) for doc, c in zip(docs, corpus, strict=False)]
    scored = [(d, s) for d, s in scored if s > 0]
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def _render_note(doc: KnowledgeDocument, score: float | None = None) -> str:
    head = f"[{doc.type}] {doc.name}"
    if score is not None:
        head += f" (score {score:.2f})"
    # Unmerged / distilled docs must not read with the same authority as
    # AST-verified facts — surface non-HIGH confidence inline.
    conf = (doc.confidence or "").upper()
    if conf and conf != "HIGH":
        head += f" (confidence: {conf} — inferred, verify before relying on it)"
    lines = [head]
    if doc.summary:
        lines.append(f"  {doc.summary}")
    files = doc.content.get("files") or []
    if files:
        lines.append(f"  files: {', '.join(files[:8])}")
    # Delivery notes / lesson docs carry validated cross-run learnings — always
    # show them.
    for key in ("wiring", "gotchas", "lessons"):
        vals = doc.content.get(key) or []
        if vals:
            lines.append(f"  {key}: " + " | ".join(str(v)[:200] for v in vals[:4]))
    return "\n".join(lines)


def notes_context(repo_url: str, query: str, *, k: int = 4) -> str:
    hits = notes(repo_url, query, k=k)
    if not hits:
        return ""
    return ("What previous delivered work on this repo already established "
            "(validated cross-run notes):\n"
            + "\n".join(_render_note(d, s) for d, s in hits))


# --- Graph localization digest ------------------------------------------------

def _hit_key(r: dict) -> str:
    """Fuse the BM25 and dense channels by node identity."""
    return r.get("qualified_name") or f"{r.get('file_path')}:{r.get('name')}"


def _rrf(rankings: list[tuple[list[dict], float]], k: int) -> dict[str, float]:
    """Weighted Reciprocal Rank Fusion: score(node) = Σ wᵢ/(k + rankᵢ).
    Scale-free — merges a dense and a lexical ranker without normalising their
    incomparable score scales.

    The weights are NOT decorative. Measured on 10 vocabulary-mismatch queries
    over rich (top-5 hit rate): dense alone 8/10, BM25 alone 5/10, but *equally*
    weighted fusion only 6/10 — BM25's confident-but-wrong top hits outrank
    dense's correct 4th/5th ones. Weighting dense ≥2× restores 8/10 while
    keeping BM25's exact-identifier precision. `k` made no difference (5→60)."""
    scores: dict[str, float] = {}
    for ranking, weight in rankings:
        if not weight:
            continue
        for rank, r in enumerate(ranking):
            key = _hit_key(r)
            if key.endswith("None") or str(r.get("file_path") or "").startswith("<"):
                continue
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)
    return scores


def fuse(repo_url: str, query: str, *, limit: int = 14) -> list[dict]:
    """Stage 1 — the fused candidate pool: the graph's BM25 (exact identifiers)
    and a local dense-embedding channel (vocabulary bridging), merged by
    weighted Reciprocal Rank Fusion. Returns hit dicts, best first.

    The weights are NOT decorative (see ``_rrf``). Falls back to BM25-only
    ordering when embeddings are unavailable."""
    if not graph.available() or not query.strip():
        return []
    bm25 = graph.search(repo_url, query, limit=limit)
    dense = embed.search(repo_url, query, limit=limit)  # [] when disabled/absent
    meta = {_hit_key(r): r for r in (bm25 + dense)}
    scores = _rrf([(bm25, 1.0), (dense, settings.rrf_dense_weight)], settings.rrf_k) \
        if dense else {_hit_key(r): 1.0 / (i + 1) for i, r in enumerate(bm25)}
    out: list[dict] = []
    for key in sorted(scores, key=lambda k: -scores[k]):
        r = meta.get(key)
        if r is None or not r.get("file_path") or str(r["file_path"]).startswith("<"):
            continue
        out.append(r)
    return out


def _identifiers(text: str, extra: list[str] | None = None) -> list[str]:
    """Identifier-shaped terms worth a lexical pass: flags, CONSTANTS, and
    CamelCase/snake_case names — never plain English, which would grep the
    whole repo for 'the'."""
    out: list[str] = []
    for m in _QUERY_TOKEN_RE.finditer(text or ""):
        tok = m.group(0)
        name = tok.lstrip("-")
        if name.lower() in _QUERY_STOP:
            continue
        if (tok.startswith("--") or name.isupper() or "_" in name
                or (name[:1].isupper() and any(c.islower() for c in name))):
            out.append(name)
    for sym in extra or []:
        ident = str(sym).split("::")[-1].split(".")[-1].strip()
        if ident and ident.isidentifier():
            out.append(ident)
    return list(dict.fromkeys(out))[:6]


def _refine(repo_url: str, root: str, idents: list[str], known: set[str]) -> list[dict]:
    """Stage 4 — lexical refinement. GrepRAG's point: exact identifier search
    recovers occurrences a semantic index misses, cheaply and with high
    precision. Only NEW files are returned; a location the ranked channels
    already found doesn't need saying twice."""
    if not (root and idents):
        return []
    out: list[dict] = []
    for ident in idents:
        for row in search.lines(root, rf"\b{re.escape(ident)}\b", max_lines=4).splitlines():
            parts = row.split(":", 2)
            if len(parts) < 3 or parts[0].endswith(_NOISE_EXT):
                continue
            key = f"{parts[0]}:{ident}"
            if key in known:
                continue
            known.add(key)
            out.append({"name": ident, "label": "", "file_path": parts[0],
                        "start_line": graph.as_int(parts[1]), "qualified_name": "",
                        "signature": "", "via": f"lexical match on {ident}",
                        "lexical": True})
    return out[:8]


def retrieve_context(repo_url: str, query: str, *, plan_symbols: list[str] | None = None,
                     limit: int = 8, snippets: bool = True,
                     exclude: set[str] | None = None) -> str:
    """The retrieval pipeline, as one call. Five stages, each switchable:

      1. **fuse**    — RRF of BM25 + dense embeddings  (always)
      2. **expand**  — 1-hop graph neighbourhood of the top hits (``graph_expansion``)
      3. **refine**  — ripgrep the query's identifiers  (``grep_refine``)
      4. **rerank**  — order the merged pool           (``rerank_mode``)
      5. **snippets**— attach real source for the winners (``snippet_context``)

    `plan_symbols` are a Planner's verified targets when one has run: a
    candidate it already named is the target, not a guess, and the reranker
    weighs it accordingly.

    Returns a prompt-ready block, or "" when nothing is known about this repo.
    """
    ranked = fuse(repo_url, query, limit=limit + 6)
    # Named for the prompt header, which must describe what actually ran. A
    # header that lists every stage regardless is a lie the moment one is turned
    # off — and these stages exist precisely to be turned off and measured.
    stages = ["hybrid search"] if ranked else []
    # Each channel keeps its OWN ordinal. The pool is a concatenation, so a
    # position in it is not a rank — see rerank._deterministic_score.
    for i, r in enumerate(ranked):
        r["rank_hint"] = i

    if settings.graph_expansion and ranked:
        seeds = [str(r.get("name")) for r in ranked[:6] if r.get("name")]
        known_keys = {_hit_key(r) for r in ranked}
        grown = [n for n in expand.neighbors(repo_url, seeds, limit=limit + 4)
                 if _hit_key(n) not in known_keys]
        if grown:
            for i, r in enumerate(grown):
                r["rank_hint"] = i
            ranked += grown
            stages.append(f"{settings.graph_hops}-hop call-graph expansion")

    if settings.grep_refine:
        idents = _identifiers(query, plan_symbols)
        root = git_ops.ref_worktree(repo_url) or str(git_ops.workdir(repo_url))
        seen_files = {f"{r.get('file_path')}:{r.get('name')}" for r in ranked}
        found = _refine(repo_url, root, idents, seen_files)
        if found:
            for i, r in enumerate(found):
                r["rank_hint"] = i
            ranked += found
            stages.append("lexical refinement")

    # Hits the caller has already been shown. A Planner's rounds ask near-identical
    # questions ("dependency picker dropdown", "dependency dropdown template",
    # "dependency sidebar dropdown list"), so their result sets overlap heavily —
    # and because its context is append-only and re-sent whole every round, each
    # duplicate snippet is paid for again on every subsequent round.
    if exclude:
        ranked = [r for r in ranked if _hit_key(r) not in exclude]

    if not ranked:
        return ""
    ranked = rerank.rank(ranked, query, plan_symbols=plan_symbols, limit=limit)
    stages.append(f"{settings.rerank_mode} rerank")

    # The set is updated in place so a caller running several queries in a burst
    # passes one set and each query sees what the previous ones already showed.
    if exclude is not None:
        exclude.update(_hit_key(r) for r in ranked)

    lines = [f"  {graph.render_hit(r)}"
             + (f"   ({r['via']})" if r.get("via") else "") for r in ranked]
    block = (f"Most relevant code locations ({' + '.join(stages)}; real, "
             "AST-verified positions):\n" + "\n".join(lines))
    if snippets:
        source = snippets_mod.prepare(repo_url, ranked)
        if source:
            block += "\n\n" + source
    return block


def localize(repo_url: str, query: str, *, limit: int = 8) -> str:
    """Ranked, line-numbered code locations for `query` — the retrieval pipeline
    without the source block, for callers that only need the map."""
    return retrieve_context(repo_url, query, limit=limit, snippets=False)


def scope_context(repo_url: str, query: str, *, budget_docs: int = 8,
                  plan_symbols: list[str] | None = None) -> str:
    """A compact, query-scoped digest for the PM/Planner agents and the Dev
    agent's KB slice: the retrieval pipeline + exact code hits + relevant
    cross-run delivery notes."""
    parts = [
        retrieve_context(repo_url, query, limit=max(4, budget_docs),
                         plan_symbols=plan_symbols),
        code_hits(repo_url, query),
        notes_context(repo_url, query),
    ]
    return "\n\n".join(p for p in parts if p)


def answer(repo_url: str, messages: list[dict], *, timeout: int = 120) -> str:
    """Answer a repo question grounded in the code graph + delivery notes.
    Retrieves ranked locations (with real source snippets for the top hits),
    then (if an LLM key is set) synthesizes a concise grounded reply citing
    file paths; else returns the ranked digest."""
    query = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    if not query.strip():
        return ""
    hits = graph.search(repo_url, query, limit=6) if graph.available() else []
    note_block = notes_context(repo_url, query, k=3)
    if not hits and not note_block:
        return ""

    context_parts: list[str] = []
    for r in hits:
        block = graph.render_hit(r)
        qn = r.get("qualified_name")
        if qn and len(context_parts) < 3:  # real source for the top hits only
            snip = graph.snippet(repo_url, qn)
            src = (snip.get("source") or "")[:1200]
            if src:
                block += f"\n{src}"
        context_parts.append(block)
    if note_block:
        context_parts.append(note_block)
    context = "\n\n".join(context_parts)

    from .. import llm, providers

    if providers.can_chat(settings.knowledge_provider):
        system = (
            "You are a code knowledge-base assistant. Answer the question using ONLY the "
            "code locations, source snippets and delivery notes provided (they come from "
            "an AST-verified code graph of the repository). Be concise and concrete. Cite "
            "the exact file paths you rely on in `backticks`. If the material is "
            "insufficient, say so."
        )
        user = f"Repository knowledge:\n{context[:9000]}\n\nQuestion: {query}"
        r = llm.chat(system, user, provider=settings.knowledge_provider,
                     model=settings.knowledge_model, timeout=timeout)
        if r.get("text"):
            return r["text"]
    return f"Most relevant repository knowledge for “{query.strip()[:120]}”:\n\n{context[:4000]}"


def overview(repo_url: str) -> str:
    """A natural-language repo overview built from the graph's deterministic
    architecture summary. Used at ingest time as the repo's `kb_overview`."""
    return graph.overview_text(repo_url)


def views(repo_url: str) -> dict:
    """Knowledge content for the API / Analysis screen: the graph's structural
    summary plus the accumulated delivery notes / lessons."""
    docs = [d for d in store.load_all(repo_url)
            if d.type in ("delivery_note", "lesson")]
    deliveries = [{
        "id": d.id, "type": d.type, "name": d.name, "summary": d.summary,
        "tags": d.tags, "content": d.content, "confidence": d.confidence,
    } for d in docs]

    structure: list[dict] = []
    arch = graph.architecture(repo_url) if graph.available() else {}
    if arch:
        structure.append({
            "id": "graph_architecture", "type": "architecture",
            "name": "Code graph", "summary": graph.overview_text(repo_url),
            "tags": [], "confidence": "HIGH",
            "content": {k: arch.get(k) for k in
                        ("total_nodes", "total_edges", "languages", "packages",
                         "entry_points", "routes", "hotspots", "clusters")
                        if arch.get(k) is not None},
        })

    return {
        "slug": git_ops.slug(repo_url),
        "total": len(docs) + len(structure),
        "labels": {"structure": "Code graph", "deliveries": "Delivery notes & lessons"},
        "order": ["structure", "deliveries"],
        "domains": {"structure": structure, "deliveries": deliveries},
    }
