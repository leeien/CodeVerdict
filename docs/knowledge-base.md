# The knowledge base

Every agent is grounded in a knowledge base built from the target repository. This
is the piece that makes the pipeline cheaper and more accurate than sending a cold
agent to explore a repo from scratch — see the
[benchmarks](../benchmarks/kb-vs-claude-code.md) for the measured effect.

The knowledge base has two layers, and — importantly — **both are deterministic to
build: no LLM, no embedding service, no per-repo ingest cost.** A large repo indexes
in seconds.

## Layer 1 — the code graph (localization + structure)

On ingest, the repo is cloned and indexed into a persistent **knowledge graph** by
the external [`codebase-memory-mcp`](https://github.com/DeusData/codebase-memory-mcp)
static binary. It parses the AST of every file across **158 languages** (via vendored
tree-sitter grammars) into nodes and edges:

- **Nodes:** functions, methods, classes, interfaces, enums, HTTP routes, files,
  modules, packages, and infrastructure resources (Dockerfiles / K8s / Kustomize).
- **Edges:** `CALLS`, `IMPORTS`, `DEFINES`, `IMPLEMENTS`, `INHERITS`, `HTTP_CALLS`,
  `DATA_FLOWS`, `SIMILAR_TO` (near-clone), `SEMANTICALLY_RELATED`, and more.

The graph is queried per-call through the binary's one-shot CLI (no daemon), wrapped
by [`services/knowledge/graph.py`](../backend/app/services/knowledge/graph.py). The
channels the pipeline uses:

- **BM25 full-text search** (`search_graph`) — camelCase/snake_case-aware ranking
  over every function/class/route node. The primary natural-language localization
  channel.
- **Semantic search** — see *Layer 1b* below. The graph binary ships its own
  `semantic_query`, but it uses **static token embeddings** (a frozen word-vector
  table, not a contextual model); measured on real queries it returned Makefile
  targets for "make text bold and coloured", so we don't use it.
- **Exact definition lookup** — name → `file:line` + signature, AST-verified.
- **Call graph** (`trace_path` / Cypher) — callers and callees of a symbol: the
  *impact* half of localization a flat index can't answer.
- **Impact analysis** (`detect_changes`) — maps a diff to the symbols it can affect
  through the call graph. Fed to QA/Review so they check the real blast radius.
- **Architecture summary** (`get_architecture`) — languages, packages, entry points,
  routes, hotspots, and Louvain functional clusters in one call. This is the PM
  agent's repo-shaped bootstrap anchor.

The engine is **RAM-first** (LZ4-compressed, in-memory SQLite, single dump at the
end; memory released after) and persists to `~/.cache/codebase-memory-mcp/`.

## Layer 1b — local dense embeddings (semantic recall)

Structural search answers *"where is `X`"*. It cannot answer *"where's the code
that shows how far along a job is"* when the request shares no vocabulary with
the identifiers. That gap is filled by
[`knowledge/embed.py`](../backend/app/services/knowledge/embed.py):

- **What is embedded:** the *graph's own nodes* (functions/methods/classes). The
  graph already did AST-accurate chunking, so we reuse its symbol inventory
  rather than re-chunking files — one vector per symbol, and every hit maps
  straight back to a real `file:line` plus its callers.
- **Doc text:** `search_document: <file>\n<label> <name><signature>\n<body head>`;
  queries use `search_query: …`. Those task prefixes are **required** by the
  nomic model family — omitting them measurably degrades retrieval.
- **Model:** `nomic-ai/nomic-embed-text-v1.5-Q` (768d, ~130 MB) run locally via
  **fastembed** (ONNX, CPU) — no Docker, no Ollama, no API key.
  `jinaai/jina-embeddings-v2-base-code` is a stronger, heavier alternative.
- **Store:** an embedded **Qdrant** collection per repo (`on_disk` vectors and
  payload — no server, no container).
- **Fusion:** results are merged with the graph's BM25 by **weighted Reciprocal
  Rank Fusion** in `retriever.localize`.

**Measured on 10 vocabulary-mismatch queries over `rich`** (top-5 hit rate for
the file a human would open):

| channel | hit rate |
|---|---|
| BM25 (keyword) alone | 4–5 / 10 |
| Dense alone | **8 / 10** |
| Fusion, equal weights | 6 / 10 |
| **Fusion, dense weighted ×2** | **8 / 10** |

Equal-weight fusion scores *worse than dense alone* — BM25's confident-but-wrong
top hits crowd out dense's correct 4th/5th ones — which is why
`rrf_dense_weight` defaults to 2. The `k` constant made no difference (5→60).

### Memory discipline (why the build runs in a subprocess)

ONNX Runtime's CPU arena grows with **sequence length** and never returns it:
2 KB inputs measured **1.2 GB and climbing**, while ~700-byte inputs at batch 8
plateau **flat at ~459 MB**. So the build caps doc text, embeds in small batches,
**scales threads to available RAM**, runs as a **subprocess** (every byte returns
to the OS on exit — the server never holds the model), and **refuses to load
under ~900 MB free**, degrading to keyword-only instead of inviting the OOM
killer. None of this touches model quality.

Indexing cost: ~1,900 symbols ≈ 15 min CPU, one-time per repo, then incremental
with the graph. Disk: a few MB per repo.

### Fallback tier

If the binary isn't installed, localization degrades to a built-in, free **symbol
map** — a deterministic, line-numbered AST index of every symbol (Python via stdlib
`ast`; other languages via the optional `[treesitter]` extra, else regex extractors)
— cross-checked against live `ripgrep`. Lower fidelity (no call graph, no semantic
search), same shape, so the pipeline keeps working. Install the binary
(npm / PyPI / Homebrew / Scoop / a GitHub release) to get the full graph.

## Layer 2 — delivery notes (compounding cross-run memory)

The graph describes the repo *as it is now*. It can't capture what a run *learned*.
So after every scope delivers, [`write_back.py`](../backend/app/services/knowledge/write_back.py)
synthesizes a compact **delivery note**: the files touched and symbols added (computed
from the git diff — deterministic), plus a short prose summary, gotchas, and wiring
notes (the only LLM call in the whole KB, cheap, with a deterministic fallback).

Notes accumulate per repo and are **ranked in-process at query time** by a pure-Python
TF-IDF over the (small) note corpus — no vector store to maintain. Run #50 starts from
what runs #1–49 already proved instead of rediscovering the repo every time. Notes for
unmerged work carry a "do not assume this exists yet" caveat and low confidence; they
are upgraded to trusted knowledge once their branch lands on the default branch. When
notes are pruned, their durable gotchas/wiring are distilled into per-module **lesson**
docs that survive.

## How the layers are used

- **PM scoping** — bootstraps on the graph's architecture summary + recent delivery
  notes, then retrieves hybrid (keyword + semantic) localizations, exact code hits,
  and relevant notes on demand.
- **Ticket grounding** — every target symbol is verified against the graph (real
  definition site) before a ticket is locked; invented names are flagged.
- **Dev prompt** — a "verified locations" brief injects exact `file:line` pins,
  callers, per-file symbol outlines, and real source snippets, so the Dev agent goes
  straight to the right code instead of grepping the whole repo.
- **Dev tools** — the same retrieval is also *callable* by the Dev agent mid-run
  (`search`, `lookup`, `callers`, `outline`, `grep`), so it queries the index instead
  of grepping — and can overrule a wrong PM localization on evidence. See below.
- **QA / Review** — receive the call-graph impact brief for the change.

## Dev-callable tools (every backend, every model)

The pre-computed brief tells the Dev agent what the PM *thinks* is relevant. The
tools let it check. Both matter: the brief is free and usually right; the tools are
how a wrong localization gets corrected instead of faithfully implemented in the
wrong file.

One implementation ([`services/knowledge/tools.py`](../backend/app/services/knowledge/tools.py)),
two delivery surfaces — deliberately **not** tied to any one vendor's agent:

| Dev backend | How the tools arrive |
|---|---|
| Any headless coding CLI (Claude Code, Codex, Cursor, Aider, Gemini CLI, …) | a `.codeverdict/kb` shim written into the working copy and hidden via `.git/info/exclude`, invoked as `.codeverdict/kb lookup <Symbol>` through the shell every such tool already has |
| The HTTP SEARCH/REPLACE loop (Groq / OpenAI / Gemini / xAI / custom) | `<<<SEARCH …>>>` / `<<<LOOKUP …>>>` / `<<<CALLERS …>>>` / `<<<EXPAND …>>>` / `<<<OUTLINE …>>>` / `<<<SNIPPET …>>>` / `<<<GREP …>>>` request blocks, answered by the harness between rounds |

No MCP server and no tool-calling support is required, so a model with neither still
gets the identical seven tools. The shim never appears in the Dev agent's diff.

## Freshness

When a pipeline run starts, both layers are brought up to the repo's origin/HEAD —
deterministically and for free: the symbol map syncs per changed file, and the code
graph reindexes whenever its SHA watermark drifts (the RAM-first pipeline makes a
plain reindex cheaper than incremental bookkeeping). Unmerged delivery notes are
reconciled here too. See [configuration.md](configuration.md) for `KB_AUTO_REFRESH`.

## Graceful degradation

The whole subsystem degrades rather than fails:

- No code-graph binary → symbol map + `ripgrep` localization tier.
- No `[semantic]` extra, feature off, or RAM under pressure → keyword-only
  retrieval (the dense channel declines rather than crashing).
- No `[treesitter]` extra → regex symbol extractors for non-Python code (Python stays
  exact via stdlib `ast`).
- No LLM key → delivery-note prose uses its deterministic fallback; everything else is
  already LLM-free.
- The knowledge base is entirely local: no external service to run or pay for.

## Where it lives in the code

| Concern | Module |
|---|---|
| Code-graph engine wrapper (index / search / trace / impact) | [`services/knowledge/graph.py`](../backend/app/services/knowledge/graph.py) |
| Local dense embeddings (fastembed + embedded Qdrant) | [`services/knowledge/embed.py`](../backend/app/services/knowledge/embed.py) |
| Symbol-map fallback tier | [`services/knowledge/symbol_map.py`](../backend/app/services/knowledge/symbol_map.py) |
| Ingest / rebuild pipeline | [`services/knowledge/pipeline.py`](../backend/app/services/knowledge/pipeline.py) |
| Retrieval, scoped context, Q&A, note ranking | [`services/knowledge/retriever.py`](../backend/app/services/knowledge/retriever.py) |
| Dev-callable tools (dispatcher, CLI shim, loop protocol) | [`services/knowledge/tools.py`](../backend/app/services/knowledge/tools.py) |
| Cross-run delivery notes / lessons | [`services/knowledge/write_back.py`](../backend/app/services/knowledge/write_back.py) |
| Freshness / reindex on drift | [`services/knowledge/freshness.py`](../backend/app/services/knowledge/freshness.py) |
| Precision (use-case-scoped) retrieval | [`services/precision.py`](../backend/app/services/precision.py) |
