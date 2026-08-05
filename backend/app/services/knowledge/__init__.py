"""Repository knowledge — a code graph plus compounding delivery notes.

Two layers, both deterministic to build (no LLM, no embedding service, no
per-repo cost):

  * **Code graph** (knowledge/graph.py): a persistent knowledge graph built by
    the `codebase-memory-mcp` static binary — tree-sitter AST across 158
    languages into definitions, CALLS/IMPORTS/USAGE edges, HTTP routes and IaC
    nodes, with BM25 + bundled-embedding search and Cypher queries. This is the
    localization + structural layer the PM and Dev agents ground on. When the
    binary is unavailable everything degrades to the symbol-map + ripgrep tier.
  * **Delivery notes / lessons** (knowledge/write_back.py + store.py): validated
    cross-run learnings synthesized after each scope delivers, ranked in-process
    at query time (retriever.notes) — so run #50 starts from what runs #1–49
    already proved instead of rediscovering the repo every time.

Pipeline (per-repo, keyed off the repo-url slug):

    clone → graph index (seconds) → symbol map → serve

The LLM only ever writes the short prose of a delivery note; every localization
fact (files, symbols, call edges) comes from the AST, so it can't be
hallucinated.
"""

from . import embed, graph, pipeline, retriever, store  # noqa: F401
