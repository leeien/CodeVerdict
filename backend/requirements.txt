# Dependencies are declared in ../pyproject.toml (the single source of truth).
# This file exists only for `pip install -r` workflows and tooling that expects
# it. Prefer:
#
#     pip install -e ".[semantic,treesitter]"     # from the repo root
#
# The knowledge base's code-graph engine is the external `codebase-memory-mcp`
# static binary (installed separately — npm/PyPI/Homebrew/Scoop/release, NOT a
# Python package). Without it the KB degrades to the symbol-map + git-grep tier;
# the optional [treesitter] extra makes that fallback exact for non-Python code.
# [semantic] adds local dense embeddings (fastembed + embedded Qdrant) for
# hybrid semantic search; without it retrieval is keyword-only.
-e ..[semantic,treesitter]
