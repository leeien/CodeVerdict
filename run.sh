#!/usr/bin/env bash
# ============================================================================
# CodeVerdict — one-command local launcher.
#   ./run.sh               create a venv, install, open the CodeVerdict shell
#   ./run.sh doctor        …print a preflight report instead and exit
#   ./run.sh <any args>    …anything else is passed straight to `codeverdict`
#
# The app is self-contained; the only external helper is the codebase-memory-mcp
# code-graph binary (auto-installed below if npm is present; otherwise the KB
# degrades to a built-in symbol map + ripgrep). Copy .env.example to .env and
# add an API key first — or skip it and point the stages at a coding CLI you're
# already logged into.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8017}"
HOST="${HOST:-127.0.0.1}"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || { echo "error: no python on PATH" >&2; exit 1; }

"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || { echo "error: Python 3.11+ is required (found $("$PY" -V 2>&1))" >&2; exit 1; }

command -v git >/dev/null 2>&1 \
  || { echo "error: git is required — the agents clone and branch working copies" >&2; exit 1; }

VENV="$ROOT/backend/.venv"
FIRST_RUN=0
if [ ! -d "$VENV" ]; then
  echo "→ creating virtualenv…"
  "$PY" -m venv "$VENV"
  FIRST_RUN=1
fi
# shellcheck disable=SC1091
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; else source "$VENV/Scripts/activate"; fi

echo "→ installing…"
python -m pip install --quiet --upgrade pip
# [semantic] adds local dense embeddings (fastembed + embedded Qdrant) for
# hybrid search; [treesitter] adds exact parse trees for non-Python languages in
# the symbol-map fallback tier. Each tier degrades gracefully if no wheels exist
# for the platform — retrieval falls back to keyword-only / regex extractors.
python -m pip install --quiet -e "$ROOT[semantic,treesitter]" || {
  echo "→ tree-sitter extras unavailable; installing semantic only"
  python -m pip install --quiet -e "$ROOT[semantic]" || {
    echo "→ semantic extras unavailable; installing core only (keyword-only retrieval)"
    python -m pip install --quiet -e "$ROOT"
  }
}

# The code-graph engine: a single static binary, the KB's primary localization
# layer. Auto-install via npm when present; otherwise the KB degrades to the
# symbol-map + git-grep tier (and you can install it later — see docs).
if ! command -v codebase-memory-mcp >/dev/null 2>&1; then
  if command -v npm >/dev/null 2>&1; then
    echo "→ installing codebase-memory-mcp (code-graph engine) via npm…"
    npm install -g codebase-memory-mcp >/dev/null 2>&1 \
      || echo "→ codebase-memory-mcp install failed; KB will use the symbol-map fallback"
  else
    echo "→ codebase-memory-mcp not found and npm unavailable — KB will use the"
    echo "  symbol-map + git-grep fallback. Install the binary (npm/Homebrew/Scoop/"
    echo "  release) for the full code graph: https://github.com/DeusData/codebase-memory-mcp"
  fi
fi

if [ ! -f "$ROOT/.env" ] && [ -f "$ROOT/.env.example" ]; then
  echo "→ no .env found — copied .env.example; add an API key to it"
  cp "$ROOT/.env.example" "$ROOT/.env"
fi

# On the very first install, show the preflight report before handing over. Every
# dependency here is optional in a different way — a missing one degrades quality
# rather than crashing — so the only honest way to tell somebody what they've got
# is to probe it and print it once, while they're still paying attention.
if [ "$FIRST_RUN" = "1" ]; then
  echo
  codeverdict doctor || true
  echo
  echo "→ (re-run any time with \`./run.sh doctor\`, or /doctor in the shell)"
  echo
fi

# No arguments opens the shell; everything else is passed through untouched.
if [ "$#" -eq 0 ]; then
  exec codeverdict
else
  exec codeverdict "$@"
fi
