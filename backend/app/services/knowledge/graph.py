"""Code-graph backend — a thin wrapper around the `codebase-memory-mcp` binary.

The deterministic localization + structural layer of the knowledge base is
served by a persistent knowledge graph (tree-sitter AST across 158 languages:
definitions, CALLS/IMPORTS/USAGE edges, HTTP routes, IaC nodes) built by the
external single-static-binary engine and queried per-call via its one-shot CLI
(`codebase-memory-mcp cli <tool>`), so no daemon is required. Indexing is
RAM-first and fast (~seconds for typical repos; memory released after), which
lets freshness simply reindex whenever the working copy's SHA drifts instead
of maintaining incremental watermarks.

Projects are keyed by the repo's slug (`--name` override), matching the
knowledge dir layout, so graph identity survives clone-path changes.

Everything is fail-open: if the binary is missing, times out, or returns
garbage, every function returns an empty value and callers degrade to the
symbol-map / ripgrep tier. No query here can break a pipeline run.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from ...config import settings
from .. import git_ops

logger = logging.getLogger(__name__)

_INDEX_TIMEOUT = 600  # big repos: README benchmarks 28M LOC in ~3 min
_QUERY_TIMEOUT = 60


def binary() -> str | None:
    """Absolute path of the graph binary, or None when unavailable/disabled."""
    if not settings.graph_enabled:
        return None
    configured = (settings.graph_binary or "").strip() or "codebase-memory-mcp"
    return shutil.which(configured) or (configured if Path(configured).is_file() else None)


def available() -> bool:
    return binary() is not None


def probe() -> dict:
    """Prove the graph engine actually runs: locate the binary and read its
    version. {ok, output} — surfaced by the Settings 'Test code graph' row."""
    if not settings.graph_enabled:
        return {"ok": False, "output": "Code graph is disabled (settings.graph_enabled=false)."}
    exe = binary()
    if exe is None:
        return {"ok": False, "output":
                f"Binary '{settings.graph_binary}' not found on PATH. Install it "
                "(e.g. `npm i -g codebase-memory-mcp`, `brew install codebase-memory-mcp`, "
                "or download a release) or set graph_binary to its absolute path."}
    try:
        proc = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15)
        ver = (proc.stdout or proc.stderr or "").strip()
        return {"ok": proc.returncode == 0, "output": f"{exe}\n{ver}"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "output": f"{exe}\nfailed to run: {exc}"}


def project(repo_url: str) -> str:
    return git_ops.slug(repo_url)


def _run(tool: str, args: dict, timeout: int = _QUERY_TIMEOUT) -> dict:
    """Run one CLI tool with JSON args on stdin; parse the JSON reply.
    Returns {} on any failure — never raises."""
    exe = binary()
    if exe is None:
        return {}
    try:
        proc = subprocess.run(
            [exe, "cli", tool], input=json.dumps(args), capture_output=True,
            text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("knowledge.graph: %s failed to run: %s", tool, exc)
        return {}
    # The reply is the last JSON object on stdout (log lines go to stderr, but
    # be defensive and scan for the payload line).
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    if proc.returncode != 0:
        logger.warning("knowledge.graph: %s exited %s: %s", tool, proc.returncode,
                       (proc.stderr or "")[-300:])
    return {}


# --- Indexing -----------------------------------------------------------------

def index(repo_url: str) -> dict:
    """(Re)index the repo's working copy into the graph. Returns the index
    summary ({nodes, edges, status, ...}) or {} on failure."""
    path = git_ops.workdir(repo_url)
    if not path.exists():
        return {}
    out = _run("index_repository", {
        "repo_path": str(path), "mode": settings.graph_index_mode,
        "name": project(repo_url),
    }, timeout=_INDEX_TIMEOUT)
    if out.get("status") == "indexed":
        logger.info("knowledge.graph: indexed %s — %s nodes, %s edges",
                    project(repo_url), out.get("nodes"), out.get("edges"))
        return out
    if out:
        logger.warning("knowledge.graph: index of %s returned %s",
                       project(repo_url), out.get("status") or out.get("hint") or out)
    return {}


def indexed(repo_url: str) -> bool:
    out = _run("index_status", {"project": project(repo_url)})
    if "indexed" in out:
        return bool(out.get("indexed"))
    # Older/other shapes: presence of node counts means the project exists.
    return bool(out.get("nodes") or out.get("total_nodes") or out.get("status") == "indexed")


def ensure_indexed(repo_url: str, sha: str | None = None) -> bool:
    """Make sure the graph matches `sha` (or just exists when sha is None).
    Indexing is cheap (seconds), so on any doubt we simply reindex."""
    if not available():
        return False
    meta_sha = _read_sha(repo_url)
    if sha and meta_sha == sha and indexed(repo_url):
        return True
    if index(repo_url):
        if sha:
            _write_sha(repo_url, sha)
        return True
    return False


def indexed_sha(repo_url: str) -> str:
    """The working-copy SHA the current graph index was built from ("" when
    unknown) — freshness compares this against origin/HEAD."""
    return _read_sha(repo_url)


def _sha_path(repo_url: str) -> Path:
    return Path(settings.knowledge_dir).resolve() / project(repo_url) / "graph_sha"


def _read_sha(repo_url: str) -> str:
    try:
        return _sha_path(repo_url).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_sha(repo_url: str, sha: str) -> None:
    try:
        fp = _sha_path(repo_url)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(sha, encoding="utf-8")
    except OSError:
        pass


def delete(repo_url: str) -> None:
    _run("delete_project", {"project": project(repo_url)})
    try:
        _sha_path(repo_url).unlink(missing_ok=True)
    except OSError:
        pass


# --- Queries ------------------------------------------------------------------

def search(repo_url: str, query: str, *, limit: int = 8) -> list[dict]:
    """BM25 full-text search over graph nodes (functions/methods/routes/classes
    boosted; camelCase split into words). The primary NL localization channel."""
    out = _run("search_graph", {"project": project(repo_url),
                                "query": query, "limit": limit})
    return [r for r in out.get("results", []) if isinstance(r, dict)]


def semantic(repo_url: str, phrase: str, *, limit: int = 5) -> list[dict]:
    """Embedding search (bundled local nomic-embed-code, int8). Works best on a
    whole phrase; scores are weak-signal — callers should treat them as
    supplementary to BM25, never authoritative."""
    out = _run("search_graph", {"project": project(repo_url),
                                "semantic_query": [phrase], "limit": limit})
    return [r for r in out.get("semantic_results", []) if isinstance(r, dict)]


def as_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def node_label(raw) -> str:
    # labels(n) serializes as a JSON string like '["Function"]'.
    if isinstance(raw, str) and raw.startswith("["):
        try:
            parsed = json.loads(raw)
            return parsed[0] if parsed else ""
        except (json.JSONDecodeError, IndexError):
            return raw
    return str(raw or "")


def lookup(repo_url: str, name: str, *, limit: int = 5) -> list[dict]:
    """Exact-name definition lookup with line numbers + signatures — the
    graph-tier replacement for symbol_map.lookup.

    `qualified_name` rides along because it is the handle every other graph
    call needs: ``snippet()`` cannot fetch source without it, and the retrieval
    pipeline dedupes on it."""
    safe = name.replace('"', "").replace("\\", "")
    rows = cypher(repo_url,
                  f'MATCH (n {{name: "{safe}"}}) '
                  "WHERE n:Function OR n:Method OR n:Class "
                  "RETURN n.name, labels(n), n.file_path, n.start_line, n.signature, "
                  f"n.qualified_name LIMIT {int(limit)}")
    return [{"name": r[0], "label": node_label(r[1]), "file_path": r[2],
             "start_line": as_int(r[3]), "signature": (r[4] or "") if len(r) > 4 else "",
             "qualified_name": (r[5] or "") if len(r) > 5 else ""}
            for r in rows if isinstance(r, list) and len(r) >= 4 and r[2]]


def cypher(repo_url: str, query: str) -> list[list]:
    """Read-only openCypher query; returns rows (or [])."""
    out = _run("query_graph", {"project": project(repo_url), "query": query})
    return out.get("rows", []) if isinstance(out.get("rows"), list) else []


def callers(repo_url: str, name: str, *, limit: int = 8) -> list[dict]:
    """Who CALLS `name` — the impact half of localization that a flat symbol
    map can never answer."""
    safe = name.replace('"', "").replace("\\", "")
    rows = cypher(repo_url,
                  f'MATCH (a)-[:CALLS]->(b {{name: "{safe}"}}) '
                  f"RETURN a.name, a.file_path, a.start_line LIMIT {int(limit)}")
    return [{"name": r[0], "file_path": r[1], "start_line": as_int(r[2]) if len(r) > 2 else None}
            for r in rows if isinstance(r, list) and len(r) >= 2 and r[1]]


def outline(repo_url: str, file_path: str, *, limit: int = 25) -> list[dict]:
    """Symbols defined in one file, in line order — graph-tier replacement for
    symbol_map.outline."""
    safe = file_path.replace('"', "").replace("\\", "")
    rows = cypher(repo_url,
                  f'MATCH (n) WHERE n.file_path = "{safe}" '
                  "AND (n:Function OR n:Method OR n:Class) "
                  "RETURN n.name, labels(n), n.start_line "
                  f"ORDER BY n.start_line LIMIT {int(limit)}")
    return [{"name": r[0], "label": node_label(r[1]), "start_line": as_int(r[2])}
            for r in rows if isinstance(r, list) and len(r) >= 3]


def impact(repo_url: str, since: str = "") -> dict:
    """Map changes to impacted symbols via the call graph — fed to QA/Review so
    they check what the change can actually break instead of re-auditing the
    whole repo. `since` is a git ref (e.g. origin/main): diffs <since>...HEAD;
    empty = uncommitted working-tree changes."""
    args: dict = {"project": project(repo_url)}
    if since:
        args["since"] = since
    out = _run("detect_changes", args)
    if not out.get("changed_count"):
        return {}
    seen: set[tuple] = set()
    symbols = []
    for s in out.get("impacted_symbols", []):
        if not isinstance(s, dict) or s.get("label") in ("Module", "Variable"):
            continue
        key = (s.get("name"), s.get("file"))
        if key in seen:
            continue
        seen.add(key)
        symbols.append(s)
    return {"changed_files": sorted(set(out.get("changed_files", []))),
            "impacted_symbols": symbols}


def snippet(repo_url: str, qualified_name: str) -> dict:
    """Exact source of one node ({source, file_path, start_line, end_line})."""
    return _run("get_code_snippet", {"project": project(repo_url),
                                     "qualified_name": qualified_name})


def architecture(repo_url: str) -> dict:
    """Deterministic whole-repo structure summary (node/edge/language/package
    counts, routes) — free, no LLM."""
    return _run("get_architecture", {"project": project(repo_url), "aspects": ["all"]})


# --- Rendering (compact, prompt-ready) ----------------------------------------

def render_hit(r: dict) -> str:
    """One search hit as a single compact line: kind name (file:line) [sig]."""
    loc = r.get("file_path") or ""
    if r.get("start_line"):
        loc += f":{r['start_line']}"
    sig = r.get("signature") or ""
    label = (r.get("label") or "").lower()
    return f"{label} {r.get('name')}{sig[:80]} — {loc}"


def _own(items: list, key: str = "qualified_name") -> list[dict]:
    """Drop builtin/stdlib nodes (qualified names outside the project) that
    pollute hotspot/cluster lists."""
    return [i for i in items if isinstance(i, dict)
            and not str(i.get(key, "")).startswith("builtins.")]


def _real_packages(arch: dict) -> list[dict]:
    """Packages worth naming: real source packages (>1 node), biggest first.
    Filters the 1-node external import-targets (Sphinx, alabaster, …) that
    get_architecture lists alongside the actual source tree."""
    pkgs = [p for p in arch.get("packages", [])
            if isinstance(p, dict) and p.get("name") and (p.get("node_count") or 0) > 1]
    return sorted(pkgs, key=lambda p: -(p.get("node_count") or 0))


def _real_routes(arch: dict, limit: int) -> list[str]:
    """HTTP routes worth naming: drop full-URL 'routes' (pre-commit hook repos,
    external links miscategorized from YAML) — keep real in-repo endpoints."""
    out = []
    for r in arch.get("routes", []):
        if not isinstance(r, dict):
            continue
        path = str(r.get("path", "")).strip()
        if not path or "://" in path:
            continue
        out.append(f"{r.get('method', '?')} {path}")
        if len(out) >= limit:
            break
    return out


def overview_text(repo_url: str) -> str:
    """A short natural-language overview from the deterministic architecture
    summary. Replaces the LLM-generated repo/architecture prose as the repo's
    `kb_overview` anchor."""
    arch = architecture(repo_url)
    if not arch:
        return ""
    parts: list[str] = []
    langs = [f"{l['language']} ({l['file_count']} files)"
             for l in arch.get("languages", [])[:4] if isinstance(l, dict)]
    if langs:
        parts.append("Languages: " + ", ".join(langs) + ".")
    pkgs = [p["name"] for p in _real_packages(arch)[:8]]
    if pkgs:
        parts.append("Main packages: " + ", ".join(pkgs) + ".")
    eps = [e["name"] for e in _own(arch.get("entry_points", []))[:5] if e.get("name")]
    if eps:
        parts.append("Entry points: " + ", ".join(eps) + ".")
    routes = [n for n in arch.get("node_labels", [])
              if isinstance(n, dict) and n.get("label") == "Route"]
    if routes and routes[0].get("count"):
        parts.append(f"{routes[0]['count']} HTTP routes detected.")
    nodes, edges = arch.get("total_nodes"), arch.get("total_edges")
    if nodes:
        parts.append(f"Code graph: {nodes} nodes / {edges} edges "
                     "(definitions, calls, imports, routes).")
    return " ".join(parts)[:1200]


def bootstrap_text(repo_url: str) -> str:
    """The PM agent's repo-shaped anchor, built entirely from the graph:
    languages, packages, entry points, routes, call-graph hotspots and Louvain
    clusters (the graph's discovered functional modules). Deterministic and
    free — replaces the LLM-generated repository/architecture/module views."""
    arch = architecture(repo_url)
    if not arch:
        return ""
    lines: list[str] = ["Repository structure (from the code graph — AST-verified, current):"]
    langs = [f"{l['language']} ({l['file_count']})" for l in arch.get("languages", [])[:5]
             if isinstance(l, dict)]
    if langs:
        lines.append("  languages: " + ", ".join(langs))
    pkgs = [f"{p['name']} ({p['node_count']} nodes)" for p in _real_packages(arch)[:10]]
    if pkgs:
        lines.append("  packages: " + ", ".join(pkgs))
    eps = [f"{e['name']} ({e.get('file', '?')})"
           for e in _own(arch.get("entry_points", []))[:6] if e.get("name")]
    if eps:
        lines.append("  entry points: " + ", ".join(eps))
    routes = _real_routes(arch, 15)
    if routes:
        lines.append("  HTTP routes: " + ", ".join(routes))
    hot = [f"{h['name']} (called by {h.get('fan_in', '?')})"
           for h in _own(arch.get("hotspots", []))[:6] if h.get("name")]
    if hot:
        lines.append("  most-called functions: " + ", ".join(hot))
    for c in [c for c in arch.get("clusters", []) if isinstance(c, dict)][:6]:
        tops = ", ".join(str(t) for t in (c.get("top_nodes") or [])[:5])
        lines.append(f"  functional cluster ({c.get('members', '?')} nodes, "
                     f"packages {', '.join((c.get('packages') or [])[:3])}): {tops}")
    return "\n".join(lines)[:3500]
