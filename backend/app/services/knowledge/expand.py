"""Graph expansion — the ego-graph around a retrieval hit.

Ranked search answers "which symbol matches these words". It does not answer
"and what else must you understand before you change it", which is the question
that decides whether an edit is correct. RepoGraph's result is that expanding
each hit into its 1-hop neighbourhood is worth a large jump in localization
quality, and that naively flattening 2 hops into the prompt is worth a loss —
the model drowns. So expansion here is deliberately shallow, typed, and capped.

Which edges count is a judgement, not a shrug at "all of them". The graph
carries 21 relationship types; most are structure, not meaning:

  * kept — CALLS (both ways: what it uses, and who breaks if it changes),
    INHERITS, DEFINES_METHOD (a class and its members are one unit of
    understanding), IMPORTS, TESTS (which tests already exercise this — the
    single most useful edge for a coding agent), DECORATES, RAISES/THROWS.
  * dropped — DEFINES and CONTAINS_* (module→everything: enormous fan-out, no
    information), WRITES/USAGE (variable traffic), and SIMILAR_TO /
    SEMANTICALLY_RELATED, which are the graph binary's own weak embedding
    layer; this repo has a real dense channel and does not need a second,
    worse one competing for the same budget.

Fail-open like every other graph consumer: no binary, no index, or a malformed
reply yields [] and the caller proceeds with unexpanded hits.
"""

from __future__ import annotations

import logging

from ...config import settings
from . import graph

logger = logging.getLogger(__name__)

# Relationship types worth traversing, in the order they are reported.
EDGES = ("CALLS", "INHERITS", "DEFINES_METHOD", "IMPORTS", "TESTS", "DECORATES",
         "RAISES", "THROWS")

# Node kinds worth returning. A Module/File/Folder neighbour is technically a
# caller ("rich/table.py CALLS Table") but tells a coding agent nothing it can
# act on, and it crowds out the function that actually calls it.
_KEEP_LABELS = frozenset(["Function", "Method", "Class", "Route", "Decorator"])

# Per-direction row cap on the cypher query itself, so a hub symbol can't
# return thousands of rows for us to throw away.
_QUERY_LIMIT = 120

_REL = "|".join(EDGES)


def _rows(repo_url: str, names: list[str], *, incoming: bool) -> list[list]:
    """One batched 1-hop query. Batching matters: each cypher call is a
    subprocess, so expanding eight hits one at a time would be sixteen process
    spawns per retrieval."""
    quoted = ", ".join('"' + n.replace('"', "").replace("\\", "") + '"' for n in names)
    arrow = f"<-[r:{_REL}]-" if incoming else f"-[r:{_REL}]->"
    return graph.cypher(repo_url, (
        f"MATCH (n){arrow}(m) WHERE n.name IN [{quoted}] "
        "RETURN n.name, type(r), m.name, labels(m), m.file_path, m.start_line, "
        f"m.qualified_name LIMIT {_QUERY_LIMIT}"))


def _node(row: list, *, incoming: bool) -> dict | None:
    if not isinstance(row, list) or len(row) < 7 or not row[4]:
        return None
    label = graph.node_label(row[3])
    if label not in _KEEP_LABELS:
        return None
    file_path = str(row[4])
    if file_path.startswith("<"):       # synthetic/builtin node
        return None
    return {
        "name": row[2], "label": label, "file_path": file_path,
        "start_line": graph.as_int(row[5]), "qualified_name": row[6], "signature": "",
        # Provenance, carried into the prompt: "you got this because X calls it"
        # is the difference between context and a pile of code.
        "via": f"{row[1]} {'from' if incoming else 'to'} {row[0]}",
        "expanded": True,
    }


def neighbors(repo_url: str, names: list[str], *, hops: int | None = None,
              limit: int = 24) -> list[dict]:
    """1-hop (or `hops`-hop) neighbourhood of `names`, as search-hit dicts.

    `hops` defaults to ``settings.graph_hops``. Depth beyond 1 exists to be
    ablated, not to be used: it is the configuration RepoGraph measured as
    actively harmful when flattened into a prompt, so each extra hop gets a
    quarter of the budget rather than an equal share.
    """
    if not graph.available() or not names:
        return []
    depth = settings.graph_hops if hops is None else hops
    if depth < 1:
        return []
    out: list[dict] = []
    seen_keys: set[str] = set()
    frontier = list(dict.fromkeys(n for n in names if n))[:8]
    seen_names = set(frontier)
    budget = limit
    try:
        for hop in range(depth):
            if not frontier or budget <= 0:
                break
            found: list[dict] = []
            for incoming in (False, True):
                for row in _rows(repo_url, frontier, incoming=incoming):
                    node = _node(row, incoming=incoming)
                    if node is None:
                        continue
                    key = node["qualified_name"] or f"{node['file_path']}:{node['name']}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    node["hop"] = hop + 1
                    found.append(node)
            out.extend(found[:budget])
            budget -= len(found[:budget])
            frontier = [n["name"] for n in found
                        if n["name"] and n["name"] not in seen_names][:4]
            seen_names.update(frontier)
            budget = budget // 4 if hop + 1 < depth else budget   # see docstring
    except Exception as exc:  # noqa: BLE001 — expansion is a bonus, never a failure
        logger.warning("knowledge.expand: neighbours of %s failed: %s", names[:2], exc)
    return out[:limit]


def ego(repo_url: str, name: str, *, limit: int = 12) -> str:
    """The `expand` tool's answer: one symbol's neighbourhood, prompt-ready.

    Answers the question a coding agent has after `lookup` — "what does this
    touch, and what touches it" — in one call instead of `callers` plus a
    guess about the rest.
    """
    if not graph.available():
        return "(code graph unavailable — use `callers` or `grep` instead)"
    rows = neighbors(repo_url, [name], hops=1, limit=limit)
    if not rows:
        return (f"{name} — no neighbours in the call graph. Either it's isolated / "
                "dynamically dispatched, or the name is wrong (try `lookup`).")
    lines = [f"{name} — 1-hop neighbourhood ({len(rows)} related symbol(s)):"]
    for r in rows:
        loc = f"{r['file_path']}:{r['start_line'] or '?'}"
        lines.append(f"  [{r['via']}] {r['label'].lower()} {r['name']} — {loc}")
    return "\n".join(lines)
