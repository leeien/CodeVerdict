"""Per-juror evidence: the facts each perspective needs to be worth its cost.

Every juror used to get the identical case file, which quietly broke two of
them. Observed on live runs:

  * The ARCHITECTURE juror is charged with "cite the existing pattern this
    should match" — and was never shown the file's other members. It blocked a
    delivery on "should match the established pattern used by all other style
    parameters" (a paid Dev+QA+Review round) without having seen a single one of
    them.
  * The SECURITY juror returned the same content-free approval on three
    consecutive rendering changes, because nothing in its brief told it whether
    the changed code is reachable from any entry point at all. A juror that
    cannot distinguish "no attack path" from "I couldn't tell" is paying rent
    with boilerplate.

So each persona gets an extra, deterministic block from the code graph — free,
AST-verified, no LLM. Everything degrades to "" when the graph is unavailable,
and a juror with no extra evidence simply reviews as before.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..knowledge import graph

# `+++ b/path` lines — the files this diff actually touches.
_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_MAX_FILES = 6


def changed_files(diff: str) -> list[str]:
    return list(dict.fromkeys(_DIFF_FILE_RE.findall(diff or "")))[:_MAX_FILES]


def _outline(repo: str, path: str, limit: int = 30) -> str:
    rows = graph.outline(repo, path, limit=limit)
    names = [f"{o['name']}:{o.get('start_line') or '?'}" for o in rows if o.get("name")]
    return ", ".join(names)


def _architecture(repo: str, files: list[str]) -> str:
    """What the neighbours look like. The juror is asked to judge consistency
    with existing patterns, so it must be able to SEE the existing patterns —
    otherwise 'matches the established pattern' is an aesthetic guess that costs
    a paid round."""
    blocks = [f"  {f}: {_outline(repo, f)}" for f in files if _outline(repo, f)]
    if not blocks:
        return ""
    return ("Existing members of the files this change touches (AST-verified, from the "
            "code graph) — this is the 'established pattern' you are asked to judge "
            "against. Cite the specific sibling you mean. If you cannot point to a "
            "concrete neighbour that does it differently, you do NOT have a consistency "
            "finding:\n" + "\n".join(blocks))


def _security(repo: str, files: list[str]) -> str:
    """Reachability. Without it a rendering helper and an auth handler look the
    same to this juror, so it defaults to boilerplate approval either way."""
    arch = graph.architecture(repo) or {}
    routes = [r.get("name") or r.get("path") for r in (arch.get("routes") or [])][:12]
    entries = [e.get("name") or e.get("file") for e in (arch.get("entry_points") or [])][:12]
    entry_files = {str(e.get("file") or "") for e in (arch.get("entry_points") or [])}
    touched_entry = [f for f in files if f in entry_files]
    lines: list[str] = []
    if routes:
        lines.append("  network-facing routes in this repo: "
                     + ", ".join(str(r) for r in routes if r))
    if entries:
        lines.append("  entry points: " + ", ".join(str(e) for e in entries if e))
    lines.append("  files changed here: " + (", ".join(files) or "(none parsed)"))
    lines.append("  changed files that are themselves entry points: "
                 + (", ".join(touched_entry) if touched_entry else "NONE"))
    return ("Attack-surface context (from the code graph). Use it to decide whether "
            "untrusted input can REACH this change at all. If no reachable path exists, "
            "say exactly that and ABSTAIN rather than returning a generic 'no issues "
            "found' — an abstention is information, boilerplate approval is not:\n"
            + "\n".join(lines))


def _reliability(repo: str, files: list[str]) -> str:
    """Which edges exist to break: the callers of what changed are already in the
    impact brief, so this adds the file's own surface."""
    blocks = [f"  {f}: {_outline(repo, f)}" for f in files if _outline(repo, f)]
    if not blocks:
        return ""
    return ("Surface of the changed files (every function/method defined there). A "
            "failure mode you claim must be reachable through one of these — name it:\n"
            + "\n".join(blocks))


def _tests(repo: str, files: list[str]) -> str:
    """The existing suite around this change, so 'missing coverage' claims name a
    real gap instead of asking for tests that already exist."""
    test_files = [f for f in files if "test" in f.lower()]
    if not test_files:
        return ("No test file appears in this diff. Before calling that a gap, check "
                "whether the change is one the repo would normally test.")
    blocks = [f"  {f}: {_outline(repo, f, limit=60)}" for f in test_files if _outline(repo, f, limit=60)]
    if not blocks:
        return ""
    return ("Tests that already exist in the touched test files (names + lines). A "
            "'missing coverage' finding must name a case NOT in this list:\n"
            + "\n".join(blocks))


def _performance(repo: str, files: list[str]) -> str:
    arch = graph.architecture(repo) or {}
    hot = [h.get("name") for h in (arch.get("hotspots") or [])][:12]
    if not hot:
        return ""
    return ("Hotspots (most-called symbols in this repo, from the call graph). Work on "
            "a path that reaches one of these is worth flagging; work that does not is "
            "not hot, whatever it looks like:\n  " + ", ".join(str(h) for h in hot if h))


_BUILDERS = {
    "architecture": _architecture,
    "security": _security,
    "reliability": _reliability,
    "tests": _tests,
    "performance": _performance,
}


def for_persona(persona_id: str, workdir: str, diff: str) -> str:
    """The extra evidence block for one juror. Never raises and never blocks a
    review: no graph, no workdir, or an unknown persona → "" and the juror
    reviews from the shared case file exactly as before."""
    build = _BUILDERS.get(persona_id or "")
    if build is None or not workdir or not graph.available():
        return ""
    try:
        files = changed_files(diff)
        if not files:
            return ""
        return build(Path(workdir).name, files)[:2500]
    except Exception:  # noqa: BLE001 — evidence is a bonus, never a failure mode
        return ""
