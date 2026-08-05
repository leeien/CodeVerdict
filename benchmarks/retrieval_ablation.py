#!/usr/bin/env python
"""Ablation harness for the retrieval pipeline — free, deterministic, repeatable.

The research's ablation checklist asks which components actually earn their
place. Most of that list can only be answered by paid end-to-end runs, but the
retrieval half can be answered for nothing: retrieval quality is
"are the files you need in the top K", and that is measurable against a labelled
query set with no LLM in the loop at all.

So this measures exactly the stages this repo added — graph expansion, lexical
refinement, reranking, hop depth — on recall@K and MRR, and it can be re-run
after any change to any of them. Nothing here costs money or calls a model.

    python benchmarks/retrieval_ablation.py                     # the default repo
    python benchmarks/retrieval_ablation.py --repo <git-url>    # any indexed repo
    python benchmarks/retrieval_ablation.py --k 5 --json out.json

Requires the repo to be cloned and indexed already (the pipeline does this on
ingest). Queries are phrased as MECHANISMS, not symptoms, because that is how
the agents are instructed to phrase them — measuring symptom queries would be
measuring a mode the system does not use.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import settings  # noqa: E402
from app.services.knowledge import graph, retriever  # noqa: E402


@dataclass(frozen=True)
class Query:
    """One labelled query. `expect` is the file that genuinely implements the
    described mechanism — ground truth, not "a file that mentions it"."""

    text: str
    expect: str


# Ground truth for Textualize/rich. Each answer is the module that IMPLEMENTS
# the mechanism, verified by reading the repo — not merely one that mentions it.
RICH_QUERIES = [
    Query("measure the printable cell width of text containing ansi escape sequences",
          "rich/cells.py"),
    Query("wrap a line of text at word boundaries to fit a given width", "rich/_wrap.py"),
    Query("parse console markup tags into styled text spans", "rich/markup.py"),
    Query("convert a truecolor rgb value down to the nearest 256-color index",
          "rich/color.py"),
    Query("highlight source code using pygments lexers and a theme", "rich/syntax.py"),
    Query("render a progress bar column showing elapsed and remaining time",
          "rich/progress.py"),
    Query("detect the terminal width and whether the output stream is a tty",
          "rich/console.py"),
    Query("pretty print a python container with indentation and repr truncation",
          "rich/pretty.py"),
    Query("turn markdown headings, lists and code blocks into renderables",
          "rich/markdown.py"),
    Query("emit the ansi escape codes for a style's color and bold attributes",
          "rich/style.py"),
    Query("align a renderable to the left, center or right inside a fixed width",
          "rich/align.py"),
    Query("draw a box border around a renderable with an optional title",
          "rich/panel.py"),
    Query("render a traceback frame stack with local variables", "rich/traceback.py"),
    Query("compute the minimum and maximum width a renderable can occupy",
          "rich/measure.py"),
    Query("lay the terminal out into split regions that resize together",
          "rich/layout.py"),
]

# A second class, deliberately separated: terms that exist ONLY as string
# literals. An AST index has no node for `JUPYTER_COLUMNS` — it is not a
# function, class or route — so the ranked channels cannot find it at all, and
# whether the pipeline answers these is a direct measurement of the lexical
# refinement pass rather than of search in general. Agents issue exactly these
# queries: the PM and Planner are instructed to name flags and env vars.
RICH_LITERAL_QUERIES = [
    Query("the JUPYTER_COLUMNS environment variable overrides the notebook width",
          "rich/console.py"),
    Query("the TTY_COMPATIBLE environment variable forces terminal detection",
          "rich/console.py"),
    Query("the FORCE_COLOR environment variable enables color on a non-tty",
          "rich/console.py"),
    Query("the TERM_PROGRAM environment variable identifies the host terminal",
          "rich/diagnose.py"),
]

REPOS = {
    "https://github.com/Textualize/rich": {
        "mechanism": RICH_QUERIES,
        "string literals": RICH_LITERAL_QUERIES,
    },
}


# The ablation matrix. Each arm names the setting overrides that define it, so
# adding a stage means adding one row here — not editing the runner.
ARMS: dict[str, dict] = {
    "full pipeline": {},
    "no graph expansion": {"graph_expansion": False},
    "no lexical refinement": {"grep_refine": False},
    "no rerank": {"rerank_mode": "off"},
    "2-hop expansion": {"graph_hops": 2},
    "fuse only (no stage 2-5)": {"graph_expansion": False, "grep_refine": False,
                                 "rerank_mode": "off"},
    "no semantic channel (BM25 only)": {"local_embeddings": False},
}


def _files_in_order(block: str) -> list[str]:
    """The ranked file paths out of a rendered retrieval block, best first,
    deduped — a query is answered by a FILE, and three hits in the right file
    are one correct answer, not three."""
    out: list[str] = []
    for line in block.splitlines():
        if " — " not in line or not line.startswith("  "):
            continue
        loc = line.rsplit(" — ", 1)[-1].strip()
        path = loc.split(":")[0].split("   (")[0].strip()
        if path and path not in out:
            out.append(path)
    return out


def score(repo: str, queries: list[Query], k: int) -> dict:
    hits = 0
    reciprocal = 0.0
    misses: list[str] = []
    for q in queries:
        block = retriever.retrieve_context(repo, q.text, limit=k, snippets=False)
        ranked = _files_in_order(block)[:k]
        if q.expect in ranked:
            hits += 1
            reciprocal += 1.0 / (ranked.index(q.expect) + 1)
        else:
            misses.append(f"{q.expect} (got {', '.join(ranked[:3]) or 'nothing'})")
    n = len(queries)
    return {"recall_at_k": hits / n, "mrr": reciprocal / n, "hits": hits, "n": n,
            "misses": misses}


def run(repo: str, suites: dict[str, list[Query]], k: int) -> dict:
    baseline = {name: getattr(settings, name) for name in
                ("graph_expansion", "grep_refine", "rerank_mode", "graph_hops",
                 "local_embeddings", "snippet_context")}
    results: dict[str, dict] = {}
    try:
        for arm, overrides in ARMS.items():
            for name, value in baseline.items():
                setattr(settings, name, value)
            settings.snippet_context = False   # source text is not part of ranking
            for name, value in overrides.items():
                setattr(settings, name, value)
            results[arm] = {suite: score(repo, qs, k) for suite, qs in suites.items()}
            cells = "   ".join(
                f"{suite}: {r['recall_at_k']:4.0%} / {r['mrr']:.2f}"
                for suite, r in results[arm].items())
            print(f"  {arm:32} {cells}")
    finally:
        for name, value in baseline.items():
            setattr(settings, name, value)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="https://github.com/Textualize/rich")
    ap.add_argument("--k", type=int, default=8, help="cut-off for recall@K (default 8)")
    ap.add_argument("--json", help="write the full results here")
    args = ap.parse_args()

    suites = REPOS.get(args.repo)
    if suites is None:
        print(f"No labelled query set for {args.repo}. Add one to REPOS in this file — "
              "an ablation without ground truth measures nothing.", file=sys.stderr)
        return 2
    if not graph.available():
        print("The code graph binary is not available; every arm would score 0 and the "
              "comparison would be meaningless.", file=sys.stderr)
        return 2
    if not graph.indexed(args.repo):
        print(f"{args.repo} is not indexed yet — ingest it first.", file=sys.stderr)
        return 2

    counts = ", ".join(f"{len(q)} {name}" for name, q in suites.items())
    print(f"Retrieval ablation — {args.repo} ({counts}), K={args.k}")
    print("Each cell is recall@K / MRR.\n")
    results = run(args.repo, suites, args.k)

    full = results["full pipeline"]
    print("\nDelta vs the full pipeline (recall@K):")
    for arm, per_suite in results.items():
        if arm == "full pipeline":
            continue
        cells = "   ".join(
            f"{suite}: {r['recall_at_k'] - full[suite]['recall_at_k']:+5.0%}"
            for suite, r in per_suite.items())
        print(f"  {arm:32} {cells}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"repo": args.repo, "k": args.k, "results": results}, indent=2))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
