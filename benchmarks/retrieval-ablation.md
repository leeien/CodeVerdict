# Which retrieval stages actually earn their place?

The retrieval pipeline has five stages. Four of them are optional, and "we added
a stage and things felt better" is not a result. This is the measurement.

It is free and deterministic — no model is called, nothing is paid for — so it
can be re-run after any change to any stage:

```bash
python benchmarks/retrieval_ablation.py --k 3
```

**Setup.** Textualize/rich (~35.6k LOC), indexed as usual. Two labelled query
suites, ground truth read out of the repo:

* **mechanism** (15 queries) — how the agents are instructed to search: "measure
  the printable cell width of text containing ansi escape sequences" → `rich/cells.py`.
* **string literals** (4 queries) — terms that exist ONLY inside string literals,
  e.g. `JUPYTER_COLUMNS`. An AST index has no node for these; they are not a
  function, class or route. Agents issue exactly these queries, because the PM
  and Planner are told to name flags and env vars.

Scored at **K=3**, not K=8. At K=8 every arm scores 100% on the mechanism suite
and the comparison says nothing — an agent reads the top few hits, so rank
position is the thing that matters.

## Results (recall@3 / MRR)

| Arm | mechanism | string literals |
|---|---|---|
| **full pipeline** | **100% / 0.90** | **100% / 1.00** |
| no lexical refinement | 100% / 0.90 | 50% / 0.50 |
| no rerank | 93% / 0.70 | 25% / 0.12 |
| no graph expansion | 100% / 0.90 | 100% / 1.00 |
| 2-hop expansion | 100% / 0.90 | 100% / 1.00 |
| fuse only (no stage 2–5) | 93% / 0.70 | 25% / 0.12 |
| no semantic channel (BM25 only) | 67% / 0.52 | 100% / 1.00 |

## What this says

**The semantic channel is the largest single contributor** to mechanism queries:
−33% recall@3 without it. That confirms the earlier 8/10-vs-5/10 measurement on
vocabulary-mismatch queries, on a different query set and a harder cut-off.

**Lexical refinement is the only thing that answers string-literal queries** —
half of them are unfindable without it, and it is worth nothing at all on the
mechanism suite. That is the expected shape: it exists for what an AST index
cannot model, and nothing else.

**Reranking is worth more than it looks.** −7% on mechanism, but −75% on string
literals, because it is what lifts a recovered lexical hit into view. Recovering
the one file that mentions `JUPYTER_COLUMNS` and then leaving it at rank 20 is
not recovering it — an earlier version of this pipeline did exactly that, and
this benchmark is what caught it.

**Graph expansion shows no measurable effect here, and neither does a second
hop.** Worth stating plainly rather than burying:

* The 2-hop result agrees with RepoGraph — extra depth does not help, and their
  measurement was that flattening it into a prompt actively hurts. 1 stays the
  default.
* The 1-hop result is **not** evidence that expansion is useless; it is evidence
  that *this metric cannot see what expansion is for*. Recall@K of one expected
  file measures localization of a single target. Expansion exists to supply the
  blast radius — the callers a change would break, the tests that already cover
  it — which lands in the plan's `blast_radius` and the `expand` tool, and which
  a single-file recall metric has no way to score.

  Measuring it honestly needs a multi-file benchmark (given a change, are ALL the
  files that must change in the top K) with labelled multi-file ground truth.
  That does not exist yet, so **the case for graph expansion is currently
  argued, not measured**. It stays on because the blast radius it produces is
  visible in the plan and used by the reviewers; it is a setting, and an operator
  who wants it off has a switch.

## What is not measured here

End-to-end effects — whether the Planner improves delivered patches, whether
reviewer reachback reduces false blocking findings — need paid pipeline runs
against a task set. The flags exist for it (`planner_enabled`, `jury_tool_calls`,
`dev_inject_file_contents`), and `kb-vs-claude-code.md` is the harness those
numbers belong in.
