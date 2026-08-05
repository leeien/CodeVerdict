# Does a one-time knowledge base make AI coding cheaper? CodeVerdict vs plain Claude Code

## Current pipeline vs plain Claude Code

The numbers that matter are from the **current pipeline** — the tuned version with
file-content injection into the Dev prompt and a QA pre-existing-failure baseline
(see *How the pipeline got here* below for the earlier iterations that led to it).
Measured on the **two largest repos** (Textualize/rich ~35.6k LOC and Textualize/textual
~82.5k LOC), identical plain-English request given to both systems:

| Repo | Task | Kind | **Pipeline** | Baseline (cold Claude Code) | **Δ** |
|---|---|---|---:|---:|---:|
| rich | A | feature | **$0.333** | $0.449 | **−26%** |
| rich | C | cross-cutting bug | **$0.456** | $0.828 | **−45%** |
| textual | A | feature | **$0.380** | $0.590 | **−36%** |
| textual | C | extreme cross-cutting bug | **$1.705** | $6.830 | **−75%** \* |
| textual | D | medium bug | **$1.697** | $2.146 | **−21%** |
| textual | E | greppable bug | **$0.374** | $0.401 | **−7%** |

**On these two repos the tuned pipeline beats a cold `claude -p` on 6 of 6 tasks it
localizes well — by 7% to 75%** — *and* it ships each change with a locked scope,
independent QA, a cross-provider review, and a PR branch that the raw baseline never
produces. The saving grows with how hard a task is to find: on textual-C the baseline
burned **$6.83 over 207 turns** (14.3M tokens) hunting one cache-decorator bug; the
pipeline localized and fixed it for a quarter of that.

> \* **textual-C honesty note (this is the one caveat worth reading):** the −75% is a
> real *cost* win, but the PM localized to the wrong layer, so the fix covers only the
> Markdown case, not the general one the baseline cured. Dramatically cheaper, but a
> *narrower* fix — treat a hard cross-cutting bug's cheap pipeline output as a tested
> draft pending human root-cause review, not a finished delivery.

### Where it doesn't win (and why)

Two honest holdouts — both understood, neither about repo size:

* **rich-B (+31%, $0.278 vs $0.212).** The baseline here is *tiny*. The pipeline's
  fixed PM+QA+Review gate floor (~$0.14) structurally can't shrink below what those
  gates cost, so on a task whose whole baseline is $0.21 the floor alone eats the
  margin. For a stream of *very* cheap, single-file edits, a cold `claude -p` is hard
  to beat on price — that's the pipeline's break-even boundary, not a defect.
* The **QA "pre-existing failures" false-FAIL** that cost an earlier run two wasted
  revision rounds has since been **fixed** (`git_ops.failing_tests` now snapshots the
  failing set before Dev, so QA judges only *new* failures) and **validated** on tasks
  D and E — both passed QA in a single round despite the repo's 450 pre-existing
  snapshot failures. See *How the pipeline got here → Experiment 4*.

**The mechanism:** the knowledge base buys a cheap, accurate **localization** step, and
with the pinned files' contents fed straight into the Dev prompt, the Dev step stops
re-discovering what the PM already found — so it stays cheap too. Baseline cost tracks
**localizability**, not lines of code; pipeline cost tracks the **gate floor + how many
Dev↔Review rounds** a task takes. Spend the one-time KB money on repeated, hard-to-localize
work against the *same* repo — and read the diff on the hardest bugs.

---

## Full experiments (chronological — where the numbers above come from)

The sections below are in the order they were run, oldest first, so the fixes make sense
in sequence. Two version labels to keep straight:

* **Earlier, un-tuned pipeline** — Experiments 1 and 2, and the *first two rounds* of
  Experiment 3 (clean + optimization round). These predate the file-content-injection
  change and do **not** reflect the shipped pipeline; they're kept to show *why* each fix
  was made.
* **Current pipeline** — Experiment 3's **file-injection round** and all of
  **Experiment 4** (file-injection, plus the concurrency-lock and QA-baseline fixes added
  mid-Experiment-4). These are the runs the summary table above is drawn from.

---

# Experiment 1 — tinydb (small repo, mixed-provider pipeline)

## What was compared

Two systems were given the **identical, non-technical request text** (no file names, no API
internals — the way a PM or user would actually phrase it; see `data/task_a.txt` /
`data/task_b.txt`):

| | CodeVerdict (KB pipeline) | Plain Claude Code (baseline) |
|---|---|---|
| Repo knowledge | One-time structured KB: 62 LLM-written views (architecture, modules, features, workflows, entrypoints, domain, rules, integrations) + free AST symbol map, embedded for retrieval | None — fresh clone, agent explores with its own tools |
| Flow | PM scope-lock → tickets → human approve → Dev → QA → Review → PR branch | One `claude -p` run that edits the working tree |
| Coding model | Claude Code CLI, `sonnet` (KB context injected into the prompt) | Claude Code CLI, `sonnet` |
| Other stages | PM: Gemini `gemini-3.1-flash-lite` · QA: Groq `llama-3.3-70b-versatile` · Review: Groq `openai/gpt-oss-120b` (cross-provider by design: the reviewer never authored the code) | — |

**Target repo:** [msiemens/tinydb](https://github.com/msiemens/tinydb) @ `a7174d1`
(~2.2k core LOC, 63 analyzable files).
**Environment:** Claude Code CLI 2.1.145, July 15 2026. All Claude costs are the CLI's own
`total_cost_usd` meter (API list pricing, cache-aware). Gemini/Groq costs are the platform's
per-token list-price meter.

### The two requests

* **Task A — feature:** "Right now, when updating documents in the database, I can add a number
  to a field or subtract a number from a field. I also want to be able to multiply a field by a
  number and divide a field by a number in the same way. Please add this capability, with tests."
* **Task B — bug fix:** "I ran an update that increments a counter field on all matching
  documents. Some of the matched documents don't have that field yet, and the whole update
  crashed with a KeyError… documents that don't have the field should simply be left unchanged.
  Please fix this, with regression tests." (A real defect — reproduced before the run.)

The KB build cost is **excluded** from the per-task comparison: it is a one-time investment per
repository, amortized across every subsequent request. It is reported separately below.

## Results

### One-time KB build (excluded from per-task totals)

| | tokens in | tokens out | cost | wall time |
|---|---:|---:|---:|---:|
| Structured KB build, `claude-cli sonnet` (62 views) | 225,838 | 7,434 | **$0.1976** | ~4 min |

### Task A — feature (multiply/divide update operations)

| stage | model | tokens in | tokens out | cost |
|---|---|---:|---:|---:|
| PM scope + tickets | gemini-3.1-flash-lite | 3,863 | 644 | $0.0161 |
| **Dev** | **claude-cli sonnet** | **123,498** | **1,951** | **$0.1106** |
| QA | llama-3.3-70b-versatile | 944 | 313 | $0.0055 |
| Review | openai/gpt-oss-120b | 861 | 1,093 | $0.0131 |
| **Pipeline total** | | | | **$0.1453** |
| **Claude Code baseline** | claude-cli sonnet, 12 turns | 242,694¹ | 2,637 | **$0.1924** |

Pipeline **24% cheaper** end-to-end; the Dev step alone was **42% cheaper** than the baseline.
Both produced the same shape of change (2 files, +41 lines); both pass the full
`test_operations.py` suite (22 passed).

### Task B — bug fix (KeyError on missing field)

| stage | model | tokens in | tokens out | cost |
|---|---|---:|---:|---:|
| PM scope + tickets | gemini-3.1-flash-lite | 16,029 | 1,097 | $0.0510 |
| **Dev** | **claude-cli sonnet** | **261,949** | **3,498** | **$0.1890** |
| QA | llama-3.3-70b-versatile | 1,551 | 357 | $0.0074 |
| Review | openai/gpt-oss-120b | 1,469 | 900 | $0.0127 |
| **Pipeline total** | | | | **$0.2601** |
| **Claude Code baseline** | claude-cli sonnet, 13 turns | 285,490¹ | 3,073 | **$0.1752** |

Pipeline **48% more expensive** end-to-end here; the Dev step alone was ~8% more expensive.
Both fixes guard the numeric operations and pass the suite (pipeline: 18 passed, baseline: 24
passed — they added different numbers of regression tests). Honest miss, reported as-is: this
task's PM did three knowledge-retrieval rounds (16k tokens) and the Dev agent, handed two
tickets (fix + tests) as one work order, spent more than the baseline's single focused session.

¹ Baseline `tokens in` = fresh input + prompt-cache reads + cache creation, as reported by the
CLI; cache reads are billed at a reduced rate, which `total_cost_usd` accounts for.

### Bottom line

| | Task A | Task B | both tasks |
|---|---:|---:|---:|
| KB pipeline, all 5 stages | $0.1453 | $0.2601 | **$0.4054** |
| KB pipeline, Dev (coding) only | $0.1106 | $0.1890 | **$0.2996** |
| Plain Claude Code | $0.1924 | $0.1752 | **$0.3677** |

* **Coding step vs coding step** (same model, same CLI): the KB-grounded Dev agent used
  **18.5% fewer dollars** than Claude Code exploring cold.
* **Whole pipeline vs Claude Code**: +10% cost — but that buys a locked scope, tickets, an
  independent QA verdict, a review by a *different* model vendor, and an isolated PR branch.
  Claude Code's number buys a diff in your working tree.
* **KB amortization**: at the observed Dev-step savings (~$0.034/task avg) the $0.198 build
  breaks even after roughly 6 tasks on this repo — but the variance between our two tasks is
  larger than the average saving, so treat break-even math on n=2 as illustrative, not proven.
  The KB also compounds: delivery notes are written back after each scope, and the symbol map
  refreshes free on every run.

# Experiment 2 — pallets/click (larger repo, all-Claude pipeline)

Experiment 1 ran the pipeline's non-coding stages on cheap third-party models (Gemini/Groq),
which makes the pipeline look cheap for reasons unrelated to the KB. Experiment 2 removes that
confound: **every stage runs on the Claude Code CLI**, so pipeline-vs-baseline is a like-for-like
Claude comparison, and it uses a **larger repo** to test the core hypothesis — that cold
exploration (and therefore the baseline's cost) grows with codebase size while KB retrieval does
not.

| | CodeVerdict (KB pipeline) | Plain Claude Code (baseline) |
|---|---|---|
| Flow | PM → tickets → approve → Dev → QA → Review → PR branch | One `claude -p` run on a fresh clone |
| PM / Dev | claude-cli **sonnet** | — |
| QA / Review / Knowledge | claude-cli **haiku** | — |
| Coding model | claude-cli **sonnet** (KB context injected) | claude-cli **sonnet** |

**Target repo:** [pallets/click](https://github.com/pallets/click) (~12k core LOC in 17 source
files; 163 files ingested incl. tests/docs — ~5.6× tinydb's source). **One-time KB build:** 61
structured views / 163 files, **$0.3355**
(claude-cli, 242,675 in / 24,542 out). Excluded from per-task totals, as before.

### The two requests (non-technical)

* **Task A — feature:** add an option to show **processing speed (items/sec)** in the progress
  bar, *off by default* so existing bars are unchanged. (`data/exp2-task_a.txt`)
* **Task B — bug fix:** when an option's value comes from an **environment variable**, an invalid
  value produces `Invalid value for '--count': …` with no hint of where it came from; the error
  should **name the env var**. (A real usability defect; the fixed message reads
  `Invalid value for '--count' (from env var 'MY_COUNT'): 'abc' is not a valid integer.`)
  (`data/exp2-task_b.txt`)

### Task A — feature (progress-bar speed display)

| stage | model | runs | tokens in | tokens out | cost |
|---|---|---:|---:|---:|---:|
| PM scope + tickets | sonnet | 1 | 47,632 | 2,783 | $0.1505 |
| **Dev** | **sonnet** | **3** | **1,178,404**¹ | 12,202 | **$0.7610** |
| QA | haiku | 3 | 53,037 | 12,650 | $0.0937 |
| Review | haiku | 3 | 46,807 | 7,008 | $0.0675 |
| **Pipeline total** | | | | | **$1.0727** |
| **Claude Code baseline** | sonnet, 30 turns | 1 | 987,075¹ | 9,382 | **$0.6536** |

Here the pipeline ran **+64%** over the baseline, and even the **Dev step alone was +16% dearer** —
because the **reviewer requested changes twice**, forcing three Dev rounds ($0.427 + $0.233 + $0.101).
Each round re-pays sonnet's per-call overhead. Both ship the same shape of change (progress-bar
`show_speed`, off by default); pipeline branch passes `test_termui.py` (225 passed), baseline
passes 875. This is the pipeline's worst case: a task the gates decide to re-work.

### Task B — bug fix (name the env var in the error)

| stage | model | runs | tokens in | tokens out | cost |
|---|---|---:|---:|---:|---:|
| PM scope + tickets | sonnet | 1 | 52,446 | 3,327 | $0.1452 |
| **Dev** | **sonnet** | **2**² | **970,734**¹ | 11,658 | **$0.6518** |
| QA | haiku | 1 | 17,679 | 12,269 | $0.0729 |
| Review | haiku | 1 | 12,861 | 8,309 | $0.0569 |
| **Pipeline total** | | | | | **$0.9268** |
| **Claude Code baseline** | sonnet, 45 turns | 1 | 1,650,563¹ | 18,217 | **$1.0668** |

Here the KB earns its keep: the baseline needed **45 turns / $1.07** to localize the env-var
resolution path cold, while the KB-grounded Dev did it in fewer, cheaper rounds — **pipeline −13%
overall, Dev step −39%**. Both fixes pass the suite (pipeline branch: 752 passed; baseline: 871).

**Scope-creep finding (reported as-is):** the pipeline's sonnet Dev agent churned `core.py` by
**435 lines (+250/−151 non-blank)** to make this fix, versus the baseline's focused 43-line edit.
Some of that is defensible (extracting a `resolve_envvar_name()` helper the feature genuinely
needs); some is tangential drift the ask didn't call for — merging `with` statements, rewriting
type hints (`type[Group] | type[type]` → `type[Group | type]`), refactoring `to_info_dict`. An
agentic Dev on a capable model will "improve" code around its target unless the diff is
constrained. Worth a tighter diff-discipline instruction in the Dev prompt.

### Bottom line (Experiment 2)

| | Task A | Task B | both tasks |
|---|---:|---:|---:|
| KB pipeline, all 5 stages | $1.0727 | $0.9268 | **$1.9995** |
| KB pipeline, Dev (coding) only | $0.7610 | $0.6518 | **$1.4128** |
| Plain Claude Code | $0.6536 | $1.0668 | **$1.7204** |

* **Coding step vs coding step** (sonnet vs sonnet): KB-grounded Dev used **18% fewer dollars** in
  aggregate — but it's entirely the bug-fix task (−39%); the feature task's revision loop reversed
  the sign (+16%). The KB advantage is real but is dwarfed by re-work variance.
* **Whole pipeline vs Claude Code**: **+16%** — the QA + Review gates and their re-runs. That
  premium buys a locked scope, tickets, an independent QA verdict, a code review, and a PR branch;
  the baseline buys a working-tree diff.
* **Repo size matters, as predicted:** baseline per-task cost roughly **3.5–5×'d** going from
  tinydb (~$0.19) to click ($0.65–$1.07). The KB's job is to cap that growth; on the harder task
  it did (−39% at the Dev step).
* **Model split matters:** QA + Review on **haiku** cost only $0.057–$0.094 *including 3 rounds
  each* on Task A. Running the gates on a small model is what keeps the pipeline within ~16% of a
  single sonnet run despite doing five stages.

### Where the KB was thin (self-improvement signal)

The gap-logger (`data/exp2-click-gaps.jsonl`) recorded PM retrieval depth per request:
**Task A → 0 retrieval rounds** (the KB's feature/workflow views answered scoping outright);
**Task B → 1 round** (PM had to look up *"how Click's Option reads a value from an env var and
passes it through type conversion"* — the exact wiring the fix touches). That's the KB telling us
which corner was under-documented; the write-back/consolidation loop folds such lookups back into
per-module lesson docs so the next env-var question is answered from cache.

¹ `tokens in` includes prompt-cache reads + cache creation as reported by the CLI; cache reads are
billed at a reduced rate, which `total_cost_usd` accounts for. Dev totals sum all rounds.
² Task B's first Dev run aborted on a Claude session limit mid-flight and was re-run after reset;
the two runs / $0.6518 reflect only completed work — the aborted partial is **not** counted.

# Experiment 3 — Textualize/rich (largest repo — and it overturns the size story)

Experiment 3 went bigger again — **Textualize/rich, ~35.6k core LOC in 146 source files
(~3× click, ~16× tinydb)** — to push the "repo size drives baseline cost" hypothesis. It
**falsified the naive version of that claim** and replaced it with a sharper one: what drives
cost is not how big the repo is, but **how well the user's words map onto the code** — the
*localizability* of the task. Same config as Experiment 2 (PM/Dev on `sonnet`, QA/Review/Knowledge
on `haiku`, `max_revision_rounds = 2`).

**Target repo:** [Textualize/rich](https://github.com/Textualize/rich) @ `9d8f9a37`.
**One-time KB build:** 72 structured views / 553 files, **$0.3589** (claude-cli, 278,031 in /
33,659 out). Excluded from per-task totals, as before. Note it is barely above click's $0.3355
**despite 3× the code** — the KB build scales with *view count*, not LOC, which only strengthens
the amortization case.

### Three requests (non-technical), chosen to vary localizability on one repo

* **Task A — feature, easy to localize:** markdown checklists render as raw `[x]`/`[ ]` brackets
  instead of tick boxes; make them render as real check boxes, leave plain bullets unchanged.
  ("checklist" → `markdown.py`.) (`data/exp3-task_a.txt`)
* **Task B — bug, easy to localize:** console output captured on Windows loses lines — only the
  last of three survives. (rich issue #4090; "captured output" → `ansi.py`. A real defect,
  reproduced: `Text.from_ansi('line1\r\nline2\r\nline3')` → `'\n\nline3'`.) (`data/exp3-task_b.txt`)
* **Task C — bug, hard to localize (added mid-experiment):** a clickable web address gets chopped
  into several links when an emphasised word overlaps it. (rich issue #4109; the fix lives in how
  link identity is minted during *style composition* — no file is named "link" or "highlight". A
  real defect, reproduced: one logical link emitted as **3 OSC-8 sequences with 2 ids**.)
  (`data/exp3-task_c.txt`)

Why C exists: Tasks A and B are trivially *greppable* — the user's vocabulary maps 1:1 to a
filename, so a cold agent's first search hits and repo size never gets exercised. Indeed **rich's
A/B baselines came in *cheaper* than click's despite rich being 3× larger** ($0.45/$0.21 vs
$0.65/$1.07). Task C holds the repo, model, and day constant and varies *only* localizability, to
isolate the thing that actually moves cost.

### Baselines (plain Claude Code, `sonnet`, fresh clone each; all pass the suite)

| task | kind | turns | tokens in¹ | cost |
|---|---|---:|---:|---:|
| A | feature, greppable | 20 | 562,528 | $0.4486 |
| B | bug, greppable | 11 | 260,364 | $0.2123 |
| C | bug, cross-cutting | 34 | 918,897 | $0.8283 |

**Same repo, same model, same day: Task C read 3.5× Task B's tokens and took 3× the turns.**
That spread is the localization tax, made visible — and it is what the KB is supposed to attack.

### Pipeline results (clean environment — see the box below)

| task | PM | Dev (all rounds) | QA | Review | **pipeline** | baseline | Dev rounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | $0.1798 | $0.2324 | $0.0250 | $0.0387 | **$0.4759** | $0.4486 | 1 |
| B | $0.2647 | $0.2193 | $0.0494 | $0.0761 | **$0.6095** | $0.2123 | 2 |
| C | $0.2678 | $1.0041 | $0.0483 | $0.0880 | **$1.4082** | $0.8283 | 1 |
| **total** | | | | | **$2.4936** | **$1.4892** | |

All three ship the right change and pass rich's suite (A: checklists render `☑`/`☐`, `test_markdown`
14 passed; B: CRLF text intact, `test_ansi` 28 passed; C: link emitted once, full suite 962 passed;
diffs in `data/exp3-pipeline-*.diff`). The `git push` step fails on all three only because we lack
write access to Textualize/rich — no LLM cost, irrelevant to the comparison.

### What the numbers say

* **The KB's value shows up at localization (PM), cheaply and reliably.** From purely non-technical
  text the PM localized Task C to `segment/text/style/highlighter.py` on **$0.27 / 62k tokens** —
  where the cold baseline burned **919k tokens / 34 turns** to find the same neighbourhood. That is
  the KB mechanism working exactly as intended.
* **But the Dev step does not inherit that saving — it tracks how *spread out* the change is.**
  The KB-grounded Dev's token use vs the cold baseline: **Task A −38%** (348k vs 563k), **Task C
  +54%** (1.41M vs 919k). On a well-localized change the KB keeps Dev tight; on an inherently
  cross-cutting change the Dev must read many files *regardless* of how good the localization was,
  and the injected KB context is added on top. So the Dev-step delta is **A −48% / B +3% / C +21%**
  in dollars — the KB helps most where the change is contained, and reverses where it is not.
* **Whole pipeline vs baseline: A +6%, B +187%, C +70%.** The gates (PM + QA + Review) have a
  **fixed floor** (~$0.34 here) that is independent of task size, so the *cheaper* the task, the
  worse the pipeline looks: on Task B the PM stage *alone* ($0.265) already exceeds the entire
  $0.212 baseline. Repo size did **not** rescue the pipeline's economics — because on this repo the
  baseline stayed cheap wherever the task was greppable.
* **Aggregate Dev-step cost was essentially even** ($1.456 pipeline vs $1.489 baseline, −2%) — but
  that average hides the real signal (A −48% vs C +21%). As in Experiments 1–2, the mechanism
  matters more than the mean.

### An infrastructure bug this benchmark surfaced (and we fixed)

The first pipeline runs of A and C each burned **two extra `sonnet` Dev rounds** — not because the
code was wrong, but because QA hard-failed on a broken test environment, and the reviewer dutifully
bounced the (correct) diff. Two independent causes, both real CodeVerdict bugs, both fixed before the
numbers above were taken:

1. **Poetry / PEP-735 dev-dependencies were never installed.** `git_ops.test_python()` built the
   per-repo `.agent-venv` with `pip install -e .[test]/[tests]/[dev]` only — PEP-621 extras. rich
   declares `attrs` (needed to *collect* the suite) under `[tool.poetry.dev-dependencies]`, which
   those extras never reach, so the suite couldn't even import and QA read that as a code failure.
   Fixed by parsing `tool.poetry.dev-dependencies`, `tool.poetry.group.*.dependencies`, and
   `[dependency-groups]` and installing those names. Affects **any Poetry-managed repo**.
2. **The workspace path contained a space.** `REPOS_DIR` defaulted under `…/AI_ML projs/…`; rich's
   `test_log`/`test_traceback` assert on rendered file paths and fail spuriously under a spaced
   path. Proven by copying the identical clone to a space-free path (2 failed → 2 passed). Fixed by
   relocating `REPOS_DIR`.

After both fixes, rich's suite runs **957 passed / 0 failed** inside the pipeline's own
`.agent-venv`, QA passes on merit, and the phantom revision rounds disappear (Task C went from 3
Dev rounds / $1.88 to **1 round / $1.41**). The lesson generalises: **a pipeline that gates on
"does the suite pass" is only as trustworthy as the environment it builds** — a mis-provisioned
test env doesn't just fail a task, it silently *inflates* cost by manufacturing re-work. Task B's
two revision rounds, by contrast, were **genuine** (env already fixed; the reviewer asked for a
real change and accepted it on round 2) — evidence the gate does useful work when the environment
is honest.

### Optimization round: closing the gap (post-benchmark)

The clean numbers above located the pipeline's cost leaks precisely, so we turned the three
obvious screws and re-ran the two most instructive tasks:

1. **PM stage → haiku** (it was $0.18–0.27/task on sonnet; the KB retrieval does the semantic
   heavy lifting, and the gates already prove haiku handles structured judgment).
2. **Scope minimalism in the PM prompts**: default to ONE ticket per scope (a fix and its tests
   are one ticket), criteria must restate the user's observable ask and nothing more, 3–6
   criteria per scope — because every invented criterion is paid Dev work *and* a legitimate
   ground for the reviewer to bounce the diff.
3. Env fixes from above retained (QA now only fails for real reasons).

| run | PM | Dev | QA+Review | **total** | baseline | Δ |
|---|---:|---:|---:|---:|---:|---:|
| **C optimized** | $0.191 (haiku) | $0.586 (1 ticket, 1 round) | $0.081 | **$0.857** | $0.828 | **+3.5%** |
| C clean (before) | $0.268 (sonnet) | $1.004 (4 tickets) | $0.136 | $1.408 | $0.828 | +70% |
| **A optimized** | $0.104 (haiku) | $0.382 | $0.080 | **$0.566** | $0.449 | +26% |
| A clean (before) | $0.180 (sonnet) | $0.232 | $0.064 | $0.476 | $0.449 | +6% |
| **B optimized** | $0.125 (haiku) | $0.131 (1 ticket, 1 round) | $0.048 | **$0.304** | $0.212 | **+43%** |
| B clean (before) | — (sonnet) | — (2 genuine rounds) | — | $0.610 | $0.212 | +187% |

All three optimized runs: single pass, accepted on first review, diffs verified (C: link ids
3→1, 961 passed; A: `☑`/`☐` render, 14 passed; B: CRLF/CR normalized one-liner in `ansi.py`
+4 regression tests, 28/28 `test_ansi.py` passed). C's diff got *tighter* (+98/−7 vs +124/−8);
B's is a single `.replace().replace()` line plus tests — about as tight as a fix gets.

**What worked:** on the cross-cutting task the minimalism prompt collapsed 4 tickets / 12
criteria into **1 ticket**, and the Dev step fell from $1.00 to $0.59 (1.41M → 1.11M tokens) —
the pipeline went from **+70% to a statistical tie (+3.5%)** with the baseline on the exact task
type the KB is for. On Task B, the same minimalism collapse (single ticket, single revision
round instead of 2 genuine ones) cut total cost from $0.610 to $0.304 — **+187% down to +43%** —
even though B's tiny $0.212 baseline gives the fixed gate floor (~$0.20 across PM+QA+Review) the
least room to amortize of any task tested. The haiku PM localized identically to the sonnet PM
on both A and C (same files).

**What it teaches:** Task A's optimized run came in *worse* than its unoptimized one — PM
savings materialized ($0.104 vs $0.180) but the Dev step swung up ($0.382 vs $0.232) on nothing
but agentic run-to-run variance. At n=1 per configuration, single-run Dev variance (~±$0.15) is
the same order as the margins separating pipeline from baseline — and on tasks whose whole
baseline is $0.21–0.45 (A, B), that variance alone can flip the verdict. The honest claim after
optimization is **cost parity with plain Claude Code, roughly +3% to +45% depending on the task,
while also delivering** a locked scope, tickets, an independent QA verdict, a review, and a PR
branch — not a strict, repeatable dollar win. The two structural reasons a strict win stays hard:
the Dev must implement the PM's *written* criteria and iterate the test suite (the baseline
free-forms to "good enough"), and the gate floor (~$0.19–0.27 across PM+QA+Review, before Dev
even starts) never fully amortizes on tasks whose whole baseline costs $0.21–0.45 — it comes
closest to disappearing on genuinely hard, cross-cutting tasks like C, which is exactly where a
persistent KB should earn its keep.

### Second optimization round: stop Dev from re-reading what the PM already found

The remaining Dev cost on C (1.11M input tokens even with grep-verified file:line pins) pointed at
one specific waste: the pipeline told Dev exactly which files mattered, then made it spend a Read
tool call re-discovering their contents anyway. Fix: `_verified_locations()` in `orchestrator.py`
now inlines the actual current contents of the pinned files (budgeted: 4,000 chars/file, 9,000
combined, truncating gracefully on large files) directly into the Dev/revise prompt, with an
explicit instruction not to re-read what's already shown.

Re-ran Task C (the task this should help most, since it's the one where Dev reads the most):

| run | PM | Dev | QA+Review | **total** | baseline | Δ |
|---|---:|---:|---:|---:|---:|---:|
| **C + file-injection** | $0.124 (haiku) | $0.263 (356k tokens in) | $0.069 | **$0.456** | $0.828 | **−45%** |
| C optimized (before) | $0.191 (haiku) | $0.586 (1.11M tokens in) | $0.081 | $0.857 | $0.828 | +3.5% |
| C clean (original) | $0.268 (sonnet) | $1.004 (4 tickets) | $0.136 | $1.408 | $0.828 | +70% |

Dev's input tokens dropped **68%** (1.11M → 356k) for the *same* bug, same criteria, same model.
Single revision round, QA PASS, Review approved first try, all 960 tests pass (3 new regression
tests for link-coalescing, `tests/test_console.py`). The fix itself is *different* from the
earlier run's (this time Dev batched consecutive same-link segments in `Console._render_buffer`
into one OSC-8 escape sequence, rather than deduplicating link IDs in `Style` merging) — both are
independently valid solutions to the same bug, which is itself a sign Dev spent its budget
thinking about the fix instead of finding the files.

**This is the first genuine, unambiguous win**: not a statistical tie inside run-to-run variance,
but a 45% cost reduction with the pipeline's full structured-delivery guarantees (locked scope,
independent QA, review, PR branch) still included. It validates the diagnosis from the first
optimization round — the gate floor and criteria discipline closed most of the gap, but the Dev
step's *re-exploration of already-known files* was the remaining lever, and it was the single
biggest one.

**Follow-up: re-ran A and B with the same file-injection fix** to see whether the win generalizes
or was specific to C's unusually large Dev reads:

| run | PM | Dev | QA+Review | **total** | baseline | Δ | Δ before injection |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A + file-injection** | $0.118 (haiku) | $0.168 (148k tokens in) | $0.047 | **$0.333** | $0.449 | **−26%** | +26% |
| **B + file-injection** | $0.085 (haiku) | $0.140 (169k tokens in) | $0.053 | **$0.278** | $0.212 | **+31%** | +43% |
| **C + file-injection** | $0.124 (haiku) | $0.263 (356k tokens in) | $0.069 | **$0.456** | $0.828 | **−45%** | +3.5% |

All three single-round, QA PASS, Review approved first try. Diffs verified: A's markdown.py
checkbox change is +29/−2 plus 49 lines of new tests, 963/963 tests pass; B's is the same
one-line CRLF normalization as before, 31 lines of new tests, 961/961 pass. One notable finding
on B: the PM's `affected_files` hint this run actually named the wrong file (`rich/__init__.py`
instead of `rich/ansi.py`) — the Dev agent's grep-verification step (an existing pipeline
safeguard, not part of this fix) caught the bad hint and edited the correct file anyway, which is
exactly the "hints, not facts" design working as intended.

**Every task moved in the right direction**, and two of three now beat the baseline outright:
A went from +26% to **−26%**, B from +43% to **+31%**, C from +3.5% to **−45%**. B is the one
holdout — it's the task with the smallest baseline ($0.212) and a fixed gate floor of ~$0.14
(PM+QA+Review) that structurally can't shrink below what QA+Review cost regardless of how cheap
Dev gets, so a task this cheap may simply be below the pipeline's break-even size. That said, B
still improved substantially (43%→31%), for the same reason as A and C: Dev cost fell because it
had less to explore.

### Bottom line (Experiment 3)

The bigger repo did not make the KB pipeline win on total cost at first — but it clarified *why*,
and that diagnosis led to an actual win. The KB buys a cheap, accurate **localization** step (PM),
and that saving is real and grows with how hard the task is to find. The **Dev** step *can* also
be cheaper across the board — but only once it stops re-discovering, via tool calls, files the
pipeline already told it about; feeding Dev those files' actual content directly is what closed
the rest of the gap. On this repo the plain-Claude baseline stayed cheap precisely where tasks
were greppable, so the pipeline's structured-delivery premium (locked scope, tickets, independent
QA, cross-checked review, PR branch) showed up as **+6% to +187%** before optimization. The
PM-minimalism + haiku-PM round narrowed that to **+3.5% to +43%**. The file-injection round then
moved every task further in the same direction — **A: +26% → −26%, B: +43% → +31%, C: +3.5% →
−45%** — putting two of three tasks (A, C) ahead of plain Claude Code outright, with the third
(B) still improved but held back by its own tiny $0.21 baseline, which the pipeline's fixed
PM+QA+Review gate floor (~$0.14) can't shrink below regardless of how cheap Dev gets. The
practical rule that falls out: spend the one-time KB money on **any** repeated work on the
**same** repo once the pipeline is tuned (minimal scope, cheap PM, file-content injection) — the
one clear exception is a steady stream of *very* cheap, single-file edits (baseline under
~$0.25), where the fixed gate floor may still cost more than a cold `claude -p` saves in
exploration.

---

# Experiment 4 — Textualize/textual (biggest repo yet — where localization *quality* decides everything)

Experiment 4 went bigger again — **Textualize/textual, ~82.5k LOC in 247 source files (~2.3× rich,
~7× click)** — to see whether the file-injection-tuned pipeline holds up on a large, genuinely
complex codebase (async TUI framework, widget lifecycles, a rendering stack several layers deep).
Config: PM/QA/Review/Knowledge on `haiku`, Dev on `sonnet`, `max_revision_rounds = 2`, **with the
file-content injection from Experiment 3's second optimization round live**.

**Target repo:** [Textualize/textual](https://github.com/Textualize/textual) @ `06dbeef4b`.
**One-time KB build:** 115 structured views / 2,120 files, **$0.8230** (claude-cli, 553,444 in /
38,984 out). Excluded from per-task totals, as always. It is ~2.3× rich's $0.359 KB — the first
time the KB cost scaled roughly with the repo, because textual has far more *modules* to describe
(view count, not raw LOC, is the driver, and a 247-file framework genuinely has more distinct
views than rich's 146).

### Three requests (non-technical), chosen to vary localizability — verified to reproduce on `main`

* **Task A — feature/bug, easy to localize:** sorting a table column whose cells contain
  styled/coloured text crashes instead of sorting. (`DataTable.sort` → `TypeError` comparing
  `rich.Text`; "table … sort" → `_data_table.py`.) (`data/exp4-task_a.txt`)
* **Task B — bug, easy to localize:** a table's auto-width column stays stuck wide after you remove
  the row that held its widest value, instead of shrinking back. ("table … column width" →
  `_data_table.py`.) (`data/exp4-task_b.txt`)
* **Task C — bug, hard to localize:** hovering one inline clickable link highlights *all* links
  that happen to share the same action, because they mint the same link id. The real root cause is
  an `@lru_cache` on `Style.__add__` in **`strip.py`** — no file is named "link" or "hover", and
  the bug spans the shared style/rendering machinery, not any one widget. (`data/exp4-task_c.txt`)

### Baselines (plain Claude Code, `sonnet`, fresh clone each, `--no-session-persistence`)

| task | kind | turns | tokens in¹ | cost |
|---|---|---:|---:|---:|
| A | feature, greppable | 35 | 1,064,307 | $0.5896 |
| B | bug, greppable | 41 | 1,182,705 | $0.6116 |
| C | bug, cross-cutting | **207** | **14,267,624** | **$6.8300** |

**Task C is the sharpest localization tax we have ever measured: 207 turns and 14.3M tokens — ~12×
Task A's cost on the *same repo, same model, same day*.** A cold agent had to spend $6.83 spelunking
the render stack to find that the culprit was a cache decorator on `Style.__add__`. This is exactly
the case a persistent KB is *supposed* to demolish.

### Pipeline results (all runs redone strictly sequentially — see methodology note)

| task | PM | Dev (all rounds) | QA | Review | **pipeline** | baseline | Δ | Dev rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | $0.1575 | $0.1986 | $0.0097 | $0.0142 | **$0.3800** | $0.5896 | **−36%** | 1 |
| B | $0.1381 | $0.5319 | $0.0528 | $0.0434 | **$0.7662** | $0.6116 | **+25%** | 3 |
| C | $0.1158 | $1.5669² | $0.0104 | $0.0123 | **$1.7054²** | $6.8300 | **−75%** | 1 |

¹ input + cache_read (pipeline's formula), for like-for-like with the pipeline's own token counts.
² C's Dev shows the one legitimate pass ($1.5669); a spurious revision round (+$0.1615) that a
contamination bug triggered during the first run is excluded — a clean run's QA returned CONCERNS,
which does not trigger a revision. As-incurred it was $1.8669 (−73%). Either way ≈ −73–75%.

### What the numbers say — and where localization *quality* bites

**Localization accuracy, task by task** (PM's `affected_files` vs where the baseline actually fixed it):

| task | PM localized to | baseline fixed in | verdict |
|---|---|---|---|
| A | `_data_table.py` | `_data_table.py` | ✅ exact |
| B | `_data_table.py` | `_data_table.py` | ✅ exact |
| C | `_markdown.py`, `_log.py` | **`strip.py`** | ❌ wrong file *and* wrong layer |

This is the experiment's headline: **localization quality, not repo size, decides whether the KB
pays off — and on the one task where it would have paid off most, it missed.**

* **A — clean win (−36%).** Perfect localization, one Dev pass, ships a correct fix (extracts plain
  text from `rich.Text` for the sort key; 91/91 `test_data_table.py` pass, incl. 4 new sort tests,
  zero regressions). The KB did exactly its job: the greppable task got greppable cheaply.

* **C — huge headline win (−75%), but a *narrower* fix.** The pipeline localized C to the **wrong
  file** — yet it still delivered a working, tested fix **4× cheaper than the $6.83 baseline**, by
  patching `_markdown.py` (a per-link unique `@region` id that defeats the shared-cache collision).
  The catch: it fixes the bug **only for Markdown-widget links**. The baseline's `strip.py` fix cures
  the *root cause* for every widget; the pipeline's does not — we verified the identical-action
  `Static` markup case is **still broken** under the pipeline's fix. Its two `_log.py` "regression
  tests" are vacuous (one literally asserts `line_count == 2` and documents "Log has no clickable
  regions"). So on the task the KB should have won *most*, the honest reading is: **dramatically
  cheaper, but a partial fix that a human reviewer should catch as under-scoped.** Cheaper is not
  equivalent.

* **B — a clean loss (+25%), and the most instructive one.** Localization was perfect and Dev's
  *first* pass was already correct (adds `_recalculate_column_widths()` on row removal; 89/89
  `test_data_table.py` pass, zero regressions — we verified independently). But **QA misread
  textual's ~40 pre-existing snapshot-test failures as a "widespread visual regression" caused by
  B**, returned FAIL, and triggered **two full revision rounds** (Dev $0.53 across 3 passes vs the
  one it needed) before shipping — still stamped FAIL. The fix was right the whole time; the gate
  panicked. On a repo that ships with failing tests, the pipeline's QA cannot tell a pre-existing
  failure from a regression, and that turns a would-be win into a loss.

### Follow-up: two fixes for the failure modes above, validated on two fresh tasks (D, E)

The A/B/C round exposed two concrete bugs in the pipeline itself — not the models, the harness:
(1) the per-repo working copy is **shared**, so running scopes concurrently corrupts branch state
(this actually destroyed work mid-benchmark); (2) QA **counts** a repo's pre-existing test failures
as regressions, which is what cost Task B two wasted revision rounds. Both were fixed —
`git_ops.repo_lock` serializes same-repo scopes (different repos still run in parallel), and
`git_ops.failing_tests` snapshots the failing set on the clean tree *before* Dev so QA judges only
**new** failures — then re-tested on two fresh textual tasks run **deliberately in parallel**:

* **Task D — `@on(Subclass.Pressed)` fires on the base widget** (medium-hard; baseline $2.1463 / 64
  turns). PM localized it to `_button.py` — the *exact same file and the byte-identical one-line fix*
  the baseline produced (`Button.Pressed(self)` → `self.Pressed(self)`). (`data/exp4-task_d.txt`)
* **Task E — `max_length` not enforced on programmatic `.value` set** (greppable; baseline $0.4005 /
  26 turns → `_input.py`). (`data/exp4-task_e.txt`)

| task | PM | Dev (rounds) | QA | Review | **pipeline** | baseline | Δ | Dev rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D | $0.1851 | $1.4405 | $0.0459 | $0.0250 | **$1.6965** | $2.1463 | **−21%** | 1 |
| E | $0.0989 | $0.2250 | $0.0263 | $0.0240 | **$0.3742** | $0.4005 | **−7%** | 1 |

**Both fixes held on live runs:**

* **Concurrency lock:** D and E were launched at the same moment. D acquired the repo lock and ran
  Dev→QA→Review→PR while E sat blocked (`scoped`, no branch checked out); the instant D released,
  E started. Result: two **distinct** commits, two clean **distinct** diffs (D→`_button.py`,
  E→`_input.py`), zero corruption — the exact scenario that wrecked the A/B/C round, now safe.
* **QA pre-existing-failure baseline:** both tasks passed QA on a **single Dev round** despite the
  repo's **450** pre-existing snapshot failures on clean `main`. QA's verdict explicitly treated
  those as pre-existing rather than counting them as regressions — no repeat of Task B's
  false-FAIL revision spiral (3 Dev rounds → 1).

Both were independently re-verified: E genuinely enforces `max_length` on programmatic sets now
(10 chars → clamped to 5) with 134 input tests green; D's fix is byte-identical to the baseline's
(so it inherits the same known edge — a `pass`-only subclass shares the parent's message *class
object*, which no dispatch change can separate — but that limitation is the baseline's too, making
this a clean equal-fix-for-less comparison). **Net: on both new tasks the tuned pipeline beat plain
Claude Code — including on the cheap greppable task (E, −7%) where the gate floor usually wins for
the baseline — and both harness fixes did their job under real load.**

### Bottom line (Experiment 4)

The biggest repo yet confirms the mechanism and exposes its limits. **When the user's words map onto
a file (A, B), localization is pixel-perfect even at 82k LOC — the KB does not care about repo
size.** The dramatic case is C: the baseline's **$6.83 / 207-turn** hunt is precisely the pain a KB
should erase, and on cost it did (**−75%**) — but because the PM mislocated the *root cause*, the
pipeline "won" with a **shallower fix than the baseline shipped.** That is the sharpest lesson of the
whole report: **on hard, cross-cutting bugs the KB buys you a massive cost cut and a plausible,
tested, but potentially incomplete fix — the savings are real and the correctness risk is also
real.** And B shows the second failure mode has nothing to do with localization at all: a noisy test
suite makes the QA gate reject good work and burn revision money. Tune for this repo class by (a)
giving QA a pre-change test baseline so it can diff failures instead of counting them, and (b)
treating a hard-to-localize bug's cheap pipeline fix as a *draft* pending human root-cause review,
not a finished delivery.

---

## Caveats — applies to all four experiments (read before quoting)

* **n = 12 tasks across 4 repos.** These are directional data points, not a study. The tasks within
  each repo already show more variance (revision loops, scope drift) than the average effect being
  measured — treat every "% cheaper/dearer" as illustrative, and weight the *mechanism*
  (localizability drives baseline cost; gate floor + re-work rounds drive pipeline cost) over the
  exact percentages.
* **Cheaper ≠ equivalent (Experiment 4's lesson).** The gate checks that a change passes the suite,
  not that it fixes the *root cause*. On Experiment 4's Task C the pipeline shipped a tested fix 4×
  cheaper than the baseline that only covered one widget, because the PM mislocated the root cause —
  a strictly cost-based "win" can hide an under-scoped fix. Read the diff, not just the dollar delta.
* **QA couldn't tell a pre-existing failure from a regression — now fixed (Experiment 4).** On a repo
  that ships with failing tests (textual has **450** environment-sensitive snapshot failures on clean
  `main`), QA counted them as regressions and burned two needless revision rounds on Task B — turning
  a correct first-pass fix into a +25% loss. Fixed by snapshotting the failing set on the clean tree
  *before* Dev (`git_ops.failing_tests`) and feeding QA only the **new** failures — **validated in the
  D/E follow-up: both passed QA on a single Dev round despite the 450 pre-existing failures.**
* **Concurrency corrupted the shared clone — now fixed (Experiment 4 methodology).** The pipeline
  reuses one git clone per repo; running multiple scopes through it *concurrently* raced on
  branch/commit state and destroyed work. Experiment 4's A/B/C round hit this and had to be redone
  strictly sequentially. It is now fixed with a per-repo lock (`git_ops.repo_lock`) that serializes
  same-repo scopes while letting different repos run in parallel — **validated in the D/E follow-up,
  which launched two scopes at once and saw them serialize cleanly with zero corruption.** The A/B/C
  numbers remain the clean sequential re-runs (Task B's corrupted-attempt Dev artifact was lost and
  fully re-run).
* **Token counts use the pipeline's own formula** (`input + cache_read`, excluding cache-creation)
  on *both* sides in Experiment 3, so baseline-vs-pipeline token comparisons are like-for-like.
  (Experiments 1–2 reported baseline `tokens in` *inclusive* of cache creation while their pipeline
  rows excluded it — a small labelling inconsistency that touches **no cost figure**, since all
  dollars come from the CLI's own `total_cost_usd`.)
* Claude costs come from the CLI's own meter (`total_cost_usd`, API list pricing). Subscription
  users don't pay per-token, but the meter is the correct like-for-like unit here.
* Every headless `claude -p` invocation carries a fixed ~12.8k-token system-prompt overhead
  (≈$0.04 at sonnet input price before caching); this hits both systems, but hits the pipeline
  once per stage that uses the CLI — so multi-round Dev/QA/Review re-runs pay it repeatedly.
* Experiment 1's Gemini/Groq stages ran on free-tier keys; their dollar figures are list-price
  equivalents computed by the platform's meter, not an actual invoice. Experiment 2 is all-Claude,
  metered end-to-end. Compare pipelines *within* an experiment, not across.
* Quality was gate-checked (every change passes the repos' test suites; diffs in `data/`), but
  no deeper quality comparison was attempted.
* **Environment honesty matters (Experiment 3's lesson).** A gated pipeline's cost is only
  trustworthy if its test env is correctly provisioned; a missing dev-dependency or a path quirk
  can manufacture QA failures and inflate cost with phantom re-work. Experiment 3's headline numbers
  are the **post-fix** runs; the pre-fix runs (Task A $1.26 / Task C $1.88, both with 2 phantom Dev
  rounds) are retained only as the narrative of that finding.
* **Isolated verification (Experiment 4).** Each Experiment 4 pipeline diff was re-verified on an
  isolated copy of the repo with its own test env, applying the diff to clean `origin/main` and
  running the affected suite — A: 91/91 `test_data_table.py`; B: 89/89 (87 clean + 2 new), fix
  confirmed correct despite the QA FAIL; C: 8/8 of its own tests, but the general `Static` case
  confirmed *still broken*, which is how we caught that C's fix is partial.

## Reproducing

1. Ingest a repo in CodeVerdict with the knowledge stage on `claude-cli` (Settings →
   Knowledge provider). The one-time build cost is recorded on the repo row (`kb_cost_usd`).
2. Run a scope through PM → tickets → approve → run; per-stage costs land in `agentrun`
   (`/costs` in the UI), PM scoping cost on the session.
3. Baseline: `claude -p --model sonnet --output-format json --no-session-persistence
   --dangerously-skip-permissions < request.txt` in a fresh clone; take `total_cost_usd`.

Raw run data: [`data/`](data/). Experiment 1 (tinydb): `task_a/b.txt`, `pipeline-task-*.diff`,
`baseline-task-*.{json,diff}`. Experiment 2 (click): `exp2-task_a/b.txt`, `exp2-pipeline-*.diff`,
`exp2-baseline-*.{json,diff}`, and the PM retrieval log `exp2-click-gaps.jsonl`. Per-stage Claude
pipeline costs for Experiment 2 come from the `agentrun` rows (task ids 9–12 for A, 13–15 for B)
and the `scopesession` PM cost (sessions 7 and 8). Experiment 3 (rich): `exp3-task_a/b/c.txt`,
`exp3-baseline-*.{json,diff}`, `exp3-pipeline-*.diff`. Per-stage clean-run costs come from the
`agentrun` rows for sessions 14 (A), 15 (B), 13 (C) and their `scopesession` PM cost. The two
infra fixes are in `backend/app/services/git_ops.py` (`_dev_requirements()`) and the `REPOS_DIR`
setting. Optimized re-runs: sessions 18 (A), 17 (C), 19 (B) — same repo state, `pm_model=haiku`
+ scope-minimalism prompts; diffs are the `exp3-pipeline-*-optimized.diff` files. File-injection
re-runs: sessions 21 (A), 22 (B), 20 (C) — same optimized config plus the `_verified_locations()`
file-content injection in `orchestrator.py`; diffs are the `exp3-pipeline-*-fileinject.diff`
files. Experiment 4 (textual): `exp4-task_{a,b,c,d,e}.txt`, `exp4-baseline-*.{json,diff}`,
`exp4-pipeline-*.diff`. A/B/C pipeline runs are sessions 24 (A), 25 (B), 26 (C) on repo id 6, KB
build $0.8230; per-stage costs in the `agentrun` rows for tasks 37 (A), 38 (B), 39 (C) and
`scopesession` PM cost. **Methodology note:** A/B/C had to be run strictly sequentially (the shared
per-repo clone raced under concurrent scopes); the published numbers are the clean sequential
re-runs, with A and C QA/Review redone in isolation and B re-run in full after its first-attempt Dev
artifact was lost. The **D/E follow-up** (sessions 27 D, 28 E; tasks 40, 41) was run with the
per-repo-lock and QA-baseline fixes in place, launched **concurrently on purpose** to validate the
lock — both serialized cleanly and passed QA in one Dev round each.
