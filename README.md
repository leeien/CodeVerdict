<div align="center">

<img src="docs/screenshots/cli-session.gif" alt="CodeVerdict in the terminal: which model owns each stage, an indexed repository, a request as tickets, the Planner's verified plan, and a four-judge jury with each juror's verdict" width="900">

<sub>One unbroken session against <a href="https://github.com/go-gitea/gitea">go-gitea/gitea</a>
— Go, TypeScript and templates, 120,521 symbols indexed. Which model owns each of
the six stages, the request that started it, the ticket it became, the Planner's
plan pinned to real symbols, then the jury: four judges on four different
providers, each with its own verdict, one of them dissenting. Every screen is
real state from a delivery that actually ran; nothing here is a mockup.</sub>

<br>

### A terminal-first coding agent that is reviewed by a **jury**, not by a judge.

**One LLM grading another LLM's code is not a review — it is a coin flip with a
confident voice.** So CodeVerdict sends every change to a panel of independent,
differently-modelled judges and a foreperson who synthesizes one verdict. And
because a panel costs tokens, the agent earns them back: it navigates your
repository through a **persistent code graph plus semantic search over graph
nodes**, so it *looks up* where a change goes instead of burning context
rediscovering it.

**And nothing about it is fixed.** Six stages — *knowledge, PM, planner, dev, QA,
review* — and you choose the provider and model for every one of them, live, from
the terminal. Claude plans, Codex writes, GPT reviews. The jury itself is a roster
you seat, re-model and re-brief yourself. All of it runs on the coding CLIs you're
already logged into, so **you may not need an API key at all**.

<br>

`pip install` · one command to run · no API key required · **macOS · Linux · Windows**

[![CI](https://github.com/leeien/CodeVerdict/actions/workflows/ci.yml/badge.svg)](https://github.com/leeien/CodeVerdict/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)
![Windows](https://img.shields.io/badge/Windows-0078D4?logo=windows&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 1. The jury: why one LLM judge is not enough

Every agentic coding tool ends up needing a reviewer, and almost all of them use
the same design: ask one model whether the diff is good. That design fails in a
predictable shape.

A single judge is **confident, fast, and blind in exactly the places the author
was blind** — same training distribution, same intuition about what "looks
right", same willingness to accept a plausible diff that never actually wires
the feature through. Asking it for a second opinion returns the first opinion
again. It is a *single point of judgement* on the one step whose whole job is to
catch what the previous step got wrong.

This isn't a hunch. It is measured.

### The evidence: SE-Jury (arXiv 2505.20854)

CodeVerdict's review stage is built on the finding in **[SE-Jury: An
LLM-as-Ensemble-Judge Metric for Narrowing the Gap with Human Evaluation in
SE](https://arxiv.org/abs/2505.20854)** (Zhou et al., 2025 — Singapore
Management University, DGIST, Monash, NUAA, Huawei, Zhejiang University). The
paper asks precisely the question a reviewing agent depends on: *how well does an
automatic judge agree with human experts about whether generated code is
correct?* — and it shows an **ensemble of judges beats every single judge tested,
including the state of the art.**

**Table I — correlation with human correctness scores** (Kendall's τ and
Spearman's *rₛ*, ×100; `avg` = mean of the two; the final column is the average
across tasks). Backbone: GPT-4o-mini, temperature 0.

| Metric | CoNaLa (codegen) | Card2Code (codegen) | APR-Assess (repair) | Summary-Assess | **Average** |
|---|---:|---:|---:|---:|---:|
| BLEU | 31.0 | 49.0 | 26.9 | 14.3 | 30.4 |
| CodeBLEU | 23.9 | 46.1 | 21.8 | 14.6 | 26.7 |
| CrystalBLEU | 28.1 | 42.7 | 31.8 | 13.7 | 29.1 |
| ROUGE-L | 47.5 | 58.8 | 22.4 | 19.6 | 37.1 |
| METEOR | 41.2 | 72.5 | 45.0 | 19.9 | 44.7 |
| BERTScore | 45.9 | 62.5 | 1.5 | 22.8 | 33.2 |
| CodeBERTScore | 44.3 | 65.1 | 6.1 | 15.9 | 32.9 |
| **Vanilla LLM** (one judge, plain prompt) | 45.6 | 70.4 | 43.5 | 33.9 | 48.4 |
| **ICE-Score** (one judge, SOTA prompt) | 55.2 | 66.5 | 43.5 | 33.1 | <ins>49.6</ins> |
| **SE-Jury (ensemble of judges)** | **63.5** | **80.3** | **76.2** | **37.3** | **64.3** |

Read the bottom three rows together. The best *single* LLM judge in the
literature scores **49.6**. The same underlying model, arranged as a panel,
scores **64.3** — and the gap is not uniform, it is largest exactly where review
is hardest:

| | SE-Jury vs. best single-judge baseline |
|---|---:|
| Code generation (CoNaLa, Card2Code) | **+17.9%** |
| **Automated program repair (APR-Assess)** | **+75.2%** |
| Code summarization (Summary-Assess) | **+12.7%** |
| Over all existing automatic metrics | **+29.6% … +140.8%** |
| Over *the same LLM used alone* | **+32.9%** |

> **Program repair — fixing a bug in existing code — is the closest task in the
> paper to what a coding agent actually ships, and it is where the ensemble's
> margin is biggest: 76.2 vs 43.5.** A single judge is barely better than
> coin-flipping on whether a patch is really a fix. This is the number that
> justifies the whole jury.

### Every individual judge is worse than the panel

The most useful table in the paper is the ablation, because it kills the obvious
objection — *maybe one of the judges is just the good one*. It isn't. SE-Jury
defines five evaluation strategies (S1 direct assess, S2 assess-then-rethink, S3
equivalence to a reference, S4 extract-key-properties-then-check, S5
generate-tests-then-check) and measures each **alone**:

**Table II — individual judges vs. the ensemble** (avg correlation, ×100)

| Judge | CoNaLa | Card2Code | APR-Assess | Summary-Assess | **Average** |
|---|---:|---:|---:|---:|---:|
| S1 — direct assess | 60.1 | 59.8 | 68.6 | 36.1 | 56.2 |
| S2 — assess + rethink | 56.4 | 76.4 | 63.1 | 34.9 | 57.7 |
| S3 — equivalence assess | 55.6 | 79.4 | 76.9 | 21.4 | <ins>58.3</ins> |
| S4 — analyze reference | 47.9 | 67.8 | 56.1 | 20.3 | 48.0 |
| S5 — generate tests | 49.3 | 55.8 | 68.2 | n/a | 57.8 |
| the underlying LLM, alone | 45.6 | 70.4 | 43.5 | 33.9 | 48.4 |
| **SE-Jury (the panel)** | **63.5** | **80.3** | **76.2** | **37.3** | **64.3** |

Three things fall out of that table, and all three are load-bearing for CodeVerdict:

1. **No single judge wins.** The best one averages 58.3; the panel averages 64.3.
   The ensemble beats S1/S2/S3/S4/S5 by **14.4%, 11.4%, 10.3%, 34.0% and 11.2%**
   respectively.
2. **Single judges are wildly unstable across tasks.** S3 is the best judge
   overall — and it is the *second worst* on summarization (21.4). S1 is
   mid-table everywhere. Whichever single judge you pick, you are picking its
   blind spot too. The panel is the only configuration that is strong on *every*
   task.
3. **The gain is the ensemble, not the prompt.** Strip the panel down to one
   plain prompt on the same model and you get 48.4. The strategy design alone
   contributes **+27.9%**; the diversity-plus-aggregation structure carries the
   rest.

### How close is a panel to a *human* reviewer?

The paper's second result is the one that matters if you want to trust a verdict
without reading the diff yourself. It compares **human↔human** agreement
(Cohen's κ between pairs of expert annotators) against **human↔tool** agreement:

| Dataset | Human ↔ human κ | SE-Jury ↔ human κ | |
|---|---:|---:|---|
| APR-Assess (program repair) | — | comparable median **and** mean | ✅ at inter-annotator level |
| Card2Code (code generation) | 30.5 | **41.9** | ✅ exceeds human agreement |
| CoNaLa (code generation) | 25.7 | 16.7 | 🟡 approaches it |
| Summary-Assess (summarization) | 15.5 | 1.9 | ❌ not a substitute |

On program repair the panel agrees with human experts about as much as human
experts agree with each other — and on Card2Code it agrees *more*. (Note the
honest failure: judging prose summaries is still not solved, by anyone. That is
also the task furthest from what CodeVerdict does.)

And it generalizes past human labels to **actual test execution** — 0 for a
failing suite, 1 for passing:

**Table IV — correlation with test-execution outcomes** (avg, ×100)

| Metric | HumanEval-X: Python | Java | C++ | JavaScript | Go | APPS |
|---|---:|---:|---:|---:|---:|---:|
| CodeBLEU | 52.5 | 41.4 | 43.7 | 47.5 | 31.8 | 20.5 |
| Vanilla LLM | 62.9 | 59.1 | 57.4 | **62.3**\* | 45.7 | 16.0 |
| ICE-Score | 60.1 | 53.0 | 59.3 | 60.5 | 53.9 | 27.7 |
| **SE-Jury** | **74.0** | **63.6** | **72.2** | 62.3 | **55.9** | **47.4** |

<sub>\* JavaScript is effectively a tie. Across the five HumanEval-X languages the
panel beats the best baseline by **+14.4%**; on the much harder APPS problems, by
**+71.1%**. Against **CodeJudge**, the previous state-of-the-art LLM code judge:
**+9.6%** average, **+36.8%** over CodeJudge without a reference.</sub>

Two more findings from the paper that shaped CodeVerdict's design directly:

- **Robust to the backbone.** Swapping GPT-4o-mini for DeepSeek-Chat moved the
  average 64.3 → 64.9. The ensemble is a *structure*, not a prompt tuned to one
  vendor — which is why CodeVerdict lets you point each judge at a different model.
- **The panel is cheap, and picking a subset is cheaper.** ~**US$0.10 per 100
  samples** at GPT-4o-mini prices; dynamic team selection cut LLM usage **~50%**
  with no loss (64.3 selected vs 61.9 for merge-everything). Judges you don't
  seat are judges you don't pay for — which is exactly how `/jury` works.

### What CodeVerdict takes from the paper — and where it differs

Being precise about this, because the numbers above are **SE-Jury's, measured on
SE-Jury** — they are the evidence for the *architecture*, not a benchmark of this
repository:

| | SE-Jury (the paper) | CodeVerdict (this repo) |
|---|---|---|
| Core thesis | An ensemble of independent LLM judges tracks human correctness judgement far better than any single judge | ✅ adopted wholesale — the review stage is a panel by default |
| Source of diversity | Five **evaluation strategies** over one backbone | **Concern + model family**: correctness, reliability & edge cases, security, architecture (plus optional performance and test-quality seats), each on its own provider/model |
| Judge independence | Judges never see each other's scores | ✅ same — all judges run in parallel on the same case file, blind to each other |
| Team selection | Trial teams on 20 annotated samples, keep the best; ~50% cost cut | Operator-selected roster (`/jury`): two specialists ship disabled by default for the same cost reason |
| Aggregation | Mean of the team's scores | **LLM foreperson** — merges duplicates, resolves conflicts in favour of quoted evidence, drops low-confidence findings, issues one verdict; a deterministic merger takes over if it fails |
| Output | One correctness score | Structured findings — evidence, severity, confidence, suggested fix — plus a verdict that gates delivery |
| Judged against | Human labels and test outcomes | The requirement, the acceptance criteria, the repo's own knowledge, call-graph impact analysis, the implementer's account, and the diff |

The strategy diversity is not a substitute for concern diversity — it is a
different axis, and extending the roster along SE-Jury's axis too (an
equivalence judge, a generate-tests judge) is on the roadmap.

### The panel, as implemented

| | |
|---|---|
| **Independent** | Every judge gets the same case file — requirement, acceptance criteria, repository knowledge, call-graph impact analysis, the implementer's own account, the diff — and reviews it in parallel, blind to the others. Correlated errors are the failure being engineered away. |
| **Specialized** | Each brief narrows one judge's attention *and names what it must leave to the others*. Correctness hunts dead wiring and unmet criteria; reliability hunts empty inputs, error paths and leaks; security wants a **reachable** attack path; architecture is the seat that knows how the rest of your repo does things. |
| **Differently-modelled** | Judges default to different providers (`spread_providers`). A panel that all runs one model agrees with itself — including where it is wrong. |
| **Evidence-bound** | Findings are structured, not prose: quoted evidence, why it matters, severity, confidence, suggested fix. That is what makes them mechanically weighable. |
| **Synthesized** | The foreperson merges duplicates — independent corroboration is the strongest signal a panel produces — resolves conflicts in favour of quoted evidence over general principle, drops findings under the confidence floor, and returns one verdict. Only real defects block; taste becomes an observation. |
| **Fail-loud** | A judge that errors, times out or returns garbage **abstains and says so**. A missing juror must never read as a clean one. All judges fail → `INCONCLUSIVE`, never `APPROVED`. Foreperson unreachable → a deterministic merger runs and labels itself as the fallback. |
| **Yours** | `/jury` seats, unseats, reorders, re-models and re-briefs judges live — including custom judges with your own charge (house API rules, a compliance checklist, whatever your team actually argues about in review). Each judge bills to its own run, so `/costs` shows what the panel costs. Turn the ensemble off and the classic single-reviewer path runs unchanged. |

---

## 2. Six stages. Any model on any one of them. And a jury you pick yourself.

Everything above describes the review seat. But CodeVerdict is not one agent with a
reviewer bolted on — it is **six specialized stages**, and *you* choose the
provider and the model for each one, live, from the terminal. This is the thing
that separates it from a pack of skills or subagents: a skill pack lives *inside*
one vendor's agent and runs one vendor's model everywhere. CodeVerdict runs *across*
them.

### The six stages

| Stage | What it does | Wants a model that's… |
|---|---|---|
| **Knowledge** | Builds and refreshes the repo's knowledge base — structured views over the code graph. Runs once per repo, then incrementally. | cheap and high-volume |
| **PM** | Runs a Socratic clarify-loop against your plain-English request, hunting ambiguity, then drafts engineering tickets. Owns the requirement, never the code. | conversational, good at pushing back |
| **Planner** | Reads the repo through the code graph and decides *how* the change is made: ordered steps, blast radius, which tests to extend. Every symbol it names is then verified against the graph and ripgrep. | strong at reasoning, cheap — it's read-only |
| **Dev** | Implements the verified plan on an isolated `agent/<key>` branch. | your best coding model |
| **QA** | Runs the test suite and reviews the result independently of Dev. | reliable at reading output, not expensive |
| **Review** | The **jury** — N specialized judges in parallel, plus the foreperson. | a *different family* than Dev (see below) |

Point each one wherever you like:

```bash
/models                                   # who owns each stage, and is it installed here?
/model dev anthropic claude-sonnet-5      # repoint one stage
/model planner gemini gemini-3.5-flash-lite
/model qa groq openai/gpt-oss-120b
/settings                                 # or drive it all from the panel — `p` applies
                                          #   a one-provider preset if you'd rather not
                                          #   choose twelve times
```

A working combination people actually run: **Claude plans, Codex writes, GPT
reviews.** Mix vendors however you like — and note that the reviewer running a
*different model family* than the author is not a cosmetic choice. It is the same
decorrelation argument as §1, applied one layer up: a model reviewing its own
output shares its blind spots. §1's paper measured what happens when you fix
that; this is the knob that lets you.

### You do not need a single API key

The coding *and* planning stages run **natively on the agents you already pay
for** — **Claude Code, Codex, Cursor, Aider, Gemini CLI** — driven headless
through *their own login*. Point a stage at Claude Code and it just uses your
existing subscription and plan: no per-token API billing, no keys to manage.
`/settings` auto-detects which CLIs are on your machine, shows the version, and
one-click installs the ones that aren't; `/doctor` tells you which are usable
right now.

And if you'd rather use keys, any of them work: the native Anthropic Messages
API, or any OpenAI-compatible endpoint — OpenAI, Groq, Gemini, xAI, OpenRouter,
Together, DeepSeek, a local Ollama. The shipped defaults target Groq's **free
tier**, so the entire pipeline can run for **$0**.

### Choosing your own jury

The panel is a roster, not a fixture. `/jury` opens it:

```
  Review jury — 4 of 6 seated

  ✓  Correctness & Requirements     claude-cli / sonnet
  ✓  Reliability & Edge Cases       groq / openai/gpt-oss-120b
  ✓  Security                       gemini / gemini-3.5-flash-lite
  ✓  Architecture & Maintainability anthropic / claude-sonnet-5
  ·  Performance & Scalability      not seated
  ·  Test Quality                   not seated

  space seat/unseat   m model   a add judge   d remove
  shift+↑/↓ reorder   R reset   q close
```

| | |
|---|---|
| **Six built-in seats** | Correctness & requirements, reliability & edge cases, security, architecture & maintainability ship **seated**. Performance & scalability and test quality ship **unseated** — they aren't worth their cost on a typical change, and an unseated judge is a judge you don't pay for. Seat them for a change where they matter. |
| **One model per seat** | `m` gives any judge its own provider and model. Judges default to *spreading across providers* — a panel that all runs one model agrees with itself, including where it's wrong. Leave a seat's model blank and it inherits the Review stage's. |
| **Write your own judge** | `a` adds a custom seat with your own charge: house API conventions, a compliance checklist, the thing your team argues about in every review. It's a text brief — you're writing the juror's instructions, not code. |
| **Order matters** | `shift+↑/↓` reorders, and roster order is the order opinions reach the foreperson. |
| **It tells you when a seat is broken** | A judge pointed at a provider with no usable key or CLI is flagged in the panel — *"this seat cannot run: it will abstain, and its perspective will be missing from every verdict."* A silently missing juror is the one failure mode that would make the whole ensemble a lie. |
| **Costs are legible** | Each judge bills to its own run, so `/costs` shows exactly what the panel costs. A jury multiplies the review stage's bill by its size; that should be visible, not buried. |
| **Or turn it off** | Unseat everything but one and the classic single-reviewer path runs unchanged. §1 is the argument for why you shouldn't — but it's your call, not the tool's. |

---

## 3. Efficiency: it looks changes up in a graph instead of hunting for them

A jury multiplies the review bill by its size. That is only a good trade if the
rest of the pipeline is *cheaper* than a cold agent — and it is, because of where
CodeVerdict spends its first tokens.

A cold coding agent pays the **localization tax on every single task**: dropped
into a repo it has never seen, it greps, opens files, guesses, backtracks, and
re-derives the same architecture from scratch every time. On a large codebase
that tax is brutal and recurring.

CodeVerdict pays it **once**. Ingesting a repo builds a durable knowledge base that
persists on disk and re-syncs as the repo moves:

- **A persistent code graph** — definitions, call edges, imports, HTTP routes
  across 158 languages — giving exact `file:line` lookups and call-graph impact
  analysis. Deterministic, and free: no per-repo LLM or embedding spend.
- **Semantic search over the graph's nodes** — the graph's symbols are embedded
  locally (fastembed ONNX + embedded Qdrant; no Docker, no API key) and fused
  with BM25 keyword search via weighted **Reciprocal Rank Fusion**. So *"show how
  far along a long-running job is"* finds `progress.py` even though it shares not
  one token with it. Measured **8/10 vs 4/10 top-5 recall** against keyword-only
  on vocabulary-mismatch queries.
- **Compounding delivery memory** — after each scope ships, the KB records what
  it *learned*: files touched, symbols added, gotchas, wiring. Run #50 starts
  from what runs #1–49 proved.

### Any language, and more than one at a time

Nothing in the pipeline is Python-shaped. The code graph parses **158 languages**
through bundled tree-sitter grammars, and every language-specific decision —
symbol extraction, import parsing, the edit-time syntax gate, what counts as a
test file, which test runner to use — lives in one registry that dispatches on
file type. Add a language by teaching that registry, not by touching the
pipeline.

That matters most on repos that are not one language. A real example: a
**6,174-file Go + TypeScript + Vue monorepo** indexes to 120,521 graph nodes and
15,991 embedded symbols, and a plain-English question reaches both halves of it —
*"issue dependency search"* returns the Go model, *"dependency dropdown
candidates"* returns the TypeScript component, from the same query interface. A
change that starts in a template and ends in a backend handler is one lookup, not
two investigations.

Two rules keep this honest rather than aspirational:

- **Fail open, never fail wrong.** An unrecognised language still gets its files
  indexed, its edits applied, and its tests reported as *couldn't run* — never as
  passing. Degraded retrieval is a worse answer; a fabricated green is a wrong
  one.
- **A manifest is not an ecosystem.** Polyglot repos carry manifests for their
  *tooling* — gitea ships a `pyproject.toml` whose entire contents pin three
  Python linters, against zero `.py` files and 3,024 `.go` files. Detection
  requires the source to actually be there, or QA runs the wrong suite and
  reports nothing useful.

### What that buys, measured

Head-to-head against plain Claude Code, identical plain-English request to both,
on the two largest repos **benchmarked** — Textualize/rich (~35.6k LOC) and
Textualize/textual (~82.5k LOC). Every number below was measured end to end.

| Repo | Task | CodeVerdict | Cold `claude -p` | Δ |
|---|---|---:|---:|---:|
| rich | feature | **$0.333** | $0.449 | **−26%** |
| rich | cross-cutting bug | **$0.456** | $0.828 | **−45%** |
| textual | feature | **$0.380** | $0.590 | **−36%** |
| textual | extreme cross-cutting bug | **$1.705** | $6.830 | **−75%** |
| textual | medium bug | **$1.697** | $2.146 | **−21%** |
| textual | greppable bug | **$0.374** | $0.401 | **−7%** |

- **The saving scales with how hard a change is to find.** On the extreme case
  the cold baseline spent **$6.83 over 207 turns and 14.3M tokens** hunting one
  cache-decorator bug. CodeVerdict localized and fixed it for a quarter of that —
  *and* shipped it with a locked scope, independent QA, a jury review and a PR
  branch that a raw `claude -p` never produces.
- **Repo size isn't the driver — localizability is.** A cross-cutting bug can
  cost a cold agent 3.5× the tokens of a greppable one in the *same* repo.
- **It's honest about the edges.** On a *very* cheap task (baseline ~$0.21) the
  fixed gate floor costs more than a cold run saves; and on the hardest bug the
  cheap win shipped a narrower fix than the baseline did. The
  [full report](benchmarks/kb-vs-claude-code.md) documents both, plus the
  earlier un-tuned iterations that led here.

> The benchmark is the most interesting artifact in this project. Read it before
> the code.

---

## 4. The code graph comes from `codebase-memory-mcp`

The graph engine is **not** written here. CodeVerdict's knowledge base is built on
top of **[`codebase-memory-mcp`](https://github.com/DeusData/codebase-memory-mcp)**
(DeusData) — a single static binary that indexes a repository into a persistent
code graph: definitions, call edges, imports and HTTP routes across **158
languages** via bundled tree-sitter grammars, with call-graph impact analysis and
exact `file:line` resolution. It indexes in seconds, needs no daemon, no Docker
and no API key, and CodeVerdict shells out to its CLI (`codebase-memory-mcp cli
<tool>`).

Using a purpose-built external indexer rather than a home-grown one is
deliberate: language coverage is the kind of problem that is only ever solved by
a project whose whole job it is.

```bash
npm i -g codebase-memory-mcp     # or Homebrew / Scoop / a release binary
```

`./run.sh` installs it for you when npm is present. **It is optional** — without
it the KB degrades to a built-in symbol map (stdlib `ast` for Python, tree-sitter
or regex extractors elsewhere) plus `ripgrep`. That tier works; it just costs
more tokens per task, which `/doctor` will tell you.

---

## 5. Getting started

### Prerequisites

Just Python and git, honestly.

- **Python 3.11+** and **git** — the only hard requirements.
- **One** way to power the agents. You do **not** need an API key:
  - an **already-logged-in coding CLI** — Claude Code, Codex, Cursor, Aider or
    Gemini CLI — driven headless on **your existing subscription**, no key, no
    per-token billing; **or**
  - **any LLM API key** — the native Anthropic API, or any OpenAI-compatible
    endpoint (OpenAI, Groq, Gemini, xAI, OpenRouter, Together, DeepSeek, local
    Ollama). Defaults target Groq's **free tier**, so the whole thing can run for
    **$0**.
- *Recommended, all optional:* `ripgrep` (sharper localization), the
  `codebase-memory-mcp` binary (the full code graph), `gh` (to open real PRs).

### Which OS?

**All three.** The application is pure Python — `pathlib` throughout, no shelling
out through a shell, no platform branches — and every dependency it needs is
cross-platform: Rich, prompt_toolkit and Textual all support Windows consoles,
and the external binaries ship for all three (`ripgrep`, `gh`, and
`codebase-memory-mcp` via npm, Homebrew *and* Scoop). **CI runs the full test
suite plus a CLI smoke test on macOS, Linux and Windows** — the badge above is
the claim, not this paragraph.

| | |
|---|---|
| **macOS** · **Linux** | `./run.sh` — nothing else to know. |
| **Windows** | `pip install -e ".[semantic,treesitter]"` then `codeverdict`. `run.sh` is bash, so use **PowerShell/CMD with pip**, or run the script under **WSL** or **Git Bash** (it already handles the `Scripts\activate` venv layout). |

Two small platform notes, for honesty: five tests that stand up a fake CLI on
disk rely on shebang dispatch and skip on Windows (the adapters under test are
portable — they only ever `subprocess.run` a resolved path). And the *coding
CLIs* themselves have their own platform support; `/doctor` tells you which ones
are actually usable on the machine you're on.

### Install and run — one command

```bash
git clone https://github.com/leeien/CodeVerdict
cd codeverdict
./run.sh
```

That's it. `run.sh` creates the venv, installs CodeVerdict and its extras, installs
the code-graph binary if npm is around, copies `.env.example` → `.env`, prints a
**preflight report** on first run, and drops you into the shell.

Prefer pip?

```bash
pip install -e ".[semantic,treesitter]"
codeverdict
```

### Is my machine set up? Ask it

```bash
codeverdict doctor          # or /doctor in the shell, or ./run.sh doctor
```

```
  ╭─  Preflight  ──────────────────────────────────────────────────────────────────╮
  │                                                                                │
  │  Environment                                                                   │
  │    ✓  python           3.12.3                                                  │
  │    ✓  git              git version 2.43.0                                      │
  │    ✓  ripgrep          ripgrep 15.2.0                                          │
  │    ✓  gh (GitHub CLI)  gh version 2.96.0                                       │
  │                                                                                │
  │  Knowledge base                                                                │
  │    ✓  code graph (codebase-memory-mcp)      codebase-memory-mcp 0.9.0          │
  │    ✓  semantic search (fastembed + qdrant)  installed                          │
  │    ✓  tree-sitter extractors                installed                          │
  │                                                                                │
  │  Pipeline stages                                                               │
  │    ✓  stage: knowledge  claude-cli / haiku                                     │
  │    ✓  stage: pm         claude-cli / haiku                                     │
  │    ✓  stage: planner    claude-cli / sonnet                                    │
  │    ✓  stage: dev        claude-cli / sonnet                                    │
  │    ✓  stage: qa         claude-cli / haiku                                     │
  │    ✓  stage: review     claude-cli / haiku                                     │
  │                                                                                │
  │  Coding CLIs detected                                                          │
  │    ✓  claude-code  2.1.220 (Claude Code)                                       │
  │    ✓  codex        codex-cli 0.145.0                                           │
  │    ✗  cursor       'cursor-agent' not found on PATH                            │
  │    ✓  aider        aider 0.86.2                                                │
  │    ✓  gemini-cli   0.52.0                                                      │
  │      • cursor: Install the Cursor CLI (cursor-agent), then run `cursor-agent   │
  │      login` — or set a Cursor API key on the Providers tab.                    │
  │                                                                                │
  │  Delivery                                                                      │
  │    ✓  demo mode  on — the PR stage is a dry run                                │
  │    ✓  jira       not configured (optional — pushes are a no-op)                │
  │                                                                                │
  │  ✓ Ready to run.  2 optional component(s) degraded — see above.                │
  │                                                                                │
  ╰──────────────────────────────────────────────────────────────────────  ready  ─╯
```

Every dependency in CodeVerdict is optional in a *different* way — some degrade
quality silently rather than crashing — so `doctor` reports all of them with a
repair hint each, and exits non-zero when something genuinely blocks a run
(handy in CI). It is the answer to "did I set this up right?".

### Your first delivery, in six lines

```
/kb add https://github.com/pallets/click     # 1. index a repo (this is the KB)
add a --dry-run flag to the CLI runner       # 2. plain English; the PM agent scopes it
/tickets                                     # 3. PM drafts engineering tickets
/approve all                                 # 4. the human gate — nothing writes code before this
/run                                         # 5. Planner → Dev → QA → jury → PR, streamed live
/review                                      # 6. the verdict, every juror's findings, the diff
```

**Demo mode is on by default**, so step 5's PR stage is a dry run. Turn it off
(and authenticate `gh`) only for a repo you own.

### The commands worth knowing

| | |
|---|---|
| `/doctor` | Preflight — what's installed, what's degraded, what blocks a run. |
| `/kb add <url>` | Index a repository: code graph, embeddings, structured views — rendered live, with the percentage and the stage it is actually in (cloning, AST, embedding *n*/*m*). Ctrl-C detaches; the build keeps going. |
| `/run` | The pipeline, streamed as a stage timeline with per-stage model, elapsed time, cost and live tool-call count — plus a rolling window of what the running agent is doing *right now*. **Ctrl-C detaches the view; the run keeps going.** |
| `/review` | The jury panel — the foreperson's verdict, every juror's own findings (including abstentions), the diff, and the Planner's plan to read it against. |
| `/jury` | The roster — seat, unseat, reorder, re-model, re-brief. See §2. |
| `/models` · `/model` | Who owns each of the six stages, and whether the CLI it points at is installed here. See §2. |
| `/settings` | Everything configurable: providers and keys, per-stage models, one-provider presets, loop bounds, delivery safety, Jira. |
| `/costs` | Real token and dollar breakdowns per ticket, scope and agent — read from each backend's own meter. |
| `/show <KEY>` | One ticket in full. Criteria the PM **assumed** rather than heard are counted and flagged — the last place to catch a drifted requirement before it costs money. |

**Everything the shell does is also a subcommand**, so it drops into a Makefile
or a CI job unchanged:

```bash
codeverdict doctor --json
codeverdict ingest https://github.com/pallets/click --wait
codeverdict scope "add a --dry-run flag to the CLI runner"
codeverdict draft --json
codeverdict approve all && codeverdict run
codeverdict review TASK-101 --json
codeverdict settings set dev_provider=anthropic
```

That is not a convenience layer — most subcommands dispatch to the *same* slash
command through the same registry, so the interactive and scripted paths cannot
drift.

### Other ways to run

- **Docker** — `docker compose run --rm app` gives you the shell in a container.
  Run it attached: CodeVerdict is a terminal program, so there is nothing to visit
  and nothing to publish a port for.
- **Extras** — `[semantic]` local dense embeddings (drop it for keyword-only
  retrieval), `[treesitter]` exact parse trees for JS/TS/Go/Rust/Java/Ruby
  (Python is exact either way via stdlib `ast`), `[rerank]` a local cross-encoder
  as a third reranking tier (the default deterministic reranker is free).

> The knowledge base uses free local embeddings in an embedded Qdrant DB — no API
> key, no Docker. The first ingest downloads a ~90 MB model once. Set
> `RAG_EMBEDDINGS=tfidf` to skip embeddings entirely.

---

## 6. Features

| | |
|---|---|
| **Multi-judge review jury** | The review stage is a **panel**, not a reviewer — independent, specialized, differently-modelled judges plus a synthesizing foreperson. See §1. |
| **Requirement-drift check** | Jurors *and* the foreperson see the user's verbatim request beside the PM's criteria, and drift between the two blocks. A panel judging only the criteria cannot notice the criteria were wrong — every juror checks the same paraphrase and agrees with itself. This is the one finding a single juror can carry alone. |
| **Per-stage provider *and* model** | All six stages independently repointable at 5 headless coding CLIs (on their own login, no key) or any API model — the Anthropic Messages API and any OpenAI-compatible endpoint. One-click install for a missing CLI. See §2. |
| **Repo code graph** | A persistent knowledge graph — definitions, call edges, imports, HTTP routes across 158 languages — via the external [`codebase-memory-mcp`](https://github.com/DeusData/codebase-memory-mcp) binary. Exact `file:line` lookups and call-graph impact analysis, deterministic and free. Degrades to a symbol map + `ripgrep`. |
| **Hybrid semantic search** | Graph nodes embedded locally (fastembed + embedded Qdrant) and RRF-fused with BM25. **8/10 vs 4/10** top-5 recall against keyword-only on vocabulary-mismatch queries. |
| **Incremental re-index** | The graph reindexes on SHA drift; the dense index re-embeds only the nodes in files the diff touched, keeping the rest. A merge that touches four files costs seconds, not the hour a full rebuild takes on a large repo — and the build is resumable, so a crash costs minutes rather than everything. |
| **Compounding delivery memory** | Each shipped scope records what it learned — files, symbols, gotchas, wiring — and those notes rank into future scoping. Unmerged work is flagged and upgraded once it lands. |
| **Agentic PM scoping** | A PM agent runs a Socratic clarify-loop hunting ambiguity before locking a scope, then drafts concrete engineering tickets. It owns the requirement, not the code. |
| **Planner agent** | Before any code is written, a separate agent reads the repo through the graph and decides *how* the change is made — ordered steps, blast radius, which tests to extend. Every symbol it names is deterministically verified against the graph and ripgrep, so Dev starts from checked locations, not a guess. |
| **Human approval gate** | No agent touches code until a human approves the ticket (optionally pushed to Jira). |
| **Dev → QA → Review → PR** | Runs on an isolated branch of a cloned working copy; opens a real PR via `gh`. |
| **Bounded revise loop** | QA/jury feedback returns to Dev for up to N rounds, with deliberately conservative verdict parsing — an errored agent is `INCONCLUSIVE`, never silently a pass. A Dev that fails stops the scope even if it committed something first: a truncated edit set is indistinguishable from a finished one in a diff. |
| **No silent green** | QA sees a diff and the test output, so "no failures" and "the suite never ran" look identical. A suite that times out or is missing is announced as such, and the stored verdict is stamped `NO TEST SIGNAL` — it reaches the board, the PR body and the knowledge write-back saying it was never verified. |
| **Language-agnostic** | Symbol extraction, edit-time syntax gates and test running all dispatch through one language registry — Python (exact `ast`), JS/TS, Go, Rust, Java, Ruby. Unsupported languages fail open rather than breaking the run. A manifest alone does not decide the ecosystem: polyglot repos carry them for *tooling* (gitea ships a `pyproject.toml` pinning three Python linters and zero `.py` files), so the source has to be there too. |
| **Trivial fast path** | Deterministic triage skips LLM QA and Review on trivially-scoped green-gate changes, so the gate floor doesn't eat small tasks. |
| **Terminal-native** | A conversation, not a web app. Full-screen panels for the three things that need two dimensions — the jury's review, settings, the roster. Real line editing, history and completion. Every command is also a subcommand. |
| **Honest cost accounting** | Real tokens, cost and duration per agent run, read from each backend's own meter, rolled up per ticket/scope/agent. Backends that report tokens but not dollars say so rather than showing a fake $0.00. |
| **Runtime settings** | Keys, per-stage models, loop bounds, demo mode, Jira — all editable live, no restart. The settings UI is *generated* from one field spec so terminal and web can't drift. Keys encrypted at rest (Fernet). |
| **Degrades instead of breaking** | No graph binary → symbol map. No embeddings → TF-IDF. No ripgrep → `git grep`. No Jira → a no-op. Runs fully offline, zero-CDN, on a free-tier key. |

---

## 7. How it works

```
ingest repo → build knowledge base (code graph + embeddings + views)
  → PM agent clarifies + drafts tickets → human approval (+ optional Jira)
  → Planner reads the repo: ordered steps, verified file:line pins, blast radius
  → Dev implements the plan on an agent/<key> branch
  → QA runs tests + reviews                       ┐
  → Jury: N specialized judges review in parallel ├─ revise loop ×N on failure
  → Foreperson synthesizes one verdict            │
  → PR pushes + opens a real PR                   ┘
  → human merges
```

The pipeline runs on a worker thread and records everything to SQLite — that is
the durable record, and it is what a rejoined session reads back. It *also* publishes to an in-process event bus, so the terminal client
follows a run as it happens instead of polling a database for something that
happened in the thread next to it.

**It's a system, not a prompt wrapper:** a real state machine with explicit SDLC
lanes and a bounded revise loop, not a chain of `if`s. Conservative verdict
parsing, so an unreviewed change can't slip through looking clean. Provider keys
encrypted at rest. Demo mode dry-runs delivery until you opt in. And the structured views take
their *facts* from static analysis, letting the LLM supply only interpretation —
so the map the agents navigate by is anchored to real code.

Full write-ups:

- **[docs/architecture.md](docs/architecture.md)** — components, the pipeline,
  the request flow, cross-provider review, the revise loop.
- **[docs/knowledge-base.md](docs/knowledge-base.md)** — the code graph, the
  retrieval pipeline, the structured views, and how each tier degrades.
- **[docs/configuration.md](docs/configuration.md)** — every environment
  variable and runtime setting.
- **[benchmarks/kb-vs-claude-code.md](benchmarks/kb-vs-claude-code.md)** — the
  head-to-head, including where it loses.
- **[benchmarks/retrieval-ablation.md](benchmarks/retrieval-ablation.md)** — what
  each retrieval stage actually contributes.

---

## 8. Development

```bash
pip install -e ".[dev]"                 # pytest, pytest-cov, ruff
pytest                                  # no network; the LLM boundary is mocked
ruff check backend/app tests
```

Tests cover the security-critical and pipeline-logic paths — encryption at rest,
password hashing and the bootstrap admin, role-based access control,
runtime-settings validation and masking, revise-loop verdict parsing, jury
synthesis and abstention, retrieval, and the preflight checks.

CI runs ruff plus the suite on **Linux (3.11 and 3.12), macOS and Windows**. The
suite is installed **core-only** there — no `[semantic]`, no `[treesitter]` —
because that is what a plain `pip install` gets, so anything needing an optional
package is `importorskip`ped rather than assumed present. Two things this
catches that a local run does not: an import that only exists with the extras,
and reading a file without naming an encoding (Windows defaults to cp1252, and
every CLI source is full of box glyphs).

See **[CONTRIBUTING.md](CONTRIBUTING.md)** and **[SECURITY.md](SECURITY.md)**.

### Tech stack

| Area | Stack |
|---|---|
| Terminal client | Python, **Rich** (rendering), **prompt_toolkit** (input, history, completion), **Textual** (full-screen panels) |
| Data | **SQLModel** (SQLAlchemy + Pydantic) over SQLite |
| Auth | Cookie sessions, PBKDF2 (stdlib), role-based access control, Fernet encryption at rest |
| LLMs | Five headless coding CLIs (Claude Code, Codex, Cursor, Aider, Gemini CLI), the Anthropic Messages API, and any OpenAI-compatible Chat Completions endpoint |
| Knowledge base | [`codebase-memory-mcp`](https://github.com/DeusData/codebase-memory-mcp) code graph (158 languages, call graph, impact analysis) + local dense embeddings (fastembed ONNX, embedded Qdrant) RRF-fused with BM25; symbol-map + `ripgrep` fallback; compounding delivery notes |
| Integrations | GitHub `gh` CLI, `git`, Jira Cloud REST v3 (optional) |

---

## 9. Roadmap

- Extend the jury along SE-Jury's *strategy* axis too — an equivalence judge and
  a generate-tests-then-check judge alongside the concern-specialized seats.
- Automatic team selection from a handful of labelled reviews, per the paper's
  ~50% cost cut.
- Swap the in-process thread pool for a durable queue (Celery / RQ / Arq).
- Sandboxed test execution (containers) for untrusted repos.
- A pluggable agent graph for richer branching and parallelism.

---

## How this repository was built

CodeVerdict was developed using its own review pipeline as a validation loop:
tasks are scoped before implementation, plans are pinned to real symbols, changes
are checked by independent judges, and every accepted result remains under human
release control. The project demonstrates both the tool and the workflow it is
designed to support.

## Citation

The ensemble-judge design is grounded in:

```bibtex
@article{zhou2025sejury,
  title  = {SE-Jury: An LLM-as-Ensemble-Judge Metric for Narrowing the Gap
            with Human Evaluation in SE},
  author = {Zhou, Xin and Kim, Kisub and Zhang, Ting and Weyssow, Martin and
            Gomes, Lu{\'i}s F. and Yang, Guang and Liu, Kui and Xia, Xin and
            Lo, David},
  journal = {arXiv preprint arXiv:2505.20854},
  year   = {2025},
  url    = {https://arxiv.org/abs/2505.20854}
}
```

All correlation, agreement and ablation figures in §1 are from that paper and
describe **SE-Jury**, not this implementation. They are cited as evidence for the
architecture CodeVerdict adopts.

---

<div align="center">
<sub>MIT Licensed · CLI first · provider-agnostic · code graph by
<a href="https://github.com/DeusData/codebase-memory-mcp">codebase-memory-mcp</a></sub>
</div>
