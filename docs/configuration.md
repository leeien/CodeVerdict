# Configuration

There are two ways to configure CodeVerdict, and they layer:

1. **Environment / `.env`** — the baseline, read at startup. Copy
   [`.env.example`](../.env.example) to `.env` and edit. Every setting has a sensible
   default except the API keys.
2. **The Settings screen (admin, runtime)** — most values are also editable live from
   the UI. UI-saved values are stored in the database and **override** the env
   baseline; API keys saved this way are encrypted at rest (Fernet). A value never
   touched in the UI keeps its env/`.env` value.

So the effective value of a setting is: *UI override, if one exists; otherwise the
env/`.env` value; otherwise the built-in default.*

## First-boot admin

On first boot an `admin` account is created. Its password comes from
`ADMIN_PASSWORD` if you set it; otherwise a random one is generated and printed to
the server log **once**. There is deliberately no guessable default. Change it from
**Settings → Access & users** (the UI nags until you do), or preset it:

```bash
ADMIN_PASSWORD=choose-your-own
```

## LLM providers

The pipeline calls LLMs through an OpenAI-compatible client, the native Anthropic
Messages API, and/or the Claude Code CLI. You only need keys for the providers you
actually use. The defaults are tuned to run the non-Dev stages on **Groq's free
tier**.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Primary OpenAI-compatible key (PM/QA agents, KB synthesis). |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Point at any OpenAI-compatible endpoint. |
| `ANTHROPIC_API_KEY` | — | Optional; lets the Dev/Review agents use the Claude Code CLI. |
| `GROQ_API_KEY` | — | Groq free-tier key (fast small models for QA/Review/PM). |
| `GEMINI_API_KEY` | — | Google Gemini (OpenAI-compatible endpoint). |
| `XAI_API_KEY` | — | xAI / Grok. |
| `CUSTOM_API_KEY` / `CUSTOM_BASE_URL` | — | Any other OpenAI-compatible provider. |

### Per-stage model + provider

Each pipeline stage picks its own provider and model, so you can put a frontier model
where it matters (Dev) and cheap models everywhere else:

| Stage | Provider var | Model var |
|---|---|---|
| Knowledge synthesis | `KNOWLEDGE_PROVIDER` | `KNOWLEDGE_MODEL` |
| PM (scoping) | `PM_PROVIDER` | `PM_MODEL` |
| Planner (how to build it) | `PLANNER_PROVIDER` | `PLANNER_MODEL` |
| Dev (coding) | `CODING_PROVIDER` | `DEV_MODEL` / `CLAUDE_MODEL` |
| QA | `QA_PROVIDER` | `QA_MODEL` |
| Review | `REVIEW_PROVIDER` | `REVIEW_MODEL` |
| Jury foreperson | `JURY_SYNTHESIS_PROVIDER` | `JURY_SYNTHESIS_MODEL` |

The Review stage's provider/model is what a **judge inherits** when it has no override
of its own; the individual judges on the panel are configured per judge (Settings →
Review jury, or `/api/jury`), not through environment variables. With the jury off,
`REVIEW_PROVIDER`/`REVIEW_MODEL` is the single reviewer.

The **Planner** decides where every edit lands, and Dev, QA and the jury all work
from its output — so a weak model there is paid for by every stage downstream. It
defaults to the PM stage's routing (the strongest reasoner you have configured).
Planning is read-only, so an agentic CLI is a valid choice for it: it reaches the
same index tools through the `.codeverdict/kb` shim.

`CODING_PROVIDER=auto` (recommended) uses the Claude Code CLI when it's installed and
authenticated, and falls back to the OpenAI-compatible edit loop otherwise. Code
writing is the one stage that reliably needs a frontier model; small free-tier models
fail at precise multi-file edits.

## Knowledge base

| Variable | Default | Purpose |
|---|---|---|
| `GRAPH_ENABLED` | `true` | Index each repo into a `codebase-memory-mcp` code graph (localization + structure). Off = symbol-map + `ripgrep` fallback. |
| `GRAPH_BINARY` | `codebase-memory-mcp` | Binary name on PATH, or an absolute path to it. |
| `GRAPH_INDEX_MODE` | `full` | `full` (structural + similarity + semantic-embedding edges), `moderate`, or `fast` (structural only). |
| `LOCAL_EMBEDDINGS` | `true` | Embed the graph's symbols locally (fastembed + embedded Qdrant) and fuse with keyword search. Needs the `[semantic]` extra; off/absent → keyword-only. |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5-Q` | fastembed model id. Heavier code-specialised option: `jinaai/jina-embeddings-v2-base-code`. |
| `QDRANT_PATH` | `./.qdrant` | On-disk embedded Qdrant storage (no server). |
| `RRF_DENSE_WEIGHT` | `2.0` | Weight of the semantic channel vs keyword in RRF fusion. Equal weights measurably underperform. |
| `RRF_K` | `60` | RRF rank offset (measured to make no difference between 5 and 60). |
| `RIPGREP_ENABLED` | `true` | Use ripgrep as the lexical engine (identifier-aware, per-language definition patterns, `.gitignore`-aware). Off or binary missing → `git grep`: tracked files only; definition searches retain language filters, but the fallback has no `.gitignore`-aware walk. |
| `RIPGREP_PATH` | `rg` | Binary name on PATH, or an absolute path to it. |

### Retrieval pipeline

Five stages; four are switchable so their contribution can be measured rather than
assumed (`benchmarks/retrieval_ablation.py`, results in `benchmarks/retrieval-ablation.md`).

| Variable | Default | Purpose |
|---|---|---|
| `GRAPH_EXPANSION` | `true` | Pull each top hit's 1-hop neighbourhood (callers, callees, class members, covering tests) into the results. |
| `GRAPH_HOPS` | `1` | Expansion depth. 2 exists for ablation — measured as no better here, and as a loss in the literature when flattened into a prompt. |
| `GREP_REFINE` | `true` | Finish with a ripgrep pass over the query's identifiers. The only stage that finds terms living solely in string literals. |
| `SNIPPET_CONTEXT` | `true` | Attach the top hits' real source, so an agent edits instead of re-reading every pin. |
| `SNIPPET_SUMMARIZE` | `true` | Summarize a symbol too large to inline (cached per commit) instead of truncating mid-function. |
| `RERANK_MODE` | `deterministic` | `deterministic` (free), `llm` (one scoring call per retrieval), `cross-encoder` (needs the `[rerank]` extra), or `off`. |
| `RERANK_PROVIDER` / `RERANK_MODEL` | inherit knowledge stage | Used only by `RERANK_MODE=llm`. |

### Planner and reviewer reachback

| Variable | Default | Purpose |
|---|---|---|
| `PLANNER_ENABLED` | `true` | Run the Planner before Dev. Off = Dev works from the scope alone. |
| `PLANNER_MAX_ROUNDS` | `4` | Retrieve/decide rounds before the Planner must commit. |
| `DEV_INJECT_FILE_CONTENTS` | `true` | Put the plan-named files' real contents into the Dev prompt. |
| `JURY_TOOL_CALLS` | `3` | Repository lookups a judge (and QA) may request before voting. 0 = judge from the diff alone. |
| `KB_AUTO_REFRESH` | `true` | Reindex the graph + symbol map at run entry when origin has moved (deterministic, free). |
| `KB_WRITE_BACK` | `true` | Persist a delivery note after each scope ships, so retrieval compounds across runs. |
| `KB_DELIVERY_NOTES_MAX` | `40` | Newest N delivery notes kept per repo; older ones distill into per-module lessons. |

See [knowledge-base.md](knowledge-base.md) for what these control.

## Review jury

The panel roster itself (which judges are seated, on which models, with what briefs)
lives in the database and is edited from **Settings → Review jury** — it is a list, not
a setting, so it has no env var. These are the panel-wide knobs:

| Variable | Default | Purpose |
|---|---|---|
| `JURY_ENABLED` | `true` | Review by ensemble. Off = the classic single reviewer (`REVIEW_PROVIDER`/`REVIEW_MODEL`) does it alone. |
| `JURY_MAX_PARALLEL` | `4` | How many judges are polled at once. Lower it if a free-tier provider rate-limits the panel. |
| `JURY_MIN_CONFIDENCE` | `0.5` | Findings a judge reports below this confidence are dismissed by the foreperson instead of sent back to Dev. |
| `JURY_SYNTHESIS_PROVIDER` | *(Review stage)* | Provider for the foreperson — the call that merges opinions and decides. |
| `JURY_SYNTHESIS_MODEL` | *(Review stage)* | Model for the foreperson. Worth a strong one: it's cheap (no diff, just opinions) and it makes the call. |

> **Cost.** Every seated judge is one LLM call per review round, so a 4-judge panel plus
> the foreperson makes the review stage roughly 5× what a single reviewer costs. Each
> judge is billed to its own run, so the Costs page shows this rather than hiding it.

## Pipeline behaviour

| Variable | Default | Purpose |
|---|---|---|
| `MAX_REVISION_ROUNDS` | `2` | Dev↔QA/Review revise loops before giving up (0 disables). |
| `FAST_PATH_ENABLED` | `true` | Skip LLM QA+Review on deterministically trivial scopes whose change passes the test gate green with a small diff. |
| `DEV_MAX_ROUNDS` | `4` | Max model calls per ticket inside the Dev agent. |
| `DEV_RUN_TESTS` | `true` | Run targeted tests inside the Dev loop. |
| `DEV_FILE_CHARS` | `30000` | Cap on pinned-file content injected into the Dev prompt. |
| `PM_MAX_RETRIEVAL_ROUNDS` | `3` | KB retrieval rounds the PM agent may run while scoping. |
| `MAX_REQUEST_CHARS` | `22000` | Cap on the raw request text accepted for scoping. |

## Delivery & safety

| Variable | Default | Purpose |
|---|---|---|
| `DEMO_MODE` | `true` | Dry-run the PR stage (logs the PR it would open; never pushes). |
| `OPEN_REAL_PR` | `true` | Whether real PRs are opened *when* demo mode is off. |
| `REPOS_DIR` | `~/.codeverdict/workspace` | Where agent clones + test envs live. `~` is expanded. **Use a path with no spaces** — some repos' tests assert on rendered file paths and fail spuriously otherwise. |
| `CLAUDE_MAX_BUDGET_USD` | `2.0` | Per-run spend cap for the Claude CLI. |

To open real PRs: set `DEMO_MODE=false`, authenticate the `gh` CLI (or connect a
GitHub token per user from the UI), and point it at a repo you own.

## Optional Jira integration

Leave blank to disable. When set, approved tickets can be pushed to a Jira project.

| Variable | Purpose |
|---|---|
| `JIRA_BASE_URL` | Your Jira Cloud site, e.g. `https://acme.atlassian.net`. |
| `JIRA_EMAIL` | Account email for API auth. |
| `JIRA_API_TOKEN` | Jira API token. |
| `JIRA_PROJECT_KEY` | Target project key. |

## Secrets & storage

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLite by default; anchored to an absolute path so the launch directory doesn't matter. |
| `CODEVERDICT_SECRET_KEY` | Passphrase for Fernet encryption of stored secrets. If unset, a key file (`.secret.key`, chmod 600) is generated at the repo root. |

> If `CODEVERDICT_SECRET_KEY` and `.secret.key` are both lost, previously encrypted
> settings become unreadable (they're treated as unset, not fatal) — you'll just
> re-enter the affected API keys.

The common baseline configuration lives in [`.env.example`](../.env.example); the
table above also documents advanced settings exposed through the runtime Settings
screen and environment variables.
