import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[2]


def _default_database_url() -> str:
    """SQLite file at the project root. ``codeverdict.db`` is the current name;
    an install that predates the rename keeps using its existing
    ``sdlc_agents.db`` so no history is orphaned."""
    new = _ROOT / "codeverdict.db"
    legacy = _ROOT / "sdlc_agents.db"
    path = legacy if (legacy.exists() and not new.exists()) else new
    return f"sqlite:///{path.as_posix()}"


def _load_key_from_env_file(name: str, env_path: str) -> str:
    """Read NAME=value from an arbitrary .env-style file (used as a cross-cwd
    fallback for keys not present in our own env)."""
    try:
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


class Settings(BaseSettings):
    """Runtime configuration, read from environment / .env."""

    # Anchor .env to the project root (CodeVerdict/.env) so it's read regardless
    # of the launch directory (uvicorn may run from the repo root or backend/).
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        extra="ignore",
    )

    # Database — anchored to an ABSOLUTE path at the AGENT root so it never
    # depends on the server's launch cwd (a relative path silently created a
    # second, empty DB when the server was started from backend/).
    database_url: str = _default_database_url()

    # --- Knowledge base / RAG ---
    # There is one retrieval backend: the built-in, self-contained code-graph
    # knowledge base in services/knowledge. Nothing external to run or pay for.

    # --- Code graph (services/knowledge/graph.py) ---
    # The deterministic localization + structural layer: a persistent knowledge
    # graph (definitions, call edges, imports, HTTP routes; 158 languages) built
    # by the `codebase-memory-mcp` static binary and queried per-call via its
    # one-shot CLI. No daemon, no API key; indexing is RAM-first and takes
    # seconds on typical repos. When the binary is missing everything degrades
    # to the built-in symbol-map + ripgrep tier.
    graph_enabled: bool = True
    # Binary name on PATH or an absolute path to it.
    graph_binary: str = "codebase-memory-mcp"
    # Index mode: "full" (similarity + semantic-embedding edges — needed for
    # semantic search), "moderate", or "fast" (structural only).
    graph_index_mode: str = "full"

    # --- Lexical search (services/search.py) ---
    # ripgrep is the lexical channel: identifier-aware matching over the working
    # copy, type-filtered per language, .gitignore-aware. It is the refinement
    # pass at the end of retrieval (GrepRAG: lexical search recovers what an AST
    # index cannot model — strings, comments, config, templates) and the `grep`
    # tool every agent can call. Without the binary the same calls fall back to
    # `git grep` (tracked files only, with equivalent type filters for definition
    # searches, but no .gitignore-aware walk) — degraded, never broken.
    ripgrep_enabled: bool = True
    ripgrep_path: str = "rg"

    # Anchored to the project root, not the launch cwd. A relative path here
    # silently split the store in two — uvicorn started from backend/ built its
    # indexes under backend/.knowledge and backend/.qdrant, and the same server
    # started from the root then found nothing and re-indexed from scratch.
    knowledge_dir: str = str(_ROOT / ".knowledge")

    # --- Local dense-embedding channel (services/knowledge/embed.py) ---
    # Real contextual embeddings over the graph's nodes, run locally via
    # fastembed (ONNX/CPU, no Docker, no API) and stored in an embedded Qdrant
    # collection per repo. Fused with the graph's BM25 via RRF at query time —
    # dense for vocabulary bridging ("login" → auth code), BM25 for exact
    # identifiers. Replaces the graph binary's near-useless static-token
    # semantic layer. Optional: needs the [semantic] extra
    # (fastembed + qdrant-client); off/absent → retrieval is BM25-only.
    local_embeddings: bool = True
    # A fastembed model id. Default: the quantized nomic-embed-code-family model
    # (768d, ~130MB, 8k context, code-capable). Stronger code option:
    # jinaai/jina-embeddings-v2-base-code (768d, ~640MB).
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-Q"
    # Embedded Qdrant storage path (on-disk, no server; per-repo collections).
    qdrant_path: str = str(_ROOT / ".qdrant")   # root-anchored; see knowledge_dir
    # --- Retrieval pipeline (services/knowledge/retriever.retrieve_context) ---
    # Five stages, each independently switchable so the whole thing is ablatable
    # (see benchmarks/): fuse (RRF of BM25 + dense) → expand (graph) → refine
    # (ripgrep) → snippets → rerank. Turning all four optional stages off leaves
    # exactly the pre-pipeline behaviour, which is what makes them measurable.
    #
    # Graph expansion: pull each top hit's 1-hop neighbourhood (callers, callees,
    # class members, the tests that exercise it) into the candidate pool.
    graph_expansion: bool = True
    # Hops to expand. RepoGraph measured 1-hop as a large gain and naive 2-hop
    # flattening as a LOSS (the model drowns) — 2 exists for the ablation, not
    # for production.
    graph_hops: int = 1
    # Lexical refinement: ripgrep the identifiers named by the query/plan to
    # catch occurrences the index doesn't model (strings, config, templates).
    grep_refine: bool = True
    # Attach the top hits' real source to retrieval results, so the agent
    # doesn't spend a tool call reading back every pin it was given.
    snippet_context: bool = True
    # Per-node source cap; a node above it is summarized instead of inlined.
    snippet_max_chars: int = 2400
    # Total char budget for the whole snippet block.
    snippet_budget_chars: int = 6000
    # Summarize oversized nodes with the knowledge-stage model (cached on disk
    # per node per commit) instead of truncating them mid-function.
    snippet_summarize: bool = True
    # Reranking: "deterministic" (free, always available), "llm" (one batched
    # scoring call on rerank_provider/model), "cross-encoder" (local model, needs
    # the [rerank] extra), or "off" (the fused order untouched — the ablation arm
    # that makes the deterministic scorer's contribution measurable rather than
    # assumed). See services/knowledge/rerank.py.
    rerank_mode: str = "deterministic"
    rerank_provider: str = ""      # empty = the knowledge stage's provider
    rerank_model: str = ""         # empty = the knowledge stage's model

    # RRF fusion constant (rank offset). 60 is the standard default; measured to
    # make no difference here (5→60 scored identically).
    rrf_k: int = 60
    # Weight of the dense channel relative to BM25 in the fusion. Measured on
    # vocabulary-mismatch queries: equal weights score WORSE than dense alone
    # (BM25's confident-but-wrong hits crowd out dense's correct ones); ≥2
    # recovers full dense recall while keeping BM25's exact-identifier precision.
    rrf_dense_weight: float = 2.0

    # --- Knowledge freshness (services/knowledge/freshness.py) ---
    # Auto-refresh the KB at pipeline-run entry when origin has moved: the
    # symbol map syncs incrementally and the code graph reindexes — both
    # deterministic and free, so the KB is always exactly current.
    kb_auto_refresh: bool = True

    # --- Knowledge write-back (services/knowledge/write_back.py) ---
    # After a scope delivers, persist a delivery-note knowledge doc (files
    # touched, symbols added, gotchas) so future PM/Dev retrieval compounds
    # across runs instead of rediscovering the repo every time.
    kb_write_back: bool = True
    # Newest N delivery notes kept per repo (older ones are pruned).
    kb_delivery_notes_max: int = 40
    # When pruning, distill retiring notes' gotchas/wiring into per-module
    # `lesson` docs (survive full rebuilds) instead of discarding them — the KB
    # compounds without cluttering (one LLM call per affected module, at prune
    # time only).
    kb_consolidate: bool = True

    # --- Precision retrieval: right-sized, use-case-scoped, ranked context ---
    # Feed agents a token-budgeted slice with exact source files (instead of a
    # broad KB blob). Served by the local RAG by default.
    precision_retrieval: bool = True
    # Optional external "Deep Analysis" (AST+LLM) service for precision retrieval.
    # Off by default; the local RAG provides precision slices standalone.
    use_deep_analysis: bool = False
    functional_analysis_url: str = "http://localhost:8002"

    # --- Agents: Claude Code CLI (PM, Dev, Review, PR) ---
    claude_cli_path: str = "claude"
    # Efficient default for coding. The CLI default here is opus-4-6[1m] (very
    # expensive); "sonnet" is ~5x cheaper and excellent for code. Override with
    # CLAUDE_MODEL=opus for the hardest tasks.
    claude_model: str = "sonnet"
    # Cheaper model for test-writing subtasks (boilerplate-ish, well-specified).
    claude_test_model: str = "haiku"
    # Optional explicit API key; otherwise the CLI uses the host's Claude login.
    anthropic_api_key: str = ""
    # Autonomous file edits + tool use inside the isolated workspace clone.
    claude_skip_permissions: bool = True
    # Hard dollar ceiling per Claude agent run (--max-budget-usd). A normal
    # scope's Dev pass costs well under $1; the cap stops runaway sessions.
    # 0 disables the cap.
    claude_max_budget_usd: float = 2.0

    # --- Other headless coding agents (adapter registry: services/agent_backends).
    # Each stage can point at any of these; a tool that isn't installed is shown
    # as unavailable in Settings instead of failing a run. ---
    codex_cli_path: str = "codex"               # OpenAI Codex CLI (`codex exec`)
    cursor_cli_path: str = "cursor-agent"       # Cursor CLI headless agent
    cursor_api_key: str = ""                    # else the host's `cursor-agent login`
    aider_cli_path: str = "aider"
    gemini_cli_path: str = "gemini"             # Google Gemini CLI
    # Dev (code-writing) agent provider: "claude" | "openai" | "auto".
    # "auto" tries Claude, then falls back to the OpenAI coding agent if the
    # Claude CLI can't authenticate.
    coding_provider: str = "auto"
    # Review agent provider — intentionally a DIFFERENT model than Dev for a less
    # biased review. Defaults to OpenAI so Claude's code is reviewed by OpenAI.
    review_provider: str = "openai"

    # --- QA agent: a DIFFERENT provider (OpenAI) for less bias ---
    qa_provider: str = "openai"
    qa_model: str = "gpt-4o"
    # The "primary" OpenAI-compatible endpoint. Its base URL is configurable and
    # points at Groq's free tier out of the box (set to https://api.openai.com/v1
    # for real OpenAI). Exposed in the UI as the "openai" provider.
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # --- Additional OpenAI-compatible providers (see services/providers.py) ---
    # Groq's own key + static endpoint, so it can be selected as a distinct provider
    # from the (configurable) "openai" primary above.
    groq_api_key: str = ""
    # xAI (Grok). Static https://api.x.ai/v1 endpoint.
    xai_api_key: str = ""
    # A user-defined OpenAI-compatible provider (OpenRouter, Together, Ollama, …).
    custom_api_key: str = ""
    custom_base_url: str = ""

    # --- Google AI Studio (Gemini) via its OpenAI-compatible endpoint ---
    # Gemini's free tier has a FAR larger per-minute token budget than Groq's
    # (e.g. gemini-2.0-flash ~1M TPM vs gpt-oss-120b's 8000), so it's the natural
    # overflow for big Dev/revision requests. Any model whose name starts with
    # "gemini" is routed here automatically. Set GEMINI_API_KEY to enable.
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    # Added to the fallback pool (comma-separated) once a key is set. These absorb
    # requests Groq rejects for size (413/TPM) or daily quota.
    gemini_models: str = "gemini-3.5-flash-lite,gemini-flash-latest"

    # --- Per-stage models (Groq free tier, OpenAI-compatible) ---
    # Each stage can run on a different model. This spreads load across separate
    # free-tier rate-limit pools AND gives real model diversity (so QA/Review are
    # genuinely less correlated with Dev, not "different provider" in name only).
    # Empty = fall back to qa_model (see resolution at the bottom of this file).
    #   knowledge — high call volume, purely interpretive short JSON → fast/cheap.
    #   pm        — the "strong PM" stage: hardest reasoning + strict JSON.
    #   planner   — reads the repo and decides the approach; reasoning-heavy.
    #   dev       — precise SEARCH/REPLACE instruction-following.
    #   review    — a different pool/model from Dev for a less biased review.
    knowledge_model: str = "openai/gpt-oss-20b"
    pm_model: str = "openai/gpt-oss-120b"
    planner_model: str = ""
    dev_model: str = "openai/gpt-oss-120b"
    review_model: str = "llama-3.3-70b-versatile"

    # --- Per-stage provider (which registry provider runs each stage) ---
    # A provider id from services/providers.py (groq | openai | gemini | xai |
    # anthropic | custom), plus "claude-cli" for dev/review and "auto" for dev.
    # These start empty and are backfilled at the bottom of this file from the
    # legacy coding/qa/review_provider + model names, so existing installs keep
    # their exact current routing (gemini-* models auto-route to Gemini, etc.).
    knowledge_provider: str = ""
    pm_provider: str = ""
    planner_provider: str = ""
    dev_provider: str = ""

    # --- Review jury (services/jury) ---
    # The Review stage runs as an ensemble: several specialized judges review the
    # same change independently from different engineering perspectives, and a
    # foreperson merges their opinions into one verdict. The roster itself lives
    # in the Judge table (runtime-editable, see services/judges.py) — these are
    # the panel-wide knobs.
    # Off = the classic single-reviewer path (review_provider + review_model).
    jury_enabled: bool = True
    # How many judges are polled at once. Each judge is one LLM call, so the
    # panel multiplies the review stage's cost by its size; the cap also keeps
    # free-tier providers from rate-limiting the whole panel at once.
    jury_max_parallel: int = 4
    # Findings a judge reports below this confidence are dismissed by the
    # foreperson rather than sent back to Dev. 0.5 = "the judge was guessing".
    jury_min_confidence: float = 0.5
    # The foreperson (synthesis) stage. Empty = reuse the Review stage's
    # provider/model. This is the one call that should be a strong model: it is
    # cheap (no diff, just opinions) and it decides.
    jury_synthesis_provider: str = ""
    jury_synthesis_model: str = ""
    # Repository lookups a reviewing stage (each juror, and QA) may request
    # before it commits to a verdict. A diff shows what changed, not what
    # depends on it, so a finding about code outside the diff is a guess unless
    # it was checked — and a wrong blocking finding costs a full paid
    # Dev+QA+Review round. Only a stage that ASKS pays for the extra call; one
    # follow-up round, never a loop. 0 disables reachback for reviewers.
    jury_tool_calls: int = 3

    # --- Agentic Dev loop (openai_agent.code) ---
    # Max model calls per ticket: round 1 edits, then verify-against-diff rounds.
    dev_max_rounds: int = 4
    # Per-file char budget shown to the Dev model. 12K blind-truncated real test
    # files mid-file (the model then emitted SEARCH blocks that could never
    # match); 30K covers most source files whole. Gemini's 250K-TPM budget takes
    # this fine; Groq-routed requests are still middle-trimmed to max_request_chars.
    dev_file_chars: int = 30000
    # Run targeted pytest on touched test files inside the Dev loop (feeds real
    # failures back to the model before QA ever sees the change).
    dev_run_tests: bool = True
    # Hard cap on total request size (chars, ~4/token) sent to the LLM. Sized to
    # fit Groq free tier's tightest per-minute budget (gpt-oss-120b = 8000 TPM):
    # ~5.5k input tokens + headroom for output. Requests over this are trimmed
    # (middle-cut) instead of 413-ing. Raise it for paid tiers / roomier models.
    max_request_chars: int = 22000

    # Bounded automated back-and-forth in the pipeline: if Review requests
    # changes (or QA fails), feedback is fed back to Dev and QA+Review re-run —
    # up to this many rounds before the pipeline gives up and surfaces the state.
    max_revision_rounds: int = 2

    # Trivial-task fast path. The pipeline's fixed PM+QA+Review floor (~$0.14 in
    # the benchmarks) makes very cheap greppable edits cost more than a cold
    # `claude -p`. When a scope is deterministically trivial (one ticket, small
    # grep-pinned localization) AND the Dev change verifies through the
    # deterministic test gate (runnable suite, zero new failures) with a small
    # non-test diff, skip the paid LLM QA + Review passes and stamp an explicit
    # fast-path verdict instead. Any doubt — no runnable suite, new failures,
    # a bigger diff than triaged — falls through to the full QA+Review loop.
    fast_path_enabled: bool = True

    # PM agent (agentic retrieval loop, modeled on oxygen/PM_agent): max on-demand
    # knowledge-retrieval rounds the PM may run within a single turn before it must
    # answer / ask / draft. Bootstrap (repository+architecture+modules) is loaded
    # first, then the PM pulls more knowledge only as it decides it needs it.
    pm_max_retrieval_rounds: int = 3

    # --- Planner agent (services/planner.py) ---
    # Runs at pipeline entry, after the scope is locked and the knowledge layers
    # are refreshed, and decides HOW the change is made: which code produces the
    # behaviour, what else touches it, in what order to change it. Its symbols are
    # then verified against the graph/ripgrep before anything consumes them.
    # Off = the Dev agent works from the scope alone (the ablation the research
    # measures as falling back to single-agent performance).
    planner_enabled: bool = True
    # Retrieve/decide rounds per plan. The research puts a Planner at 3-5.
    planner_max_rounds: int = 4
    # Inject the plan-named files' real contents into the Dev prompt. Measured to
    # matter (it is what beat a cold `claude -p` on the rich benchmark); scoped to
    # what the plan names rather than everything anyone guessed. Off = pure
    # reachback, Dev pulls everything through its tools.
    dev_inject_file_contents: bool = True

    # Where agents check out working copies (cloned per repo).
    # Keep agent workspaces outside the checkout by default. Values supplied
    # through env/.env may use `~`; consumers expand it before filesystem use.
    repos_dir: str = str(Path.home() / ".codeverdict" / "workspace")

    # Optional fallback .env for OPENAI_API_KEY. Empty by default — the primary
    # source is OPENAI_API_KEY in env / .env.
    openai_env_path: str = ""
    # Optional fallback .env for ANTHROPIC_API_KEY. Empty by default — primary
    # source is ANTHROPIC_API_KEY in env / .env.
    agent_env_path: str = ""

    # Safety: in demo mode the pipeline never pushes branches or force-updates a
    # remote — it logs the PR title/body it WOULD open and stops. Turn this OFF
    # (DEMO_MODE=false) only against a repo you own and intend to push to.
    demo_mode: bool = True
    # Open a real PR via the gh CLI at the end of the pipeline. Ignored (treated
    # as a dry-run) while demo_mode is on.
    open_real_pr: bool = True

    # --- Agent identity on deliveries ---
    # Commits the pipeline makes are authored by this name/email, so the change
    # history shows the agent (not whoever's laptop the server runs on).
    agent_git_name: str = "CodeVerdict Agent"
    agent_git_email: str = "codeverdict-agent@users.noreply.github.com"
    # Optional GitHub token of a dedicated bot/machine account. When set, branch
    # pushes and `gh pr create` authenticate as that account, so the PR itself is
    # "created by" the agent on GitHub. Without it, gh falls back to the host's
    # login (commits are still authored by the agent).
    github_bot_token: str = ""

    # --- Jira Cloud (optional) — approved tickets are pushed as Stories ---
    # All four must be set for the integration to activate; editable at runtime
    # from the Settings screen (services/jira.py rebuilds its client on change).
    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = ""

    # Demo seed (real integration boots empty by default; set true for sample data)

    # --- Bootstrap admin ---
    # Password for the `admin` account created on first boot. Leave blank to have
    # one generated and printed to the server log once (see services/auth.py) —
    # there is deliberately no guessable default.
    admin_password: str = ""


settings = Settings()

# The Planner reasons about the repo like the PM reasons about the requirement,
# so it inherits the PM stage's routing rather than the generic QA default —
# an install that predates the Planner gets its strongest configured reasoner
# on it, which is where it belongs.
if not settings.planner_model:
    settings.planner_model = settings.pm_model
if not settings.planner_provider:
    settings.planner_provider = settings.pm_provider

# Per-stage models default to qa_model when left unset, so a single QA_MODEL still
# configures the whole system if the operator doesn't split them.
for _stage_field in ("knowledge_model", "pm_model", "planner_model", "dev_model",
                     "review_model"):
    if not getattr(settings, _stage_field):
        setattr(settings, _stage_field, settings.qa_model)

# OpenAI key: env first, then an optional fallback .env.
if not settings.openai_api_key:
    settings.openai_api_key = os.environ.get("OPENAI_API_KEY", "") or _load_key_from_env_file(
        "OPENAI_API_KEY", settings.openai_env_path
    )

# Gemini key: env first, then AGENT/.env (GEMINI_API_KEY or GOOGLE_API_KEY).
if not settings.gemini_api_key:
    settings.gemini_api_key = (
        os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    )

# ANTHROPIC_API_KEY: env first, then AGENT/.env (works regardless of launch cwd).
if not settings.anthropic_api_key:
    settings.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "") or _load_key_from_env_file(
        "ANTHROPIC_API_KEY", settings.agent_env_path
    )

# GROQ_API_KEY: env first. Existing installs put their Groq key in OPENAI_API_KEY with
# OPENAI_BASE_URL pointed at Groq — mirror it into GROQ_API_KEY so the explicit 'groq'
# provider works without re-entering the key.
if not settings.groq_api_key:
    settings.groq_api_key = os.environ.get("GROQ_API_KEY", "")
if not settings.groq_api_key and "groq.com" in (settings.openai_base_url or "") and settings.openai_api_key:
    settings.groq_api_key = settings.openai_api_key
if not settings.xai_api_key:
    settings.xai_api_key = os.environ.get("XAI_API_KEY", "")


# --- Per-stage provider backfill --------------------------------------------------
# Preserves today's routing exactly for installs that predate the provider registry:
# gemini-* models auto-route to Gemini, everything else to the primary 'openai'
# endpoint, and dev/review honor the legacy claude/auto coding provider.
def _provider_for_model(model: str) -> str:
    return "gemini" if (model or "").startswith("gemini") else "openai"


if not settings.knowledge_provider:
    settings.knowledge_provider = _provider_for_model(settings.knowledge_model)
if not settings.pm_provider:
    settings.pm_provider = _provider_for_model(settings.pm_model)
if not settings.planner_provider:
    settings.planner_provider = _provider_for_model(settings.planner_model)
if not settings.dev_provider:
    if settings.coding_provider == "auto":
        settings.dev_provider = "auto"
    elif settings.coding_provider == "claude":
        settings.dev_provider = "claude-cli"
    else:
        settings.dev_provider = _provider_for_model(settings.dev_model)

# qa_provider / review_provider already exist with legacy values ("openai"/"claude"/
# "auto"). Normalize them to real provider ids from services/providers.py.
if settings.review_provider in ("claude", "auto"):
    settings.review_provider = "claude-cli"
elif settings.review_provider in ("", "openai"):
    settings.review_provider = _provider_for_model(settings.review_model)
if settings.qa_provider in ("", "openai", "claude", "auto"):
    settings.qa_provider = _provider_for_model(settings.qa_model)
