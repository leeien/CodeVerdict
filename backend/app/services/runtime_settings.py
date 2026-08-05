"""Runtime-editable configuration.

The env/.env file remains the baseline (config.Settings); rows in the
AppSetting table override it. Overrides are applied by mutating the settings
singleton — every service reads ``settings.x`` at call time, so changes take
effect immediately, no restart.

Secrets are write-only through the API (reads return a masked placeholder; saving
the placeholder back is a no-op) AND encrypted at rest in the DB (see crypto.py).

Providers/models are driven by the registry in services/providers.py: each stage
has a ``<stage>_provider`` (a provider id) + ``<stage>_model``, so the operator can
run each stage on a different provider, or apply one provider to all stages via a
preset (see ``apply_provider_preset``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from sqlmodel import Session, select

from ..config import settings
from ..models import AppSetting, utcnow
from . import crypto, providers

logger = logging.getLogger(__name__)

SECRET_MASK = "••••••••"


@dataclass
class Spec:
    """One editable field: how to render, validate, and describe it."""

    group: str
    label: str
    help: str = ""
    type: str = "str"          # str | int | float | bool | enum | model | provider
    secret: bool = False
    options: list[str] = field(default_factory=list)   # for enum / provider
    min: float | None = None
    max: float | None = None
    provider_field: str = ""   # for a "model" field: name of the paired provider field
    model_field: str = ""      # for a "provider" field: name of the paired model field
    section: str = ""          # optional subsection heading within a tab
    # Conditional visibility: "field=value" (alternatives with |, bools as
    # true/false). The UI hides the row when the driver field's current value
    # doesn't match — fields that only make sense in one mode don't clutter the
    # others (e.g. the embeddings API URL only exists for the 'api' engine).
    show_if: str = ""


def _provider_spec(stage: str, label: str, help_text: str) -> Spec:
    return Spec("models", label, help_text, type="provider",
                options=providers.stage_provider_ids(stage), model_field=f"{stage}_model")


def _model_spec(stage: str, label: str, help_text: str) -> Spec:
    return Spec("models", label, help_text, type="model", provider_field=f"{stage}_provider")


# Every runtime-editable settings field. Order here is the UI order.
FIELDS: dict[str, Spec] = {
    # --- Connections → API providers ---
    "openai_base_url": Spec("providers", "Primary endpoint base URL",
                            "The 'openai' provider's base URL. Points at Groq's free tier out of "
                            "the box; set to https://api.openai.com/v1 for real OpenAI.",
                            section="API providers"),
    "openai_api_key": Spec("providers", "Primary endpoint API key",
                           "Key for the base URL above (the 'openai' provider).", secret=True,
                           section="API providers"),
    "groq_api_key": Spec("providers", "Groq API key",
                         "Groq's own key + static endpoint, selectable as the 'groq' provider.",
                         secret=True, section="API providers"),
    "gemini_api_key": Spec("providers", "Gemini API key",
                           "Google AI Studio key for the 'gemini' provider.", secret=True,
                           section="API providers"),
    "xai_api_key": Spec("providers", "xAI (Grok) API key",
                        "Key for the 'xai' provider (https://api.x.ai/v1).", secret=True,
                        section="API providers"),
    "anthropic_api_key": Spec("providers", "Anthropic API key",
                              "Used by the Anthropic API provider AND the Claude Code CLI. Without "
                              "it the CLI falls back to the host's Claude login.", secret=True,
                              section="API providers"),
    "custom_base_url": Spec("providers", "Custom provider base URL",
                            "Any OpenAI-compatible endpoint (OpenRouter, Together, DeepSeek, Ollama, "
                            "or LM Studio). Localhost endpoints may run without an API key.",
                            section="API providers"),
    "custom_api_key": Spec("providers", "Custom provider API key",
                           "Key for the 'custom' provider above.", secret=True,
                           section="API providers"),

    # --- Connections → Agentic coding CLIs (path + key per tool; the card above
    #     each shows install status and how to enable a missing one) ---
    "codex_cli_path": Spec("providers", "Codex CLI path",
                           "Binary for the Codex agent backend (default: codex).",
                           section="Agentic coding CLIs"),
    "cursor_cli_path": Spec("providers", "Cursor CLI path",
                            "Binary for the Cursor agent backend (default: cursor-agent).",
                            section="Agentic coding CLIs"),
    "cursor_api_key": Spec("providers", "Cursor API key",
                           "Key for the Cursor CLI agent backend. Without it cursor-agent uses "
                           "the host's `cursor-agent login`.", secret=True,
                           section="Agentic coding CLIs"),
    "aider_cli_path": Spec("providers", "Aider path",
                           "Binary for the Aider agent backend (default: aider).",
                           section="Agentic coding CLIs"),
    "gemini_cli_path": Spec("providers", "Gemini CLI path",
                            "Binary for the Gemini CLI agent backend (default: gemini).",
                            section="Agentic coding CLIs"),

    # --- Agent models (per stage: provider + model) ---
    "knowledge_provider": _provider_spec("knowledge", "Knowledge provider",
                                         "Builds the structured repo views on ingest — high call volume."),
    "knowledge_model": _model_spec("knowledge", "Knowledge model", "Model for the knowledge stage."),
    "pm_provider": _provider_spec("pm", "PM provider",
                                  "Talks to the human: clarifies the requirement and writes the "
                                  "acceptance criteria. It does not choose edit targets."),
    "pm_model": _model_spec("pm", "PM model", "Model for the PM stage."),
    "planner_provider": _provider_spec("planner", "Planner provider",
                                       "Reads the repository through the code graph and decides HOW "
                                       "the change is made: which code produces the behaviour, what "
                                       "else touches it, in what order. Its pins drive Dev, QA and "
                                       "the jury, so a weak model here is paid for by every stage "
                                       "downstream."),
    "planner_model": _model_spec("planner", "Planner model", "Model for the Planner stage."),
    "dev_provider": _provider_spec("dev", "Dev provider",
                                   "Who writes the code: any installed agentic CLI (Claude Code, Codex, "
                                   "Cursor, Aider, Gemini CLI) or an OpenAI-compatible loop. 'auto' tries "
                                   "Claude then falls back to the HTTP coding loop."),
    "dev_model": _model_spec("dev", "Dev model",
                             "Model for the Dev stage. For the CLI/auto provider use sonnet/opus/haiku."),
    "qa_provider": _provider_spec("qa", "QA provider",
                                  "Runs tests and reviews the change for defects — keep it a different "
                                  "provider than Dev for a less biased check."),
    "qa_model": _model_spec("qa", "QA model", "Model for the QA stage."),
    "review_provider": _provider_spec("review", "Review provider",
                                      "Reviews the diff against acceptance criteria — keep it different "
                                      "from Dev for a less biased review."),
    "review_model": _model_spec("review", "Review model", "Model for the Review stage."),
    "claude_model": Spec("models", "Claude CLI default model",
                         "Fallback CLI model when a Dev/Review stage on the CLI doesn't name one.",
                         type="enum", options=list(providers.PROVIDERS["claude-cli"].models),
                         section="Advanced"),
    "gemini_models": Spec("models", "Gemini fallback pool",
                          "Comma-separated gemini-* models added to the cross-provider fallback pool "
                          "once a Gemini key is set.", section="Advanced"),
    "claude_max_budget_usd": Spec("models", "Claude budget cap ($/run)",
                                  "Hard dollar ceiling per Claude CLI run. 0 disables the cap.",
                                  type="float", min=0, max=50, section="Advanced"),

    # --- Review jury (the panel roster itself is edited on the Jury tab) ---
    "jury_enabled": Spec("jury", "Multi-judge review",
                         "Review the change with an ensemble of specialized judges instead of "
                         "one reviewer, then synthesize their opinions into one verdict. Off = "
                         "the single Review stage model above does it alone.", type="bool"),
    "jury_synthesis_provider": Spec("jury", "Foreperson provider",
                                    "Merges the judges' opinions, resolves their disagreements "
                                    "and decides. Worth a strong model — it's a cheap call (no "
                                    "diff, just opinions) and it makes the call.",
                                    type="provider", options=providers.stage_provider_ids("review"),
                                    model_field="jury_synthesis_model",
                                    show_if="jury_enabled=true"),
    "jury_synthesis_model": Spec("jury", "Foreperson model",
                                 "Model for the synthesis stage.", type="model",
                                 provider_field="jury_synthesis_provider",
                                 show_if="jury_enabled=true"),
    "jury_min_confidence": Spec("jury", "Confidence floor",
                                "Findings a judge reports below this confidence are dismissed "
                                "instead of sent back to Dev. Raise it if the panel is blocking "
                                "on speculation; lower it if real defects are slipping through.",
                                type="float", min=0.0, max=1.0, show_if="jury_enabled=true"),
    "jury_max_parallel": Spec("jury", "Judges in parallel",
                              "How many judges are polled at once. Lower this if a free-tier "
                              "provider rate-limits the panel.",
                              type="int", min=1, max=8, show_if="jury_enabled=true"),
    "jury_tool_calls": Spec("jury", "Reviewer lookups",
                            "Repository lookups a judge (and QA) may request before voting. A "
                            "diff shows what changed, not what depends on it — without this, "
                            "\"breaks callers\" or \"doesn't match the pattern\" are guesses "
                            "about code the reviewer never saw, and a wrong blocking finding "
                            "costs a full revision round. Only a judge that ASKS pays for the "
                            "extra call. 0 = reviewers judge from the diff alone.",
                            type="int", min=0, max=8),

    # --- Pipeline limits ---
    "planner_enabled": Spec("pipeline", "Planner stage",
                            "Before Dev writes anything, a Planner agent reads the repository "
                            "and decides which code changes and in what order — then every "
                            "symbol it named is verified against the code graph. Off = Dev "
                            "works from the scope alone, which measurably drops a multi-agent "
                            "pipeline to single-agent performance.", type="bool"),
    "planner_max_rounds": Spec("pipeline", "Planner rounds",
                               "Retrieve/decide rounds the Planner may take before it must "
                               "commit to a plan.", type="int", min=1, max=8,
                               show_if="planner_enabled=true"),
    "dev_inject_file_contents": Spec("pipeline", "Pre-read the plan's files",
                                     "Put the real contents of the files the plan names into the "
                                     "Dev prompt, so it edits instead of re-reading them (~70% of "
                                     "a Dev run's input tokens used to go on rediscovery). Off = "
                                     "Dev pulls everything through its own tools.", type="bool"),
    "max_revision_rounds": Spec("pipeline", "Max revise rounds",
                                "If QA fails or Review requests changes, feedback goes back to Dev and "
                                "QA+Review re-run — up to this many times.", type="int", min=0, max=6),
    "fast_path_enabled": Spec("pipeline", "Trivial-task fast path",
                              "Skip the paid LLM QA + Review passes when a scope is deterministically "
                              "trivial (one ticket, ≤2 grep-pinned files) AND the change passes the "
                              "test gate with zero new failures and a small diff. Any doubt falls "
                              "back to the full QA+Review loop.", type="bool"),
    "dev_max_rounds": Spec("pipeline", "Dev loop rounds",
                           "Max model calls per ticket inside the Dev agent (edit, then verify rounds).",
                           type="int", min=1, max=10),
    "dev_run_tests": Spec("pipeline", "Run tests in Dev loop",
                          "Run the repo's tests (pytest / npm / go / cargo) on touched test files "
                          "inside the Dev loop, feeding real failures back to the model before QA.",
                          type="bool"),
    "pm_max_retrieval_rounds": Spec("pipeline", "PM retrieval rounds",
                                    "Max on-demand knowledge lookups the PM may run within a single turn.",
                                    type="int", min=0, max=8),
    "max_request_chars": Spec("pipeline", "Max request size (chars)",
                              "Hard cap per LLM request (~4 chars/token). Sized for Groq's tightest "
                              "free-tier budget; raise it for paid tiers.", type="int", min=4000, max=400000),
    "dev_file_chars": Spec("pipeline", "Per-file context (chars)",
                           "How much of each file the Dev model sees. 30K covers most source files whole.",
                           type="int", min=4000, max=200000),

    # --- Knowledge base → Code graph (the deterministic localization engine) ---
    "graph_enabled": Spec("knowledge", "Code graph",
                          "Index each repo into a persistent knowledge graph (definitions, call "
                          "edges, imports, HTTP routes; 158 languages) via the codebase-memory-mcp "
                          "binary — the localization layer the agents ground on. Deterministic, "
                          "free, RAM-first (seconds per repo). Off = fall back to the built-in "
                          "symbol map + ripgrep.", type="bool", section="Code graph"),
    "graph_binary": Spec("knowledge", "Graph binary",
                         "Binary name on PATH or an absolute path to it. Install via npm/PyPI/"
                         "Homebrew/Scoop (`codebase-memory-mcp`) or a GitHub release download.",
                         section="Code graph", show_if="graph_enabled=true"),
    "graph_index_mode": Spec("knowledge", "Index mode",
                             "full = structural + similarity + semantic-embedding edges (enables "
                             "semantic search). moderate = structural + similarity. fast = "
                             "structural only (quickest, no semantic search).",
                             type="enum", options=["full", "moderate", "fast"],
                             section="Code graph", show_if="graph_enabled=true"),
    # --- Knowledge base → Code search (the lexical channel) ---
    "ripgrep_enabled": Spec("knowledge", "ripgrep",
                            "Use ripgrep as the lexical search engine: identifier-aware "
                            "matching, per-language definition patterns, .gitignore-aware, "
                            "and the `grep` tool every agent can call mid-run. It is the "
                            "channel that sees what an AST index can't — strings, comments, "
                            "config, templates. Off (or binary missing) = `git grep`: "
                            "tracked files only; definition searches retain language "
                            "filters, but the fallback has no .gitignore-aware walk.",
                            type="bool", section="Code search"),
    "ripgrep_path": Spec("knowledge", "ripgrep binary",
                         "Binary name on PATH or an absolute path to it. Install with "
                         "`apt install ripgrep`, `brew install ripgrep`, or "
                         "`cargo install ripgrep`.",
                         section="Code search", show_if="ripgrep_enabled=true"),

    # --- Knowledge base → Semantic search (local dense embeddings) ---
    "local_embeddings": Spec("knowledge", "Semantic search",
                             "Embed the code graph's symbols with a local model (fastembed, CPU, "
                             "no Docker/API) and fuse with keyword search. Doubles 'describe it in "
                             "English and find the code' recall (measured 8/10 vs 4/10). Needs the "
                             "[semantic] extra; off = keyword-only.",
                             type="bool", section="Semantic search"),
    "embedding_model": Spec("knowledge", "Embedding model",
                            "A fastembed model id. Default nomic-embed-text-v1.5-Q (768d, ~130MB, "
                            "code-capable). Stronger for code: jinaai/jina-embeddings-v2-base-code "
                            "(~640MB). Re-index repos after changing this.",
                            section="Semantic search", show_if="local_embeddings=true"),
    "rrf_dense_weight": Spec("knowledge", "Semantic weight",
                             "How much the semantic channel outweighs keyword search when the two "
                             "are fused. 1 = equal (measurably worse — keyword noise crowds out "
                             "correct semantic hits); 2 is the tuned default.",
                             type="float", min=0.0, max=5.0,
                             section="Semantic search", show_if="local_embeddings=true"),
    # --- Knowledge base → Retrieval pipeline (the stages every agent calls) ---
    "graph_expansion": Spec("knowledge", "Call-graph expansion",
                            "After ranking, pull each top hit's neighbourhood — callers, callees, "
                            "class members, the tests that already cover it — into the results. "
                            "Matching finds the symbol; adjacency finds what breaks when you "
                            "change it.", type="bool", section="Retrieval pipeline"),
    "graph_hops": Spec("knowledge", "Expansion hops",
                       "How far to expand. 1 is the tuned default and what the research measured "
                       "as a clear gain; 2 was measured as a LOSS when flattened into a prompt "
                       "(the model drowns), and exists here for ablation.",
                       type="int", min=1, max=2,
                       section="Retrieval pipeline", show_if="graph_expansion=true"),
    "grep_refine": Spec("knowledge", "Lexical refinement",
                        "Finish each retrieval with a ripgrep pass over the identifiers in the "
                        "query, catching occurrences an AST index cannot model — strings, "
                        "comments, config, templates.",
                        type="bool", section="Retrieval pipeline"),
    "snippet_context": Spec("knowledge", "Attach source",
                            "Include the real source of the top-ranked locations in results, so "
                            "an agent edits instead of spending a tool call reading back every "
                            "pin it was just given.", type="bool", section="Retrieval pipeline"),
    "snippet_summarize": Spec("knowledge", "Summarize large symbols",
                              "A symbol too big to inline is summarized by the knowledge model "
                              "(cached per commit) rather than truncated mid-function.",
                              type="bool", section="Retrieval pipeline",
                              show_if="snippet_context=true"),
    "rerank_mode": Spec("knowledge", "Reranking",
                        "How the merged candidate pool is ordered. 'deterministic' is free and "
                        "always available (it weighs the fused rank, source-vs-test, and the "
                        "Planner's verified targets). 'llm' adds one scoring call per retrieval "
                        "— accurate, but retrieval happens many times per run. 'cross-encoder' "
                        "needs the [rerank] extra installed. A tier that can't run falls back to "
                        "deterministic rather than to an unranked pool. 'off' keeps the fused "
                        "order untouched — for measuring what the reranker is worth.",
                        type="enum", options=["deterministic", "llm", "cross-encoder", "off"],
                        section="Retrieval pipeline"),
    "rerank_provider": Spec("knowledge", "Rerank provider",
                            "Which provider scores the candidates. Empty = the knowledge stage's.",
                            type="provider", options=[""] + providers.stage_provider_ids("knowledge"),
                            model_field="rerank_model",
                            section="Retrieval pipeline", show_if="rerank_mode=llm"),
    "rerank_model": Spec("knowledge", "Rerank model",
                         "Model for the rerank call. Empty = the knowledge stage's.",
                         type="model", provider_field="rerank_provider",
                         section="Retrieval pipeline", show_if="rerank_mode=llm"),

    "kb_auto_refresh": Spec("knowledge", "Auto-refresh on drift",
                            "Reindex the code graph + symbol map at pipeline entry when the repo's "
                            "origin has moved (deterministic, free).", type="bool"),
    "kb_write_back": Spec("knowledge", "Delivery write-back",
                          "After a scope delivers, persist a delivery-note doc so future retrieval "
                          "compounds across runs.", type="bool"),

    # --- Delivery & safety ---
    "demo_mode": Spec("delivery", "Demo mode",
                      "Safe default: the pipeline never pushes branches or opens real PRs — it logs "
                      "what it would do. Turn off only against a repo you own.", type="bool"),
    "open_real_pr": Spec("delivery", "Open real PRs",
                         "Open a PR via the gh CLI at the end of the pipeline. Off = deliveries "
                         "wait for the Create PR button on the board.",
                         type="bool", show_if="demo_mode=false"),
    "agent_git_name": Spec("delivery", "Agent commit name",
                           "Author name on every commit the pipeline makes — the history shows the "
                           "agent, not the server's git config."),
    "agent_git_email": Spec("delivery", "Agent commit email",
                            "Author email for the agent's commits (a GitHub noreply address keeps "
                            "it unlinked from a personal account)."),
    "github_bot_token": Spec("delivery", "GitHub bot token",
                             "Token of a dedicated bot/machine account. When set, pushes and PR "
                             "creation authenticate as that account, so PRs are created BY the "
                             "agent on GitHub. Empty = the host's gh login opens them.", secret=True,
                             show_if="demo_mode=false"),

    # --- Jira (optional) ---
    "jira_base_url": Spec("jira", "Jira base URL", "e.g. https://your-org.atlassian.net"),
    "jira_email": Spec("jira", "Jira account email"),
    "jira_api_token": Spec("jira", "Jira API token", secret=True),
    "jira_project_key": Spec("jira", "Jira project key", "Approved tickets are pushed here as Stories."),
}

GROUPS: list[tuple[str, str, str]] = [
    ("providers", "Connections", "Connect your LLM providers and agentic coding CLIs. Keys are "
                                 "encrypted, stored locally, and never shown back."),
    ("models", "Agent models", "Assign a provider + model to each pipeline stage. Use the preset "
                               "to point every stage at one provider in a click."),
    ("jury", "Review jury", "Code quality should come from several independent perspectives, not "
                            "one model's judgement. Seat the judges, give each its own model, and "
                            "tune how the foreperson weighs what they find."),
    ("pipeline", "Pipeline limits", "Loop bounds and request budgets."),
    ("knowledge", "Knowledge base", "How repositories are indexed and kept fresh."),
    ("delivery", "Delivery & safety", "What the pipeline is allowed to do to real repositories."),
    ("jira", "Jira integration", "Optional: push approved tickets to a Jira board."),
]


def _coerce(name: str, value, spec: Spec):
    if spec.type == "bool":
        if isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if spec.type == "int":
        value = int(value)
    elif spec.type == "float":
        value = float(value)
    else:
        value = str(value).strip()
        if spec.type in ("enum", "provider") and spec.options and value not in spec.options:
            raise ValueError(f"{name}: '{value}' is not one of {spec.options}")
        return value
    if spec.min is not None and value < spec.min:
        raise ValueError(f"{name}: must be ≥ {spec.min:g}")
    if spec.max is not None and value > spec.max:
        raise ValueError(f"{name}: must be ≤ {spec.max:g}")
    return value


def apply_overrides(db: Session) -> None:
    """Load persisted overrides onto the settings singleton (startup)."""
    applied = 0
    for row in db.exec(select(AppSetting)).all():
        spec = FIELDS.get(row.key)
        if spec is None:
            continue  # a field that no longer exists
        try:
            raw = json.loads(row.value)
            if spec.secret and isinstance(raw, str):
                raw = crypto.decrypt(raw)  # stored encrypted; live singleton holds plaintext
            setattr(settings, row.key, _coerce(row.key, raw, spec))
            applied += 1
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Skipping bad setting override %s: %s", row.key, exc)
    if applied:
        logger.info("Applied %d runtime setting override(s)", applied)


def _persist(db: Session, name: str, value, spec: Spec) -> None:
    """Write one coerced value: encrypt secrets at rest, keep plaintext on the singleton."""
    stored = crypto.encrypt(value) if (spec.secret and isinstance(value, str)) else value
    row = db.get(AppSetting, name)
    if row is None:
        row = AppSetting(key=name, value=json.dumps(stored))
    else:
        row.value = json.dumps(stored)
        row.updated_at = utcnow()
    db.add(row)
    setattr(settings, name, value)


def update(db: Session, values: dict) -> list[str]:
    """Validate, persist, and live-apply a batch of overrides. Returns the list
    of field names actually changed. Raises ValueError on any invalid value
    (nothing is applied in that case)."""
    coerced: dict[str, object] = {}
    for name, value in values.items():
        spec = FIELDS.get(name)
        if spec is None:
            raise ValueError(f"Unknown setting '{name}'")
        if spec.secret and value == SECRET_MASK:
            continue  # untouched masked field sent back — ignore
        coerced[name] = _coerce(name, value, spec)

    for name, value in coerced.items():
        _persist(db, name, value, FIELDS[name])
    db.commit()
    return sorted(coerced)


def apply_provider_preset(db: Session, provider_id: str) -> list[str]:
    """Point every stage at one provider (with that provider's recommended model per
    stage). Persisted + live-applied like update(). Raises ValueError for an unknown
    or non-applicable provider."""
    values = providers.preset_values(provider_id)  # raises ValueError if invalid
    for name, value in values.items():
        _persist(db, name, value, FIELDS[name])
    db.commit()
    return sorted(values)


def _providers_view() -> list[dict]:
    """Registry for the UI: dependent provider→model dropdowns + key status.
    Agent-CLI providers additionally carry live availability (installed? which
    version?) so the stage dropdowns can mark tools that can't run here."""
    from . import agent_backends  # local import — avoids a cycle at module load

    avail = agent_backends.availability()
    out = []
    for pid, p in providers.PROVIDERS.items():
        base_url, _ = providers.endpoint(pid)
        entry = {
            "id": pid, "name": p.name, "kind": p.kind, "note": p.note,
            "models": list(p.models), "default_model": p.default_model,
            "key_field": p.key_field, "base_url_field": p.base_url_field,
            "base_url": base_url,
            "local_endpoint": providers.is_local_endpoint(base_url),
            "path_field": p.path_field, "key_set": providers.has_key(pid),
        }
        if p.backend:
            det = avail.get(p.backend, {"available": False, "version": "",
                                        "reason": "unknown backend"})
            entry.update(backend=p.backend, available=det["available"],
                         version=det["version"], unavailable_reason=det["reason"],
                         connect_hint=det.get("connect_hint", ""),
                         installable=det.get("installable", False))
        out.append(entry)
    return out


def view() -> dict:
    """Current values grouped for the Settings screen, secrets masked."""
    groups = []
    for gid, label, help_text in GROUPS:
        fields = []
        for name, spec in FIELDS.items():
            if spec.group != gid:
                continue
            raw = getattr(settings, name, "")
            value = (SECRET_MASK if raw else "") if spec.secret else raw
            fields.append({
                "name": name, "label": spec.label, "help": spec.help,
                "type": spec.type, "secret": spec.secret, "options": spec.options,
                "min": spec.min, "max": spec.max, "value": value,
                "set": bool(raw) if spec.secret else None,
                "provider_field": spec.provider_field, "model_field": spec.model_field,
                "section": spec.section, "show_if": spec.show_if,
            })
        groups.append({"id": gid, "label": label, "help": help_text, "fields": fields})
    return {
        "groups": groups,
        "providers": _providers_view(),
        "preset_providers": [
            {"id": pid, "name": providers.PROVIDERS[pid].name}
            for pid in providers.preset_provider_ids()
        ],
        # Legacy: a flat name→models catalog for the shared datalist.
        "model_catalog": {p.name: list(p.models) for p in providers.PROVIDERS.values() if p.models},
    }
