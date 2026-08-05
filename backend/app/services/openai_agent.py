"""OpenAI-compatible transport: chat completions with cross-provider fallback,
plus the SEARCH/REPLACE agentic coding loop.

`chat` serves any stage routed to an OpenAI-kind provider (groq/openai/gemini/
xai/custom), with per-minute-budget trimming, 429/network retries, and fallback
across the other configured stages' providers (separate free-tier quota pools).
`code` is the HTTP coding loop used when no agentic CLI backend is available:
localize → edit (SEARCH/REPLACE protocol) → verify against the applied diff and
targeted tests → iterate."""

import json
import re
import time
from pathlib import Path

import httpx

from ..config import settings
from . import lang, providers
from .knowledge import tools

_PATH_RE = re.compile(r"[A-Za-z0-9_][\w./\-]*\.[A-Za-z0-9]+")

# (provider, model) pairs that hit a DAILY quota this process (per-minute limits are
# retried in-call; daily ones can't be waited out, so we route around them until restart).
_EXHAUSTED: set[tuple[str, str]] = set()


def _endpoint(provider: str) -> tuple[str, str]:
    """(base_url, api_key) for an OpenAI-compatible provider id (see providers.py)."""
    return providers.endpoint(provider)


def _candidate_pool(provider: str, model: str) -> list[tuple[str, str]]:
    """Fallback chain of (provider, model): the requested one first, then the other
    configured OpenAI-compatible stages (each its own free-tier quota pool, and often a
    different provider entirely), then the Gemini overflow models last — Gemini has a
    much larger per-minute budget, so it absorbs requests others reject for size
    (413/TPM) or that exhaust a daily quota. Non-OpenAI-kind stages (anthropic/CLI) and
    providers with no key are skipped in the fallback tail (but the primary is always
    tried, so a missing key surfaces as a clear error rather than a silent skip)."""
    pool: list[tuple[str, str]] = [(provider, model)]
    for stage in ("dev", "pm", "qa", "review", "knowledge"):
        sp = getattr(settings, f"{stage}_provider", "")
        sm = getattr(settings, f"{stage}_model", "")
        if sm and providers.kind(sp) == "openai" and providers.has_key(sp):
            pool.append((sp, sm))
    if settings.gemini_api_key:
        for gm in (x.strip() for x in settings.gemini_models.split(",") if x.strip()):
            pool.append(("gemini", gm))
    out: list[tuple[str, str]] = []
    for pv, mdl in pool:
        if pv and mdl and (pv, mdl) not in out and providers.kind(pv) == "openai":
            out.append((pv, mdl))
    return out


def _is_daily_limit(err: str) -> bool:
    low = (err or "").lower()
    return "per day" in low or "tpd" in low or "daily" in low


def _is_too_big(err: str) -> bool:
    """A request that exceeds the model's per-minute token budget (Groq free tier
    caps e.g. gpt-oss-120b at 8000 TPM). Can't be waited out — needs a smaller
    request (we trim proactively) or a roomier model (pool fallback)."""
    low = (err or "").lower()
    return "413" in low or "request too large" in low or "tokens per minute" in low or "tpm" in low


def _is_network(err: str) -> bool:
    """Transient transport failures (DNS blip, connection reset/refused, read
    timeout) — nothing about the request itself is wrong, so they're worth
    retrying and worth trying another provider's endpoint. One un-retried DNS
    blip once silently voided a whole QA+Review stage (run log #77/78)."""
    low = (err or "").lower()
    return any(k in low for k in (
        "name resolution", "getaddrinfo", "temporary failure", "connecterror",
        "connection refused", "connection reset", "connection error", "timed out",
        "timeout", "network is unreachable", "server disconnected", "eof occurred"))


def _trim_messages(messages: list[dict], cap_chars: int) -> list[dict]:
    """Guarantee the request fits a char budget (~4 chars/token) so it never
    exceeds the smallest Groq free-tier per-minute limit. Trims the LARGEST
    message's middle first (keeps head+tail — usually a diff/file dump), leaving
    a marker. System + structure are preserved; we never drop whole messages."""
    out = [dict(m) for m in messages]

    def total() -> int:
        return sum(len(m.get("content") or "") for m in out)
    while total() > cap_chars:
        i = max(range(len(out)), key=lambda j: len(out[j].get("content") or ""))
        content = out[i].get("content") or ""
        overshoot = total() - cap_chars
        # Trim this message by the overshoot (+ a margin), cutting from the middle.
        target = max(600, len(content) - overshoot - 400)
        if target >= len(content):
            break  # can't shrink further without emptying — give up (rare)
        head = target * 2 // 3
        tail = target - head
        out[i]["content"] = (content[:head]
                             + f"\n… [trimmed {len(content) - target} chars to fit token budget] …\n"
                             + content[len(content) - tail:])
    return out


def _load_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        if text and "{" in text:
            try:
                return json.loads(text[text.find("{"): text.rfind("}") + 1])
            except (json.JSONDecodeError, ValueError):
                pass
    return {}


def chat(system: str, user: str = "", *, provider: str | None = None, model: str | None = None,
         timeout: int = 180, json_mode: bool = False, messages: list[dict] | None = None) -> dict:
    """One chat completion with provider-fallback: if the requested (provider, model)'s
    DAILY quota is exhausted (429 that can't be waited out) or it's too big / a transient
    network failure, retry on the other configured OpenAI-compatible stages — each lives
    in a separate free-tier quota pool (and often a different provider). `messages`
    (multi-turn) overrides `user`. Returns {text, tokens_in, tokens_out, cost, model, error}."""
    primary_provider = provider or settings.qa_provider
    primary_model = model or settings.qa_model
    last: dict | None = None
    for pv, m in _candidate_pool(primary_provider, primary_model):
        if (pv, m) in _EXHAUSTED:
            continue
        r = _chat_once(system, user, provider=pv, model=m, timeout=timeout,
                       json_mode=json_mode, messages=messages)
        err = r.get("error") or ""
        if not err:
            return r
        if "429" in err or _is_too_big(err):
            if _is_daily_limit(err):
                _EXHAUSTED.add((pv, m))
            last = r
            continue  # rate-limited / too-big even after in-call retries — try the next pool
        if _is_network(err):
            last = r
            continue  # transport failure even after in-call retries — another provider's endpoint may be fine
        return r  # a real error (auth, bad request…) — don't mask it by switching providers
    return last or {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
                    "error": "all configured providers are rate-limited or quota-exhausted"}


def _chat_once(system: str, user: str, *, provider: str, model: str, timeout: int = 180,
               json_mode: bool = False, messages: list[dict] | None = None) -> dict:
    """One chat completion against ONE (provider, model), routed to that provider's
    OpenAI-compatible endpoint. Returns {text, tokens_in, tokens_out, cost, error}."""
    base_url, api_key = _endpoint(provider)
    if not api_key or not base_url:
        p = providers.PROVIDERS.get(provider)
        which = (p.key_field.upper() if p and p.key_field else "API key")
        return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
                "error": f"{which} not configured for provider '{provider}' — skipping {model}"}
    # Gemini's per-minute budget is huge, so only trim to Groq's tight cap for
    # non-Gemini providers; Gemini takes the full (already file-capped) request.
    cap = settings.max_request_chars if provider != "gemini" else max(settings.max_request_chars, 120000)
    msgs = [{"role": "system", "content": system}] + (
        messages if messages is not None else [{"role": "user", "content": user}])
    # Hard fit to the per-minute token budget so we never 413 on Groq's free tier
    # (a too-big request can't be waited out; trimming context beats zero output).
    msgs = _trim_messages(msgs, cap)
    payload = {"model": model, "messages": msgs}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    # Retry on 429 (Groq's free tier is rate-limited) honoring Retry-After, and
    # on transient transport failures (DNS blip, connection reset, read timeout)
    # with backoff — a momentary network hiccup must not void a whole QA/Review
    # stage. Keeps Dev/QA/Review from failing on bursts.
    try:
        data = None
        for attempt in range(4):
            try:
                with httpx.Client(timeout=timeout) as c:
                    r = c.post(
                        f"{base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=payload,
                    )
            except Exception as exc:  # noqa: BLE001 — transport-level failure
                if attempt < 3 and _is_network(str(exc)):
                    time.sleep(3.0 * (attempt + 1))
                    continue
                return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
                        "error": str(exc)}
            if r.status_code == 429 and attempt < 3 and not _is_daily_limit(r.text):
                try:
                    wait = float(r.headers.get("retry-after", ""))
                except ValueError:
                    wait = 0.0
                time.sleep(min(max(wait, 2.0 * (attempt + 1)), 20.0))
                continue
            if r.status_code >= 400:
                return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
                        "error": f"OpenAI {r.status_code}: {r.text[:200]}"}
            data = r.json()
            break
        if data is None:
            return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
                    "error": "OpenAI 429: rate limit — retries exhausted"}
    except Exception as exc:  # noqa: BLE001
        return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "error": str(exc)}

    # Guard against 200-with-malformed-body (e.g. a gateway/proxy error envelope
    # or empty choices) so the pipeline fails gracefully instead of crashing.
    choices = data.get("choices") or []
    if not choices:
        return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
                "error": f"OpenAI returned no choices: {str(data)[:200]}"}
    text = ((choices[0] or {}).get("message") or {}).get("content") or ""
    usage = data.get("usage", {}) or {}
    tin = usage.get("prompt_tokens", 0)
    tout = usage.get("completion_tokens", 0)
    # gpt-4o list price ($2.5 / $10 per 1M) — approximate for the cost meter.
    cost = round(tin / 1_000_000 * 2.5 + tout / 1_000_000 * 10, 4)
    return {"text": text, "tokens_in": tin, "tokens_out": tout, "cost": cost, "model": model, "error": None}


# We deliberately DON'T use JSON or whole-file regeneration for the coding agent:
#   * Groq's JSON mode rejects large code payloads (`json_validate_failed`).
#   * Asking a mid-size model to re-emit a whole large file verbatim loses code
#     (it returns a shorter, simplified version — correctly rejected downstream).
# Instead we use a plain-text SEARCH/REPLACE edit protocol (the model reproduces
# only a small exact snippet) plus whole-file blocks for genuinely new files.
_CODE_SYSTEM = (
    "You are a senior software engineer implementing a ticket in an EXISTING "
    "repository, working over MULTIPLE rounds. Each round you emit edits; the "
    "harness applies them and replies with what applied, the resulting git diff, "
    "and test results. You then FIX problems or fill gaps until the ticket is "
    "fully implemented, and only then declare DONE.\n\n"
    "OUTPUT FORMAT — respond with exactly this and NOTHING else.\n"
    "First line: SUMMARY: <one sentence>\n"
    "Last line: STATUS: CONTINUE  (or STATUS: DONE — see below)\n\n"
    "To EDIT an existing file, emit one block per change:\n"
    "<<<EDIT relative/path>>>\n"
    "<<<SEARCH>>>\n"
    "<a short, UNIQUE run of lines copied VERBATIM from the shown file>\n"
    "<<<REPLACE>>>\n"
    "<the new lines that replace them>\n"
    "<<<END>>>\n\n"
    "To CREATE a new file:\n"
    "<<<FILE relative/path>>>\n"
    "<the complete file content>\n"
    "<<<END>>>\n\n"
    "To SEE a file you were not shown (next round will contain its content):\n"
    "<<<OPEN relative/path>>>\n\n"
    + tools.PROTOCOL_BLOCK + "\n"
    "Rules:\n"
    "- The localization hints you were given come from a PM working off summaries — "
    "they are a HYPOTHESIS, not fact. If the shown files don't plausibly contain the "
    "behavior, QUERY THE INDEX (SEARCH/LOOKUP/CALLERS) and edit where the code "
    "actually lives. Say so in your SUMMARY when you conclude the hints were wrong.\n"
    "- SEARCH must match the shown file EXACTLY, including indentation. Keep each "
    "SEARCH block small and unique (2-8 lines); several small edits beat one huge one.\n"
    "- Implement the ticket COMPLETELY: a new function/flag must also be WIRED IN "
    "(registered, called, added to the parser/config/exports) — check the diff for this.\n"
    "- STATUS: DONE only when the diff you were shown fully implements every "
    "acceptance criterion and tests pass. Never DONE in the same round as new edits.\n"
    "- Do NOT wrap content in markdown fences. No prose outside SUMMARY, the blocks, "
    "and STATUS."
)

_SUMMARY_RE = re.compile(r"SUMMARY:\s*(.+)")
_EDIT_RE = re.compile(
    r"<<<EDIT\s+(.+?)\s*>>>\s*\n<<<SEARCH>>>\n(.*?)\n<<<REPLACE>>>\n(.*?)\n?<<<END>>>",
    re.DOTALL,
)
_NEWFILE_RE = re.compile(r"<<<FILE\s+(.+?)\s*>>>\n(.*?)\n?<<<END>>>", re.DOTALL)
_OPEN_RE = re.compile(r"<<<OPEN\s+(.+?)\s*>>>")
_DONE_RE = re.compile(r"STATUS:\s*DONE", re.IGNORECASE)


def _clean_path(p: str) -> str:
    return p.strip().strip("`\"'")


def _parse_edits(text: str) -> tuple[str, list[tuple[str, str, str]], list[tuple[str, str]], list[str], list[tuple[str, str]], bool]:
    """Parse the edit protocol → (summary, edits, new_files, opens, queries, done).

    edits are (path, search, replace); new_files are (path, content); opens are
    file paths the model asked to see; queries are (tool, argument) index
    lookups it asked to run (search/lookup/callers/outline/grep); done is the
    STATUS: DONE flag. Falls back to the legacy JSON
    `{files:[{path,content}]}` shape (real OpenAI handles that fine) so both
    providers keep working."""
    edits = [(_clean_path(p), s, r) for p, s, r in _EDIT_RE.findall(text)]
    # Strip matched EDIT spans so the FILE regex can't re-match their bodies.
    remainder = _EDIT_RE.sub("", text)
    new_files = [(_clean_path(p), c) for p, c in _NEWFILE_RE.findall(remainder)]
    # OPEN/query requests only outside EDIT/FILE bodies (so file content can't fake one).
    outside = _NEWFILE_RE.sub("", remainder)
    opens = [_clean_path(p) for p in _OPEN_RE.findall(outside)]
    # One parser for the request protocol, shared with the reviewing stages
    # (services/knowledge/tools.py) — the tool list must not drift per surface.
    queries = tools.parse_requests(outside)
    done = bool(_DONE_RE.search(outside))
    m = _SUMMARY_RE.search(text)
    summary = m.group(1).strip() if m else ""

    if not edits and not new_files:
        data = _load_json(text)
        summary = summary or data.get("summary", "")
        new_files = [(f.get("path"), f.get("content")) for f in (data.get("files") or [])
                     if isinstance(f, dict) and f.get("path") and isinstance(f.get("content"), str)]
    return summary, edits, new_files, opens, queries, done


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _reindent(lines: list[str], delta: int) -> list[str]:
    """Shift every non-blank line's leading indentation by `delta` spaces."""
    if delta == 0:
        return lines
    out = []
    for l in lines:
        if not l.strip():
            out.append(l)
        else:
            out.append(" " * max(0, _indent(l) + delta) + l.lstrip())
    return out


def _apply_edit(original: str, search: str, replace: str) -> tuple[str, bool, str]:
    """Apply one SEARCH/REPLACE → (merged, ok, reason-if-not). Exact match first,
    then an indentation-insensitive line-window match (models often dedent); the
    tolerant path re-indents REPLACE to the matched region. An AMBIGUOUS match
    (search appears more than once) is refused rather than guessed — a wrong-spot
    replace silently corrupts code."""
    if not search.strip():
        return original, False, "empty SEARCH block"
    n_exact = original.count(search)
    if n_exact == 1:
        return original.replace(search, replace, 1), True, ""
    if n_exact > 1:
        return original, False, f"SEARCH matches {n_exact} locations — add surrounding lines to make it unique"
    o_lines = original.split("\n")
    s_lines = search.split("\n")
    n = len(s_lines)
    s_norm = [l.strip() for l in s_lines]
    matches = [i for i in range(len(o_lines) - n + 1)
               if [l.strip() for l in o_lines[i:i + n]] == s_norm]
    if len(matches) > 1:
        return original, False, f"SEARCH matches {len(matches)} locations (ignoring indentation) — make it unique"
    if len(matches) == 1:
        i = matches[0]
        # Align the replacement to the file's actual indentation: delta =
        # (indent of first matched line) − (indent of first SEARCH line).
        o_first = next((l for l in o_lines[i:i + n] if l.strip()), "")
        s_first = next((l for l in s_lines if l.strip()), "")
        delta = _indent(o_first) - _indent(s_first)
        merged = o_lines[:i] + _reindent(replace.split("\n"), delta) + o_lines[i + n:]
        return "\n".join(merged), True, ""
    return original, False, "SEARCH not found in file — copy the lines verbatim from the shown content"


def _syntax_error(path: str, content: str) -> str | None:
    """Parse gate: a merged result that no longer parses is rejected before it
    touches disk (the model gets the error back instead). Language-dispatched
    via services/lang.py (Python ast; node --check / gofmt when available);
    ungateable languages fail open — the test run still catches real breakage."""
    return lang.syntax_error(path, content)


def _apply_round(cwd: str, edits: list, new_files: list, written: list[str],
                 on_event=None) -> list[str]:
    """Apply one round of edits + new files to the working copy. Returns
    human/model-readable per-action result lines (the loop's feedback)."""
    from . import git_ops

    results: list[str] = []

    def _fail(msg: str) -> None:
        results.append(msg)
        if on_event:
            on_event("warn", msg)  # surface skipped edits so zero-change runs are diagnosable

    # 1) SEARCH/REPLACE edits into existing files (syntax-gated).
    for path, search, replace in edits:
        existing = git_ops.read_file(cwd, path, max_chars=400000)
        if existing is None:
            _fail(f"✗ EDIT {path}: file not found (use <<<FILE>>> to create new files)")
            continue
        merged, ok, reason = _apply_edit(existing, search, replace)
        if not ok:
            _fail(f"✗ EDIT {path}: {reason}")
            continue
        err = _syntax_error(path, merged)
        if err:
            _fail(f"✗ EDIT {path}: REJECTED — result no longer parses ({err})")
            continue
        try:
            git_ops.write_file(cwd, path, merged)  # enforces workspace containment
        except (ValueError, OSError) as exc:
            results.append(f"✗ EDIT {path}: blocked path ({exc})")
            continue
        if path not in written:
            written.append(path)
        results.append(f"✓ EDIT {path}: applied")
        if on_event:
            on_event("info", f"✎ edited {path}")

    # 2) Whole-file blocks — new files, or full rewrites of small files.
    for path, content in new_files:
        if not (path and isinstance(content, str)):
            continue
        from . import git_ops as _g
        existing = _g.read_file(cwd, path, max_chars=400000)
        # Truncation guard for existing non-trivial files (don't clobber real code
        # with a shorter, likely-partial rewrite). New files are always allowed.
        if existing and len(existing) > 400 and len(content) < 0.55 * len(existing):
            _fail(f"✗ FILE {path}: rewrite much shorter than the existing file — "
                  "likely truncated; use SEARCH/REPLACE edits instead")
            continue
        err = _syntax_error(path, content)
        if err:
            _fail(f"✗ FILE {path}: REJECTED — does not parse ({err})")
            continue
        try:
            _g.write_file(cwd, path, content)
        except (ValueError, OSError) as exc:
            _fail(f"✗ FILE {path}: blocked path ({exc})")
            continue
        if path not in written:
            written.append(path)
        results.append(f"✓ FILE {path}: written ({len(content)} chars)")
        if on_event:
            on_event("info", f"{'✎ wrote' if existing else '＋ created'} {path} ({len(content)} chars)")
    return results


def _looks_like_test(path: str) -> bool:
    return lang.is_test_file(path)


def code(cwd: str, task_key: str, title: str, description: str, criteria: list[str],
         on_event=None, model: str | None = None, provider: str | None = None, context: str = "",
         affected_files: list[str] | None = None, target_symbols: list[str] | None = None) -> dict:
    """Agentic Dev loop (Agentless/AutoCodeRover-style: localize → edit → verify →
    iterate). Each round the model emits SEARCH/REPLACE edits / new files / OPEN
    requests; the harness applies them and feeds back per-edit results, the real
    git diff of its work so far, and targeted test results. The model may only
    declare DONE after seeing its own applied diff — this catches the classic
    single-shot failure of defining a function but never wiring it in.

    Returns {summary, files, tokens_in, tokens_out, cost, error}.

    `affected_files` / `target_symbols` are the PM agent's localization for this
    ticket — the exact files and symbols to work on. When present they drive which
    files are shown to the model (no rediscovery), falling back to a path-regex
    over the ticket text only when the PM didn't localize."""
    from . import git_ops

    model = model or settings.dev_model
    provider = provider or settings.dev_provider
    file_chars = settings.dev_file_chars
    # The code graph / embedding index is keyed by repo slug, which is exactly
    # the working copy's directory name (git_ops.slug is idempotent on slugs).
    repo = Path(cwd).name
    tree = git_ops.list_files(cwd, 300)
    tree_set = set(tree)
    # Localization first: use the files the PM already pinned (that exist in the
    # tree). Only if none were provided/exist do we fall back to scraping paths out
    # of the ticket text + retrieved knowledge.
    pinned = [p for p in dict.fromkeys(affected_files or []) if p in tree_set]
    if pinned:
        candidates = pinned[:6]
    else:
        blob = f"{title} {description} {' '.join(criteria)} {context}"
        candidates = [p for p in dict.fromkeys(_PATH_RE.findall(blob)) if p in tree_set][:5]
    if on_event and pinned:
        on_event("info", f"Using PM-localized files: {', '.join(pinned[:6])}")

    crit = "\n".join(f"- {c}" for c in criteria) or "- (use your judgment)"
    kb = f"\nRelevant repository knowledge:\n{context}\n" if context else ""
    symbols = ", ".join(target_symbols or [])
    sym_block = (f"Target symbols — the PM's localization HYPOTHESIS, not fact. Verify with "
                 f"LOOKUP/SEARCH before trusting it; if the behavior lives elsewhere, edit "
                 f"there instead and say so in your SUMMARY:\n{symbols}\n") if symbols else ""
    new_paths = [p for p in (affected_files or []) if p not in tree_set]
    new_block = f"Files to create (do not exist yet):\n{chr(10).join(new_paths)}\n" if new_paths else ""
    header = (
        f"Ticket {task_key}: {title}\nDescription: {description}\nAcceptance criteria:\n{crit}\n"
        f"{sym_block}{new_block}{kb}\n"
        "Repository file tree (partial):\n" + "\n".join(tree[:120])
    )

    # Each round we send a SELF-CONTAINED prompt: the ticket header + the CURRENT
    # on-disk content of every file in play (re-read fresh) + last round's results
    # + the cumulative diff + verification. No reliance on chat history — so a
    # corrective edit always has the real current lines to match against (the
    # earlier windowing bug lost file content and sent the model editing blind).
    # `_open` grows as the model reaches for files (via OPEN or by editing one it
    # wasn't shown — the PM often mislocalizes and the model correctly picks another).
    open_files: list[str] = [c for c in candidates if c in tree_set]

    def _files_block() -> str:
        blocks = []
        for f in dict.fromkeys(open_files):
            content = git_ops.read_file(cwd, f, max_chars=file_chars)
            if content is not None:
                blocks.append(f"<<<FILE {f}>>>\n{content}\n<<<END>>>")
        return "\n".join(blocks) or "(no files loaded yet — OPEN one or create new files)"

    written: list[str] = []
    summary = ""
    tin = tout = 0
    cost = 0.0
    last_err: str | None = None
    verification_broken = False  # last round's import check / tests failed
    feedback = ("Implement the ticket now with SEARCH/REPLACE edits (and new files if "
                "needed). If the shown files don't contain the code you need — or you're "
                "unsure the hints point to the right place — first query the repository "
                "index (SEARCH for the mechanism, LOOKUP/CALLERS for a symbol, GREP for "
                "raw text) and edit where the behavior ACTUALLY lives. End with STATUS: "
                "CONTINUE — you'll verify against the applied diff before DONE.")
    max_rounds = max(2, settings.dev_max_rounds)

    for round_no in range(max_rounds):
        user = (f"{header}\n\nCurrent content of the files in play (edit against THESE "
                f"exact lines):\n{_files_block()}\n\n{feedback}")
        r = chat(_CODE_SYSTEM, user, provider=provider, model=model, timeout=300)
        tin += r.get("tokens_in", 0) or 0
        tout += r.get("tokens_out", 0) or 0
        cost += r.get("cost", 0.0) or 0.0
        last_err = r.get("error")
        if last_err:
            break
        text = r.get("text", "")
        rsum, edits, new_files, opens, queries, done = _parse_edits(text)
        summary = rsum or summary

        # NEVER accept DONE while a verification is red or nothing's been written.
        if done and not edits and not new_files:
            if written and not verification_broken:
                if on_event:
                    on_event("info", f"Dev loop: DONE after {round_no + 1} round(s), "
                                     f"{len(written)} file(s) changed")
                break
            if verification_broken and round_no < max_rounds - 1:
                feedback = ("You replied DONE, but the last import check or test was still "
                            "FAILING (see above). You are NOT done — emit the SEARCH/REPLACE "
                            "edits that fix it, ending with STATUS: CONTINUE.")
                continue

        results = _apply_round(cwd, edits, new_files, written, on_event)
        # Answer the model's index queries — the loop's localization tools
        # (services/knowledge/tools.py), the same ones the CLI backends get.
        for tool, arg in queries[:4]:
            out = tools.call(repo, cwd, tool, arg)
            results.append(f"{tool.upper()} {arg}:\n{out}")
            if on_event:
                on_event("info", f"⌕ {tool} {arg[:60]} → "
                                 f"{(out.splitlines() or ['(nothing)'])[0][:120]}")
        if not edits and not new_files and not opens and not queries:
            results.append("(you produced no edits, new files, OPEN or index queries)")

        # Make sure every file the model touched or asked for is loaded next round.
        for p in dict.fromkeys(opens + [p for p, _s, _r in edits] + written):
            if p in tree_set and p not in open_files:
                open_files.append(p)

        if round_no >= max_rounds - 1:
            break  # out of rounds — ship whatever applied

        # --- Verification feedback for the next round ---
        fb: list[str] = ["Result of your last round:"]
        fb += results or ["(no blocks found in your reply)"]
        applied_this_round = any(l.startswith("✓") for l in results)
        verification_broken = False
        if written:
            diff = git_ops.diff_worktree(cwd)
            fb.append("Cumulative git diff of ALL your work on this ticket:\n```diff\n"
                      f"{diff[:9000]}\n```")
            if applied_this_round:
                src_edited = [p for p in written if not _looks_like_test(p)]
                imp_ok, imp_detail = git_ops.import_check(cwd, src_edited)
                if imp_ok is False:
                    verification_broken = True
                    fb.append(f"⚠ IMPORT CHECK FAILED — the edited module no longer imports. "
                              f"Fix this before anything else:\n```\n{imp_detail[-1500:]}\n```")
                    if on_event:
                        on_event("warn", "Dev-loop import check FAILED — module broken on import")
            test_paths = [p for p in dict.fromkeys(written + (affected_files or []))
                          if _looks_like_test(p)]
            if applied_this_round and settings.dev_run_tests and test_paths:
                passed, out = git_ops.run_tests(cwd, test_paths=test_paths, timeout=300)
                verdict = "PASSED" if passed else ("could not run" if passed is None else "FAILED")
                if passed is False:
                    verification_broken = True
                fb.append(f"Targeted tests ({', '.join(test_paths[:4])}): {verdict}\n```\n{out[-2500:]}\n```")
                if on_event:
                    on_event("info" if passed else "warn", f"Dev-loop tests: {verdict}")
        fb.append(
            "Verify against the ticket and the diff above: is EVERY acceptance criterion "
            "fully implemented and wired in (new flags registered in the parser, new "
            "functions actually called, tests matching real behavior)? Fix any ✗ failures. "
            "If work remains, emit more edits with STATUS: CONTINUE. If and only if the "
            "diff is complete and tests pass, reply with just SUMMARY + STATUS: DONE."
        )
        feedback = "\n\n".join(fb)

    error = last_err if not written else None
    if not written and not error:
        error = "coding agent produced no file changes"
    return {"summary": summary or "", "files": written,
            "tokens_in": tin, "tokens_out": tout, "cost": cost, "error": error}
