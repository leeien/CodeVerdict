"""The panel roster: which judges exist, in what order, on which models.

Judges live in the ``Judge`` table so the operator can add, drop, reorder and
re-model them at runtime. This module owns everything about the roster itself;
``services/jury`` owns what the roster *does* with a diff.

Two rules shape the defaults:

* The panel is seeded once, on first boot, from ``jury.personas.defaults()``.
  Personas added in a later release do NOT retro-fit onto an existing roster —
  the operator's panel is theirs, and silently growing it would silently grow
  their bill. ``reset_to_defaults`` is the explicit opt-in.
* Judges default to *different* providers. An ensemble whose members all run on
  one model is an ensemble in name only: the members agree, including where they
  are all wrong. ``spread_providers`` assigns distinct configured providers
  round-robin; when nothing else is configured a judge inherits the Review
  stage, which is a correct panel, just a less independent one.
"""

from __future__ import annotations

import logging

from sqlmodel import Session, select

from ..config import settings
from ..models import Judge
from . import providers
from .jury import personas

logger = logging.getLogger(__name__)

# Preference order when spreading judges across providers. Cheap-and-fast first:
# a jury multiplies the review stage's cost by its size, so the default panel
# should not be five frontier models.
_SPREAD_PREFERENCE = ("groq", "gemini", "openai", "xai", "anthropic", "custom",
                      "claude-cli", "codex", "cursor-cli", "gemini-cli")


def all_judges(db: Session) -> list[Judge]:
    return list(db.exec(select(Judge).order_by(Judge.position, Judge.id)).all())


def enabled_judges(db: Session) -> list[Judge]:
    return [j for j in all_judges(db) if j.enabled]


def resolve(judge: Judge) -> tuple[str, str]:
    """(provider, model) this judge actually runs on — its own override, else
    the Review stage's. A provider set without a model borrows the provider's
    default model rather than sending an empty model id."""
    provider = (judge.provider or "").strip() or settings.review_provider
    model = (judge.model or "").strip()
    if not model:
        if (judge.provider or "").strip():
            p = providers.PROVIDERS.get(provider)
            model = (p.default_model if p else "") or settings.review_model
        else:
            model = settings.review_model
    return provider, model


def label(judge: Judge) -> str:
    """Run-row label, e.g. 'Security · groq llama-3.3-70b-versatile'."""
    provider, model = resolve(judge)
    return f"{judge.name} · {providers.label(provider, model)}"


def available_providers() -> list[str]:
    """Configured providers a judge could actually run on right now, in spread
    preference order (keyed API providers + installed agentic CLIs)."""
    return [pid for pid in _SPREAD_PREFERENCE
            if pid in providers.PROVIDERS and providers.can_chat(pid)]


def _spread_model(provider_id: str, registry_default: str) -> str:
    """The model to hand a judge on ``provider_id``.

    The registry's default model is only a guess for a provider whose base URL
    the operator controls — 'openai' commonly points at Groq or a local server,
    where a model from the public OpenAI catalog may not exist. A model id the
    endpoint rejects is an
    *abstention* at review time, i.e. a juror that silently isn't there, which is
    the one failure mode this whole subsystem exists to prevent. So:

      1. If the provider matches the Review stage's, reuse the Review stage's
         model — that pairing is already proven against this endpoint.
      2. Otherwise, when the endpoint can be asked what it serves, keep the
         registry default only if it's actually on the list.
      3. Failing that, take the Review stage's model if this endpoint serves it,
         then the best *curated* id from anywhere in the registry that it serves.
         An OPENAI_BASE_URL pointed at Groq serves Groq's catalog, so this finds
         a real reviewing model instead of whatever sorts first (which is how
         this judge briefly ended up on a 7B translation model).
      4. Only if none of that matches, anything the endpoint serves. A mediocre
         working model still beats a well-chosen id that 404s — the operator can
         always pick better, but they cannot see a juror that never spoke.
    """
    if provider_id == settings.review_provider and settings.review_model:
        return settings.review_model
    p = providers.PROVIDERS.get(provider_id)
    if not registry_default or not (p and p.base_url_field):
        return registry_default          # static endpoint: the registry is right
    live = providers.fetch_models(provider_id)
    if not live:
        return registry_default          # couldn't ask; don't second-guess
    if registry_default in live:
        return registry_default
    if settings.review_model in live:
        return settings.review_model
    served = set(live)
    for other in providers.PROVIDERS.values():
        for candidate in other.models:
            if candidate in served:
                return candidate
    return live[0]


def spread_providers(db: Session) -> int:
    """Assign the enabled judges distinct providers, round-robin over whatever is
    configured. Returns the number of judges changed.

    Judges beyond the number of available providers wrap around — two judges on
    one provider still differ by persona, which is the larger share of the
    diversity. With nothing configured this is a no-op and judges keep
    inheriting the Review stage.
    """
    pool = available_providers()
    if not pool:
        return 0
    changed = 0
    for i, judge in enumerate(enabled_judges(db)):
        pid = pool[i % len(pool)]
        p = providers.PROVIDERS.get(pid)
        model = (p.default_model if p else "") or ""
        model = _spread_model(pid, model)
        if judge.provider == pid and judge.model == model:
            continue
        judge.provider = pid
        judge.model = model
        db.add(judge)
        changed += 1
    if changed:
        db.commit()
    return changed


def ensure_seeded(db: Session) -> int:
    """First boot: seat the default panel. Returns the number of judges created
    (0 when a roster already exists — this never edits one)."""
    if db.exec(select(Judge)).first() is not None:
        return 0
    for row in personas.defaults():
        db.add(Judge(**row))
    db.commit()
    n = spread_providers(db)
    logger.info("Seeded the default review jury (%d judges, %d given a distinct provider)",
                len(personas.defaults()), n)
    return len(personas.defaults())


def reset_to_defaults(db: Session) -> list[Judge]:
    """Drop the roster and re-seat the shipped panel. Explicit operator action."""
    for judge in all_judges(db):
        db.delete(judge)
    db.commit()
    for row in personas.defaults():
        db.add(Judge(**row))
    db.commit()
    spread_providers(db)
    return all_judges(db)


def create(db: Session, *, name: str, persona: str = personas.CUSTOM, enabled: bool = True,
           provider: str = "", model: str = "", focus: str = "") -> Judge:
    if persona != personas.CUSTOM and personas.get(persona) is None:
        raise ValueError(f"Unknown persona '{persona}'")
    if not (name or "").strip():
        raise ValueError("A judge needs a name")
    if persona == personas.CUSTOM and not (focus or "").strip():
        raise ValueError("A custom judge needs a focus — what should it look for?")
    last = max((j.position for j in all_judges(db)), default=-1)
    judge = Judge(name=name.strip(), persona=persona, enabled=enabled, position=last + 1,
                  provider=provider.strip(), model=model.strip(), focus=(focus or "").strip() or None)
    db.add(judge)
    db.commit()
    db.refresh(judge)
    return judge


_EDITABLE = ("name", "persona", "enabled", "position", "provider", "model", "focus")


def update(db: Session, judge_id: int, values: dict) -> Judge:
    judge = db.get(Judge, judge_id)
    if judge is None:
        raise ValueError(f"No judge {judge_id}")
    persona = values.get("persona", judge.persona)
    if persona != personas.CUSTOM and personas.get(persona) is None:
        raise ValueError(f"Unknown persona '{persona}'")
    for key in _EDITABLE:
        if key in values:
            setattr(judge, key, values[key])
    if not (judge.name or "").strip():
        raise ValueError("A judge needs a name")
    db.add(judge)
    db.commit()
    db.refresh(judge)
    return judge


def move(db: Session, judge_id: int, delta: int) -> None:
    """Swap a judge with its neighbour. Panel order is cosmetic — it decides the
    order opinions reach the foreperson and appear in the verdict — but operators
    reorder to put the judges they care about first, so it should behave."""
    rows = all_judges(db)
    idx = next((i for i, j in enumerate(rows) if j.id == judge_id), None)
    if idx is None:
        raise ValueError(f"No judge {judge_id}")
    target = idx + (1 if delta > 0 else -1)
    if not 0 <= target < len(rows):
        return
    rows[idx], rows[target] = rows[target], rows[idx]
    for position, judge in enumerate(rows):   # renumber densely; positions may be sparse
        judge.position = position
        db.add(judge)
    db.commit()


def delete(db: Session, judge_id: int) -> None:
    judge = db.get(Judge, judge_id)
    if judge is None:
        raise ValueError(f"No judge {judge_id}")
    db.delete(judge)
    db.commit()


def view(db: Session) -> dict:
    """Roster + persona catalog + provider registry for the Settings screen."""
    rows = []
    for j in all_judges(db):
        provider, model = resolve(j)
        p = personas.get(j.persona)
        rows.append({
            "id": j.id, "name": j.name, "persona": j.persona, "enabled": j.enabled,
            "position": j.position, "provider": j.provider, "model": j.model,
            "focus": j.focus or "",
            "persona_name": p.name if p else "Custom",
            "persona_summary": p.summary if p else "Operator-defined focus.",
            "effective_provider": provider, "effective_model": model,
            "inherits": not (j.provider or "").strip(),
            "runnable": providers.can_chat(provider),
        })
    return {
        "judges": rows,
        "personas": [
            {"id": p.id, "name": p.name, "summary": p.summary,
             "default_enabled": p.default_enabled}
            for p in personas.PERSONAS.values()
        ] + [{"id": personas.CUSTOM, "name": "Custom", "default_enabled": False,
              "summary": "Write your own brief — what this juror should look for."}],
        "provider_options": [""] + providers.stage_provider_ids("review"),
        "enabled_count": sum(1 for r in rows if r["enabled"]),
    }
