"""Push approved tickets to a Jira board.

Wraps the async JiraClient for use from the (sync) approval route, and degrades
gracefully: if Jira isn't configured, pushing is a no-op so local approval
still works. Connection details come from the runtime settings (env/.env or
the Settings screen) and are re-read on every call, so configuring Jira in the
UI activates it immediately.
"""

from __future__ import annotations

import asyncio
import logging

from ..config import settings
from .jira_client import JiraClient, Settings

logger = logging.getLogger(__name__)


def _current_settings() -> Settings | None:
    """Build the Jira client settings from runtime config; None when any of
    the required fields is missing (integration disabled)."""
    required = (settings.jira_base_url, settings.jira_email,
                settings.jira_api_token, settings.jira_project_key)
    if not all(v.strip() for v in required):
        return None
    return Settings(
        jira_base_url=settings.jira_base_url.strip(),
        jira_email=settings.jira_email.strip(),
        jira_api_token=settings.jira_api_token.strip(),
        jira_project_key=settings.jira_project_key.strip(),
    )


def is_configured() -> bool:
    return _current_settings() is not None


def _story_body(title: str, description: str | None, criteria: list | None) -> str:
    parts = [description.strip()] if description and description.strip() else []
    if criteria:
        parts.append("Acceptance criteria:\n" + "\n".join(f"- {c}" for c in criteria))
    return "\n\n".join(parts)


def push_story(key: str, title: str, description: str | None = None,
               criteria: list | None = None) -> dict | None:
    """Create a Jira Story for an approved ticket. Returns {id, key, url} or
    None when Jira is disabled or the call fails (never raises)."""
    jira_settings = _current_settings()
    if jira_settings is None:
        return None
    summary = f"[{key}] {title}" if key else title
    body = _story_body(title, description, criteria)

    async def _run() -> dict:
        client = JiraClient(jira_settings)
        try:
            return await client.create_story(summary, body or None, None)
        finally:
            await client.aclose()

    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001  (HTTPException, network, etc.)
        logger.error("Jira push failed for %s: %s", key, exc)
        return None
