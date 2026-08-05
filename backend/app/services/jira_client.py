import logging
from pathlib import Path

import httpx
from fastapi import HTTPException
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Project root (CodeVerdict/.env) — read regardless of launch directory.
_ENV_FILE = str(Path(__file__).resolve().parents[3] / ".env")


class Settings(BaseSettings):
    """Jira Cloud connection settings, loaded from env vars / .env."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    jira_base_url: str
    jira_email: str
    jira_api_token: str
    jira_project_key: str

    # Custom field IDs vary per Jira site (check GET /rest/api/3/field). Override
    # via env if your instance uses different IDs; these are common defaults.
    # Comma-separated list; the first field that has a value is used.
    jira_story_points_fields: str = "customfield_10016"
    jira_sprint_field: str = "customfield_10020"

    @property
    def story_points_fields(self) -> tuple[str, ...]:
        return tuple(f.strip() for f in self.jira_story_points_fields.split(",") if f.strip())


def _from_adf(doc: dict | None) -> str | None:
    """Extract plain text from an ADF doc, recursing through nested blocks
    (headings, bullet/ordered lists, list items) to reach text nodes.

    Note: plain text only; drops formatting like bold/links/tables.
    """
    if not doc:
        return None
    parts: list[str] = []

    def walk(node: dict) -> None:
        node_type = node.get("type")
        if node_type == "text":
            parts.append(node.get("text", ""))
            return
        if node_type == "listItem":
            parts.append("- ")
        for child in node.get("content") or []:
            walk(child)
        if node_type in ("paragraph", "heading", "listItem"):
            parts.append("\n")

    for node in doc.get("content") or []:
        walk(node)

    return "".join(parts).strip() or None


def _to_adf(text: str) -> dict:
    """Wrap plain text in a minimal Atlassian Document Format doc.

    Note: single-paragraph plain text only; upgrade to real ADF nodes
    (lists, links, formatting) if callers ever need rich descriptions.
    """
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": text}]}
        ],
    }


class JiraClient:
    """Thin async wrapper over the Jira Cloud REST API v3."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.jira_base_url,
            auth=httpx.BasicAuth(settings.jira_email, settings.jira_api_token),
            timeout=10.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        logger.info("jira request %s %s", method, url)
        response = await self._client.request(method, url, **kwargs)
        logger.info("jira response %s %s -> %d", method, url, response.status_code)
        if response.status_code in (401, 403):
            logger.error("jira auth failed %s %s -> %d", method, url, response.status_code)
            raise HTTPException(status_code=502, detail="Jira upstream auth failed")
        if response.is_error:
            logger.error("jira error %s %s -> %d: %s", method, url, response.status_code, response.text)
            raise HTTPException(status_code=502, detail=f"Jira error: {response.text}")
        return response

    async def create_story(
        self, summary: str, description: str | None, project_key: str | None
    ) -> dict:
        payload = {
            "fields": {
                "project": {"key": project_key or self._settings.jira_project_key},
                "summary": summary,
                "issuetype": {"name": "Story"},
            }
        }
        if description:
            payload["fields"]["description"] = _to_adf(description)

        logger.debug("create_story payload=%s", payload)
        response = await self._request("POST", "/rest/api/3/issue", json=payload)
        data = response.json()
        return {
            "id": data["id"],
            "key": data["key"],
            "url": f"{self._settings.jira_base_url}/browse/{data['key']}",
        }

    async def search_stories(
        self,
        project_key: str | None,
        jql: str | None,
        max_results: int,
        sprint_id: int | None = None,
        status: str | None = None,
    ) -> list[dict]:
        if jql:
            query = jql
        else:
            conditions = [
                f'project = "{project_key or self._settings.jira_project_key}"',
                "issuetype = Story",
            ]
            if sprint_id is not None:
                conditions.append(f"sprint = {sprint_id}")
            if status:
                conditions.append(f'status = "{status}"')
            query = " AND ".join(conditions) + " ORDER BY created DESC"
        logger.info("search_stories jql=%s max_results=%d", query, max_results)
        response = await self._request(
            "POST",
            "/rest/api/3/search/jql",
            json={
                "jql": query,
                "maxResults": max_results,
                "fields": [
                    "summary",
                    "status",
                    "description",
                    "assignee",
                    "priority",
                    self._settings.jira_sprint_field,
                    *self._settings.story_points_fields,
                ],
            },
        )
        data = response.json()
        logger.info("search_stories jira returned %d issues", len(data.get("issues", [])))
        return [self._to_story_summary(issue) for issue in data.get("issues", [])]

    def _to_story_summary(self, issue: dict) -> dict:
        fields = issue["fields"]
        assignee = fields.get("assignee")
        priority = fields.get("priority")
        sprints = fields.get(self._settings.jira_sprint_field) or []
        story_points = next(
            (fields[f] for f in self._settings.story_points_fields if fields.get(f) is not None), None
        )
        return {
            "id": issue["id"],
            "key": issue["key"],
            "summary": fields["summary"],
            "status": fields["status"]["name"],
            "description": _from_adf(fields.get("description")),
            "assignee": assignee["displayName"] if assignee else None,
            "priority": priority["name"] if priority else None,
            "story_points": story_points,
            "sprint": sprints[-1]["name"] if sprints else None,
        }
