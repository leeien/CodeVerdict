"""Data shapes for the knowledge subsystem.

Two layers, deliberately separate:

  * **Facts** — raw structure the static analyzer extracts from source
    (`FileFacts`, `ExtractedFacts`). The LLM never sees code, only these.
  * **Knowledge** — the interpreted, stored output (`KnowledgeDocument`): one
    document per view, with a uniform envelope and an open `content` dict that
    differs per document type.

Plain dataclasses (not pydantic) keep this dependency-light and trivially
JSON-serializable via `dataclasses.asdict`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# --- Raw facts (analyzer output) --------------------------------------------
@dataclass
class Endpoint:
    method: str  # GET, POST, ...
    path: str    # e.g. /orders
    file: str    # relative path where it was found


@dataclass
class FileFacts:
    path: str    # relative to repo root
    language: str
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    endpoints: list[Endpoint] = field(default_factory=list)


@dataclass
class ExtractedFacts:
    root: str    # absolute repo path
    files: list[FileFacts] = field(default_factory=list)


# --- Knowledge (generator output, stored as JSON) ---------------------------
@dataclass
class KnowledgeDocument:
    id: str
    type: str    # repository | architecture | module | feature | workflow | ...
    name: str
    summary: str
    tags: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    content: dict = field(default_factory=dict)
    confidence: str = "MEDIUM"  # HIGH | MEDIUM | LOW

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> KnowledgeDocument:
        return cls(
            id=raw["id"],
            type=raw["type"],
            name=raw["name"],
            summary=raw.get("summary", ""),
            tags=list(raw.get("tags", [])),
            related=list(raw.get("related", [])),
            content=dict(raw.get("content", {})),
            confidence=raw.get("confidence", "MEDIUM"),
        )
