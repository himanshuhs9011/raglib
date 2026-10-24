"""Data model for GraphRAG: entities, relations, communities."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

MAX_DESCRIPTION_CHARS = 800


def normalize_entity_key(name: str) -> str:
    """Canonical entity id: NFKC, casefold, collapse whitespace, strip surrounding punctuation."""

    text = unicodedata.normalize("NFKC", name or "").casefold()
    text = re.sub(r"\s+", " ", text).strip(" \t\n\"'`.,;:()[]{}")
    return text


def merge_descriptions(existing: str, new: str, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """Append a new distinct description, bounded in length."""

    new = (new or "").strip()
    if not new:
        return existing
    if not existing:
        return new[:limit]
    if new.lower() in existing.lower():
        return existing
    merged = f"{existing} | {new}"
    return merged if len(merged) <= limit else merged[:limit]


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str
    name: str
    type: str = "CONCEPT"
    description: str = ""
    source_chunk_ids: List[str] = field(default_factory=list)
    community_ids: Dict[str, str] = field(default_factory=dict)  # level (str for JSON) -> community id

    def embedding_text(self) -> str:
        """Text used to embed the entity for query-time entity search."""

        return f"{self.name} ({self.type}): {self.description}".strip()

    def text_hash(self) -> str:
        """Hash of the embedded text (re-embed only when it changes)."""

        return hashlib.sha256(self.embedding_text().encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Deserialize from a dict."""

        return cls(**data)


@dataclass
class Relation:
    """A typed, weighted edge between two entities."""

    source: str
    target: str
    type: str = "RELATED_TO"
    description: str = ""
    weight: float = 1.0
    source_chunk_ids: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Unique key of the relation (direction-insensitive for the same type)."""

        left, right = sorted((self.source, self.target))
        return f"{left}|{self.type}|{right}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """Deserialize from a dict."""

        return cls(**data)


@dataclass
class Community:
    """A cluster of densely connected entities with an LLM-written report."""

    id: str
    level: int
    entity_ids: List[str]
    parent_id: Optional[str] = None
    signature: str = ""
    report_signature: str = ""
    title: str = ""
    summary: str = ""
    findings: List[str] = field(default_factory=list)
    rating: float = 0.0
    embedding: Optional[List[float]] = None

    @property
    def has_report(self) -> bool:
        """True when a report exists for the current membership."""

        return bool(self.summary) and self.report_signature == self.signature

    def report_text(self) -> str:
        """Full report as plain text."""

        findings = "\n".join(f"- {finding}" for finding in self.findings)
        return f"{self.title}\n{self.summary}\n{findings}".strip()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Community":
        """Deserialize from a dict."""

        return cls(**data)


@dataclass
class ExtractionResult:
    """Raw entities/relations extracted from one chunk."""

    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""

        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionResult":
        """Deserialize from a dict."""

        return cls(
            entities=[Entity.from_dict(item) for item in data.get("entities", [])],
            relations=[Relation.from_dict(item) for item in data.get("relations", [])],
        )
