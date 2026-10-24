"""Base contracts for vector store backends used by raglib retrievers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

MetadataFilter = Dict[str, Any]


@dataclass
class VectorRecord:
    """A vectorized document payload stored in a vector backend."""

    document_id: str
    embedding: List[float]
    document_text: str
    metadata: Dict[str, Any]


@dataclass
class VectorHit:
    """Represents a scored vector search match."""

    document_id: str
    score: float
    document_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def matches_filter(metadata: Dict[str, Any], filters: Optional[MetadataFilter]) -> bool:
    """Evaluate a Mongo/Chroma-style metadata filter against a metadata dict.

    Supported: ``{"key": value}``, ``{"key": {"$eq"|"$ne"|"$in"|"$nin"|"$gt"|"$gte"|"$lt"|"$lte": v}}``
    and top-level ``{"$and": [...]}`` / ``{"$or": [...]}``.
    """

    if not filters:
        return True
    for key, condition in filters.items():
        if key == "$and":
            if not all(matches_filter(metadata, item) for item in condition):
                return False
            continue
        if key == "$or":
            if not any(matches_filter(metadata, item) for item in condition):
                return False
            continue
        value = metadata.get(key)
        if isinstance(condition, dict):
            for operator, expected in condition.items():
                if not _apply_operator(value, operator, expected):
                    return False
        elif value != condition:
            return False
    return True


def _apply_operator(value: Any, operator: str, expected: Any) -> bool:
    """Apply one comparison operator."""

    try:
        if operator == "$eq":
            return value == expected
        if operator == "$ne":
            return value != expected
        if operator == "$in":
            return value in expected
        if operator == "$nin":
            return value not in expected
        if value is None:
            return False
        if operator == "$gt":
            return value > expected
        if operator == "$gte":
            return value >= expected
        if operator == "$lt":
            return value < expected
        if operator == "$lte":
            return value <= expected
    except TypeError:
        return False
    raise ValueError(f"Unsupported metadata filter operator '{operator}'")


class BaseVectorStore(ABC):
    """Abstract vector store contract for add/query/delete lifecycle operations."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored vectors from the backend."""

    @abstractmethod
    def upsert(self, records: List[VectorRecord]) -> None:
        """Insert or update vector records in the backend."""

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[MetadataFilter] = None,
    ) -> List[VectorHit]:
        """Return top_k nearest vector matches for a query embedding."""

    def delete(self, ids: Iterable[str]) -> None:
        """Delete vectors by id (backends should override)."""

        raise NotImplementedError(f"{type(self).__name__} does not support delete()")

    def count(self) -> int:
        """Return the number of stored vectors (backends should override)."""

        raise NotImplementedError(f"{type(self).__name__} does not support count()")

    def existing_ids(self, ids: Iterable[str]) -> Set[str]:
        """Return the subset of ``ids`` already stored (default: none known)."""

        return set()

    def get_embeddings(self, ids: Iterable[str]) -> Dict[str, List[float]]:
        """Return stored embeddings for ids when the backend can (default: empty)."""

        return {}
