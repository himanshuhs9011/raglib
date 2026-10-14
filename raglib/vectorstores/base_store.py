"""Base contracts for vector store backends used by raglib retrievers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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


class BaseVectorStore(ABC):
    """Abstract vector store contract for add/query lifecycle operations."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored vectors from the backend."""

    @abstractmethod
    def upsert(self, records: List[VectorRecord]) -> None:
        """Insert or update vector records in the backend."""

    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> List[VectorHit]:
        """Return top_k nearest vector matches for a query embedding."""
