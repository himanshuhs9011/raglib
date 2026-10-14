"""In-memory vector store backend with cosine similarity search."""

from __future__ import annotations

import logging
import math
from typing import Dict, List

from raglib.vectorstores.base_store import BaseVectorStore, VectorHit, VectorRecord

logger = logging.getLogger(__name__)


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector storage for local development and fallback mode."""

    def __init__(self):
        """Initialize an empty in-memory vector index."""

        self._records: Dict[str, VectorRecord] = {}

    def clear(self) -> None:
        """Remove all stored vectors from memory."""

        self._records.clear()

    def upsert(self, records: List[VectorRecord]) -> None:
        """Insert or replace vector records keyed by document id."""

        for record in records:
            self._records[record.document_id] = record
        logger.info("InMemoryVectorStore upserted %d vectors (total=%d)", len(records), len(self._records))

    def query(self, query_embedding: List[float], top_k: int) -> List[VectorHit]:
        """Return nearest vectors by cosine similarity."""

        if not query_embedding or top_k <= 0 or not self._records:
            return []

        hits: List[VectorHit] = []
        for record in self._records.values():
            score = self._cosine_similarity(query_embedding, record.embedding)
            hits.append(
                VectorHit(
                    document_id=record.document_id,
                    score=score,
                    document_text=record.document_text,
                    metadata=dict(record.metadata),
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def _cosine_similarity(self, left: List[float], right: List[float]) -> float:
        """Compute cosine similarity between two dense vectors."""

        if not left or not right:
            return 0.0

        length = min(len(left), len(right))
        if length == 0:
            return 0.0

        dot = sum(left[index] * right[index] for index in range(length))
        left_norm = math.sqrt(sum(value * value for value in left[:length]))
        right_norm = math.sqrt(sum(value * value for value in right[:length]))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)
