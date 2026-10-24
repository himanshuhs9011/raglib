"""In-memory vector store backend with cosine similarity search."""

from __future__ import annotations

import heapq
import logging
import math
from array import array
from typing import Dict, Iterable, List, Optional, Set, Tuple

from raglib.core.concurrency import ReadWriteLock
from raglib.vectorstores.base_store import (
    BaseVectorStore,
    MetadataFilter,
    VectorHit,
    VectorRecord,
    matches_filter,
)

logger = logging.getLogger(__name__)


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector storage with compact float32 vectors.

    * Vectors are stored as float32 arrays (4 bytes/dim instead of ~32 for Python floats):
      30k x 1536-d chunks use ~180 MB instead of ~1.5 GB.
    * With numpy installed, search is one matrix-vector product + ``argpartition`` top-k.
    * Thread-safe: concurrent queries, exclusive writes.
    For millions of vectors use a dedicated vector database (Chroma server, Qdrant, pgvector).
    """

    def __init__(self):
        """Initialize an empty in-memory vector index."""

        self._vectors: Dict[str, array] = {}
        self._payload: Dict[str, Tuple[str, Dict]] = {}  # id -> (text, metadata)
        self._norms: Dict[str, float] = {}
        self._matrix = None  # (normalized float32 matrix, ids) snapshot, rebuilt lazily after writes
        self._lock = ReadWriteLock()

    # ---------------------------------------------------------------- writes
    def clear(self) -> None:
        """Remove all stored vectors from memory."""

        with self._lock.write():
            self._vectors.clear()
            self._payload.clear()
            self._norms.clear()
            self._matrix = None

    def upsert(self, records: List[VectorRecord]) -> None:
        """Insert or replace vector records keyed by document id."""

        prepared = []
        for record in records:
            vector = array("f", record.embedding)
            norm = math.sqrt(sum(value * value for value in vector))
            prepared.append((record, vector, norm))
        with self._lock.write():
            for record, vector, norm in prepared:
                self._vectors[record.document_id] = vector
                self._payload[record.document_id] = (record.document_text, dict(record.metadata))
                self._norms[record.document_id] = norm
            self._matrix = None
        logger.debug("InMemoryVectorStore upserted %d vectors", len(records))

    def delete(self, ids: Iterable[str]) -> None:
        """Delete vectors by id."""

        with self._lock.write():
            for record_id in ids:
                self._vectors.pop(record_id, None)
                self._payload.pop(record_id, None)
                self._norms.pop(record_id, None)
            self._matrix = None

    # ---------------------------------------------------------------- reads
    def count(self) -> int:
        """Return the number of stored vectors."""

        with self._lock.read():
            return len(self._vectors)

    def existing_ids(self, ids: Iterable[str]) -> Set[str]:
        """Return ids already present in the store."""

        with self._lock.read():
            return {record_id for record_id in ids if record_id in self._vectors}

    def get_embeddings(self, ids: Iterable[str]) -> Dict[str, List[float]]:
        """Return stored embeddings for the given ids."""

        with self._lock.read():
            return {rid: list(self._vectors[rid]) for rid in ids if rid in self._vectors}

    def records(self) -> List[VectorRecord]:
        """Return all stored records (used for persistence)."""

        with self._lock.read():
            return [
                VectorRecord(document_id=rid, embedding=list(vector), document_text=self._payload[rid][0], metadata=dict(self._payload[rid][1]))
                for rid, vector in self._vectors.items()
            ]

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[MetadataFilter] = None,
    ) -> List[VectorHit]:
        """Return nearest vectors by cosine similarity, honoring metadata filters."""

        if not query_embedding or top_k <= 0:
            return []
        with self._lock.read():
            if not self._vectors:
                return []
            allowed: Optional[Set[str]] = None
            if filters:
                allowed = {rid for rid, (_, meta) in self._payload.items() if matches_filter(meta, filters)}
                if not allowed:
                    return []
            scored = self._top_numpy(query_embedding, top_k, allowed)
            if scored is None:
                scored = self._top_python(query_embedding, top_k, allowed)
            return [
                VectorHit(
                    document_id=rid,
                    score=float(score),
                    document_text=self._payload[rid][0],
                    metadata=dict(self._payload[rid][1]),
                )
                for score, rid in scored
            ]

    # ---------------------------------------------------------------- search kernels
    def _snapshot(self, np):
        """Return (normalized matrix, ids), building it once per batch of writes."""

        snapshot = self._matrix
        if snapshot is None:
            ids = list(self._vectors.keys())
            matrix = np.vstack([np.frombuffer(self._vectors[rid], dtype=np.float32) for rid in ids])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            snapshot = (matrix / norms, ids)
            self._matrix = snapshot
        return snapshot

    def _top_numpy(self, query: List[float], top_k: int, allowed: Optional[Set[str]]):
        """Matrix search with numpy; None when numpy is unavailable or dimensions are ragged."""

        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            matrix, ids = self._snapshot(np)
            query_vec = np.asarray(query, dtype=np.float32)
            sims = matrix @ (query_vec / (float(np.linalg.norm(query_vec)) or 1.0))
        except ValueError:
            return None  # mixed embedding dimensions: use the tolerant pure-Python path
        if allowed is not None:
            mask = np.fromiter((rid in allowed for rid in ids), dtype=bool, count=len(ids))
            sims = np.where(mask, sims, -np.inf)
        k = min(top_k, len(ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(float(sims[i]), ids[i]) for i in top if sims[i] != -np.inf]

    def _top_python(self, query: List[float], top_k: int, allowed: Optional[Set[str]]):
        """Pure-Python cosine search (no numpy)."""

        query_norm = math.sqrt(sum(value * value for value in query))
        if query_norm == 0.0:
            return []
        scored = []
        for rid, vector in self._vectors.items():
            if allowed is not None and rid not in allowed:
                continue
            length = min(len(query), len(vector))
            norm = self._norms.get(rid) or 0.0
            score = sum(query[i] * vector[i] for i in range(length)) / (query_norm * norm) if norm else 0.0
            scored.append((score, rid))
        return heapq.nlargest(top_k, scored)
