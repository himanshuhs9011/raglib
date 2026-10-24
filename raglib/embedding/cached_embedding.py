"""Caching + batching wrapper for any embedding model."""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Dict, List, Optional

from raglib.core.concurrency import parallel_map
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.resilience import RateLimiter, retry_call

logger = logging.getLogger(__name__)


class CachedEmbedding(BaseEmbedding):
    """Content-hash LRU cache and fixed-size batching in front of an embedding model.

    Re-ingesting unchanged text, or embedding the same query twice, costs nothing. Calls to the
    inner model are rate limited (optional ``rate_limiter``) and transient errors are retried.
    """

    def __init__(
        self,
        inner: BaseEmbedding,
        batch_size: int = 64,
        max_entries: int = 50_000,
        max_concurrency: int = 4,
        max_retries: int = 2,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """Wrap ``inner`` with batching, parallel batch requests, retries and an in-memory cache."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.inner = inner
        self.batch_size = batch_size
        self.max_concurrency = max(1, max_concurrency)
        self.max_entries = max_entries
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_retries = max(0, max_retries)
        self.rate_limiter = rate_limiter

    def _call(self, fn, *args):
        """Call the inner model with rate limiting and retries."""

        return retry_call(
            fn, *args, max_retries=self.max_retries, limiter=self.rate_limiter, label=f"embedding:{type(self.inner).__name__}"
        )

    @staticmethod
    def _key(text: str) -> str:
        """Hash text into a cache key."""

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get(self, key: str):
        """Return a cached vector and refresh its LRU position."""

        with self._lock:
            vector = self._cache.get(key)
            if vector is not None:
                self._cache.move_to_end(key)
            return vector

    def _put(self, key: str, vector: List[float]) -> None:
        """Store a vector, evicting the least-recently-used entry if needed."""

        with self._lock:
            self._cache[key] = vector
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Drop every cached vector (called when documents are deleted)."""

        with self._lock:
            self._cache.clear()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed texts, calling the inner model only for uncached ones, in batches."""

        keys = [self._key(text) for text in texts]
        results: Dict[int, List[float]] = {}
        missing: List[int] = []
        for index, key in enumerate(keys):
            cached = self._get(key)
            if cached is not None:
                results[index] = cached
            else:
                missing.append(index)

        batches = [missing[start : start + self.batch_size] for start in range(0, len(missing), self.batch_size)]

        def embed_batch(batch_indexes: List[int]) -> List[List[float]]:
            vectors = self._call(self.inner.embed_documents, [texts[index] for index in batch_indexes])
            if len(vectors) != len(batch_indexes):
                raise RuntimeError(f"Embedding model returned {len(vectors)} vectors for {len(batch_indexes)} texts")
            return vectors

        # Embedding APIs are I/O bound: send several batches at once.
        outputs = parallel_map(embed_batch, batches, self.max_concurrency)
        for batch_indexes, vectors in zip(batches, outputs):
            for index, vector in zip(batch_indexes, vectors):
                results[index] = list(vector)
                self._put(keys[index], results[index])

        return [results[index] for index in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query with caching."""

        key = "q:" + self._key(text)
        cached = self._get(key)
        if cached is not None:
            return cached
        vector = list(self._call(self.inner.embed_query, text))
        self._put(key, vector)
        return vector

    @property
    def dimension(self) -> int:
        """Return the wrapped model's dimension."""

        return self.inner.dimension
