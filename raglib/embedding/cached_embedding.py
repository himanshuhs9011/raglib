"""Caching + batching wrapper for any embedding model."""

from __future__ import annotations

import hashlib
import logging
from typing import Callable, Dict, List, Optional

from raglib.cache import TTLCache
from raglib.core.concurrency import parallel_map
from raglib.core.tracing import record_cache_lookup
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.resilience import RateLimiter, retry_call

logger = logging.getLogger(__name__)


class CachedEmbedding(BaseEmbedding):
    """Content-hash LRU cache and fixed-size batching in front of an embedding model.

    Re-ingesting unchanged text, or embedding the same query twice, costs nothing. Calls to the
    inner model are rate limited (optional ``rate_limiter``) and transient errors are retried.
    ``ttl`` (seconds, default ``None`` = no expiry) bounds how long a vector is reused, e.g.
    ``ttl=24 * 3600`` for query embeddings of a frequently repeated workload. Trade-off: a vector
    only goes stale if the *model* behind the same name changes, so a TTL mostly trades API
    calls for memory; ``invalidate_texts`` drops the vectors of changed documents.
    """

    def __init__(
        self,
        inner: BaseEmbedding,
        batch_size: int = 64,
        max_entries: int = 50_000,
        max_concurrency: int = 4,
        max_retries: int = 2,
        rate_limiter: Optional[RateLimiter] = None,
        ttl: Optional[float] = None,
    ):
        """Wrap ``inner`` with batching, parallel batch requests, retries and an in-memory cache."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.inner = inner
        self.batch_size = batch_size
        self.max_concurrency = max(1, max_concurrency)
        self.max_entries = max_entries
        self._cache = TTLCache(max_entries=max_entries, ttl=ttl)
        self.max_retries = max(0, max_retries)
        self.rate_limiter = rate_limiter
        # Counts tokens for cost estimates; RAG sets it to the chunker's token_counter (default: tiktoken / estimate).
        self.token_counter: Optional[Callable[[str], int]] = None

    @property
    def ttl(self) -> Optional[float]:
        """Vector lifetime in seconds (``None`` = no expiry)."""

        return self._cache.ttl

    @ttl.setter
    def ttl(self, value: Optional[float]) -> None:
        if value is not None and value <= 0:
            raise ValueError("ttl must be greater than zero (or None)")
        self._cache.ttl = value

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
        """Return a cached vector (refreshing its LRU position), or None."""

        return self._cache.get(key)

    def _put(self, key: str, vector: List[float]) -> None:
        """Store a vector, evicting the least-recently-used entry if needed."""

        self._cache.set(key, vector)

    def _record(self, texts: List[str]) -> None:
        """Report the embedding tokens of a real (uncached) call for cost tracking (estimated)."""

        from raglib.core.runtime import record_embedding_usage
        from raglib.core.text import count_tokens

        counter = self.token_counter or count_tokens
        record_embedding_usage(sum(counter(text) for text in texts), model=self.model_label)

    @property
    def model_label(self) -> str:
        """Name used to price this model (``model_name`` of the wrapped model, else its class)."""

        name = getattr(self.inner, "model_name", None)
        return str(name) if name else type(self.inner).__name__

    def invalidate_texts(self, texts: List[str]) -> int:
        """Drop the cached vectors of ``texts`` (chunks of a changed or deleted document)."""

        dropped = 0
        for text in texts:
            for key in (self._key(text), "q:" + self._key(text)):
                if key in self._cache:
                    self._cache.discard(key)
                    dropped += 1
        return dropped

    def clear(self) -> None:
        """Drop every cached vector (called when documents are deleted)."""

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
            batch = [texts[index] for index in batch_indexes]
            vectors = self._call(self.inner.embed_documents, batch)
            self._record(batch)
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
        record_cache_lookup("embedding", cached is not None)
        if cached is not None:
            return cached
        vector = list(self._call(self.inner.embed_query, text))
        self._record([text])
        self._put(key, vector)
        return vector

    @property
    def dimension(self) -> int:
        """Return the wrapped model's dimension."""

        return self.inner.dimension

    @property
    def model_name(self):
        """Return the wrapped model's name."""

        return self.inner.model_name

    def fingerprint(self):
        """Fingerprint of the wrapped model (the cache does not change vectors)."""

        return self.inner.fingerprint()
