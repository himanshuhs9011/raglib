"""Caching primitives: TTL + LRU caches with document dependencies, cache scopes, and the answer cache.

* ``TTLCache`` - thread-safe LRU with an optional time-to-live per entry and *dependency tags*
  (the document ids an entry was built from), so ``invalidate(doc_id)`` drops exactly the
  entries that used a changed document. Used by ``CachedLLMClient`` and ``CachedEmbedding``.
* ``cache_dependencies(ids)`` - while active, entries stored by the caches are tagged with ``ids``
  (the generator sets it to the documents in the prompt).
* ``cache_bypass()`` - while active, LLM and answer caches neither read nor write. The facade
  turns it on for personalized queries (a ``user`` context or caller ``filters``) unless
  ``cache_personalized=True``. Trade-off: repeated personalized questions pay full LLM cost.
* ``AnswerCache`` - a question-level cache (the "FAQ tier"): normalized question -> finished
  answer, valid while every source document keeps the version it had when the answer was made.
  Trade-off: it returns the same answer to everyone in a scope, so it is used for the public tier
  only, and freshness depends on invalidation (TTL as a backstop).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Hashable, Iterable, Iterator, Optional, Set, Tuple

_MISS = object()

_DEPS: ContextVar[Tuple[str, ...]] = ContextVar("raglib_cache_deps", default=())
_BYPASS: ContextVar[bool] = ContextVar("raglib_cache_bypass", default=False)


def _reset(var: ContextVar, token: Any) -> None:
    try:
        var.reset(token)
    except ValueError:
        pass


@contextmanager
def cache_dependencies(doc_ids: Iterable[str]) -> Iterator[None]:
    """Tag every cache entry stored inside the block with ``doc_ids`` (added to outer ones)."""

    token = _DEPS.set(tuple(dict.fromkeys((*_DEPS.get(), *(str(i) for i in doc_ids if i)))))
    try:
        yield
    finally:
        _reset(_DEPS, token)


def current_dependencies() -> Tuple[str, ...]:
    """Document ids the entries stored right now depend on."""

    return _DEPS.get()


@contextmanager
def cache_bypass(active: bool = True) -> Iterator[None]:
    """Inside the block (when ``active``), LLM and answer caches neither read nor write."""

    token = _BYPASS.set(bool(active) or _BYPASS.get())
    try:
        yield
    finally:
        _reset(_BYPASS, token)


def caching_allowed() -> bool:
    """False inside ``cache_bypass()``."""

    return not _BYPASS.get()


class TTLCache:
    """Thread-safe LRU cache with optional TTL and per-entry dependency tags.

    ``ttl`` is in seconds (``None`` = entries never expire, only LRU eviction). ``clock`` is a
    monotonic clock, replaceable in tests.
    """

    def __init__(self, max_entries: int = 2048, ttl: Optional[float] = None, clock: Callable[[], float] = time.monotonic):
        """Create an empty cache."""

        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be greater than zero (or None)")
        self.max_entries = max_entries
        self.ttl = ttl
        self._clock = clock
        self._entries: "OrderedDict[Hashable, Tuple[Any, Optional[float], Tuple[str, ...]]]" = OrderedDict()
        self._by_dep: Dict[str, Set[Hashable]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.expired = 0

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Return the cached value (refreshing its LRU position), or ``default``."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return default
            value, expires_at, _ = entry
            if expires_at is not None and self._clock() >= expires_at:
                self._drop(key)
                self.expired += 1
                self.misses += 1
                return default
            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def lookup(self, key: Hashable) -> Tuple[bool, Any]:
        """``(hit, value)``; distinguishes a cached ``None`` from a miss."""

        value = self.get(key, _MISS)
        return (False, None) if value is _MISS else (True, value)

    def set(self, key: Hashable, value: Any, deps: Iterable[str] = (), ttl: Optional[float] = None) -> None:
        """Store ``value``; ``deps`` (plus ``current_dependencies()``) tag it for ``invalidate``."""

        tags = tuple(dict.fromkeys((*current_dependencies(), *(str(d) for d in deps if d))))
        lifetime = ttl if ttl is not None else self.ttl
        expires_at = self._clock() + lifetime if lifetime is not None else None
        with self._lock:
            if key in self._entries:
                self._drop(key)
            self._entries[key] = (value, expires_at, tags)
            for tag in tags:
                self._by_dep.setdefault(tag, set()).add(key)
            while len(self._entries) > self.max_entries:
                self._drop(next(iter(self._entries)))

    def invalidate(self, doc_ids: Iterable[str]) -> int:
        """Drop every entry that depends on one of ``doc_ids``; returns how many were dropped."""

        dropped = 0
        with self._lock:
            for doc_id in doc_ids:
                for key in list(self._by_dep.pop(str(doc_id), ())):
                    if key in self._entries:
                        self._drop(key)
                        dropped += 1
        return dropped

    def discard(self, key: Hashable) -> None:
        """Drop one entry if present."""

        with self._lock:
            if key in self._entries:
                self._drop(key)

    def clear(self) -> None:
        """Drop every entry."""

        with self._lock:
            self._entries.clear()
            self._by_dep.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        hit, _ = self.lookup(key)
        return hit

    def _drop(self, key: Hashable) -> None:
        """Remove ``key`` and its dependency links (call with the lock held)."""

        _, _, tags = self._entries.pop(key)
        for tag in tags:
            keys = self._by_dep.get(tag)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del self._by_dep[tag]


_SPACE_RE = re.compile(r"\s+")


def normalize_question(question: str) -> str:
    """Case- and whitespace-insensitive form of a question (the answer-cache key)."""

    return _SPACE_RE.sub(" ", (question or "").strip().casefold()).rstrip(" ?!.")


class AnswerCache:
    """Question-level answer cache (the FAQ tier).

    An entry stores the finished ``GenerationResult`` and the version of every document it was built
    from, *as retrieved* (chunks carry ``document_version``; GraphRAG summaries carry
    ``source_document_versions`` for the chunks behind them). It is served only while the corpus
    still has those versions (``version_of`` is the RAG's document version lookup) - so a document
    updated while the answer was being generated never leaves a stale answer behind - and dropped
    by ``invalidate(doc_id)``, ``on_document_update`` or after ``ttl``.

    ``refresh_on_new_documents=True`` (default): adding a document that was not in the index
    before makes every cached answer stale (a new document can answer, or supersede the answer
    to, a cached question). Trade-off: frequent ingestion of new documents lowers the hit rate;
    pass ``False`` when new documents never change existing answers.

    Trade-offs: the same answer goes to everyone asking the same question in a scope (so raglib
    uses it for the public tier only), wording variants miss (keys are normalized text, not
    meaning), and conversation follow-ups bypass it (their answer depends on history).
    """

    def __init__(
        self,
        max_entries: int = 5_000,
        ttl: Optional[float] = 24 * 3600.0,
        clock: Callable[[], float] = time.monotonic,
        refresh_on_new_documents: bool = True,
    ):
        """Create an empty answer cache (default TTL: 24 hours)."""

        self._cache = TTLCache(max_entries=max_entries, ttl=ttl, clock=clock)
        self.version_of: Optional[Callable[[str], Optional[str]]] = None
        self.refresh_on_new_documents = refresh_on_new_documents
        self._generation = 0

    def new_documents_added(self) -> None:
        """The corpus gained documents: answers cached before are stale (``refresh_on_new_documents``)."""

        if self.refresh_on_new_documents:
            self._generation += 1

    @staticmethod
    def key(question: str, scope: Any = None, pipeline: str = "") -> str:
        """Cache key from the normalized question, the permission scope and the pipeline."""

        raw = json.dumps([normalize_question(question), scope, pipeline], sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, question: str, scope: Any = None, pipeline: str = "") -> Optional[Any]:
        """A copy of the cached result, or ``None`` (also when a source document changed)."""

        if not caching_allowed():
            return None
        key = self.key(question, scope, pipeline)
        hit, entry = self._cache.lookup(key)
        if not hit:
            return None
        result, versions, generation = entry if len(entry) == 3 else (*entry, self._generation)
        stale = generation != self._generation or (
            self.version_of is not None and any(self.version_of(doc) != version for doc, version in versions.items())
        )
        if stale:
            self._cache.discard(key)
            return None
        return copy.deepcopy(result)

    def put(self, question: str, result: Any, scope: Any = None, pipeline: str = "") -> bool:
        """Store ``result`` with the current versions of its source documents; False when not stored."""

        if not caching_allowed() or not getattr(result, "sources", None):
            return False  # answers without sources ("I don't know") are not worth serving again
        versions = answer_dependencies(result.sources, self.version_of)
        entry = (copy.deepcopy(result), versions, self._generation)
        self._cache.set(self.key(question, scope, pipeline), entry, deps=set(versions))
        return True

    def invalidate(self, doc_ids: Iterable[str]) -> int:
        """Drop the answers that used one of ``doc_ids``."""

        return self._cache.invalidate(doc_ids)

    def clear(self) -> None:
        """Drop every answer."""

        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> Dict[str, int]:
        """``hits``, ``misses``, ``expired`` and ``entries``."""

        return {"hits": self._cache.hits, "misses": self._cache.misses, "expired": self._cache.expired, "entries": len(self)}


def answer_dependencies(sources: Iterable[Any], version_of: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, Optional[str]]:
    """``{document id: version}`` an answer was built from, as the sources carried them at retrieval.

    A chunk depends on its ``parent_id`` at its ``document_version``; a GraphRAG summary on every
    document behind it (``source_document_versions``). Only a source without a recorded version is
    looked up now (``version_of``).
    """

    versions: Dict[str, Optional[str]] = {}
    for doc in sources:
        metadata = getattr(doc, "metadata", {}) or {}
        behind = metadata.get("source_document_versions")
        if isinstance(behind, dict) and behind:
            for parent, version in behind.items():
                versions.setdefault(str(parent), version)
            continue
        parent = str(metadata.get("parent_id") or getattr(doc, "id", ""))
        version = metadata.get("document_version")
        if version is None and version_of is not None:
            version = version_of(parent)
        versions.setdefault(parent, version)
    return versions
