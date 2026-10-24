"""Abstract interfaces for LLM clients used by raglib."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Defines the contract for language model clients.

    Subclasses must implement ``complete`` and ``chat``. ``stream``, ``acomplete``
    and ``complete_json`` have working defaults built on ``complete``.
    """

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Generate a completion for a single prompt."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from a structured chat transcript."""

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Yield the completion incrementally (default: one chunk)."""

        yield self.complete(prompt=prompt, system=system)

    async def acomplete(self, prompt: str, system: str = "") -> str:
        """Async completion; defaults to running ``complete`` in a worker thread."""

        return await asyncio.to_thread(self.complete, prompt, system)

    def complete_json(self, prompt: str, system: str = "", default: Any = None) -> Any:
        """Complete and parse the first JSON value in the response, or return ``default``."""

        from raglib.core.parsing import parse_json

        try:
            raw = self.complete(prompt=prompt, system=system)
        except Exception as exc:  # noqa: BLE001
            logger.warning("complete_json LLM call failed: %s", exc)
            return default
        return parse_json(raw, default=default)


class CachedLLMClient(BaseLLMClient):
    """Thread-safe LRU cache in front of any LLM client (keyed on system + prompt).

    Useful for deterministic (temperature=0) calls repeated across queries, such as
    query rewriting, relevance grading and GraphRAG extraction of unchanged chunks.
    """

    def __init__(self, inner: BaseLLMClient, max_entries: int = 2048):
        """Wrap ``inner`` with an in-memory LRU cache."""

        self.inner = inner
        self.max_entries = max_entries
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def complete(self, prompt: str, system: str = "") -> str:
        """Return a cached completion or call the wrapped client."""

        key = hashlib.sha256(f"{system}\x00{prompt}".encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
        self.misses += 1
        value = self.inner.complete(prompt=prompt, system=system)
        with self._lock:
            self._cache[key] = value
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop every cached response (called when documents are deleted)."""

        with self._lock:
            self._cache.clear()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Delegate chat calls without caching (transcripts rarely repeat)."""

        return self.inner.chat(messages)

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Delegate streaming to the wrapped client."""

        return self.inner.stream(prompt=prompt, system=system)

    def __getattr__(self, name: str) -> Any:
        """Expose wrapped-client attributes (e.g. provider metadata)."""

        return getattr(self.inner, name)


class FallbackLLMClient(BaseLLMClient):
    """Try a list of clients in order; the next one is used when a call fails (after its own retries).

    Useful with free / shared endpoints that are often rate limited or temporarily down.
    """

    def __init__(self, clients: List[BaseLLMClient]):
        """Wrap ``clients`` (primary first)."""

        if not clients:
            raise ValueError("FallbackLLMClient needs at least one client")
        self.clients = list(clients)

    def _try(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call ``method`` on each client until one succeeds."""

        last: Optional[BaseException] = None
        for index, client in enumerate(self.clients):
            try:
                return getattr(client, method)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if index + 1 < len(self.clients):
                    logger.warning(
                        "LLM %s failed (%s); falling back to %s",
                        getattr(client, "model_name", type(client).__name__),
                        type(exc).__name__,
                        getattr(self.clients[index + 1], "model_name", "next model"),
                    )
        assert last is not None
        raise last

    def complete(self, prompt: str, system: str = "") -> str:
        """Complete with the first client that succeeds."""

        return self._try("complete", prompt=prompt, system=system)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat with the first client that succeeds."""

        return self._try("chat", messages)

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Stream from the first client that produces a first chunk."""

        last: Optional[BaseException] = None
        for client in self.clients:
            try:
                iterator = iter(client.stream(prompt=prompt, system=system))
                first = next(iterator, None)
            except Exception as exc:  # noqa: BLE001
                last = exc
                continue
            if first is not None:
                yield first
            yield from iterator
            return
        assert last is not None
        raise last

    @property
    def provider(self) -> Optional[str]:
        """Provider of the primary client."""

        return getattr(self.clients[0], "provider", None)

    @property
    def model_name(self) -> Optional[str]:
        """Model of the primary client."""

        return getattr(self.clients[0], "model_name", None)

    @property
    def rate_limiter(self) -> Any:
        """Rate limiter of the primary client."""

        return getattr(self.clients[0], "rate_limiter", None)


def unwrap_llm(client: Optional[BaseLLMClient]) -> Optional[BaseLLMClient]:
    """Return the innermost client behind caching wrappers."""

    while isinstance(client, CachedLLMClient):
        client = client.inner
    return client
