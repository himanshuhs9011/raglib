"""Per-run state shared by every component of one pipeline run: token usage and degradations.

Both live in context variables, so concurrent requests never mix their numbers, and
``raglib.core.concurrency.parallel_map`` copies the context into its worker threads.

* ``usage_scope()`` / ``record_usage()`` - LLM clients report each call's tokens (the
  provider's real counts when it returns them, else an estimate); every open scope adds it.
* ``run_scope(strict)`` / ``report_degraded()`` - optional steps (grader, router, reranker,
  query rewrite, web search, ...) that fail and fall back report it here. The failure is
  listed in ``result.metadata["degraded"]``; with ``strict=True`` it is raised instead.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class UsageMeter:
    """Accumulates usage for one scope: LLM calls (also per model), embedding tokens, search calls."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_calls: int = 0
    embedding_tokens: int = 0
    llm_by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)
    embedding_by_model: Dict[str, int] = field(default_factory=dict)
    searches: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        estimated: bool,
        model: Optional[str] = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Add one LLM call (``prompt_tokens`` includes the cached prompt tokens)."""

        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += int(prompt_tokens or 0)
            self.completion_tokens += int(completion_tokens or 0)
            if estimated:
                self.estimated_calls += 1
            row = self.llm_by_model.setdefault(model or "unknown", _empty_llm_row())
            _add_llm(row, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens)

    def record_embedding(self, tokens: int, model: Optional[str] = None) -> None:
        """Add the tokens of one uncached embedding call."""

        with self._lock:
            self.embedding_tokens += int(tokens or 0)
            key = model or "unknown"
            self.embedding_by_model[key] = self.embedding_by_model.get(key, 0) + int(tokens or 0)

    def record_search(self, provider: str, count: int = 1) -> None:
        """Add billable search calls (API rerankers, web search providers)."""

        with self._lock:
            self.searches[provider] = self.searches.get(provider, 0) + int(count)

    def merge(self, other: "UsageMeter") -> None:
        """Add another meter's counts to this one."""

        snapshot = other.breakdown()
        with self._lock:
            self.llm_calls += other.llm_calls
            self.prompt_tokens += other.prompt_tokens
            self.completion_tokens += other.completion_tokens
            self.estimated_calls += other.estimated_calls
            self.embedding_tokens += other.embedding_tokens
            for model, row in snapshot["llm"].items():
                target = self.llm_by_model.setdefault(model, _empty_llm_row())
                for key, value in row.items():
                    target[key] = target.get(key, 0) + value
            for model, row in snapshot["embedding"].items():
                self.embedding_by_model[model] = self.embedding_by_model.get(model, 0) + row["tokens"]
            for provider, row in snapshot["search"].items():
                self.searches[provider] = self.searches.get(provider, 0) + row["calls"]

    def breakdown(self) -> Dict[str, Any]:
        """``{"llm": {model: counts}, "embedding": {model: {"tokens": n}}, "search": {provider: {"calls": n}}}``."""

        with self._lock:
            return {
                "llm": {model: dict(row) for model, row in self.llm_by_model.items()},
                "embedding": {model: {"tokens": tokens} for model, tokens in self.embedding_by_model.items()},
                "search": {provider: {"calls": calls} for provider, calls in self.searches.items()},
            }

    def as_dict(self) -> Dict[str, Any]:
        """Totals for ``GenerationResult.usage``."""

        return {
            "llm_calls": self.llm_calls,
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "usage_estimated": self.estimated_calls > 0,
            "embedding_tokens": self.embedding_tokens,
        }


def _empty_llm_row() -> Dict[str, int]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}


def _add_llm(row: Dict[str, int], prompt: int, completion: int, cache_read: int, cache_write: int) -> None:
    row["calls"] += 1
    row["prompt_tokens"] += int(prompt or 0)
    row["completion_tokens"] += int(completion or 0)
    row["cache_read_tokens"] += int(cache_read or 0)
    row["cache_write_tokens"] += int(cache_write or 0)


@dataclass
class RunState:
    """Degradations and strictness for one pipeline run."""

    strict: bool = False
    degraded: List[Dict[str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


_METERS: ContextVar[Tuple[UsageMeter, ...]] = ContextVar("raglib_usage_meters", default=())
_RUN: ContextVar[Optional[RunState]] = ContextVar("raglib_run_state", default=None)


def _reset(var: ContextVar, token: Any) -> None:
    """Reset a context variable, tolerating generators closed from another context."""

    try:
        var.reset(token)
    except ValueError:
        pass


@contextmanager
def usage_scope(meter: Optional[UsageMeter] = None) -> Iterator[UsageMeter]:
    """Collect the usage of every LLM call made inside the block (nested scopes all count it).

    ``meter`` reuses an existing meter (e.g. one opened by ``RAG.query`` for its guards); it is
    not added twice when it is already collecting, so nothing is double-counted.
    """

    if meter is not None and any(active is meter for active in _METERS.get()):
        yield meter
        return
    meter = meter or UsageMeter()
    token = _METERS.set(_METERS.get() + (meter,))
    try:
        yield meter
    finally:
        _reset(_METERS, token)


def record_usage(
    prompt_tokens: int,
    completion_tokens: int,
    estimated: bool = False,
    model: Optional[str] = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> None:
    """Report one LLM call (custom ``BaseLLMClient`` implementations may call this too).

    ``model`` (``"provider/model"``) lets cost tracking price the call; ``cache_*_tokens`` are the
    parts of ``prompt_tokens`` read from / written to the provider's prompt cache.
    """

    for meter in _METERS.get():
        meter.record(prompt_tokens, completion_tokens, estimated, model, cache_read_tokens, cache_write_tokens)
    from raglib.core.tracing import add_stage_usage

    add_stage_usage(
        "llm",
        model or "unknown",
        calls=1,
        prompt_tokens=int(prompt_tokens or 0),
        completion_tokens=int(completion_tokens or 0),
        cache_read_tokens=int(cache_read_tokens or 0),
        cache_write_tokens=int(cache_write_tokens or 0),
    )


def record_embedding_usage(tokens: int, model: Optional[str] = None) -> None:
    """Report embedding tokens (``CachedEmbedding`` calls it for every uncached call)."""

    for meter in _METERS.get():
        meter.record_embedding(tokens, model)
    from raglib.core.tracing import add_stage_usage

    add_stage_usage("embedding", model or "unknown", tokens=int(tokens or 0))


def record_search_usage(provider: str, count: int = 1) -> None:
    """Report billable search calls (Cohere rerank, web search APIs)."""

    for meter in _METERS.get():
        meter.record_search(provider, count)
    from raglib.core.tracing import add_stage_usage

    add_stage_usage("search", provider, calls=int(count))


@contextmanager
def run_scope(strict: bool = False) -> Iterator[RunState]:
    """Open the degradation log for one pipeline run."""

    state = RunState(strict=strict)
    token = _RUN.set(state)
    try:
        yield state
    finally:
        _reset(_RUN, token)


def report_degraded(component: str, exc: BaseException, log: Optional[logging.Logger] = None) -> None:
    """Record that ``component`` failed and fell back; re-raise ``exc`` in strict mode.

    Call it from inside the ``except`` block that handles ``exc``.
    """

    (log or logger).warning("%s failed (%s: %s); continuing with a fallback", component, type(exc).__name__, exc)
    state = _RUN.get()
    if state is None:
        return
    if state.strict:
        raise exc
    with state._lock:
        state.degraded.append({"component": component, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
