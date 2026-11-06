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
    """Accumulates LLM token usage for one scope."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, prompt_tokens: int, completion_tokens: int, estimated: bool) -> None:
        """Add one call."""

        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += int(prompt_tokens or 0)
            self.completion_tokens += int(completion_tokens or 0)
            if estimated:
                self.estimated_calls += 1

    def as_dict(self) -> Dict[str, Any]:
        """Totals for ``GenerationResult.usage``."""

        return {
            "llm_calls": self.llm_calls,
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "usage_estimated": self.estimated_calls > 0,
        }


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
def usage_scope() -> Iterator[UsageMeter]:
    """Collect the usage of every LLM call made inside the block (nested scopes all count it)."""

    meter = UsageMeter()
    token = _METERS.set(_METERS.get() + (meter,))
    try:
        yield meter
    finally:
        _reset(_METERS, token)


def record_usage(prompt_tokens: int, completion_tokens: int, estimated: bool = False) -> None:
    """Report one LLM call (custom ``BaseLLMClient`` implementations may call this too)."""

    for meter in _METERS.get():
        meter.record(prompt_tokens, completion_tokens, estimated)


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
