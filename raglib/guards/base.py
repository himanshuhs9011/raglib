"""Guard framework: decisions, the guard base class, the runner (cheapest first), and the per-query scope.

A guard checks one value - the question (``input``), the retrieved documents (``retrieval``), the
answer (``output``) or a tool call (``tool``) - and returns a ``Decision``:

* ``allow``  - continue;
* ``block``  - stop; ``message`` is shown to the user instead of an answer;
* ``modify`` - continue with ``value`` (e.g. the question with PII masked).

Tiers order the work by cost: tier 0 rules (free), tier 1 cheap checks, tier 2 LLM judge. The
runner sorts guards by tier, stops at the first block, caches decisions of ``cacheable`` guards by
input hash, and traces every guard as its own stage ``guard.<name>`` (so it shows up in the trace,
``on_event``, ``LatencyStats`` and - when it calls a model - ``CostReport``).

Failure policy: a guard that raises is *fail closed* (blocks) when ``kind="security"`` and
*fail open* (allows, error recorded) when ``kind="quality"``; ``fail_closed=`` overrides it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from raglib.cache import TTLCache
from raglib.core.tracing import trace_stage

logger = logging.getLogger(__name__)

ALLOW, BLOCK, MODIFY = "allow", "block", "modify"
STAGES = ("input", "retrieval", "output", "tool")


@dataclass
class Decision:
    """The outcome of one guard check."""

    action: str = ALLOW
    message: Optional[str] = None  # user-facing text when blocked
    value: Any = None  # the replacement value when modified
    confidence: float = 1.0  # how sure the guard is (rules: 1.0; models: their score)
    reason: str = ""  # short, never contains the checked text itself
    guard: str = ""
    stage: str = ""
    tier: int = 0
    mode: str = "rule"  # rule | local | api | skipped
    cached: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0

    def summary(self) -> Dict[str, Any]:
        """JSON-safe record for ``result.metadata["guards"]`` (no checked values)."""

        info = asdict(self)
        info.pop("value", None)
        return info


def allow(reason: str = "", confidence: float = 1.0) -> Decision:
    return Decision(ALLOW, reason=reason, confidence=confidence)


def block(message: str, reason: str = "", confidence: float = 1.0) -> Decision:
    return Decision(BLOCK, message=message, reason=reason, confidence=confidence)


def modify(value: Any, reason: str = "", confidence: float = 1.0) -> Decision:
    return Decision(MODIFY, value=value, reason=reason, confidence=confidence)


@dataclass
class GuardContext:
    """What a guard may look at besides the checked value."""

    stage: str = "input"
    question: str = ""
    user: Any = None  # raglib.access.UserContext or None
    user_key: str = "anonymous"  # rate limits and budgets are kept per user_key
    filters: Optional[Dict[str, Any]] = None
    permission_filter: Optional[Dict[str, Any]] = None  # the user's access-control filter (None = off)
    documents: Sequence[Any] = ()
    result: Any = None
    tool: Optional[str] = None
    retrieval_skipped: bool = False  # retrieval was skipped on purpose (small talk, "no retrieval needed")
    reranker: Optional[str] = None  # the reranker backend that scored ``documents`` (None / "none" = not reranked)


class Guard:
    """Base class. Subclasses set ``name``/``tier``/``kind`` and implement ``check``.

    ``cacheable=True`` means the decision depends only on the checked value (and ``cache_scope``),
    so the runner may reuse it for the same input within ``RAG(guard_cache_ttl=...)``.
    """

    name = "guard"
    tier = 0
    kind = "security"  # "security" guards fail closed, "quality" guards fail open
    stages: Tuple[str, ...] = ("input",)
    cacheable = False
    block_message = "This request was blocked by a safety check."

    def __init__(self, fail_closed: Optional[bool] = None, message: Optional[str] = None):
        """``fail_closed`` overrides the kind default; ``message`` overrides the user-facing block message."""

        self._fail_closed = fail_closed
        if message is not None:
            self.block_message = message

    @property
    def fail_closed(self) -> bool:
        return self.kind == "security" if self._fail_closed is None else self._fail_closed

    def check(self, value: Any, context: GuardContext) -> Decision:
        raise NotImplementedError

    def cache_scope(self, context: GuardContext) -> str:
        """Extra cache-key material besides the value (e.g. the user's permissions)."""

        return ""

    def bind(self, rag: Any) -> None:
        """Called once by ``RAG`` (e.g. to pick up its price table or models)."""

    def after(self, result: Any, context: GuardContext) -> None:
        """Called after the query finished (stateful guards: budgets)."""


class GuardRunner:
    """Runs guards cheapest tier first, stops at the first block, caches cacheable decisions."""

    def __init__(self, cache_ttl: Optional[float] = 3600.0, max_entries: int = 10_000):
        self.cache = TTLCache(max_entries=max_entries, ttl=cache_ttl) if cache_ttl else None

    def run(
        self, stage: str, guards: Sequence[Guard], value: Any, context: GuardContext
    ) -> Tuple[Any, List[Decision], Optional[Decision]]:
        """``(possibly modified value, decisions, the blocking decision or None)``."""

        decisions: List[Decision] = []
        for guard in sorted((g for g in guards if stage in g.stages), key=lambda g: g.tier):
            decision = self._check(guard, stage, value, context)
            decisions.append(decision)
            if decision.action == MODIFY:
                value = decision.value
            elif decision.action == BLOCK:
                return value, decisions, decision
        return value, decisions, None

    def _check(self, guard: Guard, stage: str, value: Any, context: GuardContext) -> Decision:
        key = None
        if self.cache is not None and guard.cacheable:
            raw = json.dumps([guard.name, stage, _fingerprint(value), guard.cache_scope(context)], default=str)
            key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            hit, cached = self.cache.lookup(key)
            if hit:
                decision = replace(cached, cached=True, latency_ms=0.0)
                with trace_stage(f"guard.{guard.name}", **_meta(decision)):
                    pass
                return decision
        started = time.perf_counter()
        with trace_stage(f"guard.{guard.name}", stage=stage, tier=guard.tier) as meta:
            try:
                decision = guard.check(value, context)
            except Exception as exc:  # noqa: BLE001 - the failure policy decides
                logger.warning("Guard %s failed (%s: %s)", guard.name, type(exc).__name__, exc)
                decision = (
                    block("This request could not be checked safely. Please try again.", reason="guard error")
                    if guard.fail_closed
                    else allow(reason="guard error (fail open)")
                )
                decision.error = f"{type(exc).__name__}: {str(exc)[:200]}"
            decision.guard, decision.stage = guard.name, stage
            if decision.tier == 0 and guard.tier:
                decision.tier = guard.tier
            decision.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            meta.update(_meta(decision))
        if key is not None and decision.error is None:
            self.cache.set(key, decision)
        return decision


def _fingerprint(value: Any) -> str:
    if isinstance(value, str):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _meta(decision: Decision) -> Dict[str, Any]:
    return {
        "stage": decision.stage,
        "tier": decision.tier,
        "action": decision.action,
        "mode": decision.mode,
        "confidence": round(decision.confidence, 3),
        "cached": decision.cached,
        "reason": decision.reason,
        **({"error": decision.error} if decision.error else {}),
    }


@dataclass
class GuardScope:
    """The guards of one query, visible to the pipeline (retrieval / output / tool stages)."""

    runner: GuardRunner
    context: GuardContext
    retrieval_guards: Sequence[Guard] = ()
    output_guards: Sequence[Guard] = ()
    tool_policy: Any = None
    decisions: List[Decision] = field(default_factory=list)

    def run(self, stage: str, guards: Sequence[Guard], value: Any, **context: Any) -> Tuple[Any, Optional[Decision]]:
        """Run a stage with this scope's context (fields overridden by ``context``); records decisions."""

        stage_context = replace(self.context, stage=stage, **context)
        value, decisions, blocked = self.runner.run(stage, guards, value, stage_context)
        self.decisions.extend(decisions)
        return value, blocked


_SCOPE: ContextVar[Optional[GuardScope]] = ContextVar("raglib_guard_scope", default=None)


@contextmanager
def guard_scope(scope: Optional[GuardScope]) -> Iterator[Optional[GuardScope]]:
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        try:
            _SCOPE.reset(token)
        except ValueError:
            pass


def current_scope() -> Optional[GuardScope]:
    """The running query's guard scope (``None`` when no guards are configured)."""

    return _SCOPE.get()
