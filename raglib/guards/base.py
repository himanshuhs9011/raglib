"""Guard framework: decisions, the guard base class, the runner (cheapest first), and the per-query scope.

A guard checks one value - the question (``input``), the retrieved documents (``retrieval``), the
answer (``output``) or a tool call (``tool``) - and returns a ``Decision``:

* ``allow``  - continue;
* ``block``  - stop; ``message`` is shown to the user instead of an answer;
* ``modify`` - continue with ``value`` (e.g. the question with PII masked).

Tiers order the work by cost: tier 0 rules (free), tier 1 cheap checks, tier 2 LLM judge. The
runner sorts guards by ``(tier, order)`` - PII masking has ``order=-1``, so no other guard ever
sees the raw question - stops at the first block, caches decisions of ``cacheable`` guards by
input hash, and traces every guard as its own stage ``guard.<name>`` (so it shows up in the trace,
``on_event``, ``LatencyStats`` and - when it calls a model - ``CostReport``).

Failure policy: a guard that raises is *fail closed* (blocks) when ``kind="security"`` and
*fail open* (allows, error recorded) when ``kind="quality"``; ``fail_closed=`` overrides it.

Nothing a guard reports (message, reason, error, log line) can carry the checked text: the runner
removes the raw value from it and masks PII and secrets before it is traced, logged or returned.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

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
    # Flag-only guards: what the guard would have done ("block" / "modify"); the action stays "allow".
    flag: Optional[str] = None
    # Tier 2: why the guard escalated to the LLM judge, and how that went (ok / cached / budget / unsure).
    escalated: Optional[str] = None
    judge: Optional[str] = None
    # Request-private data (the PII placeholder map): never traced, cached, logged or returned.
    private: Any = field(default=None, repr=False, compare=False)

    def summary(self) -> Dict[str, Any]:
        """JSON-safe record for ``result.metadata["guards"]`` (no checked values, nothing private)."""

        info = asdict(replace(self, value=None, private=None))
        info.pop("value", None)
        info.pop("private", None)
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
    # Identity for rate limits and budgets: RAG(rate_limit_key=fn), else the user id; None = anonymous.
    user_key: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request: Mapping[str, Any] = field(default_factory=dict)  # query(..., request={"ip": ..., "tenant": ...})
    filters: Optional[Dict[str, Any]] = None
    permission_filter: Optional[Dict[str, Any]] = None  # the user's access-control filter (None = off)
    documents: Sequence[Any] = ()
    result: Any = None
    tool: Optional[str] = None
    retrieval_skipped: bool = False  # retrieval was skipped on purpose (small talk, "no retrieval needed")
    reranker: Optional[str] = None  # the reranker backend that scored ``documents`` (None / "none" = not reranked)
    high_risk: Optional[str] = None  # why this request is high-risk (e.g. "tool calls"): tier 1 always escalates


class Guard:
    """Base class. Subclasses set ``name``/``tier``/``kind`` and implement ``check``.

    ``cacheable=True`` means the decision depends only on the checked value (and ``cache_scope``),
    so the runner may reuse it for the same input within ``RAG(guard_cache_ttl=...)``.
    ``streaming_safe=True`` (output guards) means the guard can check an answer piece by piece
    (``check_stream``), so ``rag.stream`` keeps streaming; otherwise the answer is buffered.
    """

    name = "guard"
    tier = 0
    order = 0  # within a tier, lower runs first
    kind = "security"  # "security" guards fail closed, "quality" guards fail open
    stages: Tuple[str, ...] = ("input",)
    cacheable = False
    streaming_safe = False
    # Characters a streaming-safe guard needs held back (None = RAG(stream_holdback_chars=...)).
    stream_holdback: Optional[int] = None
    # Quiet guards leave no trace stage and no record when they allow (like flag-only guards), so
    # an always-on guard does not change normal results; a modification or block is recorded.
    quiet = False
    block_message = "This request was blocked by a safety check."

    def __init__(self, fail_closed: Optional[bool] = None, message: Optional[str] = None, flag_only: bool = False):
        """``fail_closed`` overrides the kind default; ``message`` overrides the user-facing block message.

        ``flag_only=True``: never block or modify; what the guard would have done is reported in
        ``result.guard_flags`` and as a ``guard_flag`` event on ``on_event``. Flag-only output
        guards run once on the finished answer, so they never hold back a stream.
        """

        self._fail_closed = fail_closed
        self.flag_only = flag_only
        if message is not None:
            self.block_message = message

    @property
    def fail_closed(self) -> bool:
        return self.kind == "security" if self._fail_closed is None else self._fail_closed

    def check(self, value: Any, context: GuardContext) -> Decision:
        raise NotImplementedError

    def check_stream(self, chunk: str, context: GuardContext, state: Dict[str, Any]) -> Decision:
        """Check one released piece of a streamed answer (``state`` persists for the whole stream)."""

        return self.check(chunk, context)

    def cache_scope(self, context: GuardContext) -> str:
        """Extra cache-key material besides the value (e.g. the user's permissions)."""

        return ""

    def bind(self, rag: Any) -> None:
        """Called once by ``RAG`` (e.g. to pick up its price table, store or models)."""

    def after(self, result: Any, context: GuardContext) -> None:
        """Called after the query finished (stateful guards: budgets)."""

    def config(self) -> Dict[str, Any]:
        """The guard's settings (public attributes), for the configuration hash of the answer cache."""

        settings = {key: _plain(value) for key, value in sorted(vars(self).items()) if not key.startswith("_")}
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "tier": self.tier,
            "fail_closed": self.fail_closed,
            "message": self.block_message,
            **settings,
        }


def _plain(value: Any) -> Any:
    """A stable, JSON-able description of a setting (objects by type, functions by name)."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted(str(_plain(v)) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, re.Pattern):
        return value.pattern
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{getattr(value, '__module__', '')}.{value.__qualname__}"
    return type(value).__qualname__


def sanitize(text: Optional[str], raw: Any = None) -> Optional[str]:
    """``text`` without the raw checked value and with PII / secrets masked (safe to trace and log)."""

    if not text:
        return text
    from raglib.guards.pii import redact

    values: List[str] = []
    if isinstance(raw, str):
        values.append(raw)
    elif isinstance(raw, (list, tuple)):
        values.extend(item for item in raw if isinstance(item, str))
    return redact(text, *values)


def guard_sort_key(guard: Guard) -> Tuple[int, int]:
    return guard.tier, guard.order


class GuardRunner:
    """Runs guards cheapest tier first, stops at the first block, caches cacheable decisions."""

    def __init__(self, cache_ttl: Optional[float] = 3600.0, max_entries: int = 10_000):
        self.cache = TTLCache(max_entries=max_entries, ttl=cache_ttl) if cache_ttl else None

    def run(
        self, stage: str, guards: Sequence[Guard], value: Any, context: GuardContext
    ) -> Tuple[Any, List[Decision], Optional[Decision]]:
        """``(possibly modified value, decisions, the blocking decision or None)``."""

        decisions: List[Decision] = []
        for guard in sorted((g for g in guards if stage in g.stages), key=guard_sort_key):
            decision = self._check(guard, stage, value, context)
            if decision.flag or decision.action != ALLOW or not (guard.flag_only or guard.quiet):
                decisions.append(decision)  # a flag-only or quiet guard that found nothing leaves no record
            if decision.action == MODIFY:
                value = decision.value
                if stage == "input" and isinstance(value, str):
                    context = replace(context, question=value)  # later guards see the masked question only
            elif decision.action == BLOCK:
                return value, decisions, decision
        return value, decisions, None

    def _check(self, guard: Guard, stage: str, value: Any, context: GuardContext) -> Decision:
        if guard.flag_only or guard.quiet:
            # Flag-only and quiet guards: no trace stage, no cache; a finding becomes an on_event notice.
            started = time.perf_counter()
            decision = run_guarded(guard, lambda: guard.check(value, context), value)
            decision.guard, decision.stage = guard.name, stage
            decision.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            _report_flag(decision)
            return decision
        key = None
        if self.cache is not None and guard.cacheable:
            raw = json.dumps([guard.name, stage, _fingerprint(value), guard.cache_scope(context)], default=str)
            key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            hit, cached = self.cache.lookup(key)
            if hit:
                decision = replace(cached, cached=True, latency_ms=0.0)
                with trace_stage(f"guard.{guard.name}", **_meta(decision)):
                    pass
                _report_flag(decision)
                return decision
        started = time.perf_counter()
        with trace_stage(f"guard.{guard.name}", stage=stage, tier=guard.tier) as meta:
            decision = run_guarded(guard, lambda: guard.check(value, context), value)
            decision.guard, decision.stage = guard.name, stage
            decision.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            meta.update(_meta(decision))
        if key is not None and decision.error is None:
            self.cache.set(key, replace(decision, private=None))
        _report_flag(decision)
        return decision


def _report_flag(decision: Decision) -> None:
    """A flag-only guard's finding as an ``on_event`` notice (``stage="guard_flag"``)."""

    if not decision.flag:
        return
    from raglib.core.tracing import current_tracer

    tracer = current_tracer()
    if tracer is not None:
        tracer.notice(
            "guard_flag", guard=decision.guard, guard_stage=decision.stage, would=decision.flag,
            reason=decision.reason, confidence=round(decision.confidence, 3),
        )


def run_guarded(guard: Guard, call: Any, raw: Any) -> Decision:
    """Call a guard with the failure policy applied and its texts sanitized."""

    try:
        decision = call()
    except Exception as exc:  # noqa: BLE001 - the failure policy decides
        error = (sanitize(f"{type(exc).__name__}: {exc}", raw) or "")[:300]  # sanitize the whole text, then cut
        logger.warning("Guard %s failed (%s)", guard.name, error)
        decision = (
            block("This request could not be checked safely. Please try again.", reason="guard error")
            if guard.fail_closed
            else allow(reason="guard error (fail open)")
        )
        decision.error = error
    if decision.tier == 0 and guard.tier:
        decision.tier = guard.tier
    decision.reason = sanitize(decision.reason, raw) or ""
    if decision.action == BLOCK:
        decision.message = sanitize(decision.message or guard.block_message, raw)
    if getattr(guard, "flag_only", False) and decision.action in (BLOCK, MODIFY):
        decision.flag, decision.action, decision.value, decision.private = decision.action, ALLOW, None, None
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
        **({"flag": decision.flag} if decision.flag else {}),
        **({"escalated": decision.escalated, "judge": decision.judge} if decision.escalated else {}),
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
    stream_holdback: int = 64  # RAG(stream_holdback_chars=...)

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
