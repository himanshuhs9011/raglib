"""Structured per-stage tracing for RAG pipelines.

One ``Tracer`` per pipeline run. While a run is active (``active_tracer``), any component can
time a stage with ``trace_stage("dense_search")`` without being handed the tracer; the stage's
parent is the stage that was open when it started (``TraceEvent.parent``), also across the
worker threads of ``parallel_map``. ``Tracer.finish`` emits a final ``"total"`` event (wall time
of the run) whose meta holds the run's tags: ``cache_hit``, ``query_tokens``,
``num_candidates``, ``reranker_used``, ``pipeline``.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from raglib.schemas import TraceEvent

logger = logging.getLogger(__name__)

EventCallback = Callable[[TraceEvent], None]
OnEvent = Union[EventCallback, Sequence[EventCallback], None]

TOTAL_STAGE = "total"

_ACTIVE: ContextVar[Optional["Tracer"]] = ContextVar("raglib_active_tracer", default=None)
_RUN_TAGS: ContextVar[Dict[str, Any]] = ContextVar("raglib_run_tags", default={})
_STAGE: ContextVar[Optional[Tuple[str, Dict[str, Any]]]] = ContextVar("raglib_open_stage", default=None)


def _reset(var: ContextVar, token: Any) -> None:
    """Reset a context variable, tolerating generators closed from another context."""

    try:
        var.reset(token)
    except ValueError:
        pass


def _callbacks(on_event: OnEvent) -> List[EventCallback]:
    """Normalize ``on_event`` (one callable, a list of them, or None) to a flat list."""

    if on_event is None:
        return []
    if callable(on_event):
        return [on_event]
    flat: List[EventCallback] = []
    for item in on_event:
        flat.extend(_callbacks(item))
    return flat


class Tracer:
    """Collects TraceEvents for one pipeline run and forwards them to ``on_event`` observers.

    ``on_event`` may be one callable or a list (e.g. ``[LatencyStats(), PrometheusExporter()]``).
    ``clock`` (monotonic seconds) and ``wall_clock`` (epoch seconds) can be replaced in tests.
    """

    def __init__(
        self,
        on_event: OnEvent = None,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        run_id: Optional[str] = None,
    ):
        """Start the run clock; no event is emitted until a stage ends."""

        self.events: List[TraceEvent] = []
        self._callbacks = _callbacks(on_event)
        self._clock = clock
        self._wall_clock = wall_clock
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.tags: Dict[str, Any] = dict(_RUN_TAGS.get())
        self._cache: Dict[str, List[int]] = {}  # kind -> [hits, misses]
        self._lock = threading.Lock()
        self._started = clock()
        self._started_wall = wall_clock()
        self._finished: Optional[TraceEvent] = None

    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[Dict[str, Any]]:
        """Time a pipeline stage; callers may add fields to the yielded meta dict."""

        details: Dict[str, Any] = dict(meta)
        parent = _STAGE.get()
        token = _STAGE.set((name, details))
        started = self._clock()
        started_wall = self._wall_clock()
        error: Optional[str] = None
        try:
            yield details
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _reset(_STAGE, token)
            self._emit(
                TraceEvent(
                    stage=name,
                    latency_ms=round((self._clock() - started) * 1000, 3),
                    meta=details,
                    error=error,
                    parent=parent[0] if parent else None,
                    run_id=self.run_id,
                    started_at=started_wall,
                )
            )

    def notice(self, stage: str, error: Optional[str] = None, **meta: Any) -> None:
        """Emit a warning event (``meta["event"] == "warning"``): a notice, not a timed stage."""

        self._emit(
            TraceEvent(
                stage=stage,
                latency_ms=0.0,
                meta={"event": "warning", **meta},
                error=error,
                parent=None,
                run_id=self.run_id,
                started_at=self._wall_clock(),
            )
        )

    def record(self, name: str, latency_ms: float, /, **meta: Any) -> None:
        """Emit a stage measured elsewhere (e.g. a guard summed over a stream) as a timed event."""

        parent = _STAGE.get()
        self._emit(
            TraceEvent(
                stage=name,
                latency_ms=round(float(latency_ms), 3),
                meta=dict(meta),
                error=None,
                parent=parent[0] if parent else None,
                run_id=self.run_id,
                started_at=self._wall_clock(),
            )
        )

    def tag(self, key: str, value: Any) -> None:
        """Set a run-level tag (reported on the ``total`` event)."""

        with self._lock:
            self.tags[key] = value

    def record_cache(self, kind: str, hit: bool) -> None:
        """Count one cache lookup of ``kind`` (``"embedding"``, ``"llm"``)."""

        with self._lock:
            counts = self._cache.setdefault(kind, [0, 0])
            counts[0 if hit else 1] += 1

    def cache_hits(self) -> Dict[str, bool]:
        """Per cache kind: True when every lookup in this run was a hit."""

        with self._lock:
            return {kind: misses == 0 and hits > 0 for kind, (hits, misses) in self._cache.items()}

    def finish(self, error: Optional[str] = None, **meta: Any) -> TraceEvent:
        """Emit the ``total`` event once (wall time of the run, run tags in meta) and return it."""

        if self._finished is not None:
            return self._finished
        with self._lock:
            tags = dict(self.tags)
        details = {"cache_hit": self.cache_hits(), **tags, **meta}
        event = TraceEvent(
            stage=TOTAL_STAGE,
            latency_ms=round((self._clock() - self._started) * 1000, 3),
            meta=details,
            error=error,
            parent=None,
            run_id=self.run_id,
            started_at=self._started_wall,
        )
        self._finished = event
        self._emit(event)
        return event

    def _emit(self, event: TraceEvent) -> None:
        """Record the event and notify observers (an observer error never breaks the pipeline)."""

        with self._lock:
            self.events.append(event)
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as exc:  # noqa: BLE001 - observers must never break the pipeline
                logger.warning("Trace callback failed: %s", exc)

    def total_latency_ms(self) -> float:
        """Summed latency of the top-level stages (nested stages and ``total`` are not added again)."""

        return round(sum(e.latency_ms for e in self.events if e.parent is None and e.stage != TOTAL_STAGE), 2)


@contextmanager
def run_tags(**tags: Any) -> Iterator[None]:
    """Tags every Tracer created inside the block starts with (e.g. the routing ``tier``)."""

    token = _RUN_TAGS.set({**_RUN_TAGS.get(), **tags})
    try:
        yield
    finally:
        _reset(_RUN_TAGS, token)


@contextmanager
def active_tracer(tracer: Optional[Tracer]) -> Iterator[Optional[Tracer]]:
    """Make ``tracer`` the target of ``trace_stage`` / ``trace_tag`` inside the block."""

    token = _ACTIVE.set(tracer)
    try:
        yield tracer
    finally:
        _reset(_ACTIVE, token)


def current_tracer() -> Optional[Tracer]:
    """The tracer of the running pipeline, if any."""

    return _ACTIVE.get()


@contextmanager
def trace_stage(name: str, **meta: Any) -> Iterator[Dict[str, Any]]:
    """Time ``name`` on the active tracer; a no-op (still yields a meta dict) outside a run."""

    tracer = _ACTIVE.get()
    if tracer is None:
        yield dict(meta)
        return
    with tracer.stage(name, **meta) as details:
        yield details


def annotate_stage(**fields: Any) -> None:
    """Add fields to the meta of the innermost open stage (no-op outside a stage)."""

    current = _STAGE.get()
    if current is not None:
        current[1].update(fields)


def trace_tag(key: str, value: Any) -> None:
    """Set a run-level tag on the active tracer (no-op outside a run)."""

    tracer = _ACTIVE.get()
    if tracer is not None:
        tracer.tag(key, value)


def record_cache_lookup(kind: str, hit: bool) -> None:
    """Report a cache lookup: tags the open stage (``cache_hit``) and the run's cache counters."""

    current = _STAGE.get()
    if current is not None:
        previous = current[1].get("cache_hit")
        current[1]["cache_hit"] = hit if previous is None else bool(previous and hit)
    tracer = _ACTIVE.get()
    if tracer is not None:
        tracer.record_cache(kind, hit)


STAGE_USAGE_KEY = "_usage"


def add_stage_usage(kind: str, name: str, **counts: int) -> None:
    """Add usage (``kind``: llm / embedding / search) to the innermost open stage's meta.

    Usage is attributed to the innermost stage only, so summing every event never counts a call
    twice; cost tracking prices each stage from ``meta["_usage"]``.
    """

    current = _STAGE.get()
    if current is None:
        return
    usage = current[1].setdefault(STAGE_USAGE_KEY, {})
    row = usage.setdefault(kind, {}).setdefault(name, {})
    for key, value in counts.items():
        row[key] = row.get(key, 0) + int(value)


def record_stage(name: str, latency_ms: float, /, **meta: Any) -> None:
    """``Tracer.record`` on the active tracer (no-op outside a run)."""

    tracer = _ACTIVE.get()
    if tracer is not None:
        tracer.record(name, latency_ms, **meta)
