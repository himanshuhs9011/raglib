"""Per-stage latency percentiles over many queries, built on ``on_event``.

::

    from raglib import RAG, LatencyStats

    stats = LatencyStats()
    rag = RAG("docs/", llm="openai", on_event=stats)
    for question in questions:
        rag.query(question)
    stats.print_report()                     # count, P50, P95, P99, min, max per stage (ms)
    stats.slowest("generate", 3)             # the 3 slowest generations, with their tags

Every pipeline stage ends with an event (``retrieve``, ``query_embedding``, ``dense_search``,
``rerank``, ``generate``, ...). One sample is recorded per stage execution, so a multi-query run
that embeds four queries records four ``query_embedding`` samples; ``total`` is the wall time of
the whole query. Samples are buffered per run and recorded when its ``total`` event arrives, so
each one carries the run's tags: ``cache_hit`` (``{"embedding": bool, "llm": bool}``),
``query_tokens``, ``num_candidates``, ``reranker_used`` and ``is_cold_start``.
"""

from __future__ import annotations

import math
import sys
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import IO, Any, Deque, Dict, Iterable, List, Optional, Sequence

from raglib.core.tracing import TOTAL_STAGE
from raglib.schemas import TraceEvent

STAGE_ORDER: Sequence[str] = (
    "condense", "rewrite", "query_rewrite", "retrieve", "query_embedding", "dense_search", "filtering",
    "bm25_search", "fusion", "web_search", "grade", "process_documents", "rerank", "prompt_build",
    "generate", "post_process", "tool", TOTAL_STAGE,
)
RUN_TAGS: Sequence[str] = (
    "cache_hit", "query_tokens", "num_candidates", "reranker_used", "rerank_skipped", "pipeline", "tier",
)


@dataclass
class LatencySample:
    """One timed stage execution and the context it ran in."""

    stage: str
    duration_ms: float
    run_id: Optional[str] = None
    started_at: Optional[float] = None
    error: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)  # run tags + is_cold_start
    meta: Dict[str, Any] = field(default_factory=dict)  # the stage's own fields (e.g. cache_hit, hits)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (NumPy's default method); ``q`` in [0, 100]."""

    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class LatencyStats:
    """Thread-safe rolling per-stage latency collector; pass it as ``RAG(on_event=...)``.

    ``window``: samples kept per stage (oldest dropped first). ``cold_start_after_s``: a query
    that starts after this many idle seconds (or the first one seen) is tagged
    ``is_cold_start=True``. ``max_pending_runs`` bounds the buffer of runs that never finished.
    """

    def __init__(self, window: int = 10_000, cold_start_after_s: float = 300.0, max_pending_runs: int = 1_000):
        """Create an empty collector."""

        if window <= 0:
            raise ValueError("window must be greater than zero")
        self.window = window
        self.cold_start_after_s = cold_start_after_s
        self.max_pending_runs = max(1, max_pending_runs)
        self._samples: Dict[str, Deque[LatencySample]] = {}
        self._pending: "OrderedDict[str, List[TraceEvent]]" = OrderedDict()
        self._last_activity: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ recording

    def __call__(self, event: TraceEvent) -> None:
        """``on_event`` callback."""

        self.observe(event)

    def observe(self, event: TraceEvent) -> None:
        """Buffer a stage event; record the whole run when its ``total`` event arrives."""

        if event.meta.get("event") == "warning":
            return  # e.g. chunking_fallback: a notice, not a timed stage
        if event.run_id is None:  # a stand-alone Tracer, not a pipeline run
            self._add(LatencySample(event.stage, event.latency_ms, None, event.started_at, event.error, {}, dict(event.meta)))
            return
        with self._lock:
            if event.stage != TOTAL_STAGE:
                self._pending.setdefault(event.run_id, []).append(event)
                self._pending.move_to_end(event.run_id)
                while len(self._pending) > self.max_pending_runs:
                    self._pending.popitem(last=False)
                return
            events = self._pending.pop(event.run_id, [])
            cold = self._is_cold(event)
        tags = {key: event.meta[key] for key in RUN_TAGS if key in event.meta}
        tags["is_cold_start"] = cold
        for item in [*events, event]:
            self._add(
                LatencySample(
                    stage=item.stage,
                    duration_ms=item.latency_ms,
                    run_id=item.run_id,
                    started_at=item.started_at,
                    error=item.error,
                    tags=dict(tags),
                    meta=dict(item.meta),
                )
            )

    def record(self, stage: str, duration_ms: float, meta: Optional[Dict[str, Any]] = None, **tags: Any) -> None:
        """Record one sample directly (custom stages, imports from logs, tests)."""

        self._add(LatencySample(stage=stage, duration_ms=float(duration_ms), tags=dict(tags), meta=dict(meta or {})))

    def _is_cold(self, total: TraceEvent) -> bool:
        """Cold start: the first run, or one that began after ``cold_start_after_s`` idle seconds. Call under the lock."""

        if total.started_at is None:
            return False
        ended = total.started_at + total.latency_ms / 1000.0
        cold = self._last_activity is None or total.started_at - self._last_activity > self.cold_start_after_s
        self._last_activity = ended if self._last_activity is None else max(self._last_activity, ended)
        return cold

    def _add(self, sample: LatencySample) -> None:
        with self._lock:
            bucket = self._samples.get(sample.stage)
            if bucket is None:
                bucket = self._samples[sample.stage] = deque(maxlen=self.window)
            bucket.append(sample)

    def reset(self) -> None:
        """Drop every sample and pending run."""

        with self._lock:
            self._samples.clear()
            self._pending.clear()
            self._last_activity = None

    # ------------------------------------------------------------------ reading

    def stages(self) -> List[str]:
        """Stages with samples, in pipeline order (unknown stages last, alphabetically)."""

        with self._lock:
            names = list(self._samples)
        known = [name for name in STAGE_ORDER if name in names]
        return known + sorted(name for name in names if name not in STAGE_ORDER)

    def samples(self, stage: str) -> List[LatencySample]:
        """A snapshot of the samples kept for ``stage``."""

        with self._lock:
            return list(self._samples.get(stage, ()))

    def report(self, stages: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, float]]:
        """Per stage: ``count``, ``p50``, ``p95``, ``p99``, ``min``, ``max`` and ``mean`` (ms)."""

        result: Dict[str, Dict[str, float]] = {}
        for stage in list(stages) if stages is not None else self.stages():
            values = [sample.duration_ms for sample in self.samples(stage)]
            if not values:
                continue
            result[stage] = {
                "count": len(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
        return result

    def slowest(self, stage: str, n: int = 5) -> List[LatencySample]:
        """The ``n`` slowest samples of ``stage``, slowest first, with their tags and stage meta."""

        return sorted(self.samples(stage), key=lambda sample: sample.duration_ms, reverse=True)[: max(0, n)]

    def format_report(self, stages: Optional[Iterable[str]] = None) -> str:
        """The report as a fixed-width text table."""

        report = self.report(stages)
        if not report:
            return "No latency samples recorded."
        width = max(len("stage"), *(len(name) for name in report))
        header = f"{'stage':<{width}}  {'count':>7}  {'p50 ms':>10}  {'p95 ms':>10}  {'p99 ms':>10}  {'min ms':>10}  {'max ms':>10}"
        lines = [header, "-" * len(header)]
        for name, row in report.items():
            lines.append(
                f"{name:<{width}}  {int(row['count']):>7}  {row['p50']:>10.2f}  {row['p95']:>10.2f}  "
                f"{row['p99']:>10.2f}  {row['min']:>10.2f}  {row['max']:>10.2f}"
            )
        return "\n".join(lines)

    def print_report(self, stages: Optional[Iterable[str]] = None, file: Optional[IO[str]] = None) -> None:
        """Print ``format_report()``."""

        print(self.format_report(stages), file=file or sys.stdout)


def benchmark(
    rag: Any,
    queries: Sequence[str],
    repeats: int = 3,
    warmup: int = 1,
    warm_cache: bool = False,
    stats: Optional[LatencyStats] = None,
    **query_kwargs: Any,
) -> LatencyStats:
    """Run every query ``repeats`` times through ``rag`` and return the collected ``LatencyStats``.

    ``warmup`` runs of the first query are not recorded (model loading, connections). With
    ``warm_cache=False`` (default) the LLM and embedding caches are cleared before every query,
    so repeats measure real work instead of cache hits; ``True`` measures the cached path.
    ``query_kwargs`` go to ``rag.query`` (e.g. ``user=...``, ``filters=...``).
    """

    if repeats <= 0:
        raise ValueError("repeats must be greater than zero")
    stats = stats or LatencyStats()
    pipeline = getattr(rag, "pipeline", rag)
    previous = pipeline.on_event
    purge = getattr(rag, "_purge_caches", None)

    def run(question: str) -> None:
        if not warm_cache and purge is not None:
            purge()
        rag.query(question, **query_kwargs)

    try:
        for _ in range(max(0, warmup)):
            if queries:
                run(queries[0])
        pipeline.on_event = [previous, stats] if previous is not None else stats
        for _ in range(repeats):
            for question in queries:
                run(question)
    finally:
        pipeline.on_event = previous
    return stats
