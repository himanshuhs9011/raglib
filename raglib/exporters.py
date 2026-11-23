"""Optional exporters for pipeline stage timings: Prometheus histograms and OpenTelemetry spans.

Both are ``on_event`` callbacks and can be combined with ``LatencyStats``::

    rag = RAG("docs/", on_event=[LatencyStats(), PrometheusExporter(), OpenTelemetryExporter()])

Neither dependency is required by raglib: ``pip install raglib-py[prometheus]`` /
``pip install raglib-py[otel]`` (or install ``prometheus-client`` / ``opentelemetry-api`` yourself).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

from raglib.core.tracing import TOTAL_STAGE
from raglib.schemas import TraceEvent

DEFAULT_BUCKETS: Sequence[float] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0,
)


class PrometheusExporter:
    """Observe every stage duration in a Prometheus histogram labelled ``stage`` and ``status``.

    ``registry`` defaults to prometheus_client's global registry; pass your own to avoid
    "duplicated timeseries" errors when you create several exporters (e.g. in tests).
    """

    def __init__(
        self,
        metric_name: str = "raglib_stage_duration_seconds",
        registry: Optional[Any] = None,
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ):
        """Create the histogram (``ImportError`` with an install hint without prometheus-client)."""

        try:
            from prometheus_client import Histogram  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("PrometheusExporter requires: pip install raglib-py[prometheus]") from exc
        kwargs: Dict[str, Any] = {"buckets": tuple(buckets)}
        if registry is not None:
            kwargs["registry"] = registry
        self.histogram = Histogram(
            metric_name, "Duration of raglib pipeline stages in seconds", ["stage", "status"], **kwargs
        )

    def __call__(self, event: TraceEvent) -> None:
        """``on_event`` callback."""

        status = "error" if event.error else "ok"
        self.histogram.labels(stage=event.stage, status=status).observe(event.latency_ms / 1000.0)


class OpenTelemetryExporter:
    """Export each query as an OpenTelemetry trace: a ``raglib.query`` root span with one child
    span per stage, nested like the stages (``dense_search`` under ``retrieve``) and with their
    real start and end times.

    Spans are created when the run's ``total`` event arrives, so they are complete and correctly
    nested even though stages end child-first. ``tracer`` defaults to
    ``opentelemetry.trace.get_tracer("raglib")``.
    """

    def __init__(self, tracer: Optional[Any] = None, max_pending_runs: int = 1_000):
        """Bind to an OpenTelemetry tracer (``ImportError`` with an install hint without the API)."""

        try:
            from opentelemetry import trace  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("OpenTelemetryExporter requires: pip install raglib-py[otel]") from exc
        self._trace = trace
        self.tracer = tracer or trace.get_tracer("raglib")
        self.max_pending_runs = max(1, max_pending_runs)
        self._pending: "OrderedDict[str, List[TraceEvent]]" = OrderedDict()
        self._lock = threading.Lock()

    def __call__(self, event: TraceEvent) -> None:
        """``on_event`` callback."""

        if event.run_id is None:
            self._export([event], root=None)
            return
        with self._lock:
            if event.stage != TOTAL_STAGE:
                self._pending.setdefault(event.run_id, []).append(event)
                while len(self._pending) > self.max_pending_runs:
                    self._pending.popitem(last=False)
                return
            events = self._pending.pop(event.run_id, [])
        self._export(events, root=event)

    def _export(self, events: List[TraceEvent], root: Optional[TraceEvent]) -> None:
        """Create the spans of one run top-down."""

        children: Dict[int, List[int]] = {}
        top: List[int] = []
        for index, event in enumerate(events):
            # A stage ends before its parent, so the parent is the next event with that name.
            parent_index = next(
                (j for j in range(index + 1, len(events)) if event.parent and events[j].stage == event.parent),
                None,
            )
            if parent_index is None:
                top.append(index)
            else:
                children.setdefault(parent_index, []).append(index)

        def start(event: TraceEvent, context: Any) -> Any:
            attributes = {
                f"raglib.{key}": value
                for key, value in event.meta.items()
                if isinstance(value, (str, bool, int, float))
            }
            if event.run_id:
                attributes["raglib.run_id"] = event.run_id
            start_ns = int((event.started_at or 0.0) * 1e9)
            span = self.tracer.start_span(f"raglib.{event.stage}", context=context, start_time=start_ns, attributes=attributes)
            if event.error:
                from opentelemetry.trace import Status, StatusCode  # type: ignore[import-not-found]

                span.set_status(Status(StatusCode.ERROR, event.error))
            return span, start_ns + int(event.latency_ms * 1e6)

        def emit(index: int, context: Any) -> None:
            span, end_ns = start(events[index], context)
            child_context = self._trace.set_span_in_context(span)
            for child in children.get(index, []):
                emit(child, child_context)
            span.end(end_time=end_ns)

        if root is None:
            for index in top:
                emit(index, None)
            return
        root_event = TraceEvent(
            stage="query", latency_ms=root.latency_ms, meta=root.meta, error=root.error,
            run_id=root.run_id, started_at=root.started_at,
        )
        root_span, root_end = start(root_event, None)
        root_context = self._trace.set_span_in_context(root_span)
        for index in top:
            emit(index, root_context)
        root_span.end(end_time=root_end)
