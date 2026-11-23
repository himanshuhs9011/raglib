"""Per-stage tracing, LatencyStats percentiles, exporters and benchmark() (offline, fake clocks)."""

from __future__ import annotations

import io
import sys
import threading
import types
from typing import Any, Dict, List, Optional

import pytest

from raglib import RAG, LatencyStats, benchmark
from raglib.core.tracing import Tracer, active_tracer, trace_stage
from raglib.latency import percentile
from raglib.schemas import TraceEvent

KNOWN_MS = [38, 39, 40, 41, 43, 45, 47, 52, 95, 2100]
CORPUS = [
    "Acme Corp offices close on Friday afternoons in summer and the cafeteria opens at eight.",
    "The Orion launch window opens in March and closes in April for the whole launch team.",
]


class FakeClock:
    """Monotonic clock that only moves when told to."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


# ----------------------------------------------------------------------------- percentiles


def test_percentiles_use_linear_interpolation_on_a_known_list() -> None:
    assert percentile(KNOWN_MS, 50) == pytest.approx(44.0)
    assert percentile(KNOWN_MS, 95) == pytest.approx(1197.75)
    assert percentile(KNOWN_MS, 99) == pytest.approx(1919.55)
    assert percentile(KNOWN_MS, 0) == 38 and percentile(KNOWN_MS, 100) == 2100
    assert percentile([7.0], 99) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_report_from_fake_stages_timed_with_a_fake_clock() -> None:
    clock, wall = FakeClock(), FakeClock(start=1_700_000_000.0)
    stats = LatencyStats()
    for ms in reversed(KNOWN_MS):  # order must not matter
        tracer = Tracer(on_event=stats, clock=clock, wall_clock=wall)
        with tracer.stage("generate"):
            clock.advance_ms(ms)
        with tracer.stage("post_process"):
            clock.advance_ms(2)
        tracer.finish(pipeline="fake")
        wall.advance_ms(10_000)
    row = stats.report()["generate"]
    assert row["count"] == 10
    assert row["p50"] == pytest.approx(44.0)
    assert row["p95"] == pytest.approx(1197.75)
    assert row["p99"] == pytest.approx(1919.55)
    assert (row["min"], row["max"]) == (38, 2100)
    assert stats.report()["post_process"]["p99"] == pytest.approx(2.0)
    assert stats.report()["total"]["max"] == pytest.approx(2102.0)
    assert stats.stages() == ["generate", "post_process", "total"]


def test_rolling_window_keeps_the_latest_samples() -> None:
    stats = LatencyStats(window=3)
    for ms in [1, 2, 3, 4, 5]:
        stats.record("rerank", ms)
    assert [s.duration_ms for s in stats.samples("rerank")] == [3, 4, 5]
    with pytest.raises(ValueError):
        LatencyStats(window=0)


def test_recording_is_thread_safe() -> None:
    stats = LatencyStats(window=100_000)

    def worker() -> None:
        for index in range(1_000):
            tracer = Tracer(on_event=stats)
            with tracer.stage("dense_search"):
                pass
            tracer.finish()
            stats.record("custom", index)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    report = stats.report()
    assert report["dense_search"]["count"] == report["total"]["count"] == report["custom"]["count"] == 8_000


def test_print_report_and_slowest_show_tags() -> None:
    stats = LatencyStats()
    for ms in KNOWN_MS:
        stats.record("generate", ms, meta={"cache_hit": ms < 50}, query_tokens=ms // 10)
    buffer = io.StringIO()
    stats.print_report(file=buffer)
    table = buffer.getvalue()
    assert "generate" in table and "p95 ms" in table and "1197.75" in table
    slowest = stats.slowest("generate", 2)
    assert [s.duration_ms for s in slowest] == [2100, 95]
    assert slowest[0].tags["query_tokens"] == 210 and slowest[0].meta["cache_hit"] is False
    assert LatencyStats().format_report() == "No latency samples recorded."


# ---------------------------------------------------------------------------------- tracer


def test_nested_stages_record_their_parent_and_are_not_double_counted() -> None:
    clock = FakeClock()
    tracer = Tracer(clock=clock)
    with active_tracer(tracer):
        with tracer.stage("retrieve"):
            with trace_stage("query_embedding"):
                clock.advance_ms(30)
            with trace_stage("dense_search"):
                clock.advance_ms(20)
        with trace_stage("generate"):
            clock.advance_ms(100)
    events = {event.stage: event for event in tracer.events}
    assert events["query_embedding"].parent == "retrieve" and events["retrieve"].parent is None
    assert events["retrieve"].latency_ms == pytest.approx(50)
    assert tracer.total_latency_ms() == pytest.approx(150)  # retrieve + generate, children not added again
    total = tracer.finish()
    assert total.stage == "total" and total.latency_ms == pytest.approx(150)
    assert tracer.finish() is total  # emitted once
    with trace_stage("outside_a_run") as meta:  # no active tracer: no-op
        meta["ok"] = True


def test_cold_start_is_tagged_after_idle_time() -> None:
    stats = LatencyStats(cold_start_after_s=60)

    def run(started_at: float) -> None:
        stats.observe(TraceEvent("generate", 10.0, run_id=str(started_at), started_at=started_at))
        stats.observe(TraceEvent("total", 12.0, run_id=str(started_at), started_at=started_at))

    for started in (0.0, 5.0, 10.0, 200.0, 205.0):
        run(started)
    cold = [s.tags["is_cold_start"] for s in stats.samples("generate")]
    assert cold == [True, False, False, True, False]


# ------------------------------------------------------------------------ pipeline stages


def _rag(**kwargs: Any) -> RAG:
    return RAG(CORPUS, offline=True, **kwargs)


def test_pipeline_emits_the_fine_stages_with_tags() -> None:
    stats = LatencyStats()
    rag = _rag(reranker="llm", rewrite_query=True, on_event=[stats])
    result = rag.query("When do Acme offices close?", filters={"chunk_index": 1})
    stages = [event.stage for event in result.trace_events]
    for stage in ("query_rewrite", "query_embedding", "dense_search", "filtering", "bm25_search", "fusion",
                  "rerank", "prompt_build", "generate", "post_process", "total"):
        assert stage in stages, stage
    assert stages[-1] == "total" and stages.count("generate") == 1
    parents = {event.stage: event.parent for event in result.trace_events}
    assert parents["dense_search"] == "retrieve" and parents["filtering"] == "dense_search"
    assert parents["rerank"] == "process_documents" and parents["generate"] is None
    total = stats.samples("total")[0]
    assert total.tags["reranker_used"] is True and total.tags["num_candidates"] == 2
    # Caller filters make the query personalized: the LLM cache is bypassed (no lookup at all).
    assert total.tags["query_tokens"] > 0 and total.tags["cache_hit"] == {"embedding": False}
    assert result.usage["wall_latency_ms"] >= result.usage["latency_ms"] >= 0


def test_repeated_query_is_tagged_as_cache_hit() -> None:
    stats = LatencyStats()
    rag = _rag(on_event=stats)
    rag.query("When does the Orion launch window open?")
    rag.query("When does the Orion launch window open?")
    first, second = stats.samples("total")
    assert first.tags["cache_hit"] == {"embedding": False, "llm": False}
    assert second.tags["cache_hit"] == {"embedding": True, "llm": True}
    assert [s.meta.get("cache_hit") for s in stats.samples("generate")] == [False, True]
    assert stats.samples("total")[0].tags["reranker_used"] is False


def test_streaming_and_failed_runs_are_traced() -> None:
    stats = LatencyStats()
    rag = _rag(on_event=stats)
    "".join(rag.stream("When do Acme offices close?"))
    assert {"prompt_build", "generate", "post_process", "total"} <= set(stats.stages())

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("vector store down")

    rag.retriever.retrieve = boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        rag.query("When do Acme offices close?")
    failed = stats.samples("total")[-1]
    assert failed.error and "vector store down" in failed.error


def test_benchmark_runs_every_query_and_restores_on_event() -> None:
    seen: List[str] = []
    rag = _rag(on_event=lambda event: seen.append(event.stage))
    stats = benchmark(rag, ["When do Acme offices close?", "When does Orion open?"], repeats=3, warmup=2)
    assert stats.report()["total"]["count"] == 6  # warm-up runs are not recorded
    assert all(not s.tags["cache_hit"].get("llm") for s in stats.samples("total"))  # cold cache by default
    assert rag.pipeline.on_event is not stats and seen.count("total") == 8  # original callback kept firing
    warm = benchmark(rag, ["When do Acme offices close?"], repeats=2, warmup=1, warm_cache=True)
    assert all(s.tags["cache_hit"]["llm"] for s in warm.samples("total"))


# -------------------------------------------------------------------------------- exporters


class _FakeHistogram:
    instances: List["_FakeHistogram"] = []

    def __init__(self, name: str, doc: str, labels: List[str], buckets: Any = None, registry: Any = None) -> None:
        self.name, self.labelnames, self.observed = name, labels, []
        _FakeHistogram.instances.append(self)

    def labels(self, **labels: str) -> "_FakeHistogram":
        self._labels = labels
        return self

    def observe(self, value: float) -> None:
        self.observed.append((dict(self._labels), value))


def test_prometheus_exporter_observes_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    from raglib.exporters import PrometheusExporter

    monkeypatch.setitem(sys.modules, "prometheus_client", types.SimpleNamespace(Histogram=_FakeHistogram))
    exporter = PrometheusExporter()
    exporter(TraceEvent("rerank", 250.0))
    exporter(TraceEvent("generate", 1500.0, error="TimeoutError"))
    assert exporter.histogram.observed == [
        ({"stage": "rerank", "status": "ok"}, 0.25),
        ({"stage": "generate", "status": "error"}, 1.5),
    ]


def test_exporters_explain_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    from raglib.exporters import OpenTelemetryExporter, PrometheusExporter

    monkeypatch.setitem(sys.modules, "prometheus_client", None)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    with pytest.raises(ImportError, match=r"raglib-py\[prometheus\]"):
        PrometheusExporter()
    with pytest.raises(ImportError, match=r"raglib-py\[otel\]"):
        OpenTelemetryExporter()


class _FakeSpan:
    def __init__(self, name: str, parent: Optional["_FakeSpan"], start_time: int, attributes: Dict[str, Any]) -> None:
        self.name, self.parent, self.start_time, self.attributes = name, parent, start_time, attributes
        self.end_time: Optional[int] = None
        self.status: Any = None

    def set_status(self, status: Any) -> None:
        self.status = status

    def end(self, end_time: int) -> None:
        self.end_time = end_time


class _FakeOtelTracer:
    def __init__(self) -> None:
        self.spans: List[_FakeSpan] = []

    def start_span(self, name: str, context: Any = None, start_time: int = 0, attributes: Any = None) -> _FakeSpan:
        span = _FakeSpan(name, context, start_time, dict(attributes or {}))
        self.spans.append(span)
        return span


def _install_fake_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_module = types.ModuleType("opentelemetry.trace")
    trace_module.get_tracer = lambda name: _FakeOtelTracer()  # type: ignore[attr-defined]
    trace_module.set_span_in_context = lambda span: span  # type: ignore[attr-defined]
    trace_module.Status = lambda code, description: (code, description)  # type: ignore[attr-defined]
    trace_module.StatusCode = types.SimpleNamespace(ERROR="ERROR")  # type: ignore[attr-defined]
    package = types.ModuleType("opentelemetry")
    package.trace = trace_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry", package)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_module)


def test_opentelemetry_exporter_builds_a_nested_trace_per_query(monkeypatch: pytest.MonkeyPatch) -> None:
    from raglib.exporters import OpenTelemetryExporter

    _install_fake_otel(monkeypatch)
    exporter = OpenTelemetryExporter()
    clock, wall = FakeClock(), FakeClock(start=1_700_000_000.0)

    def tick() -> float:  # wall clock follows the monotonic clock
        return wall.now + (clock.now - 1_000.0)

    tracer = Tracer(on_event=exporter, clock=clock, wall_clock=tick)
    with active_tracer(tracer):
        with tracer.stage("retrieve"):
            with trace_stage("dense_search", hits=3):
                clock.advance_ms(40)
        with trace_stage("generate"):
            clock.advance_ms(200)
    tracer.finish(pipeline="advanced")
    spans = {span.name: span for span in exporter.tracer.spans}
    assert set(spans) == {"raglib.query", "raglib.retrieve", "raglib.dense_search", "raglib.generate"}
    assert spans["raglib.dense_search"].parent is spans["raglib.retrieve"]
    assert spans["raglib.retrieve"].parent is spans["raglib.query"] and spans["raglib.query"].parent is None
    assert spans["raglib.dense_search"].attributes["raglib.hits"] == 3
    generate = spans["raglib.generate"]
    assert generate.end_time - generate.start_time == 200_000_000
    assert spans["raglib.query"].end_time - spans["raglib.query"].start_time == 240_000_000
