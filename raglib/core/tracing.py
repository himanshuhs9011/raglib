"""Structured per-stage tracing for RAG pipelines."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from raglib.schemas import TraceEvent

logger = logging.getLogger(__name__)

EventCallback = Callable[[TraceEvent], None]


class Tracer:
    """Collects TraceEvents for one pipeline run and forwards them to an optional callback."""

    def __init__(self, on_event: Optional[EventCallback] = None):
        """Initialize an empty event list with an optional observer callback."""

        self.events: List[TraceEvent] = []
        self._on_event = on_event

    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[Dict[str, Any]]:
        """Time a pipeline stage; callers may add fields to the yielded meta dict."""

        details: Dict[str, Any] = dict(meta)
        started = time.perf_counter()
        error: Optional[str] = None
        try:
            yield details
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event = TraceEvent(
                stage=name,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                meta=details,
                error=error,
            )
            self.events.append(event)
            if self._on_event is not None:
                try:
                    self._on_event(event)
                except Exception as exc:  # noqa: BLE001 - observers must never break the pipeline
                    logger.warning("Trace callback failed: %s", exc)

    def total_latency_ms(self) -> float:
        """Return the summed latency of all recorded stages."""

        return round(sum(event.latency_ms for event in self.events), 2)
