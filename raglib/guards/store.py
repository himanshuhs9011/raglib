"""Where stateful guards (rate limits, budgets) keep their counters.

``InMemoryGuardStore`` (the default) keeps them in this process: each worker process, container
or replica has its own counters, and they reset on restart. For a limit shared across workers,
implement ``GuardStore`` on a shared backend (Redis, a database) and pass it as
``RAG(guard_store=...)`` or ``UserRateLimit(store=...)``. Each method must be atomic.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, Tuple


class GuardStore(ABC):
    """Counters for stateful guards. Keys are opaque strings (guard name + user key)."""

    @abstractmethod
    def hit(self, key: str, now: float, window: float, limit: int) -> Tuple[bool, int, float]:
        """Sliding window: record a request at ``now`` unless ``limit`` requests already fall in the last
        ``window`` seconds. Returns ``(allowed, requests in the window, seconds until a slot frees)``."""

    @abstractmethod
    def spent(self, key: str, day: str) -> float:
        """USD recorded for ``key`` on ``day``."""

    @abstractmethod
    def add_spend(self, key: str, day: str, amount: float) -> float:
        """Add ``amount`` USD for ``key`` on ``day``; returns the new total."""


class InMemoryGuardStore(GuardStore):
    """Thread-safe, per-process counters (bounded: idle keys and past days are dropped)."""

    MAX_KEYS = 100_000

    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = {}
        self._spend: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, now: float, window: float, limit: int) -> Tuple[bool, int, float]:
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= window:
                hits.popleft()
            if len(hits) >= limit:
                return False, len(hits), max(0.0, window - (now - hits[0]))
            hits.append(now)
            if len(self._hits) > self.MAX_KEYS:
                idle = [k for k, v in self._hits.items() if not v or now - v[-1] >= window]
                for stale in idle[: self.MAX_KEYS // 2]:
                    self._hits.pop(stale, None)
            return True, len(hits), 0.0

    def spent(self, key: str, day: str) -> float:
        with self._lock:
            return self._spend.get((key, day), 0.0)

    def add_spend(self, key: str, day: str, amount: float) -> float:
        with self._lock:
            for stale in [k for k in self._spend if k[1] != day]:
                del self._spend[stale]
            total = self._spend.get((key, day), 0.0) + float(amount)
            self._spend[(key, day)] = total
            return total
