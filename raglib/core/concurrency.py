"""Concurrency primitives."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterator, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ReadWriteLock:
    """Many concurrent readers OR one writer (writer-preferring, so ingestion can't starve).

    Queries take the read side, so concurrent chat sessions don't block each other;
    ``add`` / ``delete`` take the write side for the short moment they mutate indexes.
    Re-entrant for the writer thread (a writer may also read).
    """

    def __init__(self) -> None:
        """Create an unlocked lock."""

        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer: int = 0  # thread id of the active writer
        self._writer_depth = 0
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        """Shared access."""

        me = threading.get_ident()
        with self._cond:
            if self._writer == me:  # writer reading its own state
                self._writer_depth += 1
                reentrant = True
            else:
                reentrant = False
                while self._writer or self._waiting_writers:
                    self._cond.wait()
                self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                if reentrant:
                    self._writer_depth -= 1
                else:
                    self._readers -= 1
                    if self._readers == 0:
                        self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        """Exclusive access."""

        me = threading.get_ident()
        with self._cond:
            if self._writer == me:
                self._writer_depth += 1
            else:
                self._waiting_writers += 1
                while self._writer or self._readers:
                    self._cond.wait()
                self._waiting_writers -= 1
                self._writer = me
                self._writer_depth = 1
        try:
            yield
        finally:
            with self._cond:
                self._writer_depth -= 1
                if self._writer_depth == 0:
                    self._writer = 0
                    self._cond.notify_all()


# Browsers (Pyodide) and WASI runtimes cannot start threads.
THREADS_AVAILABLE = sys.platform not in ("emscripten", "wasi")


def parallel_map(fn: Callable[[T], R], items: Sequence[T], max_workers: int) -> List[R]:
    """``[fn(x) for x in items]``, run on a thread pool when that helps and is possible.

    Falls back to sequential execution when threads are unavailable (Pyodide / WASI) or
    cannot be started, so raglib also runs inside a browser.
    """

    items = list(items)
    workers = min(max_workers, len(items))
    if workers <= 1 or not THREADS_AVAILABLE:
        return [fn(item) for item in items]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))
    except RuntimeError as exc:
        if "thread" not in str(exc).lower():
            raise
        return [fn(item) for item in items]
