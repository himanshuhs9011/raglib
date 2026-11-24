"""Tier 1 input guards in parallel with retrieval, with cancellation.

``RAG.query`` starts the query's parallel input guards (``Guard.parallel = True``, tier >= 1) in
worker threads and runs retrieval at the same time. ``ParallelInputGuards`` is the query's *gate*
(``raglib.core.runtime.guard_gate``):

* every traced stage ``poll()``s it at its start: once a guard has blocked, the next stage
  (embedding, dense search, rerank, a tool call ...) raises ``GuardCancelled`` instead of running;
* every LLM call ``join()``s it first: it waits for all parallel guards, so a blocked question
  never reaches a model - not even the agent's planner or the query rewriter;
* generation, the "nothing relevant" exit and ``_finish`` join it too.

``GuardCancelled`` derives from ``BaseException`` so that components which catch ``Exception``
to degrade gracefully (reranker, router, ...) cannot swallow it; ``RAG`` turns it into the
guard's blocked answer.

Trade-off: a question that gets blocked may already have paid for retrieval work that started
before the guard finished (embedding the query, a vector search) - never for an LLM call.
"""

from __future__ import annotations

import contextvars
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List, Optional, Sequence, Tuple

from raglib.guards.base import BLOCK, Decision, Guard, GuardContext, GuardRunner, guard_sort_key

_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="raglib-guard")


class GuardCancelled(BaseException):
    """A parallel input guard blocked the query; carries its decision."""

    def __init__(self, decision: Decision):
        super().__init__(decision.message or decision.guard)
        self.decision = decision


class ParallelInputGuards:
    """Run ``guards`` on ``value`` in worker threads; acts as the query's guard gate."""

    def __init__(self, runner: GuardRunner, guards: Sequence[Guard], value: Any, context: GuardContext):
        self.guards = sorted(guards, key=guard_sort_key)
        self._lock = threading.Lock()
        self._blocked: Optional[Decision] = None
        self._joined = False
        self._decisions: List[Decision] = []
        self.futures: List[Tuple[Guard, Future]] = []
        for guard in self.guards:
            snapshot = contextvars.copy_context()  # the run's tracer and usage meter (not the guard scope: set later)
            self.futures.append((guard, _EXECUTOR.submit(snapshot.run, self._work, runner, guard, value, context)))

    @staticmethod
    def _work(runner: GuardRunner, guard: Guard, value: Any, context: GuardContext) -> Decision:
        from raglib.core.runtime import _GUARD_GATE

        _GUARD_GATE.set(None)  # a guard's own model call must not wait for the guards
        return runner._check(guard, "input", value, context)

    def poll(self) -> None:
        """Raise ``GuardCancelled`` if a finished guard blocked (never waits)."""

        with self._lock:
            if self._blocked is None:
                for guard, future in self.futures:
                    if future.done() and future.exception() is None and future.result().action == BLOCK:
                        self._blocked = future.result()
                        break
            blocked = self._blocked
        if blocked is not None:
            raise GuardCancelled(blocked)

    def join(self) -> None:
        """Wait for every guard; raise ``GuardCancelled`` with the first block (in guard order)."""

        with self._lock:
            if not self._joined:
                for guard, future in self.futures:
                    decision = future.result()  # run_guarded applied the failure policy: never raises
                    self._decisions.append(decision)
                    if self._blocked is None and decision.action == BLOCK:
                        self._blocked = decision
                self._joined = True
            blocked = self._blocked
        if blocked is not None:
            raise GuardCancelled(blocked)

    def decisions(self) -> List[Decision]:
        """The guards' decisions (after ``join``), in guard order."""

        with self._lock:
            return list(self._decisions)
