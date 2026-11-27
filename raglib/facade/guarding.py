"""Guardrails on the facade: running input, retrieval and output guards around a query or a stream, and recording their decisions."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from raglib.access import UserContext
from raglib.cache import cache_bypass
from raglib.core.runtime import UsageMeter, guard_gate, usage_scope
from raglib.core.tracing import Tracer, active_tracer
from raglib.guards import (
    GuardContext,
    GuardScope,
    guard_scope,
)
from raglib.guards.parallel import GuardCancelled, ParallelInputGuards
from raglib.schemas import GenerationResult
from raglib.vectorstores.base_store import MetadataFilter

logger = logging.getLogger("raglib.facade")  # one logger name for the whole facade, as before the split



def _restoring(tokens: Iterator[str], restorer: Any) -> Iterator[str]:
    """Yield ``tokens`` with placeholders restored; returns the wrapped generator's return value."""

    while True:
        try:
            token = next(tokens)
        except StopIteration as stop:
            tail = restorer.flush()
            if tail:
                yield tail
            return stop.value
        piece = restorer.feed(token)
        if piece:
            yield piece


class GuardingMixin:
    """Guardrails on the facade: running input, retrieval and output guards around a query or a stream, and recording their decisions (mixin of ``RAG``)."""

    def _guarded(
        self,
        question: str,
        session_id: Optional[str],
        filters: Optional[MetadataFilter],
        user: Any,
        personalized: bool,
        session_key: Optional[str],
        user_id: Optional[str] = None,
        request: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[GenerationResult, str, Dict[str, str]]:
        """Run the input guards, then the pipeline with retrieval / output / tool guards in scope.

        One trace and one usage meter cover the guards and the pipeline, so guard stages show up
        in ``result.trace_events``, ``on_event``, ``LatencyStats`` and ``CostReport``. Returns the
        result (masked), the masked question and the request's PII placeholder map.
        """

        guards = self._guards
        assert guards is not None
        tracer = Tracer(on_event=self._rag.on_event)
        context = self._guard_context(question, session_id, filters, user, user_id, request)
        with usage_scope() as meter, active_tracer(tracer):
            question, scope, blocked, mapping, pending = self._run_input_guards(question, context)
            if blocked is not None:
                result = self._input_blocked(blocked, tracer, meter)
            else:
                try:
                    with guard_scope(scope), guard_gate(pending), cache_bypass(personalized and not self._cache_personalized):
                        result = self._answer(question, session_key, filters, personalized, tracer=tracer, meter=meter)
                        if pending is not None:
                            pending.join()  # e.g. an answer-cache hit made no LLM call that waited
                except GuardCancelled as cancelled:
                    self._merge_parallel(scope, pending)  # every guard's stage and usage before the trace ends
                    result = self._input_blocked(cancelled.decision, tracer, meter, cancelled=True)
                finally:
                    self._merge_parallel(scope, pending)
                if result.metadata.get("answer_cache_hit") and tracer._finished is None:
                    self._finish_faq(result, tracer, meter)
        self._after_guards(result, scope)
        return result, question, mapping

    def _run_input_guards(
        self, question: str, context: GuardContext
    ) -> Tuple[str, GuardScope, Any, Dict[str, str], Optional[ParallelInputGuards]]:
        """Input guards on the raw question; the PII map is taken out of the decisions right away.

        Tier 0 rules and non-parallel tier 1 guards (``TopicGuard``) run here, in order. Parallel
        tier 1 guards (classifiers) are started on the masked question and returned as the query's
        gate: they run while retrieval does, and every LLM call waits for them.
        """

        guards = self._guards
        assert guards is not None
        sequential = [guard for guard in guards.input if not (guard.tier >= 1 and getattr(guard, "parallel", False))]
        parallel = [guard for guard in guards.input if guard not in sequential]
        masked, decisions, blocked = guards.runner.run("input", sequential, question, context)
        mapping: Dict[str, str] = {}
        for decision in decisions:
            if decision.private:
                mapping.update(decision.private.get("pii_map", {}))
                decision.private = None
        masked_context = replace(context, question=masked)
        scope = GuardScope(
            runner=guards.runner, context=masked_context, retrieval_guards=guards.retrieval,
            output_guards=guards.output, tool_policy=guards.tool_policy, decisions=list(decisions),
            stream_holdback=self._stream_holdback,
        )
        pending = None
        if blocked is None and parallel:
            pending = ParallelInputGuards(guards.runner, parallel, masked, masked_context)
        return masked, scope, blocked, mapping, pending

    @staticmethod
    def _merge_parallel(scope: GuardScope, pending: Optional[ParallelInputGuards]) -> None:
        """Wait for the parallel input guards; put their decisions after the sequential input ones (once)."""

        if pending is None or getattr(pending, "_merged", False):
            return
        pending._merged = True
        try:
            pending.join()
        except GuardCancelled:
            pass
        position = sum(1 for decision in scope.decisions if decision.stage == "input")
        scope.decisions[position:position] = pending.decisions()

    def _after_guards(self, result: GenerationResult, scope: GuardScope) -> None:
        self._attach_guards(result, scope)
        guards = self._guards
        assert guards is not None
        for guard in guards.all():
            try:
                guard.after(result, scope.context)
            except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail the answer
                logger.warning("Guard %s after-hook failed: %s", guard.name, type(exc).__name__)

    def _guard_context(
        self,
        question: str,
        session_id: Optional[str],
        filters: Optional[MetadataFilter],
        user: Any,
        user_id: Optional[str] = None,
        request: Optional[Mapping[str, Any]] = None,
    ) -> GuardContext:
        """Who is asking. ``user_key`` (rate limits, budgets): ``rate_limit_key(context)``, else the user
        id (``user_id=`` or ``UserContext.user_id``), else ``None`` (anonymous)."""

        context_user = UserContext.coerce(user)
        uid = user_id or (context_user.user_id if context_user is not None else None)
        context = GuardContext(
            stage="input", question=question, user=context_user, user_id=uid, session_id=session_id,
            request=dict(request or {}), filters=filters,
            high_risk="tool calls" if self._rag_type == "agentic" else None,
            permission_filter=self.permission_filter(context_user) if self._access is not None else None,
        )
        if self._rate_limit_key is not None:
            key = self._rate_limit_key(replace(context, question=""))  # identity, never the question text
        else:
            key = f"user:{uid}" if uid else None
        context.user_key = str(key) if key else None
        return context

    def _input_blocked(self, decision: Any, tracer: Tracer, meter: UsageMeter, cancelled: bool = False) -> GenerationResult:
        """The result for a question an input guard refused: its message, no LLM call.

        ``cancelled``: a parallel guard blocked while retrieval was running; the stages that had
        already finished stay in the trace, the rest never started.
        """

        tracer.tag("blocked_by", decision.guard)
        if cancelled:
            tracer.tag("retrieval_cancelled", True)
        costs: Dict[str, Any] = {}
        if self._prices is not None:
            from raglib.costs import price_run

            costs = price_run(self._prices, meter.breakdown(), tracer.events)
        total = tracer.finish(pipeline=self._rag_type, docs=0, **costs)
        result = GenerationResult(answer=decision.message or "", sources=[], reasoning_trace=[f"blocked_by:{decision.guard}"])
        result.trace_events = list(tracer.events)
        result.usage = {
            **meter.as_dict(),
            "usage_by_model": meter.breakdown(),
            "latency_ms": tracer.total_latency_ms(),
            "wall_latency_ms": total.latency_ms,
            **costs,
        }
        result.metadata.update({"pipeline": self._rag_type, "blocked_by": decision.guard})
        return result

    @staticmethod
    def _attach_guards(result: GenerationResult, scope: GuardScope) -> None:
        """``result.metadata["guards"]``: every decision (no checked values); ``blocked_by`` if any."""

        if scope.decisions:  # quiet flag-only guards that found nothing leave no record
            result.metadata["guards"] = [decision.summary() for decision in scope.decisions]
        result.guard_flags = [decision.summary() for decision in scope.decisions if decision.flag]
        blocking = next((d for d in scope.decisions if d.action == "block"), None)
        if blocking is not None:
            result.metadata["blocked_by"] = blocking.guard
            result.metadata["block_reason"] = blocking.reason

    def _guarded_stream(
        self,
        question: str,
        session_id: Optional[str],
        filters: Optional[MetadataFilter],
        user: Any,
        bypass: bool,
        session_key: Optional[str],
        user_id: Optional[str] = None,
        request: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[str]:
        """``stream`` with guards: input guards first, then the pipeline; tokens are restored for the caller."""

        from raglib.guards.pii import PlaceholderRestorer

        guards = self._guards
        assert guards is not None
        tracer = Tracer(on_event=self._rag.on_event)
        context = self._guard_context(question, session_id, filters, user, user_id, request)
        with usage_scope() as meter, active_tracer(tracer):
            question, scope, blocked, mapping, pending = self._run_input_guards(question, context)
            if blocked is not None:
                result = self._input_blocked(blocked, tracer, meter)
                yield result.answer
            else:
                cancelled = None
                try:
                    with guard_scope(scope), guard_gate(pending), cache_bypass(bypass):
                        tokens = self._rag.stream(question, session_id=session_key, filters=filters, tracer=tracer, meter=meter)
                        if mapping:
                            result = yield from _restoring(tokens, PlaceholderRestorer(mapping))
                        else:
                            result = yield from tokens
                        if pending is not None:
                            pending.join()
                except GuardCancelled as stop:
                    cancelled = stop
                finally:
                    self._merge_parallel(scope, pending)
                if cancelled is not None:  # no token was shown: generation waits for the guards
                    result = self._input_blocked(cancelled.decision, tracer, meter, cancelled=True)
                    yield result.answer
        if result is not None:
            self._after_guards(result, scope)
        return result, question, mapping
