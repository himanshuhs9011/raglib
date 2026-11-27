"""Querying on the facade: query, aquery and stream (with tier routing and the answer cache), chat, conversation memory and saving results."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Union

from raglib.access import UserContext
from raglib.cache import cache_bypass
from raglib.core.runtime import UsageMeter
from raglib.core.tracing import Tracer, run_tags
from raglib.facade.extras import _accepts_keyword
from raglib.routing import TIER_CHEAP, TIER_FAQ, TIER_FULL, decide
from raglib.schemas import GenerationResult
from raglib.vectorstores.base_store import MetadataFilter

logger = logging.getLogger("raglib.facade")  # one logger name for the whole facade, as before the split



def _restored(result: GenerationResult, mapping: Mapping[str, str]) -> GenerationResult:
    """A copy of ``result`` with the caller's PII put back into the answer (the original - which
    caches, memory and ``output_dir`` hold - keeps the placeholders)."""

    if not mapping:
        return result
    from raglib.guards.pii import restore

    return replace(result, answer=restore(result.answer, mapping), metadata=dict(result.metadata))


class QueryMixin:
    """Querying on the facade: query, aquery and stream (with tier routing and the answer cache), chat, conversation memory and saving results (mixin of ``RAG``)."""

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
        user: Union[UserContext, Mapping[str, Any], None] = None,
        user_id: Optional[str] = None,
        request: Optional[Mapping[str, Any]] = None,
    ) -> GenerationResult:
        """Answer one question. ``session_id`` isolates conversation memory per user.

        With ``access_control`` on, ``user`` (a ``UserContext`` or a dict with ``user_id``,
        ``roles``, ``departments``, ``max_access_level``) decides which documents are visible;
        ``filters`` can only narrow that further.

        ``user_id`` identifies the caller for rate limits and budgets without access control;
        ``request`` (e.g. ``{"ip": ..., "api_key_id": ..., "tenant": ...}``) is handed to guards and
        to ``RAG(rate_limit_key=...)``. Both are ignored when no guards are configured.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        question = question.strip()
        effective_filters, session_key = self._scope(filters, user, session_id)
        personalized = self._personalized(filters, user)
        mapping: Dict[str, str] = {}
        if self._guards is None:
            with cache_bypass(personalized and not self._cache_personalized):
                result = self._answer(question, session_key, effective_filters, personalized)
        else:
            result, question, mapping = self._guarded(
                question, session_id, effective_filters, user, personalized, session_key, user_id, request
            )
        result.metadata["session_id"] = session_id
        if self._output_dir:
            self._save_result(question, result)  # masked: placeholders, never the restored values
        return _restored(result, mapping)

    def _personalized(self, filters: Optional[MetadataFilter], user: Any) -> bool:
        """True when the answer may depend on who asks (a user context or caller filters)."""

        return user is not None or bool(filters)

    def _answer_cache_key(self) -> str:
        """The answer cache's pipeline key: the rag type plus the guard configuration hash."""

        return self._rag_type if self._guard_signature is None else f"{self._rag_type}|guards:{self._guard_signature}"

    def _answer(
        self,
        question: str,
        session_key: Optional[str],
        filters: Optional[MetadataFilter],
        personalized: bool,
        tracer: Optional[Tracer] = None,
        meter: Optional[UsageMeter] = None,
    ) -> GenerationResult:
        """Route to the FAQ / cheap / full tier and run it (``tracer`` / ``meter``: continue the guards' run)."""

        run_kwargs: Dict[str, Any] = {"tracer": tracer, "meter": meter} if tracer is not None else {}
        if self._tier_router is None:
            return self._rag.run(question, session_id=session_key, filters=filters, **run_kwargs)
        cache = self._answer_cache
        # Answers that depend on who asks, or on the conversation so far, are never served from cache.
        cacheable = cache is not None and (self._cache_personalized or not personalized) and not self._rag.use_memory
        scope = json.dumps(filters, sort_keys=True, default=str) if filters else None
        cache_key = self._answer_cache_key()
        cached = cache.get(question, scope, cache_key) if cacheable and cache is not None else None
        context = {
            "answer_cached": cached is not None,
            "cheap_available": self._cheap_rag is not None,
            "personalized": personalized,
        }
        decision = decide(self._tier_router, question, context)
        logger.info("tier=%s (%s)", decision.tier, decision.reason)
        if decision.tier == TIER_FAQ and cached is not None:
            return self._faq_result(cached, decision.reason, tracer=tracer)
        tier = TIER_CHEAP if decision.tier in (TIER_FAQ, TIER_CHEAP) and self._cheap_rag is not None else TIER_FULL
        pipeline = self._cheap_rag if tier == TIER_CHEAP else self._rag
        if tracer is not None:
            tracer.tag("tier", tier)
        generation = getattr(cache, "generation", None) if cache is not None else None  # before retrieval starts
        with run_tags(tier=tier):
            result = pipeline.run(question, session_id=session_key, filters=filters, **run_kwargs)
        result.metadata.update({"tier": tier, "tier_reason": decision.reason, "answer_cache_hit": False})
        if cacheable and cache is not None and not result.metadata.get("blocked_by"):
            # A guard's refusal is not cached: new documents may answer the question next time.
            if generation is not None and _accepts_keyword(cache.put, "generation"):
                cache.put(question, result, scope, cache_key, generation=generation)
            else:  # an AnswerCache subclass written before put(generation=) existed
                cache.put(question, result, scope, cache_key)
        return result

    def _faq_result(self, cached: GenerationResult, reason: str, tracer: Optional[Tracer] = None) -> GenerationResult:
        """Return a cached answer, traced as its own short run (tier ``faq``).

        The cached answer already passed the output guards when it was generated (the answer-cache
        key includes the guard configuration hash, so changed guards never reuse it). In a guarded
        query (``tracer`` given) the trace is finished by ``_guarded`` once every guard is done, so
        the usage and cost they caused are reported (``_finish_faq``).
        """

        cached.metadata.update({"tier": TIER_FAQ, "tier_reason": reason, "answer_cache_hit": True})
        own = tracer is None
        tracer = tracer or Tracer(on_event=self._rag.on_event)
        tracer.tag("tier", TIER_FAQ)
        with tracer.stage("answer_cache"):
            pass
        if own:
            self._finish_faq(cached, tracer, None)
        return cached

    def _finish_faq(self, cached: GenerationResult, tracer: Tracer, meter: Optional[UsageMeter]) -> None:
        """End an answer-cache hit's trace with the usage actually spent (guard calls, no answer call)."""

        costs: Dict[str, Any] = {}
        if self._prices is not None:
            from raglib.costs import price_run

            costs = price_run(self._prices, meter.breakdown() if meter is not None else {}, tracer.events)
        total = tracer.finish(pipeline=self._rag_type, cache_hit={"answer": True}, **costs)
        cached.trace_events = list(tracer.events)
        usage = meter.as_dict() if meter is not None else {"llm_calls": 0, "total_tokens": 0}
        if meter is not None:
            usage["usage_by_model"] = meter.breakdown()
        cached.usage = {**usage, "latency_ms": tracer.total_latency_ms(), "wall_latency_ms": total.latency_ms, **costs}
        if self._prices is not None:
            cached.usage.setdefault("cost_usd", 0.0)

    async def aquery(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
        user: Union[UserContext, Mapping[str, Any], None] = None,
        user_id: Optional[str] = None,
        request: Optional[Mapping[str, Any]] = None,
    ) -> GenerationResult:
        """Async ``query`` (runs the pipeline in a worker thread)."""

        import asyncio

        return await asyncio.to_thread(self.query, question, session_id, filters, user, user_id, request)

    def stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
        user: Union[UserContext, Mapping[str, Any], None] = None,
        user_id: Optional[str] = None,
        request: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[str]:
        """Stream answer tokens: ``for token in rag.stream(q): print(token, end="")``.

        ``user_id`` / ``request``: as in ``query``. With output guards the answer streams in
        checked pieces (see ``raglib.guards.streaming``); a non-streaming-safe output guard
        (``JSONSchemaGuard``) makes it arrive in one piece.

        Each step runs in the stream's own context, so iterating it from a web framework that
        uses a fresh context per step, or interleaving it with other queries, is safe.
        """

        from raglib.core.runtime import isolated

        return (yield from isolated(self._stream_steps(question, session_id, filters, user, user_id, request)))

    def _stream_steps(
        self,
        question: str,
        session_id: Optional[str],
        filters: Optional[MetadataFilter],
        user: Union[UserContext, Mapping[str, Any], None],
        user_id: Optional[str],
        request: Optional[Mapping[str, Any]],
    ) -> Iterator[str]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        question = question.strip()
        effective_filters, session_key = self._scope(filters, user, session_id)
        bypass = self._personalized(filters, user) and not self._cache_personalized
        mapping: Dict[str, str] = {}
        if self._guards is None:
            # Streaming always runs the full pipeline (tier routing and the answer cache are query() features).
            with cache_bypass(bypass):
                result = yield from self._rag.stream(question, session_id=session_key, filters=effective_filters)
        else:
            result, question, mapping = yield from self._guarded_stream(
                question, session_id, effective_filters, user, bypass, session_key, user_id, request
            )
        if result is not None:
            result.metadata["session_id"] = session_id
        if self._output_dir and result is not None:
            self._save_result(question, result)  # masked
        return _restored(result, mapping) if result is not None else None

    def chat(self, session_id: str = "interactive") -> None:
        """Open an interactive terminal Q&A loop with conversation memory."""

        from raglib.session.interactive import InteractiveSession

        previous = self._rag.use_memory
        self._rag.use_memory = True
        try:
            InteractiveSession(rag=self, session_id=session_id).start()
        finally:
            self._rag.use_memory = previous  # later rag.query() calls keep their own setting

    def reset_memory(
        self, session_id: Optional[str] = None, user: Union[UserContext, Mapping[str, Any], None] = None
    ) -> None:
        """Clear conversation memory for one session (or all sessions).

        With access control on, sessions are kept per user: pass the same ``user`` as to ``query``.
        """

        if self._access is not None and (session_id is not None or user is not None):
            session_id = self._scope(None, user, session_id)[1]
        self._conversations.clear(session_id)

    def _save_result(self, question: str, result: GenerationResult) -> None:
        """Persist query output as JSON in output_dir."""

        output_path = Path(self._output_dir or ".")
        output_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rag_type": self._rag_type,
            "question": question,
            "answer": result.answer,
            "citations": [asdict(citation) for citation in result.citations],
            "reasoning_trace": result.reasoning_trace,
            "usage": result.usage,
            "sources": [
                {"id": document.id, "score": document.score, "source": document.source, "metadata": document.metadata}
                for document in result.sources
            ],
        }
        target = output_path / f"result_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
