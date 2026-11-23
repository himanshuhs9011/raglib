"""Shared RAG pipeline that every strategy builds on.

Pipeline (each step is an overridable method):

    start (load session history, condense follow-up into a standalone query)
      -> answer(ctx):
           retrieve(ctx)            # strategies mostly override this
           post_retrieve hook
           process_documents(ctx)   # rerank -> dedupe -> context reduce
           pre_generate hook
           generate(ctx, docs)      # grounded, cited generation
           post_generate hook
      -> finish (store memory turn, attach trace events + usage)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from typing import Generator as GeneratorType

from raglib.components.context_reducer import ContextReducer
from raglib.components.decision import (
    RETRIEVAL_HYBRID,
    RETRIEVAL_NONE,
    RETRIEVAL_VECTOR,
    RETRIEVAL_WEB,
    DecisionEngine,
)
from raglib.components.deduplicator import Deduplicator
from raglib.components.evaluator import Evaluator
from raglib.components.generator import Generator
from raglib.components.grader import RelevanceGrader
from raglib.components.hybrid_retriever import HybridRetriever
from raglib.components.memory import ConversationStore, MemoryModule
from raglib.components.multi_hop_retriever import MultiHopRetriever
from raglib.components.multi_query_retriever import MultiQueryRetriever
from raglib.components.planner import Planner
from raglib.components.query_expander import QueryExpander
from raglib.components.query_rewriter import QueryCondenser, QueryRewriter
from raglib.components.refiner import Refiner
from raglib.components.reflection import ReflectionModule
from raglib.components.reranker import Reranker
from raglib.components.retriever import Retriever
from raglib.components.router_retriever import RouterRetriever
from raglib.components.web_retriever import WebRetriever
from raglib.core.intents import is_greeting
from raglib.core.runtime import RunState, UsageMeter, run_scope, usage_scope
from raglib.core.text import count_tokens
from raglib.core.tracing import OnEvent, Tracer, active_tracer
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document, GenerationResult
from raglib.vectorstores.base_store import MetadataFilter

logger = logging.getLogger(__name__)

_INTERNAL_METADATA = {"graph_context_docs"}


class GuardBlockedError(Exception):
    """Raised inside a run when a retrieval guard blocks; ``run`` / ``stream`` catch it and return
    the guard's message. Its text is the guard's (sanitized) user-facing message, never the input."""

    def __init__(self, decision: Any):
        super().__init__(decision.message or decision.guard)
        self.decision = decision


GuardBlocked = GuardBlockedError  # short alias


@dataclass
class PipelineContext:
    """Mutable state for one pipeline run."""

    query: str
    search_query: str
    session_id: Optional[str]
    filters: Optional[MetadataFilter]
    tracer: Tracer
    history: str = ""
    trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage: Optional[UsageMeter] = None
    run_state: Optional[RunState] = None
    output_checked: bool = False  # the output guards already ran (piece by piece, while streaming)


class BaseRAG:
    """Defines the shared orchestration behaviour for all RAG strategies."""

    pipeline_name = "base"
    uses_decision_engine = True

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        web_retriever: Optional[WebRetriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        router_retriever: Optional[RouterRetriever] = None,
        multi_query_retriever: Optional[MultiQueryRetriever] = None,
        multi_hop_retriever: Optional[MultiHopRetriever] = None,
        reranker: Optional[Reranker] = None,
        evaluator: Optional[Evaluator] = None,
        refiner: Optional[Refiner] = None,
        decision_engine: Optional[DecisionEngine] = None,
        reflection_module: Optional[ReflectionModule] = None,
        memory_module: Optional[MemoryModule] = None,
        planner: Optional[Planner] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        query_expander: Optional[QueryExpander] = None,
        deduplicator: Optional[Deduplicator] = None,
        context_reducer: Optional[ContextReducer] = None,
        generator: Optional[Generator] = None,
        llm_client: Optional[BaseLLMClient] = None,
        max_retries: int = 2,
        conversation_store: Optional[ConversationStore] = None,
        condenser: Optional[QueryCondenser] = None,
        grader: Optional[RelevanceGrader] = None,
        graph_retriever: Optional[Any] = None,
        use_memory: bool = False,
        skip_retrieval_for_smalltalk: bool = True,
        on_event: OnEvent = None,
        strict: bool = False,
        rerank_candidates: Optional[int] = None,
        price_table: Optional[Any] = None,
        private_document: Optional[Callable[[Document], bool]] = None,
        allow_web_after_private: bool = False,
    ):
        """Initialize optional pipeline components via dependency injection.

        ``strict=True`` re-raises failures of optional steps instead of degrading
        (degradations are otherwise listed in ``result.metadata["degraded"]``).
        ``rerank_candidates``: how many chunks to retrieve for the reranker to cut down to its
        ``top_k`` (default ``max(4 * top_k, 20)``); ignored without a reranker.
        ``price_table`` (``raglib.costs.PriceTable``): adds ``cost_usd`` / ``cost_by_stage`` to usage.
        """

        self.price_table = price_table
        # Access control: which retrieved documents are non-public (web search is then off for the
        # query unless allow_web_after_private=True). Set by RAG(access_control=...).
        self.private_document = private_document
        self.allow_web_after_private = allow_web_after_private

        if rerank_candidates is not None and rerank_candidates <= 0:
            raise ValueError("rerank_candidates must be greater than zero")
        self.rerank_candidates = rerank_candidates

        self.retriever = retriever
        self.web_retriever = web_retriever
        self.hybrid_retriever = hybrid_retriever
        self.router_retriever = router_retriever
        self.multi_query_retriever = multi_query_retriever
        self.multi_hop_retriever = multi_hop_retriever
        self.reranker = reranker
        self.evaluator = evaluator
        self.refiner = refiner
        self.decision_engine = decision_engine
        self.reflection_module = reflection_module
        self.memory_module = memory_module
        self.planner = planner
        self.query_rewriter = query_rewriter
        self.query_expander = query_expander
        self.deduplicator = deduplicator
        self.context_reducer = context_reducer
        self.generator = generator
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.conversation_store = conversation_store
        self.condenser = condenser
        self.grader = grader
        self.graph_retriever = graph_retriever
        self.use_memory = use_memory
        self.skip_retrieval_for_smalltalk = skip_retrieval_for_smalltalk
        self.on_event = on_event
        self.strict = strict

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
        tracer: Optional[Tracer] = None,
        meter: Optional[UsageMeter] = None,
    ) -> GenerationResult:
        """Execute the pipeline for one query and return the answer with sources.

        ``tracer`` / ``meter`` continue a trace and usage count started by the caller (``RAG.query``
        passes them so its input guards are part of the same run).
        """

        tracer = tracer or Tracer(on_event=self.on_event)
        with run_scope(self.strict) as state, usage_scope(meter) as meter, active_tracer(tracer):
            try:
                ctx = self._start(query, session_id, filters, tracer=tracer)
                ctx.usage, ctx.run_state = meter, state
                try:
                    result = self.answer(ctx)
                except GuardBlockedError as blocked:
                    result = self._blocked_result(ctx, blocked.decision)
                return self._finish(ctx, result)
            except Exception as exc:
                tracer.finish(error=f"{type(exc).__name__}: {exc}", pipeline=self.pipeline_name)
                raise

    async def arun(
        self,
        query: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> GenerationResult:
        """Async variant of ``run`` (executes the pipeline in a worker thread)."""

        return await asyncio.to_thread(self.run, query, session_id, filters)

    def stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
        tracer: Optional[Tracer] = None,
        meter: Optional[UsageMeter] = None,
    ) -> GeneratorType[str, None, Optional[GenerationResult]]:
        """Yield answer tokens; the generator's return value is the final GenerationResult.

        Strategies that loop over generation (e.g. Self-RAG) yield the final answer once.
        Output guards: when all are ``streaming_safe`` (leak masking, citation access, truncation)
        the answer streams in checked pieces (``raglib.guards.streaming``: up to a sentence behind
        the model); any other output guard (JSON schema, ...) makes the answer arrive in one piece
        after it was checked.
        """

        if type(self).answer is not BaseRAG.answer or self.generator is None:
            result = self.run(query, session_id=session_id, filters=filters, tracer=tracer, meter=meter)
            yield result.answer
            return result

        from raglib.guards.base import current_scope
        from raglib.guards.streaming import StreamingOutput, stream_context, streaming_mode

        scope = current_scope()
        # Flag-only guards run once on the finished answer (in _finish), so they never hold the stream.
        enforcing = [guard for guard in scope.output_guards if not guard.flag_only] if scope is not None else []
        mode = streaming_mode(enforcing)
        buffered = mode == "buffered"
        tracer = tracer or Tracer(on_event=self.on_event)
        with run_scope(self.strict) as state, usage_scope(meter) as meter, active_tracer(tracer):
            try:
                ctx = self._start(query, session_id, filters, tracer=tracer)
                ctx.usage, ctx.run_state = meter, state
                documents = self._prepare_documents(ctx)
                try:
                    self._check_retrieval(ctx, documents)
                except GuardBlockedError as blocked:
                    result = self._finish(ctx, self._blocked_result(ctx, blocked.decision))
                    yield result.answer
                    return result
                # prompt_build is traced by the generator; "generate" times the streamed LLM reply.
                context_docs, tokens = self.generator.stream(ctx.query, documents, memory_context=ctx.history)
                monitor = (
                    StreamingOutput(
                        enforcing, stream_context(scope.context, ctx.query, context_docs), holdback=scope.stream_holdback
                    )
                    if mode == "incremental" and scope is not None
                    else None
                )
                with ctx.tracer.stage("generate", streaming=True) as meta, usage_scope() as generation:
                    parts: List[str] = []
                    for token in tokens:
                        parts.append(token)
                        if monitor is not None:
                            piece = monitor.feed(token)
                            if piece:
                                yield piece
                            if monitor.stopped:
                                close = getattr(tokens, "close", None)
                                if close is not None:
                                    close()  # stop reading (and paying for) the model's stream
                                break
                        elif not buffered:
                            yield token
                    if monitor is not None:
                        tail = monitor.close()
                        if tail:
                            yield tail
                    meta["docs"] = len(context_docs)
                answer = monitor.answer if monitor is not None else "".join(parts)
                result = self.generator.finalize(
                    answer, context_docs, ctx.trace + [f"generated_with_docs={len(context_docs)}"], meter=generation
                )
                if monitor is not None:
                    ctx.output_checked = True
                    scope.decisions.extend(monitor.decisions())  # type: ignore[union-attr]
                    if monitor.blocked is not None:
                        ctx.metadata["blocked_by"] = monitor.blocked.guard
                        ctx.tracer.tag("blocked_by", monitor.blocked.guard)
                        result.sources, result.citations = [], []
                result = self.post_generate(result)
                result = self._finish(ctx, result)
                if buffered:
                    yield result.answer
                return result
            except Exception as exc:
                tracer.finish(error=f"{type(exc).__name__}: {exc}", pipeline=self.pipeline_name)
                raise

    def answer(self, ctx: PipelineContext) -> GenerationResult:
        """Default pipeline: retrieve -> process -> generate."""

        documents = self._prepare_documents(ctx)
        result = self.generate(ctx, documents)
        return self.post_generate(result)

    def _prepare_documents(self, ctx: PipelineContext) -> List[Document]:
        """Run retrieval and document post-processing with hooks."""

        if ctx.metadata.get("smalltalk"):
            ctx.trace.append("retrieval_skipped:smalltalk")
            return []
        with ctx.tracer.stage("retrieve") as meta:
            documents = self.retrieve(ctx)
            meta["docs"] = len(documents)
        ctx.tracer.tag("num_candidates", len(documents))
        documents = self.post_retrieve(ctx.search_query, documents)
        with ctx.tracer.stage("process_documents") as meta:
            documents = self.process_documents(ctx, documents)
            meta["docs"] = len(documents)
        return self.pre_generate(ctx.search_query, documents)

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Default retrieval for custom subclasses: decision-engine routing + evaluate/refine loop.

        Every built-in ``rag_type`` overrides this step with its own strategy, so the
        ``DecisionEngine`` and the lexical evaluate -> refine retry loop below only run in
        your own ``BaseRAG`` subclasses that do not override ``retrieve``.
        """

        mode = RETRIEVAL_VECTOR
        if self.uses_decision_engine and self.decision_engine is not None:
            mode = self.decision_engine.retrieval_type(ctx.search_query)
            ctx.trace.append(f"decision:{mode}")
            if mode == RETRIEVAL_NONE:
                ctx.trace.append("retrieval_skipped")
                ctx.metadata["retrieval_skipped"] = True
                return []

        documents = self._retrieve_by_mode(ctx, ctx.search_query, mode)
        if self.evaluator is None:
            return documents

        retries = 0
        query = ctx.search_query
        while True:
            filtered = self.evaluator.evaluate(query, documents)
            if filtered:
                ctx.trace.append("evaluation_passed")
                return filtered
            ctx.trace.append("evaluation_failed")
            if self.refiner is None or retries >= self.max_retries:
                return documents
            refined = self.refiner.refine(query, documents)
            if not refined or refined == query:
                return documents
            retries += 1
            query = refined
            ctx.trace.append(f"retry_{retries}:{query}")
            documents = self._retrieve_by_mode(ctx, query, mode)

    def process_documents(self, ctx: PipelineContext, documents: List[Document]) -> List[Document]:
        """Rerank, deduplicate and reduce documents."""

        if self.reranker is not None and documents:
            documents = self.reranker.rerank(ctx.search_query, documents)
            ctx.trace.append("reranked")
        if self.deduplicator is not None and documents:
            documents = self.deduplicator.deduplicate(documents)
        if self.context_reducer is not None and documents:
            documents = self.context_reducer.reduce(documents)
        return documents

    def generate(self, ctx: PipelineContext, documents: List[Document]) -> GenerationResult:
        """Generate the final answer for the original question (after the retrieval guards)."""

        self._check_retrieval(ctx, documents)
        if self.generator is None:
            if self.llm_client is None:
                raise ValueError("Either generator or llm_client must be provided")
            self.generator = Generator(llm_client=self.llm_client)
        if type(self.generator).generate is Generator.generate:
            # The built-in generator traces prompt_build, generate (the LLM call) and post_process.
            return self.generator.generate(
                query=ctx.query,
                documents=documents,
                memory_context=ctx.history,
                reasoning_trace=ctx.trace,
            )
        with ctx.tracer.stage("generate") as meta:
            result = self.generator.generate(
                query=ctx.query,
                documents=documents,
                memory_context=ctx.history,
                reasoning_trace=ctx.trace,
            )
            meta.update(result.usage)
        return result

    def _check_retrieval(self, ctx: PipelineContext, documents: List[Document]) -> None:
        """Run the query's retrieval guards; raises ``GuardBlockedError`` (no LLM call follows) on a block."""

        from raglib.core.runtime import guard_join
        from raglib.guards.base import current_scope

        guard_join()  # a parallel input guard's block wins over the "nothing relevant" exit
        scope = current_scope()
        if scope is None or not scope.retrieval_guards:
            return
        _, blocked = scope.run(
            "retrieval",
            scope.retrieval_guards,
            list(documents),
            question=ctx.query,
            documents=list(documents),
            retrieval_skipped=bool(ctx.metadata.get("smalltalk") or ctx.metadata.get("retrieval_skipped")),
            reranker=self.reranker.backend if self.reranker is not None else None,
        )
        if blocked is not None:
            raise GuardBlockedError(blocked)

    def _blocked_result(self, ctx: PipelineContext, decision: Any) -> GenerationResult:
        """The answer returned when a guard stops the run: the guard's message, no sources."""

        ctx.metadata["blocked_by"] = decision.guard
        ctx.tracer.tag("blocked_by", decision.guard)
        ctx.trace.append(f"blocked_by:{decision.guard}")
        return GenerationResult(answer=decision.message or "", sources=[], reasoning_trace=list(ctx.trace))

    def _check_output(self, ctx: PipelineContext, result: GenerationResult) -> GenerationResult:
        """Run the query's output guards on the answer (mask / truncate / withhold)."""

        from raglib.guards.base import current_scope

        scope = current_scope()
        if scope is None or not scope.output_guards or ctx.metadata.get("blocked_by"):
            return result
        guards = [guard for guard in scope.output_guards if guard.flag_only] if ctx.output_checked else list(scope.output_guards)
        if not guards:
            return result
        answer, blocked = scope.run(
            "output", guards, result.answer, question=ctx.query, documents=list(result.sources), result=result
        )
        if blocked is not None:
            ctx.metadata["blocked_by"] = blocked.guard
            ctx.tracer.tag("blocked_by", blocked.guard)
            result.answer, result.sources, result.citations = blocked.message or "", [], []
        elif answer != result.answer:
            result.answer = answer
            result.citations = Generator.extract_citations(answer, result.sources)
        return result

    def pre_retrieve(self, query: str) -> str:
        """Hook executed before retrieval starts."""

        return query

    def post_retrieve(self, query: str, documents: List[Document]) -> List[Document]:
        """Hook executed after retrieval finishes."""

        return documents

    def pre_generate(self, query: str, documents: List[Document]) -> List[Document]:
        """Hook executed before generation starts."""

        return documents

    def post_generate(self, result: GenerationResult) -> GenerationResult:
        """Hook executed after generation finishes."""

        return result

    def candidate_k(self) -> Optional[int]:
        """How many candidates to retrieve: more when a real reranker will cut them down."""

        if self.reranker is not None and self.reranker.backend != "none":
            if self.rerank_candidates is not None:
                return max(self.rerank_candidates, self.reranker.top_k)
            return max(self.reranker.top_k * 4, 20)
        return None

    def retrieve_corpus(self, ctx: PipelineContext, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Retrieve from the corpus honoring filters and candidate_k."""

        if self.retriever is None:
            return []
        return self.retriever.retrieve(query, top_k=top_k or self.candidate_k(), filters=ctx.filters)

    def _retrieve_by_mode(self, ctx: PipelineContext, query: str, mode: str) -> List[Document]:
        """Dispatch retrieval to the selected mode with graceful fallbacks."""

        from raglib.components.web_retriever import web_retrieve

        if mode == RETRIEVAL_WEB and self.web_retriever is not None:
            return web_retrieve(self.web_retriever, query, ctx.filters)
        if mode == RETRIEVAL_HYBRID and self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(query, filters=ctx.filters)
        if self.retriever is not None:
            return self.retrieve_corpus(ctx, query)
        if self.web_retriever is not None:
            return web_retrieve(self.web_retriever, query, ctx.filters)
        logger.warning("No retriever available for mode=%s", mode)
        return []

    def memory_for(self, session_id: Optional[str]) -> Optional[MemoryModule]:
        """Return the memory for a session (conversation store first, legacy module second)."""

        if self.conversation_store is not None:
            return self.conversation_store.get(session_id)
        return self.memory_module

    def _flag_injections(self, ctx: PipelineContext, result: GenerationResult) -> None:
        """Flag (never block) sources with instruction-like text: result metadata + an on_event notice."""

        from raglib.core.prompt_safety import injection_signals

        flags = []
        for doc in result.sources:
            signals = injection_signals(doc.content)
            if signals:
                flags.append({"document_id": doc.id, "source": doc.source, "signals": signals})
        if not flags:
            return
        ctx.metadata["injection_flags"] = flags
        ctx.tracer.notice("injection_flag", flags=flags, documents=len(flags))
        logger.warning("%d retrieved source(s) contain instruction-like text: %s", len(flags), flags)

    def _start(
        self,
        query: str,
        session_id: Optional[str],
        filters: Optional[MetadataFilter],
        tracer: Optional[Tracer] = None,
    ) -> PipelineContext:
        """Validate input, load history, and build the retrieval query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query = query.strip()
        tracer = tracer or Tracer(on_event=self.on_event)
        tracer.tag("pipeline", self.pipeline_name)
        tracer.tag("query_tokens", count_tokens(query))
        tracer.tag("reranker_used", False)
        tracer.tag("num_candidates", 0)
        ctx = PipelineContext(query=query, search_query=query, session_id=session_id, filters=filters, tracer=tracer)

        if self.use_memory:
            memory = self.memory_for(session_id)
            if memory is not None:
                ctx.history = memory.get_context()
            if ctx.history and self.condenser is not None:
                with tracer.stage("condense") as meta:
                    ctx.search_query = self.condenser.condense(query, ctx.history)
                    meta["standalone_query"] = ctx.search_query
                if ctx.search_query != query:
                    ctx.trace.append(f"condensed:{ctx.search_query[:80]}")

        if self.query_rewriter is not None:
            with tracer.stage("rewrite") as meta, tracer.stage("query_rewrite"):
                rewritten = self.query_rewriter.rewrite(ctx.search_query)
                meta["rewritten_query"] = rewritten
            if rewritten and rewritten != ctx.search_query:
                ctx.trace.append(f"query_rewritten:{rewritten[:80]}")
                ctx.search_query = rewritten

        ctx.search_query = self.pre_retrieve(ctx.search_query)
        if self.skip_retrieval_for_smalltalk and is_greeting(query) and not ctx.history:
            ctx.metadata["smalltalk"] = True
        return ctx

    def _finish(self, ctx: PipelineContext, result: GenerationResult) -> GenerationResult:
        """Run output guards, persist memory and attach diagnostics."""

        from raglib.core.runtime import guard_join

        guard_join()  # nothing is stored for a question a parallel input guard blocked
        result = self._check_output(ctx, result)
        if self.use_memory:
            memory = self.memory_for(ctx.session_id)
            if memory is not None:
                memory.add(query=ctx.query, answer=result.answer, documents=result.sources)
        self._flag_injections(ctx, result)
        costs: Dict[str, Any] = {}
        if ctx.usage is not None and self.price_table is not None:
            from raglib.costs import price_run

            costs = price_run(self.price_table, ctx.usage.breakdown(), ctx.tracer.events)
        total = ctx.tracer.finish(error=None, pipeline=self.pipeline_name, docs=len(result.sources), **costs)
        result.trace_events = list(ctx.tracer.events)
        result.usage = {
            **result.usage,
            "latency_ms": ctx.tracer.total_latency_ms(),
            "wall_latency_ms": total.latency_ms,
        }
        if ctx.usage is not None:
            # Totals over every LLM call of the run (grading, routing, rewriting, generation, ...).
            result.usage.update(ctx.usage.as_dict())
            result.usage["usage_by_model"] = ctx.usage.breakdown()
        result.usage.update(costs)
        if ctx.run_state is not None and ctx.run_state.degraded:
            ctx.metadata["degraded"] = list(ctx.run_state.degraded)
        # Pipeline diagnostics (crag_verdict, agent_steps, hop_queries, graph_mode, ...) are part of the
        # result; internal Document lists stay out so metadata remains JSON-serializable.
        diagnostics = {key: value for key, value in ctx.metadata.items() if key not in _INTERNAL_METADATA}
        result.metadata = {
            **diagnostics,
            **result.metadata,
            "pipeline": self.pipeline_name,
            "search_query": ctx.search_query,
            "session_id": ctx.session_id,
        }
        logger.info("%s run completed (docs=%d, %.0fms)", self.pipeline_name, len(result.sources), result.usage["latency_ms"])
        return result
