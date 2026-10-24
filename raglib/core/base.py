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
from typing import Any, Callable, Dict, Iterator, List, Optional

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
from raglib.core.tracing import Tracer
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document, GenerationResult, TraceEvent
from raglib.vectorstores.base_store import MetadataFilter

logger = logging.getLogger(__name__)

_INTERNAL_METADATA = {"graph_context_docs"}


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
        on_event: Optional[Callable[[TraceEvent], None]] = None,
    ):
        """Initialize optional pipeline components via dependency injection."""

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

    # ================================================================== entry points
    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> GenerationResult:
        """Execute the pipeline for one query and return the answer with sources."""

        ctx = self._start(query, session_id, filters)
        result = self.answer(ctx)
        return self._finish(ctx, result)

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
    ) -> Iterator[str]:
        """Yield answer tokens; the generator's return value is the final GenerationResult.

        Strategies that loop over generation (e.g. Self-RAG) yield the final answer once.
        """

        if type(self).answer is not BaseRAG.answer or self.generator is None:
            result = self.run(query, session_id=session_id, filters=filters)
            yield result.answer
            return result

        ctx = self._start(query, session_id, filters)
        documents = self._prepare_documents(ctx)
        with ctx.tracer.stage("generate", streaming=True) as meta:
            context_docs, tokens = self.generator.stream(ctx.query, documents, memory_context=ctx.history)
            parts: List[str] = []
            for token in tokens:
                parts.append(token)
                yield token
            meta["docs"] = len(context_docs)
        result = self.generator.finalize("".join(parts), context_docs, ctx.trace + [f"generated_with_docs={len(context_docs)}"])
        result = self.post_generate(result)
        return self._finish(ctx, result)

    # ================================================================== pipeline steps
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
        documents = self.post_retrieve(ctx.search_query, documents)
        with ctx.tracer.stage("process_documents") as meta:
            documents = self.process_documents(ctx, documents)
            meta["docs"] = len(documents)
        return self.pre_generate(ctx.search_query, documents)

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Default retrieval: optional decision engine routing + corrective refine loop."""

        mode = RETRIEVAL_VECTOR
        if self.uses_decision_engine and self.decision_engine is not None:
            mode = self.decision_engine.retrieval_type(ctx.search_query)
            ctx.trace.append(f"decision:{mode}")
            if mode == RETRIEVAL_NONE:
                ctx.trace.append("retrieval_skipped")
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
        """Generate the final answer for the original question."""

        if self.generator is None:
            if self.llm_client is None:
                raise ValueError("Either generator or llm_client must be provided")
            self.generator = Generator(llm_client=self.llm_client)
        with ctx.tracer.stage("generate") as meta:
            result = self.generator.generate(
                query=ctx.query,
                documents=documents,
                memory_context=ctx.history,
                reasoning_trace=ctx.trace,
            )
            meta.update(result.usage)
        return result

    # ================================================================== hooks
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

    # ================================================================== helpers
    def candidate_k(self) -> Optional[int]:
        """How many candidates to retrieve: more when a real reranker will cut them down."""

        if self.reranker is not None and self.reranker.backend != "none":
            return max(self.reranker.top_k * 4, 20)
        return None

    def retrieve_corpus(self, ctx: PipelineContext, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Retrieve from the corpus honoring filters and candidate_k."""

        if self.retriever is None:
            return []
        return self.retriever.retrieve(query, top_k=top_k or self.candidate_k(), filters=ctx.filters)

    def _retrieve_by_mode(self, ctx: PipelineContext, query: str, mode: str) -> List[Document]:
        """Dispatch retrieval to the selected mode with graceful fallbacks."""

        if mode == RETRIEVAL_WEB and self.web_retriever is not None:
            return self.web_retriever.retrieve(query)
        if mode == RETRIEVAL_HYBRID and self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(query)
        if self.retriever is not None:
            return self.retrieve_corpus(ctx, query)
        if self.web_retriever is not None:
            return self.web_retriever.retrieve(query)
        logger.warning("No retriever available for mode=%s", mode)
        return []

    def memory_for(self, session_id: Optional[str]) -> Optional[MemoryModule]:
        """Return the memory for a session (conversation store first, legacy module second)."""

        if self.conversation_store is not None:
            return self.conversation_store.get(session_id)
        return self.memory_module

    def _start(self, query: str, session_id: Optional[str], filters: Optional[MetadataFilter]) -> PipelineContext:
        """Validate input, load history, and build the retrieval query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query = query.strip()
        tracer = Tracer(on_event=self.on_event)
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
            with tracer.stage("rewrite") as meta:
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
        """Persist memory and attach diagnostics."""

        if self.use_memory:
            memory = self.memory_for(ctx.session_id)
            if memory is not None:
                memory.add(query=ctx.query, answer=result.answer, documents=result.sources)
        result.trace_events = list(ctx.tracer.events)
        result.usage = {**result.usage, "latency_ms": ctx.tracer.total_latency_ms()}
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
