"""Base orchestration class for all RAG pipeline variants."""

from __future__ import annotations

import logging
from typing import List, Optional

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
from raglib.components.hybrid_retriever import HybridRetriever
from raglib.components.memory import MemoryModule
from raglib.components.multi_hop_retriever import MultiHopRetriever
from raglib.components.multi_query_retriever import MultiQueryRetriever
from raglib.components.planner import Planner
from raglib.components.query_expander import QueryExpander
from raglib.components.query_rewriter import QueryRewriter
from raglib.components.refiner import Refiner
from raglib.components.reflection import ReflectionModule
from raglib.components.reranker import Reranker
from raglib.components.retriever import Retriever
from raglib.components.router_retriever import RouterRetriever
from raglib.components.web_retriever import WebRetriever
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)


class BaseRAG:
    """Defines shared orchestration behavior for RAG strategies."""

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

    def run(self, query: str) -> GenerationResult:
        """Execute the default pipeline: decision, retrieve, rerank, evaluate, refine, generate."""

        reasoning_trace: List[str] = []
        working_query = self.pre_retrieve(query)

        if self.query_rewriter is not None:
            rewritten = self.query_rewriter.rewrite(working_query)
            if rewritten != working_query:
                reasoning_trace.append(f"query_rewritten:{rewritten}")
                working_query = rewritten

        should_retrieve = True
        retrieval_mode = RETRIEVAL_VECTOR
        if self.decision_engine is not None:
            retrieval_mode = self.decision_engine.retrieval_type(working_query)
            should_retrieve = retrieval_mode != RETRIEVAL_NONE
            reasoning_trace.append(f"decision:{retrieval_mode}")

        documents: List[Document] = []
        if should_retrieve and retrieval_mode != RETRIEVAL_NONE:
            documents = self._retrieve_by_mode(working_query, retrieval_mode)
            documents = self.post_retrieve(working_query, documents)
        else:
            reasoning_trace.append("retrieval_skipped")

        if self.reranker is not None and documents:
            documents = self.reranker.rerank(working_query, documents)
            reasoning_trace.append("reranked")

        retries = 0
        while self.evaluator is not None:
            filtered = self.evaluator.evaluate(working_query, documents)
            if filtered:
                documents = filtered
                reasoning_trace.append("evaluation_passed")
                break
            reasoning_trace.append("evaluation_failed")
            if self.refiner is None or retries >= self.max_retries:
                documents = filtered
                break
            refined = self.refiner.refine(working_query, documents)
            if not refined or refined == working_query:
                break
            retries += 1
            working_query = refined
            reasoning_trace.append(f"retry_{retries}:{working_query}")
            documents = self._retrieve_by_mode(working_query, retrieval_mode)
            if self.reranker is not None and documents:
                documents = self.reranker.rerank(working_query, documents)

        if self.deduplicator is not None and documents:
            documents = self.deduplicator.deduplicate(documents)
            reasoning_trace.append("deduplicated")

        if self.context_reducer is not None and documents:
            documents = self.context_reducer.reduce(documents)
            reasoning_trace.append("context_reduced")

        documents = self.pre_generate(working_query, documents)
        memory_context = self.memory_module.get_context() if self.memory_module is not None else ""

        if self.generator is not None:
            result = self.generator.generate(
                query=working_query,
                documents=documents,
                memory_context=memory_context,
                reasoning_trace=reasoning_trace,
            )
        elif self.llm_client is not None:
            answer = self.llm_client.complete(
                prompt=f"Query: {working_query}\nContext: {memory_context}",
                system="Provide a concise grounded answer.",
            )
            result = GenerationResult(answer=answer, sources=documents, reasoning_trace=reasoning_trace)
        else:
            raise ValueError("Either generator or llm_client must be provided")

        result = self.post_generate(result)

        if self.memory_module is not None:
            self.memory_module.add(query=query, answer=result.answer, documents=result.sources)

        logger.info("BaseRAG run completed for query=%s", query)
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

    def _retrieve_by_mode(self, query: str, mode: str) -> List[Document]:
        """Dispatch retrieval to the selected mode with graceful fallbacks."""

        if mode == RETRIEVAL_WEB and self.web_retriever is not None:
            return self.web_retriever.retrieve(query)
        if mode == RETRIEVAL_HYBRID and self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(query)
        if mode == RETRIEVAL_VECTOR and self.retriever is not None:
            return self.retriever.retrieve(query)

        if self.retriever is not None:
            logger.warning("Falling back to vector retriever for mode=%s", mode)
            return self.retriever.retrieve(query)
        if self.web_retriever is not None:
            logger.warning("Falling back to web retriever for mode=%s", mode)
            return self.web_retriever.retrieve(query)

        logger.warning("No retriever available for mode=%s", mode)
        return []
