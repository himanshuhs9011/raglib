"""Agentic RAG orchestration using query planning and iterative retrieval."""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional

from raglib.components.context_reducer import ContextReducer
from raglib.components.deduplicator import Deduplicator
from raglib.components.generator import Generator
from raglib.components.planner import Planner
from raglib.components.reranker import Reranker
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)


class AgenticRAG(BaseRAG):
    """Runs a planned multi-step retrieval workflow before generation."""

    def __init__(
        self,
        planner: Optional[Planner] = None,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
        deduplicator: Optional[Deduplicator] = None,
        context_reducer: Optional[ContextReducer] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize AgenticRAG with planning, ranking, and context controls."""

        super().__init__(
            planner=planner,
            retriever=retriever,
            reranker=reranker,
            deduplicator=deduplicator,
            context_reducer=context_reducer,
            generator=generator,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Plan sub-queries, retrieve per step, merge, and generate."""

        if self.planner is None or self.retriever is None or self.generator is None:
            raise ValueError("AgenticRAG requires planner, retriever, and generator")

        active_query = self.pre_retrieve(query)
        plan = self.planner.plan(active_query)

        if not plan:
            logger.warning("AgenticRAG planner returned empty plan; falling back to single-step query")
            plan = [active_query]

        logger.info("AgenticRAG plan=%s", plan)
        trace = [f"plan_steps:{len(plan)}"]
        if len(plan) == 1 and plan[0] == active_query:
            trace.append("plan_fallback:single_step")

        merged: Dict[str, Document] = {}
        for index, sub_query in enumerate(plan, start=1):
            logger.info("AgenticRAG executing step=%d query=%s", index, sub_query)
            docs = self.retriever.retrieve(sub_query)
            trace.append(f"step_{index}:{sub_query[:50]}->{len(docs)}_docs")
            for doc in docs:
                digest = hashlib.sha256(doc.content.lower().strip().encode("utf-8")).hexdigest()[:16]
                key = doc.id or f"doc-{digest}"
                if key not in merged or doc.score > merged[key].score:
                    merged[key] = doc

        documents = list(merged.values())
        if self.deduplicator is not None:
            documents = self.deduplicator.deduplicate(documents)

        if self.reranker is not None:
            documents = self.reranker.rerank(active_query, documents)
        else:
            documents = sorted(documents, key=lambda doc: doc.score, reverse=True)

        if self.context_reducer is not None:
            documents = self.context_reducer.reduce(documents)

        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(query=active_query, documents=documents, reasoning_trace=trace)
        return self.post_generate(result)
