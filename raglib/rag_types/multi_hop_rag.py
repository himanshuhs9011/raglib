"""Multi-hop RAG: iterative retrieval where each hop is planned from evidence so far."""

from __future__ import annotations

import logging
from typing import Any, List

from raglib.components.multi_hop_retriever import MultiHopRetriever
from raglib.core.base import BaseRAG, PipelineContext
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class MultiHopRAG(BaseRAG):
    """For questions whose answer needs a bridge fact (A -> B -> answer)."""

    pipeline_name = "multi_hop"

    def __init__(self, max_hops: int = 3, **kwargs: Any):
        """Initialize MultiHopRAG."""

        super().__init__(**kwargs)
        self.max_hops = max_hops

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Run dependent retrieval hops."""

        retriever = self.multi_hop_retriever
        if retriever is None:
            if self.planner is None or self.retriever is None:
                raise ValueError("MultiHopRAG requires planner and retriever")
            retriever = MultiHopRetriever(retriever=self.retriever, planner=self.planner, max_hops=self.max_hops)
            self.multi_hop_retriever = retriever
        result = retriever.retrieve(ctx.search_query, filters=ctx.filters)
        ctx.trace.append(f"hops:{result.hop_count}")
        if result.rewritten_query:
            ctx.metadata["hop_queries"] = result.rewritten_query.split(" | ")
        return result.documents
