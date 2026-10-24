"""Hybrid RAG: sparse (BM25) + dense retrieval fused with RRF, or corpus + web ("hybrid_web")."""

from __future__ import annotations

import logging
from typing import Any, List

from raglib.core.base import BaseRAG, PipelineContext
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class HybridRAG(BaseRAG):
    """Hybrid retrieval followed by reranking and generation.

    ``include_web=False`` (default): BM25 + dense vector search over the corpus.
    ``include_web=True``: corpus + live web results fused with RRF (needs a hybrid_retriever).
    """

    pipeline_name = "hybrid"

    def __init__(self, include_web: bool = False, **kwargs: Any):
        """Initialize HybridRAG."""

        super().__init__(**kwargs)
        self.include_web = include_web

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Run sparse+dense (or corpus+web) retrieval."""

        if self.include_web:
            if self.hybrid_retriever is None:
                raise ValueError("HybridRAG(include_web=True) requires hybrid_retriever (set web_search_provider)")
            ctx.trace.append("hybrid_web_retrieve")
            return self.hybrid_retriever.retrieve(ctx.search_query, filters=ctx.filters)
        if self.retriever is None:
            if self.hybrid_retriever is not None:
                return self.hybrid_retriever.retrieve(ctx.search_query)
            raise ValueError("HybridRAG requires a retriever")
        ctx.trace.append("hybrid_sparse_dense_retrieve")
        mode = "hybrid" if self.retriever.embedding is not None else "sparse"
        return self.retriever.retrieve(ctx.search_query, top_k=self.candidate_k(), filters=ctx.filters, mode=mode)
