"""Multi-query RAG: paraphrase the query, retrieve per variant in parallel, fuse with RRF."""

from __future__ import annotations

import logging
from typing import List

from raglib.components.multi_query_retriever import MultiQueryRetriever
from raglib.core.base import BaseRAG, PipelineContext
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class MultiQueryRAG(BaseRAG):
    """Improves recall for vague or ambiguously-worded questions."""

    pipeline_name = "multi_query"

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Expand, retrieve in parallel, and fuse."""

        retriever = self.multi_query_retriever
        if retriever is None:
            if self.query_expander is None or self.retriever is None:
                raise ValueError("MultiQueryRAG requires query_expander and retriever")
            retriever = MultiQueryRetriever(
                retriever=self.retriever,
                query_expander=self.query_expander,
                top_k=self.candidate_k() or self.retriever.top_k,
            )
            self.multi_query_retriever = retriever
        documents, variants = retriever.retrieve_with_variants(ctx.search_query, filters=ctx.filters)
        ctx.trace.append(f"variants:{len(variants)}")
        return documents
