"""Advanced RAG: hybrid retrieval -> rerank -> dedupe -> budgeted, cited generation."""

from __future__ import annotations

import logging
from typing import List

from raglib.core.base import BaseRAG, PipelineContext
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class AdvancedRAG(BaseRAG):
    """The recommended default production pipeline."""

    pipeline_name = "advanced"

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Retrieve reranking candidates from the corpus."""

        if self.retriever is None:
            raise ValueError("AdvancedRAG requires a retriever")
        ctx.trace.append("advanced_pipeline")
        return self.retrieve_corpus(ctx, ctx.search_query)
