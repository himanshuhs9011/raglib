"""Naive RAG: retrieve top-k chunks, then generate."""

from __future__ import annotations

import logging
from typing import List

from raglib.core.base import BaseRAG, PipelineContext
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class NaiveRAG(BaseRAG):
    """The simplest pipeline: retrieval then generation, no reranking or filtering."""

    pipeline_name = "naive"

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Retrieve top-k chunks for the query."""

        ctx.trace.append("naive_retrieve_generate")
        return self.retrieve_corpus(ctx, ctx.search_query)

    def process_documents(self, ctx: PipelineContext, documents: List[Document]) -> List[Document]:
        """Naive RAG does no post-processing."""

        return documents
