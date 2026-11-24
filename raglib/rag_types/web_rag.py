"""Web RAG: live web search -> rerank -> cited generation."""

from __future__ import annotations

import logging
from typing import List

from raglib.components.web_retriever import web_retrieve
from raglib.core.base import BaseRAG, PipelineContext
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class WebRAG(BaseRAG):
    """Answers from web search results only."""

    pipeline_name = "web"

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Search the web for the query."""

        if self.web_retriever is None:
            raise ValueError("WebRAG requires web_retriever (set web_search_provider)")
        ctx.trace.append("web_retrieve_rerank_generate")
        return web_retrieve(self.web_retriever, ctx.search_query, ctx.filters)
