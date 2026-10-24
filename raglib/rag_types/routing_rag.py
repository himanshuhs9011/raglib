"""Routing RAG: pick the best source (corpus, web, corpus+web, graph) per query."""

from __future__ import annotations

import logging
from typing import List

from raglib.core.base import BaseRAG, PipelineContext
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class RoutingRAG(BaseRAG):
    """Routes each query once, then retrieves from the chosen backend."""

    pipeline_name = "routing"

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Route and retrieve."""

        if self.router_retriever is None:
            raise ValueError("RoutingRAG requires router_retriever")
        route = self.router_retriever.route(ctx.search_query)
        ctx.trace.append(f"route:{route}")
        if route in {"vector", "graph"}:
            return self.router_retriever.retrieve_route(route, ctx.search_query, filters=ctx.filters)
        return self.router_retriever.retrieve_route(route, ctx.search_query)
