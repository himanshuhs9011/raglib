"""Routing RAG orchestration using retriever routing decisions."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.components.generator import Generator
from raglib.components.router_retriever import (
    ROUTE_HYBRID,
    ROUTE_WEB,
    RouterRetriever,
)
from raglib.core.base import BaseRAG
from raglib.schemas import GenerationResult

logger = logging.getLogger(__name__)


class RoutingRAG(BaseRAG):
    """Routes each query to the best retriever before generation."""

    def __init__(
        self,
        router_retriever: Optional[RouterRetriever] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize RoutingRAG with router and generator components."""

        super().__init__(router_retriever=router_retriever, generator=generator, **kwargs)

    def run(self, query: str) -> GenerationResult:
        """Route, retrieve, and generate for a query without duplicate routing calls."""

        if self.router_retriever is None or self.generator is None:
            raise ValueError("RoutingRAG requires router_retriever and generator")

        active_query = self.pre_retrieve(query)
        route = self.router_retriever.route(active_query)
        logger.info("RoutingRAG selected route=%s", route)

        if route == ROUTE_WEB:
            documents = self.router_retriever.web_retriever.retrieve(active_query)
        elif route == ROUTE_HYBRID and self.router_retriever.hybrid_retriever is not None:
            documents = self.router_retriever.hybrid_retriever.retrieve(active_query)
        else:
            documents = self.router_retriever.vector_retriever.retrieve(active_query)

        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(
            query=active_query,
            documents=documents,
            reasoning_trace=[f"route:{route}"],
        )
        return self.post_generate(result)
