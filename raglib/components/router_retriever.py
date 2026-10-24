"""Query router that dispatches to vector, web, hybrid, or graph retrieval."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from raglib.components.hybrid_retriever import HybridRetriever
from raglib.components.retriever import Retriever
from raglib.components.web_retriever import WebRetriever
from raglib.core.intents import is_comparison, is_global_question, needs_fresh_information
from raglib.core.parsing import parse_label
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)

ROUTE_VECTOR: str = "vector"
ROUTE_WEB: str = "web"
ROUTE_HYBRID: str = "hybrid"
ROUTE_GRAPH: str = "graph"


class RouterRetriever:
    """Routes retrieval requests to the most suitable retriever (one LLM call max)."""

    def __init__(
        self,
        vector_retriever: Retriever,
        web_retriever: Optional[WebRetriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        llm_client: Optional[BaseLLMClient] = None,
        graph_retriever: Optional[Any] = None,
    ):
        """Initialize routing dependencies; routes without a backend are never chosen."""

        self.vector_retriever = vector_retriever
        self.web_retriever = web_retriever
        self.hybrid_retriever = hybrid_retriever
        self.graph_retriever = graph_retriever
        self.llm_client = llm_client

    def available_routes(self) -> List[str]:
        """Return the routes that have a configured backend."""

        routes = [ROUTE_VECTOR]
        if self.web_retriever is not None:
            routes.append(ROUTE_WEB)
        if self.hybrid_retriever is not None:
            routes.append(ROUTE_HYBRID)
        if self.graph_retriever is not None:
            routes.append(ROUTE_GRAPH)
        return routes

    def route(self, query: str) -> str:
        """Return the best available retrieval route for the query."""

        routes = self.available_routes()
        if len(routes) == 1:
            return ROUTE_VECTOR
        if self.llm_client is not None:
            descriptions = {
                ROUTE_VECTOR: "vector = the private document corpus",
                ROUTE_WEB: "web = live internet search for recent/public information",
                ROUTE_HYBRID: "hybrid = corpus plus web",
                ROUTE_GRAPH: "graph = knowledge graph, best for relationships between entities or corpus-wide themes",
            }
            prompt = (
                "Choose the single best retrieval route for the question.\n"
                + "\n".join(descriptions[route] for route in routes)
                + f"\nQuestion: {query}\nReply with exactly one word from: {', '.join(routes)}."
            )
            try:
                label = parse_label(self.llm_client.complete(prompt=prompt), routes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("RouterRetriever LLM call failed: %s", exc)
                label = None
            if label is not None:
                logger.info("RouterRetriever route selected by LLM: %s", label)
                return label

        if ROUTE_WEB in routes and needs_fresh_information(query):
            route = ROUTE_WEB
        elif ROUTE_GRAPH in routes and is_global_question(query):
            route = ROUTE_GRAPH
        elif ROUTE_HYBRID in routes and is_comparison(query):
            route = ROUTE_HYBRID
        else:
            route = ROUTE_VECTOR
        logger.info("RouterRetriever route selected by heuristic: %s", route)
        return route

    def retrieve_route(self, route: str, query: str, **kwargs: Any) -> List[Document]:
        """Retrieve from a specific route."""

        if route == ROUTE_WEB and self.web_retriever is not None:
            return self.web_retriever.retrieve(query)
        if route == ROUTE_HYBRID and self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(query)
        if route == ROUTE_GRAPH and self.graph_retriever is not None:
            return self.graph_retriever.retrieve(query, **kwargs)
        return self.vector_retriever.retrieve(query, **kwargs)

    def retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        """Route then retrieve."""

        return self.retrieve_route(self.route(query), query, **kwargs)
