"""Iterative multi-hop retrieval over planned sub-queries."""

from __future__ import annotations

import hashlib
import logging
from typing import Dict

from raglib.components.planner import Planner
from raglib.components.retriever import Retriever
from raglib.schemas import Document, QueryResult

logger = logging.getLogger(__name__)


class MultiHopRetriever:
    """Performs staged retrieval across multiple reasoning hops."""

    def __init__(self, retriever: Retriever, planner: Planner, max_hops: int = 3):
        """Initialize multi-hop retrieval dependencies."""

        self.retriever = retriever
        self.planner = planner
        self.max_hops = max_hops

    def retrieve(self, query: str) -> QueryResult:
        """Retrieve documents iteratively over planned hops."""

        sub_queries = self.planner.plan(query)[: self.max_hops]
        logger.info("MultiHopRetriever planned %d hops", len(sub_queries))

        merged: Dict[str, Document] = {}
        for hop_index, sub_query in enumerate(sub_queries, start=1):
            logger.info("MultiHopRetriever hop=%d query=%s", hop_index, sub_query)
            for doc in self.retriever.retrieve(sub_query):
                digest = hashlib.sha256(doc.content.lower().strip().encode("utf-8")).hexdigest()[:16]
                key = doc.id or f"doc-{digest}"
                if key not in merged or doc.score > merged[key].score:
                    merged[key] = doc

        ranked = sorted(merged.values(), key=lambda doc: doc.score, reverse=True)
        return QueryResult(query=query, documents=ranked, hop_count=len(sub_queries))
