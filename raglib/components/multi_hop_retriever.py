"""Iterative multi-hop retrieval: each hop is planned from the evidence gathered so far."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, List

from raglib.components.planner import Planner
from raglib.components.retriever import Retriever
from raglib.core.fusion import interleave
from raglib.schemas import Document, QueryResult

logger = logging.getLogger(__name__)


class MultiHopRetriever:
    """Retrieve -> inspect evidence -> plan the next query (bridge entity) -> repeat."""

    def __init__(self, retriever: Retriever, planner: Planner, max_hops: int = 3, docs_per_hop: int = 4):
        """Initialize multi-hop retrieval dependencies."""

        self.retriever = retriever
        self.planner = planner
        self.max_hops = max(1, max_hops)
        self.docs_per_hop = docs_per_hop

    def retrieve(self, query: str, **kwargs: Any) -> QueryResult:
        """Run up to ``max_hops`` dependent retrieval hops and fuse their results."""

        queries: List[str] = [query]
        hop_results: List[List[Document]] = []
        evidence: List[Document] = []
        current = query
        for hop in range(1, self.max_hops + 1):
            docs = self.retriever.retrieve(current, top_k=self.docs_per_hop, **kwargs)
            hop_results.append(
                [replace(doc, metadata={**doc.metadata, "hop": hop, "hop_query": current}) for doc in docs]
            )
            evidence.extend(hop_results[-1])
            logger.debug("MultiHopRetriever hop=%d docs=%d", hop, len(docs))
            if hop == self.max_hops:
                break
            next_query = self.planner.next_step(query, evidence, queries)
            if next_query is None:
                break
            queries.append(next_query)
            current = next_query
        fused = interleave(hop_results)  # each hop answers a different sub-question
        return QueryResult(query=query, documents=fused, hop_count=len(hop_results), rewritten_query=" | ".join(queries))
