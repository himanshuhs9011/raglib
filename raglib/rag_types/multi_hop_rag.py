"""Multi-hop RAG orchestration with sequential sub-query retrieval."""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional

from raglib.components.deduplicator import Deduplicator
from raglib.components.generator import Generator
from raglib.components.multi_hop_retriever import MultiHopRetriever
from raglib.components.planner import Planner
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)


class MultiHopRAG(BaseRAG):
    """Executes sequential retrieval hops planned from the user query."""

    def __init__(
        self,
        planner: Optional[Planner] = None,
        retriever: Optional[Retriever] = None,
        deduplicator: Optional[Deduplicator] = None,
        generator: Optional[Generator] = None,
        multi_hop_retriever: Optional[MultiHopRetriever] = None,
        max_hops: int = 3,
        **kwargs,
    ):
        """Initialize MultiHopRAG with planning and retrieval dependencies."""

        super().__init__(
            planner=planner,
            retriever=retriever,
            deduplicator=deduplicator,
            generator=generator,
            multi_hop_retriever=multi_hop_retriever,
            **kwargs,
        )
        self.max_hops = max_hops

    def run(self, query: str) -> GenerationResult:
        """Plan hops, retrieve sequentially, accumulate context, and generate."""

        if self.generator is None or (
            self.multi_hop_retriever is None and (self.planner is None or self.retriever is None)
        ):
            raise ValueError("MultiHopRAG requires planner, retriever, and generator")

        active_query = self.pre_retrieve(query)

        if self.multi_hop_retriever is not None:
            query_result = self.multi_hop_retriever.retrieve(active_query)
            documents = query_result.documents
            trace = [f"hops:{query_result.hop_count}"]
        else:
            assert self.planner is not None and self.retriever is not None
            hops = self.planner.plan(active_query)[: max(1, self.max_hops)]
            trace = [f"hops:{len(hops)}"]

            merged: Dict[str, Document] = {}
            for hop_index, hop_query in enumerate(hops, start=1):
                logger.info("MultiHopRAG hop=%d query=%s", hop_index, hop_query)
                hop_docs = self.retriever.retrieve(hop_query)
                for doc in hop_docs:
                    digest = hashlib.sha256(doc.content.lower().strip().encode("utf-8")).hexdigest()[:16]
                    key = doc.id or f"doc-{digest}"
                    if key not in merged or doc.score > merged[key].score:
                        merged[key] = doc

            documents = sorted(merged.values(), key=lambda d: d.score, reverse=True)

        if self.deduplicator is not None:
            documents = self.deduplicator.deduplicate(documents)

        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(query=active_query, documents=documents, reasoning_trace=trace)
        return self.post_generate(result)
