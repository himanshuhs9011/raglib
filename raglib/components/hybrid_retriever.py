"""Corpus + web retrieval with proportional slot allocation ("hybrid_web")."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from raglib.core.concurrency import parallel_map
from raglib.components.retriever import Retriever
from raglib.components.web_retriever import WebRetriever
from raglib.core.fusion import interleave
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Queries the local corpus and the web concurrently and fuses the rankings.

    Note: sparse+dense hybrid search over the corpus is ``Retriever(mode="hybrid")``;
    this class blends the corpus with *web* results.
    """

    def __init__(
        self,
        vector_retriever: Retriever,
        web_retriever: WebRetriever,
        vector_weight: float = 0.6,
        web_weight: float = 0.4,
        top_k: Optional[int] = None,
    ):
        """Initialize weighted retriever dependencies."""

        self.vector_retriever = vector_retriever
        self.web_retriever = web_retriever
        total = max(vector_weight + web_weight, 1e-9)
        self.vector_weight = vector_weight / total
        self.web_weight = web_weight / total
        self.top_k = top_k or max(vector_retriever.top_k, web_retriever.top_k)

    def allocate(self, vector_docs: List[Document], web_docs: List[Document]) -> List[Document]:
        """Give each source a share of the top_k slots proportional to its weight, then interleave.

        Weighted RRF (k=60) cannot be used here: with 0.6/0.4 weights the best web hit only beats
        corpus hits ranked ~31st or worse, so web results never reached the answer.
        """

        top_k = self.top_k
        web_slots = min(len(web_docs), max(1, round(top_k * self.web_weight))) if web_docs and self.web_weight > 0 else 0
        vector_slots = min(len(vector_docs), top_k - web_slots)
        web_slots = min(len(web_docs), top_k - vector_slots)  # unused corpus slots go to the web
        return interleave([vector_docs[:vector_slots], web_docs[:web_slots]])[:top_k]

    def retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        """Retrieve from both sources in parallel and allocate top_k slots by weight."""

        tasks = [
            (lambda: self.vector_retriever.retrieve(query, **kwargs)) if self.vector_weight > 0 else (lambda: []),
            (lambda: self.web_retriever.retrieve(query)) if self.web_weight > 0 else (lambda: []),
        ]
        vector_docs, web_docs = parallel_map(lambda task: task(), tasks, max_workers=2)
        fused = self.allocate(vector_docs, web_docs)
        logger.info("HybridRetriever fused %d corpus + %d web docs", len(vector_docs), len(web_docs))
        return fused
