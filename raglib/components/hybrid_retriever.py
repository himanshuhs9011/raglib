"""Corpus + web retrieval fused with Reciprocal Rank Fusion ("hybrid_web")."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from raglib.components.retriever import Retriever
from raglib.components.web_retriever import WebRetriever
from raglib.core.fusion import reciprocal_rank_fusion
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

    def retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        """Retrieve from both sources in parallel and fuse with weighted RRF."""

        with ThreadPoolExecutor(max_workers=2) as pool:
            vector_future = pool.submit(self.vector_retriever.retrieve, query, **kwargs) if self.vector_weight > 0 else None
            web_future = pool.submit(self.web_retriever.retrieve, query) if self.web_weight > 0 else None
            vector_docs = vector_future.result() if vector_future else []
            web_docs = web_future.result() if web_future else []
        fused = reciprocal_rank_fusion(
            [vector_docs, web_docs],
            weights=[self.vector_weight, self.web_weight],
            list_names=["vector", "web"],
            top_k=self.top_k,
        )
        logger.info("HybridRetriever fused %d corpus + %d web docs", len(vector_docs), len(web_docs))
        return fused
