"""Multi-query retrieval: parallel retrieval per query variant, fused with RRF."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from raglib.components.deduplicator import Deduplicator
from raglib.components.query_expander import QueryExpander
from raglib.components.retriever import Retriever
from raglib.core.fusion import reciprocal_rank_fusion
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """Expands a query into variants, retrieves for each in parallel, and fuses with RRF."""

    def __init__(
        self,
        retriever: Retriever,
        query_expander: QueryExpander,
        deduplicator: Optional[Deduplicator] = None,
        top_k: int = 5,
        max_workers: int = 4,
    ):
        """Initialize variant retrieval dependencies."""

        self.retriever = retriever
        self.query_expander = query_expander
        self.deduplicator = deduplicator
        self.top_k = top_k
        self.max_workers = max_workers

    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs: Any) -> List[Document]:
        """Retrieve documents across query variants and fuse rankings."""

        return self.retrieve_with_variants(query, top_k=top_k, **kwargs)[0]

    def retrieve_with_variants(self, query: str, top_k: Optional[int] = None, **kwargs: Any):
        """Return (fused documents, query variants) - thread-safe, no shared state."""

        k = top_k or self.top_k
        variants = self.query_expander.expand(query)
        if len(variants) == 1:
            ranked_lists = [self.retriever.retrieve(variants[0], top_k=k, **kwargs)]
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(variants))) as pool:
                ranked_lists = list(pool.map(lambda v: self.retriever.retrieve(v, top_k=k, **kwargs), variants))
        # The original query gets a higher weight than paraphrases.
        weights = [1.5] + [1.0] * (len(ranked_lists) - 1)
        fused = reciprocal_rank_fusion(ranked_lists, weights=weights)
        if self.deduplicator is not None:
            fused = self.deduplicator.deduplicate(fused)
        return fused[:k], variants
