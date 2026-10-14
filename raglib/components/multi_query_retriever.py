"""Multi-query retrieval via semantic expansion and result merging."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import Dict, List, Optional

from raglib.components.deduplicator import Deduplicator
from raglib.components.query_expander import QueryExpander
from raglib.components.retriever import Retriever
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """Expands a query into variants and merges retrieval results."""

    def __init__(
        self,
        retriever: Retriever,
        query_expander: QueryExpander,
        deduplicator: Optional[Deduplicator] = None,
        top_k: int = 5,
    ):
        """Initialize variant retrieval dependencies."""

        self.retriever = retriever
        self.query_expander = query_expander
        self.deduplicator = deduplicator
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents across query variants and merge scores."""

        variants = self.query_expander.expand(query)
        logger.info("MultiQueryRetriever generated %d variants", len(variants))

        merged: Dict[str, Document] = {}
        counts: Dict[str, int] = {}

        for variant in variants:
            docs = self.retriever.retrieve(variant)
            for doc in docs:
                digest = hashlib.sha256(doc.content.lower().strip().encode("utf-8")).hexdigest()[:16]
                key = doc.id or f"doc-{digest}"
                if key in merged:
                    previous = merged[key]
                    merged[key] = replace(previous, score=max(previous.score, doc.score))
                    counts[key] += 1
                else:
                    merged[key] = doc
                    counts[key] = 1

        blended = [
            replace(doc, score=doc.score * (1.0 + 0.1 * (counts[key] - 1)))
            for key, doc in merged.items()
        ]

        ranked = sorted(blended, key=lambda d: d.score, reverse=True)
        if self.deduplicator:
            ranked = self.deduplicator.deduplicate(ranked)
        return ranked[: self.top_k]
