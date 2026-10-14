"""Hybrid retrieval that blends vector and web search signals."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import Dict, List

from raglib.components.retriever import Retriever
from raglib.components.web_retriever import WebRetriever
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines vector and web retrieval with weighted re-scoring."""

    def __init__(
        self,
        vector_retriever: Retriever,
        web_retriever: WebRetriever,
        vector_weight: float = 0.6,
        web_weight: float = 0.4,
    ):
        """Initialize weighted retriever dependencies."""

        self.vector_retriever = vector_retriever
        self.web_retriever = web_retriever
        total = max(vector_weight + web_weight, 1e-9)
        self.vector_weight = vector_weight / total
        self.web_weight = web_weight / total

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve and merge vector/web results with weighted scores."""

        vector_docs = self.vector_retriever.retrieve(query) if self.vector_weight > 0 else []
        web_docs = self.web_retriever.retrieve(query) if self.web_weight > 0 else []

        normalized_vector = self._normalize_scores(vector_docs)
        normalized_web = self._normalize_scores(web_docs)

        merged: Dict[str, Document] = {}

        for doc in normalized_vector:
            key = self._doc_key(doc)
            weighted = doc.score * self.vector_weight
            merged[key] = replace(
                doc,
                score=weighted,
                source=doc.source or "vector",
                metadata={**doc.metadata, "hybrid_components": ["vector"]},
            )

        for doc in normalized_web:
            key = self._doc_key(doc)
            weighted = doc.score * self.web_weight
            if key in merged:
                base = merged[key]
                components = list(base.metadata.get("hybrid_components", []))
                components.append("web")
                merged[key] = replace(
                    base,
                    score=base.score + weighted,
                    source="hybrid",
                    metadata={**base.metadata, "hybrid_components": components},
                )
            else:
                merged[key] = replace(
                    doc,
                    score=weighted,
                    source=doc.source or "web",
                    metadata={**doc.metadata, "hybrid_components": ["web"]},
                )

        ranked = sorted(merged.values(), key=lambda d: d.score, reverse=True)
        top_k = max(self.vector_retriever.top_k, self.web_retriever.top_k)
        logger.info("HybridRetriever merged %d documents", len(ranked))
        return ranked[:top_k]

    def _normalize_scores(self, documents: List[Document]) -> List[Document]:
        """Normalize scores into [0, 1] for stable weighting."""

        if not documents:
            return []
        scores = [doc.score for doc in documents]
        max_score = max(scores)
        min_score = min(scores)
        span = max(max_score - min_score, 1e-9)
        normalized: List[Document] = []
        for doc in documents:
            scaled = (doc.score - min_score) / span if max_score != min_score else 1.0
            normalized.append(replace(doc, score=scaled))
        return normalized

    def _doc_key(self, doc: Document) -> str:
        """Build a stable merge key for duplicate detection."""

        if doc.id:
            return doc.id
        digest = hashlib.sha256(doc.content.lower().strip().encode("utf-8")).hexdigest()[:16]
        return f"doc-{digest}"
