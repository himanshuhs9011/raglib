"""Near-duplicate removal for retrieved document sets."""

from __future__ import annotations

import logging
from typing import List, Set

from raglib.core.text import token_set
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class Deduplicator:
    """Removes semantically similar duplicate document chunks."""

    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize deduplication threshold."""

        self.similarity_threshold = similarity_threshold

    def deduplicate(self, documents: List[Document]) -> List[Document]:
        """Return documents after near-duplicate filtering."""

        unique_docs: List[Document] = []
        signatures: List[Set[str]] = []

        for doc in documents:
            signature = self._signature(doc.content)
            if not signatures:
                unique_docs.append(doc)
                signatures.append(signature)
                continue

            is_duplicate = any(
                self._jaccard(signature, existing) >= self.similarity_threshold
                for existing in signatures
            )
            if not is_duplicate:
                unique_docs.append(doc)
                signatures.append(signature)

        logger.info("Deduplicator kept %d/%d documents", len(unique_docs), len(documents))
        return unique_docs

    def _signature(self, text: str) -> Set[str]:
        """Build a token-set signature for similarity checks."""

        return token_set(text)

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        """Compute Jaccard similarity between token signatures."""

        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
