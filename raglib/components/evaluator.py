"""Retrieval quality evaluator used for corrective feedback loops."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Set

from raglib.core.text import token_set
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class Evaluator:
    """Scores relevance and filters low-quality retrieval results."""

    def __init__(self, relevance_threshold: float = 0.3):
        """Initialize evaluator with a minimum relevance threshold."""

        self.relevance_threshold = relevance_threshold

    def evaluate(self, query: str, documents: List[Document]) -> List[Document]:
        """Filter documents by relevance score threshold."""

        passed: List[Document] = []
        for doc in documents:
            score = self.score_document(query, doc)
            if score >= self.relevance_threshold:
                passed.append(replace(doc, score=score))

        logger.info(
            "Evaluator kept %d/%d documents with threshold %.2f",
            len(passed),
            len(documents),
            self.relevance_threshold,
        )
        return passed

    def score_document(self, query: str, doc: Document) -> float:
        """Compute a relevance score for a single document."""

        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(doc.content)
        if not query_terms or not doc_terms:
            return 0.0

        overlap = len(query_terms & doc_terms) / len(query_terms)
        density = len(query_terms & doc_terms) / max(len(doc_terms), 1)
        phrase_bonus = 0.2 if query.lower() in doc.content.lower() else 0.0
        score = min(1.0, overlap * 0.7 + density * 0.3 + phrase_bonus)
        return score

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into a normalized set of terms."""

        return token_set(text, remove_stopwords=True) or token_set(text)
