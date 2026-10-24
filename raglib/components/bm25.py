"""Incremental Okapi BM25 index with an inverted list (pure Python, Unicode-aware)."""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from raglib.core.text import tokenize

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 sparse index supporting add / remove / search without full rebuilds.

    Search only touches postings for the query terms, so cost scales with the
    number of matching documents, not the corpus size.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, remove_stopwords: bool = True):
        """Initialize BM25 hyperparameters and empty index structures."""

        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self._postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._doc_lengths: Dict[str, int] = {}
        self._doc_terms: Dict[str, List[str]] = {}
        self._total_length = 0

    def __len__(self) -> int:
        """Return the number of indexed documents."""

        return len(self._doc_lengths)

    def __contains__(self, doc_id: str) -> bool:
        """Return True when ``doc_id`` is indexed."""

        return doc_id in self._doc_lengths

    def _tokens(self, text: str) -> List[str]:
        """Tokenize text for indexing/search."""

        tokens = tokenize(text, remove_stopwords=self.remove_stopwords)
        # Never index an empty doc because of stopword removal (e.g. "What is this?").
        return tokens or tokenize(text)

    def add(self, doc_id: str, text: str) -> None:
        """Index (or re-index) one document."""

        if doc_id in self._doc_lengths:
            self.remove(doc_id)
        tokens = self._tokens(text)
        counts = Counter(tokens)
        for term, frequency in counts.items():
            self._postings[term][doc_id] = frequency
        self._doc_terms[doc_id] = list(counts)
        self._doc_lengths[doc_id] = len(tokens)
        self._total_length += len(tokens)

    def add_many(self, items: Iterable[Tuple[str, str]]) -> None:
        """Index many (doc_id, text) pairs."""

        for doc_id, text in items:
            self.add(doc_id, text)

    def remove(self, doc_id: str) -> None:
        """Remove one document from the index."""

        length = self._doc_lengths.pop(doc_id, None)
        if length is None:
            return
        self._total_length -= length
        for term in self._doc_terms.pop(doc_id, []):
            postings = self._postings.get(term)
            if postings is None:
                continue
            postings.pop(doc_id, None)
            if not postings:
                del self._postings[term]

    def score(self, query: str, doc_ids: Iterable[str]) -> Dict[str, float]:
        """BM25 score of ``query`` for specific documents (0.0 when nothing matches)."""

        targets = {doc_id for doc_id in doc_ids if doc_id in self._doc_lengths}
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in targets}
        if not targets:
            return scores
        total_docs = len(self._doc_lengths)
        avg_length = (self._total_length / total_docs) or 1.0
        for term in set(self._tokens(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (total_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc_id in targets:
                frequency = postings.get(doc_id)
                if not frequency:
                    continue
                length = self._doc_lengths[doc_id]
                denom = frequency + self.k1 * (1 - self.b + self.b * length / avg_length)
                scores[doc_id] += idf * frequency * (self.k1 + 1) / denom
        return scores

    def search(
        self,
        query: str,
        top_k: int,
        allow: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, float]]:
        """Return up to ``top_k`` (doc_id, score) pairs, best first."""

        if not self._doc_lengths or top_k <= 0:
            return []
        query_terms = set(self._tokens(query))
        if not query_terms:
            return []

        total_docs = len(self._doc_lengths)
        avg_length = (self._total_length / total_docs) or 1.0
        scores: Dict[str, float] = defaultdict(float)
        for term in query_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for doc_id, frequency in postings.items():
                if allow is not None and not allow(doc_id):
                    continue
                length = self._doc_lengths[doc_id]
                denom = frequency + self.k1 * (1 - self.b + self.b * length / avg_length)
                scores[doc_id] += idf * frequency * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
