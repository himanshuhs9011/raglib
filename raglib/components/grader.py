"""Batched LLM relevance grading of retrieved documents (one call for N documents)."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from raglib.components.evaluator import Evaluator
from raglib.core.parsing import parse_json, parse_label, parse_label_map
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)

LABEL_CORRECT = "correct"
LABEL_AMBIGUOUS = "ambiguous"
LABEL_INCORRECT = "incorrect"
GRADE_LABELS = (LABEL_CORRECT, LABEL_AMBIGUOUS, LABEL_INCORRECT)


class RelevanceGrader:
    """Labels each document correct / ambiguous / incorrect for a query.

    Uses a single batched LLM call (instead of one call per document). Without an
    LLM - or if the call fails - it falls back to lexical scoring with an upper and
    lower threshold, mirroring the CRAG paper's evaluator thresholds.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        evaluator: Optional[Evaluator] = None,
        upper_threshold: float = 0.5,
        lower_threshold: float = 0.2,
        max_chars_per_doc: int = 700,
    ):
        """Configure the grader."""

        self.llm_client = llm_client
        self.evaluator = evaluator or Evaluator()
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_chars_per_doc = max_chars_per_doc

    def grade(self, query: str, documents: Sequence[Document]) -> List[str]:
        """Return one label per document, in order."""

        if not documents:
            return []
        if self.llm_client is not None:
            labels = self._grade_llm(query, documents)
            if labels is not None:
                return labels
        return [self._grade_lexical(query, doc) for doc in documents]

    def _grade_llm(self, query: str, documents: Sequence[Document]) -> Optional[List[str]]:
        """Grade all documents in one LLM call; None on failure."""

        assert self.llm_client is not None
        numbered = "\n\n".join(
            f"[{index}] {doc.content[: self.max_chars_per_doc].strip()}" for index, doc in enumerate(documents, start=1)
        )
        prompt = (
            "Grade each retrieved document for the query.\n"
            "CORRECT = directly contains information that answers the query.\n"
            "AMBIGUOUS = partially relevant or uncertain.\n"
            "INCORRECT = irrelevant or misleading.\n\n"
            f"Query: {query}\n\n"
            f"Documents:\n{numbered}\n\n"
            'Reply with JSON only, e.g. {"grades": ["CORRECT", "INCORRECT"]} with exactly one grade per document, in order.'
        )
        try:
            raw = self.llm_client.complete(prompt=prompt, system="You are a strict retrieval relevance grader.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("RelevanceGrader LLM call failed: %s", exc)
            return None

        data = parse_json(raw)
        grades = data.get("grades") if isinstance(data, dict) else data if isinstance(data, list) else None
        labels: List[Optional[str]] = [None] * len(documents)
        if isinstance(grades, list):
            for index, value in enumerate(grades[: len(documents)]):
                labels[index] = parse_label(str(value), [label.upper() for label in GRADE_LABELS])
        else:
            for index, label in parse_label_map(raw, [label.upper() for label in GRADE_LABELS]).items():
                if 1 <= index <= len(documents):
                    labels[index - 1] = label
        if all(label is None for label in labels):
            if len(documents) == 1:
                single = parse_label(raw, [label.upper() for label in GRADE_LABELS])
                if single is not None:
                    return [single.lower()]
            logger.warning("RelevanceGrader could not parse grades; using lexical fallback")
            return None
        return [
            label.lower() if label is not None else self._grade_lexical(query, doc)
            for label, doc in zip(labels, documents)
        ]

    def _grade_lexical(self, query: str, document: Document) -> str:
        """Threshold lexical relevance into the three labels."""

        score = self.evaluator.score_document(query, document)
        if score >= self.upper_threshold:
            return LABEL_CORRECT
        if score < self.lower_threshold:
            return LABEL_INCORRECT
        return LABEL_AMBIGUOUS
