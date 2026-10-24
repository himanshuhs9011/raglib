"""Context reduction utilities for prompt token budgeting."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Optional

from raglib.core.text import count_tokens, truncate_to_tokens
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class ContextReducer:
    """Trims retrieved context to fit a target token budget."""

    def __init__(self, max_context_tokens: int = 3000):
        """Initialize context token budget."""

        self.max_context_tokens = max_context_tokens

    def reduce(self, documents: List[Document], max_tokens: Optional[int] = None) -> List[Document]:
        """Reduce documents until the token budget is satisfied."""

        budget = max_tokens if max_tokens is not None else self.max_context_tokens
        reduced: List[Document] = []
        used_tokens = 0

        for doc in documents:
            doc_tokens = self._estimate_tokens(doc.content)
            if used_tokens + doc_tokens <= budget:
                reduced.append(doc)
                used_tokens += doc_tokens
                continue

            remaining = budget - used_tokens
            if remaining <= 0:
                break

            trimmed_content = self._truncate_to_tokens(doc.content, remaining)
            if trimmed_content.strip():
                reduced.append(
                    replace(
                        doc,
                        content=trimmed_content,
                        metadata={**doc.metadata, "trimmed": True},
                    )
                )
                used_tokens = budget
                break

        logger.info("ContextReducer used %d/%d tokens", used_tokens, budget)
        return reduced

    def _estimate_tokens(self, text: str) -> int:
        """Count tokens with the shared tokenizer."""

        return count_tokens(text)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to a token budget."""

        return truncate_to_tokens(text, max_tokens)
