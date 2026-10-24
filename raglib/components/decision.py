"""Decision engine for retrieval gating and mode selection."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.core.intents import is_comparison, is_greeting, needs_fresh_information
from raglib.core.parsing import parse_label
from raglib.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

RETRIEVAL_VECTOR: str = "vector"
RETRIEVAL_WEB: str = "web"
RETRIEVAL_HYBRID: str = "hybrid"
RETRIEVAL_NONE: str = "none"

ALLOWED_RETRIEVAL_TYPES = {
    RETRIEVAL_VECTOR,
    RETRIEVAL_WEB,
    RETRIEVAL_HYBRID,
    RETRIEVAL_NONE,
}


class DecisionEngine:
    """Decides whether retrieval is needed and what retrieval mode to use."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None, allow_web: bool = True):
        """Initialize decision policy with an optional LLM client."""

        self.llm_client = llm_client
        self.allow_web = allow_web

    def should_retrieve(self, query: str) -> bool:
        """Return whether retrieval should run for this query."""

        return self.retrieval_type(query) != RETRIEVAL_NONE

    def retrieval_type(self, query: str) -> str:
        """Return one of: vector, web, hybrid, none."""

        if not query or not query.strip():
            return RETRIEVAL_NONE
        if self.llm_client is not None:
            allowed = "vector, web, hybrid, none" if self.allow_web else "vector, none"
            prompt = (
                f"Decide retrieval_type for the user message. Allowed values: {allowed}.\n"
                "vector = search the private document corpus; web = needs live internet info; "
                "hybrid = both; none = small talk that needs no documents.\n"
                f"Message: {query}\n"
                "Reply with exactly one word."
            )
            try:
                label = parse_label(self.llm_client.complete(prompt=prompt), sorted(ALLOWED_RETRIEVAL_TYPES))
            except Exception as exc:  # noqa: BLE001
                logger.warning("DecisionEngine LLM call failed: %s", exc)
                label = None
            if label is not None:
                if not self.allow_web and label in {RETRIEVAL_WEB, RETRIEVAL_HYBRID}:
                    label = RETRIEVAL_VECTOR
                return label

        if is_greeting(query):
            return RETRIEVAL_NONE
        if self.allow_web and needs_fresh_information(query):
            return RETRIEVAL_WEB
        if self.allow_web and is_comparison(query):
            return RETRIEVAL_HYBRID
        return RETRIEVAL_VECTOR
