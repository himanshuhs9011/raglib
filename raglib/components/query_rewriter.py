"""Pre-retrieval query rewriting and history-aware question condensation."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

_PREFIXES = ("refined query:", "rewritten query:", "standalone question:", "search query:", "query:")


def clean_llm_query(text: str, fallback: str, max_length: int = 400) -> str:
    """Strip labels/quotes from an LLM-produced query; fall back when unusable."""

    candidate = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""
    lowered = candidate.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix):].strip()
            break
    candidate = candidate.strip().strip('"').strip("'").strip()
    if not candidate or len(candidate) > max_length:
        return fallback
    return candidate


class QueryRewriter:
    """Rewrites incoming queries into retrieval-friendly forms."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        """Initialize an optional LLM-backed query rewriter."""

        self.llm_client = llm_client

    def rewrite(self, query: str) -> str:
        """Rewrite a query while preserving user intent."""

        if self.llm_client is None:
            return query.strip()
        prompt = (
            "Rewrite the user question as a concise search query for a document index, "
            "keeping every entity, number and constraint.\n"
            f"Question: {query}\n"
            "Return only the rewritten query."
        )
        try:
            return clean_llm_query(self.llm_client.complete(prompt=prompt), fallback=query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("QueryRewriter failed: %s", exc)
            return query


class QueryCondenser:
    """Turns a follow-up message into a standalone question using chat history.

    Only called when history exists, so single-turn queries pay no extra LLM call.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        """Initialize with an optional LLM client (without one, the query is returned unchanged)."""

        self.llm_client = llm_client

    def condense(self, query: str, history: str) -> str:
        """Return a self-contained version of ``query`` given ``history``."""

        if self.llm_client is None or not history.strip():
            return query
        prompt = (
            "Given the conversation so far and a follow-up message, write a standalone question "
            "that can be understood without the conversation. Resolve pronouns and references. "
            "If the follow-up is already standalone, return it unchanged.\n\n"
            f"Conversation:\n{history}\n\n"
            f"Follow-up message: {query}\n"
            "Standalone question:"
        )
        try:
            return clean_llm_query(self.llm_client.complete(prompt=prompt), fallback=query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("QueryCondenser failed: %s", exc)
            return query
