"""Query refinement component for corrective retrieval loops."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.components.query_rewriter import clean_llm_query
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class Refiner:
    """Refines user queries after poor retrieval quality."""

    def __init__(self, llm_client: BaseLLMClient, max_retries: int = 2):
        """Initialize the refiner with an LLM client and retry policy."""

        self.llm_client = llm_client
        self.max_retries = max_retries

    def refine(self, query: str, failed_docs: Optional[List[Document]] = None) -> str:
        """Generate a refined query from failed retrieval context."""

        snippets = "\n".join(f"- {doc.content[:180]}" for doc in (failed_docs or [])[:3]) or "(no results)"
        prompt = (
            "The search below returned poor results. Rewrite it as a better search query "
            "(different wording, key entities, synonyms) that keeps the original intent.\n"
            f"Original query: {query}\n"
            f"Poor results:\n{snippets}\n"
            "Return only the improved search query."
        )
        try:
            refined = clean_llm_query(self.llm_client.complete(prompt=prompt), fallback=query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Refiner failed: %s", exc)
            return query
        logger.debug("Refiner produced a refined query (changed=%s)", refined != query)
        return refined
