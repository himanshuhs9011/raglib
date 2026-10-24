"""Query expansion component for multi-query retrieval."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from raglib.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class QueryExpander:
    """Generates alternative phrasings of a user query (the original is always first)."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None, max_variants: int = 4):
        """Initialize expansion settings and optional LLM client."""

        self.llm_client = llm_client
        self.max_variants = max_variants

    def expand(self, query: str) -> List[str]:
        """Expand a query into up to ``max_variants`` retrieval variants."""

        variants = [query]
        if self.llm_client is not None:
            prompt = (
                f"Write {self.max_variants - 1} alternative search queries for the question below. "
                "Use different wording and synonyms; keep the same meaning and all constraints.\n"
                f"Question: {query}\n"
                "Return one query per line, no numbering, no explanations."
            )
            try:
                raw = self.llm_client.complete(prompt=prompt)
            except Exception as exc:  # noqa: BLE001
                logger.warning("QueryExpander failed: %s", exc)
                raw = ""
            for line in raw.splitlines():
                cleaned = re.sub(r"^\s*(?:[-*•]|\d+[\).:-])\s*", "", line).strip().strip('"')
                if cleaned:
                    variants.append(cleaned)

        deduped: List[str] = []
        seen = set()
        for variant in variants:
            key = variant.lower().strip()
            if key and key not in seen:
                deduped.append(variant.strip())
                seen.add(key)
        return deduped[: self.max_variants]
