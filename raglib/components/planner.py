"""Query decomposition and iterative next-hop planning."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence

from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class Planner:
    """Breaks complex queries into sub-queries and decides the next retrieval hop."""

    def __init__(self, llm_client: BaseLLMClient, max_sub_queries: int = 4):
        """Initialize planner with an LLM client."""

        self.llm_client = llm_client
        self.max_sub_queries = max_sub_queries

    def plan(self, query: str) -> List[str]:
        """Decompose a query into ordered, self-contained sub-queries."""

        prompt = (
            "Decompose the user question into the minimal set of self-contained search queries "
            f"needed to answer it (at most {self.max_sub_queries}). If it is already simple, return it unchanged.\n"
            f"Question: {query}\n"
            "Return one search query per line, nothing else."
        )
        try:
            raw = self.llm_client.complete(prompt=prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Planner failed: %s", exc)
            raw = ""
        sub_queries: List[str] = []
        for line in raw.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*•]|\d+[\).:-])\s*", "", line).strip()
            if cleaned and cleaned.lower() not in {q.lower() for q in sub_queries}:
                sub_queries.append(cleaned)
        if not sub_queries:
            sub_queries = [query]
        return sub_queries[: self.max_sub_queries]

    def next_step(self, question: str, evidence: Sequence[Document], previous_queries: Sequence[str]) -> Optional[str]:
        """Return the next search query needed, or None when the evidence is sufficient."""

        snippets = "\n".join(f"- {doc.content[:300].strip()}" for doc in evidence[:8]) or "(none yet)"
        asked = "\n".join(f"- {q}" for q in previous_queries) or "(none)"
        prompt = (
            "You are planning multi-hop retrieval.\n"
            f"Question: {question}\n\n"
            f"Searches already run:\n{asked}\n\n"
            f"Evidence collected so far:\n{snippets}\n\n"
            "If the evidence is enough to answer the question, reply with JSON "
            '{"done": true}. Otherwise reply with JSON {"done": false, "next_query": "<the next search query, '
            'using entities found in the evidence>"}.'
        )
        data = self.llm_client.complete_json(prompt, default={"done": True})
        if not isinstance(data, dict) or data.get("done", True):
            return None
        next_query = str(data.get("next_query") or "").strip()
        if not next_query or next_query.lower() in {q.lower() for q in previous_queries}:
            return None
        return next_query
