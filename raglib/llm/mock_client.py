"""Offline mock LLM client with deterministic, keyword-aware responses."""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from raglib.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class MockLLMClient(BaseLLMClient):
    """Provides realistic mock completions for local testing without API keys."""

    def complete(self, prompt: str, system: str = "") -> str:
        """Return a deterministic completion using prompt intent heuristics."""

        text = f"{system}\n{prompt}".lower()
        logger.debug("Mock complete invoked with prompt length=%d", len(prompt))

        if "rewrite" in text or "refine" in text:
            return self._rewrite_query(prompt)
        if "chunk:" in text or "semantic chunks" in text or "chunking" in text:
            return self._chunk_response(prompt)
        if "plan" in text or "sub-quer" in text or "decompose" in text:
            return self._plan_response(prompt)
        if "expand" in text or "variants" in text:
            return self._expand_query(prompt)
        if "should retrieve" in text or "retrieval_type" in text:
            return self._decision_response(prompt)
        if "reflect" in text or "sufficient" in text:
            return self._reflection_response(prompt)
        if "answer" in text or "context" in text:
            return self._answer_response(prompt)

        return "I can help with that. Please provide more context and any relevant constraints."

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Return a chat response by analyzing the latest user/system messages."""

        if not messages:
            return "I do not have any messages to respond to."

        combined = "\n".join(message.get("content", "") for message in messages)
        logger.debug("Mock chat invoked with %d messages", len(messages))
        return self.complete(prompt=combined)

    def _chunk_response(self, prompt: str) -> str:
        """Return deterministic CHUNK ranges for semantic chunking tests."""

        sentence_indexes = [
            int(match.group(1))
            for match in re.finditer(r"(?m)^\s*(\d+)\.\s+", prompt)
        ]
        if not sentence_indexes:
            return "CHUNK: 1-1"

        group_size = 3
        lines: List[str] = []
        start = sentence_indexes[0]
        last = sentence_indexes[-1]
        while start <= last:
            end = min(start + group_size - 1, last)
            lines.append(f"CHUNK: {start}-{end}")
            start = end + 1
        return "\n".join(lines)

    def _rewrite_query(self, prompt: str) -> str:
        """Return an improved query based on extracted keywords."""

        query = self._extract_query(prompt)
        tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 3]
        focus = " ".join(tokens[:4]) if tokens else "core concepts"
        return f"Refined query: Explain {focus} with practical examples and concrete trade-offs."

    def _plan_response(self, prompt: str) -> str:
        """Return a compact multi-step plan for agentic retrieval."""

        query = self._extract_query(prompt)
        return (
            f"1) Identify core entities in: {query}\n"
            "2) Retrieve authoritative facts for each entity\n"
            "3) Synthesize findings into a single answer"
        )

    def _expand_query(self, prompt: str) -> str:
        """Return query variants for multi-query retrieval."""

        query = self._extract_query(prompt)
        return (
            f"{query}\n"
            f"{query} best practices\n"
            f"{query} practical implementation\n"
            f"{query} common pitfalls"
        )

    def _decision_response(self, prompt: str) -> str:
        """Return retrieval decision guidance based on question intent."""

        lower = prompt.lower()
        if any(token in lower for token in ("latest", "current", "today", "news", "web")):
            return "hybrid"
        if any(token in lower for token in ("define", "what is", "explain")):
            return "vector"
        if any(token in lower for token in ("hello", "thanks", "hi")):
            return "none"
        return "vector"

    def _reflection_response(self, prompt: str) -> str:
        """Return a reflection decision for post-retrieval quality checks."""

        lower = prompt.lower()
        if "no documents" in lower or "0 documents" in lower:
            return "web_fallback"
        if "low score" in lower or "poor" in lower:
            return "retry"
        return "sufficient"

    def _answer_response(self, prompt: str) -> str:
        """Return a concise answer-style response with source-aware language."""

        query = self._extract_query(prompt)
        return (
            f"Based on the retrieved context, the key answer to '{query}' is to combine "
            "targeted retrieval, relevance filtering, and concise synthesis. "
            "Use trusted sources, verify overlap across documents, and keep the final "
            "response aligned with the user's goal."
        )

    def _extract_query(self, prompt: str) -> str:
        """Extract a likely query line from a prompt payload."""

        for line in prompt.splitlines():
            if "query" in line.lower() and ":" in line:
                return line.split(":", 1)[1].strip() or "the request"
        return prompt.strip().splitlines()[-1] if prompt.strip() else "the request"
