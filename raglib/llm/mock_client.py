"""Offline mock LLM client with deterministic responses for every raglib prompt.

Used only when explicitly requested (``offline=True`` / ``chat_llm="mock"``) and in tests.
It never pretends to be a real model: answers are templated from the retrieved sources.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

from raglib.core.text import token_set
from raglib.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

_QUERY_LABELS = ("follow-up message:", "original query:", "original question:", "question:", "query:", "message:")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b")
_ENTITY_STOP = {"The", "A", "An", "This", "That", "These", "Those", "It", "In", "On", "At", "For", "And", "But", "When", "Where", "What", "Which", "Who", "How", "Why", "If", "As", "By", "With", "From", "Its", "Each", "All"}


class MockLLMClient(BaseLLMClient):
    """Deterministic offline LLM for tests and demos (no network, no API keys)."""

    def complete(self, prompt: str, system: str = "") -> str:
        """Return a deterministic completion keyed on the prompt's purpose."""

        text = f"{system}\n{prompt}"
        lower = text.lower()

        if "standalone question" in lower:
            return self._extract_query(prompt)
        if "grade each retrieved document" in lower:
            return self._grade_documents(prompt)
        if "knowledge strips" in lower:
            return "ALL"
        if "knowledge graph extraction" in lower:
            return self._extract_graph(prompt)
        if "key points" in lower and "community" in lower:
            return json.dumps({"points": [{"description": self._first_sentence(prompt.split("Community report:")[-1]), "score": 60}]})
        if "community report" in lower:
            return self._community_report(prompt)
        if "research agent" in lower:
            return self._agent_action(prompt)
        if "planning multi-hop retrieval" in lower:
            return json.dumps({"done": True})
        if "rank the passages" in lower:
            count = len(re.findall(r"(?m)^\[(\d+)\]", prompt))
            return json.dumps({"ranking": list(range(1, count + 1))})
        if "evaluation judge" in lower:
            return json.dumps({"score": 0.8, "reason": "mock judge"})
        if "retrieval decision module" in lower:
            return "YES"
        if "relevance evaluator" in lower:
            return "RELEVANT"
        if "faithfulness evaluator" in lower:
            return "SUPPORTED"
        if "utility evaluator" in lower:
            return "USEFUL"
        if "claims not supported" in lower:
            match = re.search(r"Current answer:\n(.*?)\n\n", prompt, re.DOTALL)
            return match.group(1).strip() if match else "Revised answer."
        if "choose the single best retrieval route" in lower:
            return "vector"
        if "retrieval_type" in lower:
            return self._decision_response(prompt)
        if "decompose the user question" in lower:
            return self._plan_response(prompt)
        if "alternative search queries" in lower:
            query = self._extract_query(prompt)
            return f"{query} explained\n{query} details\n{query} overview"
        if "chunk:" in lower or "semantic chunks" in lower or "chunking" in lower:
            return self._chunk_response(prompt)
        if "rewrite" in lower or "refine" in lower:
            return self._extract_query(prompt)
        if "<sources>" in lower:
            return self._answer_response(prompt)
        if "answer" in lower:
            return f"Mock answer to: {self._extract_query(prompt)}"
        return "I can help with that. Please provide more context."

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Return a chat response by analyzing the concatenated messages."""

        if not messages:
            return "I do not have any messages to respond to."
        return self.complete(prompt="\n".join(message.get("content", "") for message in messages))

    def _extract_query(self, prompt: str) -> str:
        """Extract the user query from a prompt."""

        for label in _QUERY_LABELS:
            for line in prompt.splitlines():
                if line.strip().lower().startswith(label):
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return value
        stripped = prompt.strip()
        return stripped.splitlines()[-1] if stripped else "the request"

    def _grade_documents(self, prompt: str) -> str:
        """Grade documents by lexical overlap with the query."""

        query_terms = token_set(self._extract_query(prompt), remove_stopwords=True)
        documents = re.findall(r"(?ms)^\[(\d+)\]\s*(.*?)(?=^\[\d+\]|\Z|^Reply with)", prompt)
        grades = []
        for _, content in documents:
            overlap = len(query_terms & token_set(content)) / max(1, len(query_terms))
            grades.append("CORRECT" if overlap >= 0.5 else "AMBIGUOUS" if overlap > 0 else "INCORRECT")
        return json.dumps({"grades": grades})

    def _extract_graph(self, prompt: str) -> str:
        """Heuristic entity/relation extraction: capitalized phrases + sentence co-occurrence."""

        body = prompt.split("Text:", 1)[-1]
        entities: Dict[str, Dict[str, str]] = {}
        relations = []
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            names = []
            for match in _ENTITY_RE.finditer(sentence):
                name = match.group(0)
                words = [w for w in name.split() if w not in _ENTITY_STOP]
                name = " ".join(words)
                if len(name) < 3:
                    continue
                names.append(name)
                entities.setdefault(name, {"name": name, "type": "CONCEPT", "description": sentence.strip()[:200]})
            for left, right in zip(names, names[1:]):
                if left != right:
                    relations.append({"source": left, "target": right, "type": "RELATED_TO", "description": sentence.strip()[:200], "weight": 1.0})
        return json.dumps({"entities": list(entities.values())[:15], "relations": relations[:20]})

    def _community_report(self, prompt: str) -> str:
        """Summarize a community from its entity list."""

        names = re.findall(r"(?m)^- ([^:]+):", prompt)
        title = ", ".join(names[:3]) or "Community"
        return json.dumps(
            {
                "title": title,
                "summary": f"This community groups {', '.join(names[:6])}.",
                "rating": 5.0,
                "findings": [f"{name} is a key entity." for name in names[:3]],
            }
        )

    def _agent_action(self, prompt: str) -> str:
        """Search once, then finish."""

        question = self._extract_query(prompt)
        if "Step 1:" in prompt:
            return json.dumps({"thought": "Evidence gathered.", "action": "finish", "input": ""})
        return json.dumps({"thought": "Search the documents.", "action": "search_docs", "input": question})

    def _chunk_response(self, prompt: str) -> str:
        """Return deterministic CHUNK ranges for semantic chunking tests."""

        indexes = [int(match.group(1)) for match in re.finditer(r"(?m)^\s*(\d+)\.\s+", prompt)]
        if not indexes:
            return "CHUNK: 1-1"
        lines: List[str] = []
        start, last = indexes[0], indexes[-1]
        while start <= last:
            end = min(start + 2, last)
            lines.append(f"CHUNK: {start}-{end}")
            start = end + 1
        return "\n".join(lines)

    def _plan_response(self, prompt: str) -> str:
        """Split a question on conjunctions into sub-queries."""

        query = self._extract_query(prompt)
        parts = [part.strip(" ?.") for part in re.split(r"\band\b|;", query) if part.strip(" ?.")]
        return "\n".join(parts or [query])

    def _decision_response(self, prompt: str) -> str:
        """Return a retrieval decision using whole-word cues."""

        from raglib.core.intents import is_greeting, needs_fresh_information

        message = self._extract_query(prompt)
        if is_greeting(message):
            return "none"
        if needs_fresh_information(message) and "web" in prompt.split("Message:")[0]:
            return "web"
        return "vector"

    def _answer_response(self, prompt: str) -> str:
        """Compose an extractive answer from the first source, with a citation."""

        query = self._extract_query(prompt)
        match = re.search(r'<source id="1"[^>]*>\n(.*?)\n</source>', prompt, re.DOTALL)
        if not match:
            return f"I don't know based on the available documents (query: {query})."
        return f"Based on the sources, regarding '{query}': {self._first_sentence(match.group(1))} [1]"

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Return the first sentence of ``text`` (max 300 chars)."""

        sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
        return sentence[:300]
