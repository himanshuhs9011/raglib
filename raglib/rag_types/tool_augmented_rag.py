"""Tool-augmented RAG orchestration with external tool hooks."""

from __future__ import annotations

import hashlib
import logging
from typing import Callable, Dict, List, Optional

from raglib.components.deduplicator import Deduplicator
from raglib.components.generator import Generator
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)


class ToolAugmentedRAG(BaseRAG):
    """Merges retrieval context with tool-derived documents before generation."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        deduplicator: Optional[Deduplicator] = None,
        tool_call_hook: Optional[Callable[[str, List[Document]], List[Document]]] = None,
        **kwargs,
    ):
        """Initialize ToolAugmentedRAG with optional tool integration hook."""

        super().__init__(
            retriever=retriever,
            generator=generator,
            deduplicator=deduplicator,
            **kwargs,
        )
        self.tool_call_hook = tool_call_hook

    def run(self, query: str) -> GenerationResult:
        """Retrieve, augment with tool results, merge, and generate."""

        if self.retriever is None or self.generator is None:
            raise ValueError("ToolAugmentedRAG requires retriever and generator")

        active_query = self.pre_retrieve(query)
        retrieved = self.retriever.retrieve(active_query)
        tool_docs = self.tool_call_hook(active_query, retrieved) if self.tool_call_hook else []

        merged: Dict[str, Document] = {}
        for doc in retrieved + tool_docs:
            digest = hashlib.sha256(doc.content.lower().strip().encode("utf-8")).hexdigest()[:16]
            key = doc.id or f"doc-{digest}"
            if key not in merged or doc.score > merged[key].score:
                merged[key] = doc

        documents = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        if self.deduplicator is not None:
            documents = self.deduplicator.deduplicate(documents)

        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(
            query=active_query,
            documents=documents,
            reasoning_trace=[f"tool_docs:{len(tool_docs)}"],
        )
        return self.post_generate(result)
