"""Memory RAG: multi-turn conversational RAG with per-session history."""

from __future__ import annotations

import logging
from typing import Any, List

from raglib.core.base import BaseRAG, PipelineContext
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class MemoryRAG(BaseRAG):
    """Condenses follow-ups into standalone queries and conditions answers on session history."""

    pipeline_name = "memory"

    def __init__(self, **kwargs: Any):
        """Initialize MemoryRAG; memory is always enabled."""

        kwargs["use_memory"] = True
        super().__init__(**kwargs)
        if self.conversation_store is None and self.memory_module is None:
            raise ValueError("MemoryRAG requires conversation_store or memory_module")

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Retrieve with the condensed standalone query."""

        if self.retriever is None:
            raise ValueError("MemoryRAG requires a retriever")
        ctx.trace.append("memory_augmented")
        return self.retrieve_corpus(ctx, ctx.search_query)
