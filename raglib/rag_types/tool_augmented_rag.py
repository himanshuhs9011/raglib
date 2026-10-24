"""Tool-augmented RAG: merge corpus retrieval with documents produced by a tool hook."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from raglib.core.base import BaseRAG, PipelineContext
from raglib.core.fusion import reciprocal_rank_fusion
from raglib.schemas import Document

logger = logging.getLogger(__name__)

ToolHook = Callable[[str, List[Document]], List[Document]]


class ToolAugmentedRAG(BaseRAG):
    """Calls ``tool_call_hook(query, retrieved_docs)`` and fuses its documents with retrieval.

    For LLM-driven selection among several tools use ``AgenticRAG(tools=[...])``.
    """

    pipeline_name = "tool"

    def __init__(self, tool_call_hook: Optional[ToolHook] = None, **kwargs: Any):
        """Initialize with an optional tool hook."""

        super().__init__(**kwargs)
        self.tool_call_hook = tool_call_hook

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Retrieve, call the tool hook, and fuse both rankings."""

        retrieved = self.retrieve_corpus(ctx, ctx.search_query)
        tool_docs: List[Document] = []
        if self.tool_call_hook is not None:
            try:
                tool_docs = list(self.tool_call_hook(ctx.search_query, retrieved) or [])
            except Exception as exc:  # noqa: BLE001
                logger.warning("tool_call_hook failed: %s", exc)
                ctx.trace.append("tool_hook_failed")
        ctx.trace.append(f"tool_docs:{len(tool_docs)}")
        if not tool_docs:
            return retrieved
        return reciprocal_rank_fusion([tool_docs, retrieved], list_names=["tool", "retrieval"])
