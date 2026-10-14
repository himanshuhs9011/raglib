"""Answer generation component with prompt template support."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE: str = (
    "You are a precise assistant. Use only the provided context when possible.\n"
    "Query: {query}\n"
    "Memory:\n{memory_context}\n"
    "Context:\n{context}\n"
    "Answer:"
)


class Generator:
    """Builds prompts and generates final answers from retrieved context."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_template: Optional[str] = None,
        max_context_tokens: int = 3000,
    ):
        """Initialize generator with LLM client and prompt settings."""

        self.llm_client = llm_client
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.max_context_tokens = max_context_tokens

    def generate(
        self,
        query: str,
        documents: List[Document],
        memory_context: str = "",
        reasoning_trace: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate an answer from query, documents, and optional memory context."""

        trace = list(reasoning_trace) if reasoning_trace is not None else []
        context_docs = self._truncate_documents(documents)
        context_block = self._build_context(context_docs)

        prompt = self._render_prompt(
            query=query,
            memory_context=memory_context,
            context=context_block,
        )

        context_tokens = sum(self._estimate_tokens(doc.content) for doc in context_docs)
        logger.info(
            "Generator sending prompt with %d context docs and %d context tokens",
            len(context_docs),
            context_tokens,
        )
        answer = self.llm_client.complete(prompt=prompt, system="Answer with grounded reasoning.")
        trace.append(f"generated_with_docs={len(context_docs)}")

        return GenerationResult(answer=answer, sources=context_docs, reasoning_trace=trace)

    def _render_prompt(self, query: str, memory_context: str, context: str) -> str:
        """Safely render prompt template without failing on stray braces."""

        try:
            return self.prompt_template.format(
                query=query,
                memory_context=memory_context,
                context=context,
            )
        except (KeyError, ValueError, IndexError):
            rendered = self.prompt_template
            rendered = rendered.replace("{query}", query)
            rendered = rendered.replace("{memory_context}", memory_context)
            rendered = rendered.replace("{context}", context)
            return rendered

    def _truncate_documents(self, documents: List[Document]) -> List[Document]:
        """Trim document set to fit max context token budget."""

        selected: List[Document] = []
        used = 0
        for doc in documents:
            tokens = self._estimate_tokens(doc.content)
            if used + tokens > self.max_context_tokens:
                break
            selected.append(doc)
            used += tokens
        return selected

    def _build_context(self, documents: List[Document]) -> str:
        """Build a source-tagged context block from documents."""

        lines: List[str] = []
        for idx, doc in enumerate(documents, start=1):
            lines.append(f"[{idx}] ({doc.source}) {doc.content}")
        return "\n\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token usage from text length."""

        return max(1, len(text) // 4)
