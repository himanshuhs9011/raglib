"""Answer generation: grounded prompt, token-budgeted context packing, citations, streaming."""

from __future__ import annotations

import logging
import ntpath
import re
from dataclasses import replace
from typing import Iterator, List, Optional, Tuple

from raglib.cache import cache_dependencies
from raglib.core.prompt_safety import STABLE_RULE, data_rule, delimiter_for, wrap
from raglib.core.runtime import UsageMeter, usage_scope
from raglib.core.text import count_tokens, truncate_to_tokens
from raglib.core.tracing import trace_stage
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Citation, Document, GenerationResult

logger = logging.getLogger(__name__)


def source_parents(documents: List[Document]) -> List[str]:
    """Ids of the documents (not chunks) behind ``documents``: what an answer depends on."""

    return list(dict.fromkeys(str(doc.metadata.get("parent_id") or doc.id) for doc in documents))


def source_label(doc: Document) -> str:
    """Human-readable source name: title, file name (never the full local path) + page, or URL."""

    meta = doc.metadata or {}
    label = meta.get("title") or meta.get("source")
    if not label and meta.get("file"):
        label = ntpath.basename(str(meta["file"]))  # splits on / and \ on every OS
    label = str(label or meta.get("url") or doc.source or doc.id)
    if meta.get("page") is not None:
        label = f"{label} p.{meta['page']}"
    return label

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a careful assistant that answers strictly from the provided sources.\n"
    "Rules:\n"
    "1. Use only facts stated in the source blocks (and the conversation, if given).\n"
    "2. Cite every factual sentence with the source number in square brackets, e.g. [1] or [2][3].\n"
    "3. When the sources contain the answer, state it directly and do not hedge (the question may use "
    "a shorter or different name for something in the sources). Only if they do not contain it, say you don't know based on "
    "the available documents. Do not guess.\n"
    "4. " + STABLE_RULE
)

DEFAULT_PROMPT_TEMPLATE: str = (
    "{memory_context}"
    "<sources>\n{context}\n</sources>\n\n"
    "{data_rule}\n\n"
    "Query: {query}\n\n"
    "Write a concise, well-structured answer with citations:"
)

NO_CONTEXT_TEMPLATE: str = "{memory_context}Query: {query}\n\nAnswer concisely:"

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


class Generator:
    """Builds grounded prompts and generates final answers from retrieved context."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_template: Optional[str] = None,
        max_context_tokens: int = 3000,
        system_prompt: Optional[str] = None,
        reorder: str = "edges",
        min_doc_tokens: int = 40,
        max_supplementary_tokens: int = 1200,
    ):
        """Initialize generator.

        ``reorder="edges"`` places the strongest sources at the start and end of the
        context (mitigates "lost in the middle"); use ``"none"`` to keep ranking order.
        ``max_supplementary_tokens``: separate budget for documents marked
        ``metadata["supplementary"]`` (e.g. GraphRAG's entity/relation/community summaries). They come
        first in the context but can never push the evidence chunks out of ``max_context_tokens``.
        """

        if reorder not in {"edges", "none"}:
            raise ValueError("reorder must be 'edges' or 'none'")
        self.llm_client = llm_client
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.max_context_tokens = max_context_tokens
        self.reorder = reorder
        self.min_doc_tokens = min_doc_tokens
        self.max_supplementary_tokens = max(0, max_supplementary_tokens)

    def generate(
        self,
        query: str,
        documents: List[Document],
        memory_context: str = "",
        reasoning_trace: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate an answer from query, documents, and optional conversation context."""

        trace = list(reasoning_trace) if reasoning_trace is not None else []
        context_docs, prompt = self.build_prompt(query, documents, memory_context)
        # Cached answers are tagged with their source documents, so a document update invalidates them.
        with trace_stage("generate", docs=len(context_docs)) as meta, usage_scope() as meter, cache_dependencies(
            source_parents(context_docs)
        ):
            answer = self.llm_client.complete(prompt=prompt, system=self.system_prompt)
            usage = self._usage(answer, prompt, meter)
            meta.update(usage)
        trace.append(f"generated_with_docs={len(context_docs)}")
        return self._result(answer, context_docs, trace, prompt, meter, usage=usage)

    def stream(
        self,
        query: str,
        documents: List[Document],
        memory_context: str = "",
    ) -> Tuple[List[Document], Iterator[str]]:
        """Return the packed sources and a token iterator for the answer."""

        context_docs, prompt = self.build_prompt(query, documents, memory_context)
        return context_docs, self.llm_client.stream(prompt=prompt, system=self.system_prompt)

    def build_prompt(self, query: str, documents: List[Document], memory_context: str = "") -> Tuple[List[Document], str]:
        """Pack documents into the token budget and render the prompt (traced as ``prompt_build``)."""

        with trace_stage("prompt_build", candidates=len(documents)) as meta:
            context_docs, prompt = self._build_prompt(query, documents, memory_context)
            meta["docs"] = len(context_docs)
        return context_docs, prompt

    def _build_prompt(self, query: str, documents: List[Document], memory_context: str) -> Tuple[List[Document], str]:
        """Untraced body of ``build_prompt``."""

        context_docs = self._pack_documents(documents)
        # Unguessable per-prompt tag (stable for the same question + sources: cache-friendly).
        delimiter = delimiter_for(query, *(doc.content for doc in context_docs))
        # Earlier answers may quote documents: the conversation is data too.
        memory_block = f"<conversation>\n{wrap(memory_context, delimiter)}\n</conversation>\n\n" if memory_context.strip() else ""
        use_no_context = not context_docs and self.prompt_template == DEFAULT_PROMPT_TEMPLATE
        template = NO_CONTEXT_TEMPLATE if use_no_context else self.prompt_template
        prompt = self._render_prompt(
            template=template,
            query=query,
            memory_context=memory_block,
            context=self._build_context(context_docs, delimiter),
            data_rule=data_rule(delimiter),
        )
        return context_docs, prompt

    def finalize(
        self,
        answer: str,
        context_docs: List[Document],
        trace: Optional[List[str]] = None,
        meter: Optional[UsageMeter] = None,
    ) -> GenerationResult:
        """Build a GenerationResult (with citations) from a streamed answer."""

        return self._result(answer, context_docs, list(trace or []), prompt="", meter=meter)

    def _result(
        self,
        answer: str,
        context_docs: List[Document],
        trace: List[str],
        prompt: str,
        meter: Optional[UsageMeter] = None,
        usage: Optional[dict] = None,
    ) -> GenerationResult:
        """Assemble the result with parsed citations (traced as ``post_process``)."""

        with trace_stage("post_process") as meta:
            result = GenerationResult(
                answer=answer,
                sources=context_docs,
                reasoning_trace=trace,
                citations=self.extract_citations(answer, context_docs),
                usage=usage if usage is not None else self._usage(answer, prompt, meter),
            )
            meta["citations"] = len(result.citations)
        return result

    def _usage(self, answer: str, prompt: str, meter: Optional[UsageMeter]) -> dict:
        """Token usage of the answer call.

        ``prompt_tokens`` / ``completion_tokens`` are the provider's counts when it reports
        them (``generation_usage_estimated`` is then False), otherwise local estimates.
        """

        if meter is not None and meter.llm_calls and not meter.estimated_calls:
            return {
                "prompt_tokens": meter.prompt_tokens,
                "completion_tokens": meter.completion_tokens,
                "generation_usage_estimated": False,
            }
        return {
            "prompt_tokens": count_tokens(prompt) + count_tokens(self.system_prompt) if prompt else 0,
            "completion_tokens": count_tokens(answer),
            "generation_usage_estimated": True,
        }

    @staticmethod
    def extract_citations(answer: str, context_docs: List[Document]) -> List[Citation]:
        """Map ``[n]`` markers in the answer to the n-th packed source."""

        citations: List[Citation] = []
        seen = set()
        for match in _CITATION_RE.finditer(answer or ""):
            index = int(match.group(1))
            if index in seen or not 1 <= index <= len(context_docs):
                continue
            seen.add(index)
            doc = context_docs[index - 1]
            citations.append(
                Citation(index=index, document_id=doc.id, source=source_label(doc), snippet=doc.content[:200])
            )
        return citations

    def _render_prompt(self, template: str, query: str, memory_context: str, context: str, data_rule: str = "") -> str:
        """Render the template without failing on stray braces in custom templates.

        A custom template without ``{data_rule}`` still gets the rule, right after the context.
        """

        if context and "{data_rule}" not in template:
            template = template.replace("{context}", "{context}\n\n{data_rule}", 1)
        try:
            return template.format(query=query, memory_context=memory_context, context=context, data_rule=data_rule)
        except (KeyError, ValueError, IndexError):
            rendered = template.replace("{query}", query)
            rendered = rendered.replace("{memory_context}", memory_context)
            rendered = rendered.replace("{data_rule}", data_rule)
            return rendered.replace("{context}", context)

    def _pack_documents(self, documents: List[Document]) -> List[Document]:
        """Fit evidence into ``max_context_tokens`` and supplementary context into its own budget."""

        supplementary = [doc for doc in documents if doc.metadata.get("supplementary")]
        evidence = [doc for doc in documents if not doc.metadata.get("supplementary")]
        selected = self._fit(evidence, self.max_context_tokens)
        if self.reorder == "edges" and len(selected) > 2:
            selected = self._edge_order(selected)
        return self._fit(supplementary, self.max_supplementary_tokens) + selected

    def _fit(self, documents: List[Document], budget: int) -> List[Document]:
        """Greedily fit documents into ``budget`` tokens, trimming (not dropping) the last one."""

        selected: List[Document] = []
        used = 0
        for doc in documents:
            if not doc.content.strip():
                continue
            tokens = count_tokens(doc.content) + 12  # block markup overhead
            remaining = budget - used
            if tokens <= remaining:
                selected.append(doc)
                used += tokens
                continue
            if remaining >= self.min_doc_tokens:
                trimmed = truncate_to_tokens(doc.content, remaining - 12)
                if trimmed.strip():
                    selected.append(replace(doc, content=trimmed, metadata={**doc.metadata, "trimmed": True}))
            break
        return selected

    @staticmethod
    def _edge_order(documents: List[Document]) -> List[Document]:
        """Interleave ranks so the best documents sit at both ends: 1,3,5,...,6,4,2."""

        front = documents[0::2]
        back = documents[1::2]
        return front + list(reversed(back))

    @staticmethod
    def _build_context(documents: List[Document], delimiter: Optional[str] = None) -> str:
        """Render numbered source blocks in an unguessable tag, with escaped content and labels."""

        tag = delimiter or delimiter_for(*(doc.content for doc in documents))
        return "\n".join(
            wrap(doc.content, tag, id=idx, name=source_label(doc)) for idx, doc in enumerate(documents, start=1)
        )
