"""Self-RAG (Asai et al. 2023) as an LLM-as-judge state machine with a hard call budget.

True Self-RAG uses a fine-tuned model that emits reflection tokens; here a general LLM
plays each critic role ([Retrieve], [IsRel], [IsSup], [IsUse]) via short judge prompts.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from raglib.core.base import BaseRAG, PipelineContext
from raglib.core.intents import is_greeting
from raglib.core.parsing import parse_label
from raglib.core.text import token_set
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)

RETRIEVE_YES = "yes"
RETRIEVE_NO = "no"
ISREL_RELEVANT = "relevant"
ISREL_NOT = "not_relevant"
ISREL_REWRITE = "rewrite"
ISSUP_SUPPORTED = "supported"
ISSUP_NOT = "not_supported"
ISSUP_CONFLICT = "conflicting"
ISUSE_USEFUL = "useful"
ISUSE_NOT = "not_useful"
ISUSE_MORE = "needs_more"

MAX_RETRIEVE_RETRIES = 3
MAX_REVISE_RETRIES = 1
MAX_REWRITE_RETRIES = 2
DEFAULT_MAX_LLM_CALLS = 10

NO_ANSWER_TEXT = (
    "I was unable to find a well-supported answer in the available documents. "
    "Please rephrase the question or provide more context."
)


@dataclass
class RunState:
    """Carries mutable state for one Self-RAG execution."""

    original_query: str
    active_query: str
    documents: List[Document] = field(default_factory=list)
    answer: str = ""
    result: Optional[GenerationResult] = None
    trace: List[str] = field(default_factory=list)
    retrieve_count: int = 0
    revise_count: int = 0
    rewrite_count: int = 0
    llm_calls: int = 0
    terminal: str = ""  # end | direct | no_answer | budget


class SelfRAG(BaseRAG):
    """Self-reflective RAG: decide -> retrieve -> IsRel -> generate -> IsSup -> IsUse loop."""

    pipeline_name = "self"
    uses_decision_engine = False

    def __init__(
        self,
        max_retrieve: int = MAX_RETRIEVE_RETRIES,
        max_revise: int = MAX_REVISE_RETRIES,
        max_rewrite: int = MAX_REWRITE_RETRIES,
        max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
        **kwargs: Any,
    ):
        """Initialize SelfRAG with retry guards and a total judge-call budget."""

        super().__init__(**kwargs)
        self.max_retrieve = max_retrieve
        self.max_revise = max_revise
        self.max_rewrite = max_rewrite
        self.max_llm_calls = max_llm_calls
        self._local = threading.local()

    @property
    def _state(self) -> Optional[RunState]:
        """Per-thread run state, so one SelfRAG instance can serve concurrent requests."""

        return getattr(self._local, "state", None)

    @_state.setter
    def _state(self, value: Optional[RunState]) -> None:
        self._local.state = value

    # ------------------------------------------------------------------ state machine
    def answer(self, ctx: PipelineContext) -> GenerationResult:
        """Execute the Self-RAG graph."""

        if self.generator is None:
            raise ValueError("SelfRAG requires a generator.")
        state = RunState(original_query=ctx.query, active_query=ctx.search_query, trace=["__start__"])
        self._state = state

        verdict = RETRIEVE_NO if ctx.metadata.get("smalltalk") else self._llm_decide_retrieval(state.active_query)
        state.trace.append(f"decide_retrieval:{verdict}")
        if verdict == RETRIEVE_NO:
            state.result = self.generate(ctx, [])
            state.terminal = "direct"
            return self._finalize(ctx, state)

        self._retrieve(ctx, state)
        while True:
            relevance = self._judge_relevance(state)
            if relevance in {ISREL_NOT, ISREL_REWRITE}:
                if state.retrieve_count >= self.max_retrieve or not self._rewrite(state):
                    state.terminal = "no_answer"
                    state.trace.append("no_answer_found:retrieve_exhausted")
                    return self._finalize(ctx, state)
                self._retrieve(ctx, state)
                continue

            self._generate(ctx, state)
            support = self._judge_support(state)
            while support == ISSUP_NOT and state.revise_count < self.max_revise:
                self._revise(state)
                support = self._judge_support(state)
            if support == ISSUP_CONFLICT:
                if state.retrieve_count < self.max_retrieve and self._rewrite(state):
                    self._retrieve(ctx, state)
                    continue
            if support == ISSUP_NOT:
                state.trace.append("warning:answer_not_fully_supported")

            usefulness = self._judge_usefulness(state)
            if usefulness == ISUSE_NOT:
                state.terminal = "no_answer"
                state.trace.append("no_answer_found:not_useful")
                return self._finalize(ctx, state)
            if usefulness == ISUSE_MORE and state.retrieve_count < self.max_retrieve and self._rewrite(state):
                self._retrieve(ctx, state)
                continue
            state.terminal = "end"
            state.trace.append("END")
            return self._finalize(ctx, state)

    # ------------------------------------------------------------------ nodes
    def _retrieve(self, ctx: PipelineContext, state: RunState) -> None:
        """Retrieve for the active query."""

        state.retrieve_count += 1
        with ctx.tracer.stage("retrieve", attempt=state.retrieve_count) as meta:
            documents = self.retrieve_corpus(ctx, state.active_query) if self.retriever is not None else []
            documents = self.post_retrieve(state.active_query, documents)
            documents = self.process_documents(ctx, documents)
            meta["docs"] = len(documents)
        state.documents = documents
        state.trace.append(f"retrieve:{len(documents)}_docs")

    def _generate(self, ctx: PipelineContext, state: RunState) -> None:
        """Generate an answer from the current documents."""

        ctx.trace = state.trace
        docs = self.pre_generate(state.active_query, state.documents)
        state.result = self.generate(ctx, docs)
        state.answer = state.result.answer
        state.documents = state.result.sources
        state.trace = list(state.result.reasoning_trace)
        state.trace.append("generate_from_context:done")

    def _judge_relevance(self, state: RunState) -> str:
        """[IsRel] node."""

        if not state.documents:
            state.trace.append("is_relevant:no_docs")
            return ISREL_NOT
        verdict = self._llm_is_relevant(state.active_query, state.documents)
        state.trace.append(f"is_relevant:{verdict}")
        return verdict

    def _judge_support(self, state: RunState) -> str:
        """[IsSup] node."""

        verdict = self._llm_is_sup(state.active_query, state.documents, state.answer)
        state.trace.append(f"is_sup:{verdict}")
        return verdict

    def _judge_usefulness(self, state: RunState) -> str:
        """[IsUse] node."""

        verdict = self._llm_is_use(state.original_query, state.answer)
        state.trace.append(f"is_use:{verdict}")
        return verdict

    def _revise(self, state: RunState) -> None:
        """Revise the answer to remove unsupported claims."""

        state.revise_count += 1
        revised = self._call(
            "The answer below contains claims not supported by the sources.\n\n"
            f"Query: {state.active_query}\n\n"
            f"Sources:\n{self._snippets(state.documents, 400)}\n\n"
            f"Current answer:\n{state.answer}\n\n"
            "Rewrite the answer so every claim is supported by the sources, keeping the [n] citations. "
            "Do not add new facts.\n\nRevised answer:"
        )
        if revised and revised.strip():
            state.answer = revised.strip()
            if state.result is not None:
                state.result.answer = state.answer
        state.trace.append(f"revise_answer:{state.revise_count}")

    def _rewrite(self, state: RunState) -> bool:
        """Rewrite the active query; returns False when the rewrite budget is exhausted."""

        if state.rewrite_count >= self.max_rewrite:
            state.trace.append("rewrite_exhausted")
            return False
        state.rewrite_count += 1
        rewritten = self._llm_rewrite_question(state.active_query).strip() or state.active_query
        state.active_query = rewritten
        state.trace.append(f"rewrite_question:{state.rewrite_count}->{rewritten[:60]}")
        return True

    def _finalize(self, ctx: PipelineContext, state: RunState) -> GenerationResult:
        """Build the final result for every terminal path."""

        state.trace.append("__end__")
        if state.terminal == "no_answer" or state.result is None:
            result = GenerationResult(answer=NO_ANSWER_TEXT, sources=state.documents, reasoning_trace=state.trace)
        else:
            result = state.result
            result.reasoning_trace = state.trace
            if state.terminal == "direct":
                result.sources = []
        result.metadata = {**result.metadata, "self_rag_llm_calls": state.llm_calls, "terminal": state.terminal}
        self._state = None
        return self.post_generate(result)

    # ------------------------------------------------------------------ judges
    def _call(self, prompt: str) -> Optional[str]:
        """Call the judge LLM within the per-run budget; None when unavailable/exhausted/failed."""

        state = self._state
        if self.llm_client is None:
            return None
        if state is not None:
            if state.llm_calls >= self.max_llm_calls:
                if "budget_exhausted" not in state.trace:
                    state.trace.append("budget_exhausted")
                return None
            state.llm_calls += 1
        try:
            return self.llm_client.complete(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SelfRAG judge call failed: %s", exc)
            if state is not None:
                state.trace.append("judge_error")
            return None

    def _llm_decide_retrieval(self, query: str) -> str:
        """[Retrieve] token: should we retrieve for this query?"""

        response = self._call(
            "You are a retrieval decision module.\n\n"
            f"Question: {query}\n\n"
            "Should retrieve external documents to answer the user query accurately?\n"
            "Reply with exactly one word: YES or NO."
        )
        label = parse_label(response or "", ["YES", "NO"])
        if label == "NO":
            return RETRIEVE_NO
        if label == "YES":
            return RETRIEVE_YES
        return RETRIEVE_NO if is_greeting(query) else RETRIEVE_YES

    def _llm_is_relevant(self, query: str, documents: List[Document]) -> str:
        """[IsRel] token: are the retrieved docs relevant enough to proceed?"""

        response = self._call(
            "You are a relevance evaluator.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved documents:\n{self._snippets(documents, 300)}\n\n"
            "Reply with exactly one word: RELEVANT (they help answer), NOT_RELEVANT, "
            "or REWRITE (the query should be reformulated)."
        )
        label = parse_label(response or "", ["RELEVANT", "NOT_RELEVANT", "IRRELEVANT", "REWRITE"])
        if label in {"NOT_RELEVANT", "IRRELEVANT"}:
            return ISREL_NOT
        if label == "REWRITE":
            return ISREL_REWRITE
        if label == "RELEVANT":
            return ISREL_RELEVANT
        return self._lexical_relevance(query, documents)

    def _llm_is_sup(self, query: str, documents: List[Document], answer: str) -> str:
        """[IsSup] token: is the answer supported by the documents?"""

        if not answer.strip():
            return ISSUP_NOT
        response = self._call(
            "You are a faithfulness evaluator.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved documents:\n{self._snippets(documents, 300)}\n\n"
            f"Generated answer: {answer}\n\n"
            "Reply with exactly one word: SUPPORTED, NOT_SUPPORTED, or CONFLICTING."
        )
        label = parse_label(response or "", ["SUPPORTED", "NOT_SUPPORTED", "UNSUPPORTED", "CONFLICTING", "CONFLICT"])
        if label in {"CONFLICTING", "CONFLICT"}:
            return ISSUP_CONFLICT
        if label in {"NOT_SUPPORTED", "UNSUPPORTED"}:
            return ISSUP_NOT
        return ISSUP_SUPPORTED

    def _llm_is_use(self, original_query: str, answer: str) -> str:
        """[IsUse] token: is the answer useful, not useful, or does it need more info?"""

        response = self._call(
            "You are a utility evaluator.\n\n"
            f"Original question: {original_query}\n\n"
            f"Answer: {answer}\n\n"
            "Reply with exactly one word: USEFUL, NOT_USEFUL, or NEEDS_MORE."
        )
        label = parse_label(response or "", ["USEFUL", "NOT_USEFUL", "USELESS", "NEEDS_MORE"])
        if label == "NEEDS_MORE":
            return ISUSE_MORE
        if label in {"NOT_USEFUL", "USELESS"}:
            return ISUSE_NOT
        if label == "USEFUL":
            return ISUSE_USEFUL
        return ISUSE_USEFUL if len(answer.strip()) > 20 else ISUSE_NOT

    def _llm_rewrite_question(self, query: str) -> str:
        """Rewrite the active query for a better next retrieval step."""

        if self.query_rewriter is not None and self.llm_client is None:
            return self.query_rewriter.rewrite(query)
        response = self._call(
            "The search query did not retrieve useful documents. Rewrite it as a more specific "
            "search query that keeps the intent.\n"
            f"Original query: {query}\n"
            "Return only the rewritten query."
        )
        if response:
            from raglib.components.query_rewriter import clean_llm_query

            return clean_llm_query(response, fallback=query, max_length=300)
        if self.refiner is not None:
            return self.refiner.refine(query, [])
        return query

    # ------------------------------------------------------------------ utilities
    def _lexical_relevance(self, query: str, documents: Sequence[Document]) -> str:
        """Fallback relevance verdict without an LLM."""

        if self.evaluator is not None:
            return ISREL_RELEVANT if self.evaluator.evaluate(query, list(documents)) else ISREL_NOT
        query_terms = token_set(query, remove_stopwords=True)
        matched = sum(1 for doc in documents if query_terms & token_set(doc.content))
        return ISREL_RELEVANT if matched else ISREL_NOT

    @staticmethod
    def _snippets(documents: Sequence[Document], limit: int) -> str:
        """Render numbered document snippets."""

        return "\n".join(f"[{idx + 1}] {doc.content[:limit]}" for idx, doc in enumerate(documents[:5]))
