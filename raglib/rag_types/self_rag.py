"""Self-RAG orchestration using an explicit 8-node state machine."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from raglib.components.evaluator import Evaluator
from raglib.components.generator import Generator
from raglib.components.query_rewriter import QueryRewriter
from raglib.components.refiner import Refiner
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)

# Reflection token verdicts
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
MAX_REVISE_RETRIES = 2
MAX_REWRITE_RETRIES = 2


@dataclass
class RunState:
    """Carries mutable state for one Self-RAG execution."""

    original_query: str
    active_query: str
    documents: List[Document] = field(default_factory=list)
    answer: str = ""
    trace: List[str] = field(default_factory=list)
    retrieve_count: int = 0
    revise_count: int = 0
    rewrite_count: int = 0
    terminal: str = ""  # end | direct | no_answer


class SelfRAG(BaseRAG):
    """Implements a faithful Self-RAG state machine with reflection-token loops."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        llm_client: Optional[BaseLLMClient] = None,
        evaluator: Optional[Evaluator] = None,
        refiner: Optional[Refiner] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        max_retrieve: int = MAX_RETRIEVE_RETRIES,
        max_revise: int = MAX_REVISE_RETRIES,
        max_rewrite: int = MAX_REWRITE_RETRIES,
        **kwargs,
    ):
        """Initialize SelfRAG with reflection model and retry guards."""

        super().__init__(
            retriever=retriever,
            generator=generator,
            llm_client=llm_client,
            evaluator=evaluator,
            refiner=refiner,
            query_rewriter=query_rewriter,
            **kwargs,
        )
        self.max_retrieve = max_retrieve
        self.max_revise = max_revise
        self.max_rewrite = max_rewrite

    def run(self, query: str) -> GenerationResult:
        """Execute the Self-RAG graph from __start__ to __end__."""

        if self.generator is None:
            raise ValueError("SelfRAG requires a generator.")

        state = RunState(
            original_query=query,
            active_query=self.pre_retrieve(query),
            trace=["__start__"],
        )

        state = self._node_decide_retrieval(state)
        if state.terminal == "direct":
            return self._finalize(state)

        state = self._node_retrieve(state)

        while True:
            rel_verdict = self._node_is_relevant(state)

            if rel_verdict == ISREL_NOT:
                if state.retrieve_count < self.max_retrieve and state.rewrite_count < self.max_rewrite:
                    state = self._node_rewrite_question(state)
                    if state.terminal == "no_answer":
                        return self._finalize(state)
                    state = self._node_retrieve(state)
                    continue

                state.terminal = "no_answer"
                state.trace.append("no_answer_found:retrieve_exhausted")
                return self._finalize(state)

            if rel_verdict == ISREL_REWRITE:
                state = self._node_rewrite_question(state)
                if state.terminal == "no_answer":
                    return self._finalize(state)
                state = self._node_retrieve(state)
                continue

            state = self._node_generate_from_context(state)

            sup_verdict = self._node_is_sup(state)
            if sup_verdict == ISSUP_NOT:
                while state.revise_count < self.max_revise:
                    state = self._node_revise_answer(state)
                    sup_verdict = self._node_is_sup(state)
                    if sup_verdict != ISSUP_NOT:
                        break

            if sup_verdict == ISSUP_CONFLICT:
                state = self._node_rewrite_question(state)
                if state.terminal == "no_answer":
                    return self._finalize(state)
                state = self._node_retrieve(state)
                continue

            use_verdict = self._node_is_use(state)
            if use_verdict == ISUSE_USEFUL:
                state.terminal = "end"
                state.trace.append("END")
                return self._finalize(state)

            if use_verdict == ISUSE_NOT:
                state.terminal = "no_answer"
                state.trace.append("no_answer_found:not_useful")
                return self._finalize(state)

            state = self._node_rewrite_question(state)
            if state.terminal == "no_answer":
                return self._finalize(state)
            state = self._node_retrieve(state)

    def _node_decide_retrieval(self, state: RunState) -> RunState:
        """Node: decide_retrieval using the [Retrieve] reflection token."""

        generator = self.generator
        if generator is None:
            raise ValueError("SelfRAG requires a generator.")

        verdict = self._llm_decide_retrieval(state.active_query)
        state.trace.append(f"decide_retrieval:{verdict}")

        if verdict == RETRIEVE_NO:
            state.trace.append("generate_direct")
            result = generator.generate(
                query=state.original_query,
                documents=[],
                reasoning_trace=state.trace,
            )
            state.answer = result.answer
            state.trace = result.reasoning_trace
            state.terminal = "direct"

        return state

    def _node_retrieve(self, state: RunState) -> RunState:
        """Node: retrieve from vector retriever using active query."""

        if self.retriever is None:
            state.documents = []
            state.trace.append("retrieve:no_retriever")
            logger.warning("SelfRAG has no retriever configured.")
            return state

        state.retrieve_count += 1
        active = self.pre_retrieve(state.active_query)
        documents = self.retriever.retrieve(active)
        state.documents = self.post_retrieve(active, documents)
        state.trace.append(f"retrieve:{len(state.documents)}_docs")
        logger.info("SelfRAG retrieve #%d -> %d docs", state.retrieve_count, len(state.documents))
        return state

    def _node_is_relevant(self, state: RunState) -> str:
        """Node: relevance check using the [IsRel] reflection token."""

        if not state.documents:
            state.trace.append("is_relevant:no_docs")
            return ISREL_NOT

        verdict = self._llm_is_relevant(state.active_query, state.documents)
        state.trace.append(f"is_relevant:{verdict}")
        return verdict

    def _node_generate_from_context(self, state: RunState) -> RunState:
        """Node: generate answer from retrieved context."""

        generator = self.generator
        if generator is None:
            raise ValueError("SelfRAG requires a generator.")

        docs = self.pre_generate(state.active_query, state.documents)
        result = generator.generate(
            query=state.original_query,
            documents=docs,
            reasoning_trace=state.trace,
        )
        state.answer = result.answer
        state.documents = result.sources
        state.trace = result.reasoning_trace
        state.trace.append("generate_from_context:done")
        return state

    def _node_is_sup(self, state: RunState) -> str:
        """Node: support check using the [IsSup] reflection token."""

        verdict = self._llm_is_sup(state.active_query, state.documents, state.answer)
        state.trace.append(f"is_sup:{verdict}")
        return verdict

    def _node_revise_answer(self, state: RunState) -> RunState:
        """Node: revise answer when support check fails."""

        state.revise_count += 1
        state.answer = self._llm_revise_answer(
            state.active_query,
            state.documents,
            state.answer,
        )
        state.trace.append(f"revise_answer:{state.revise_count}")
        return state

    def _node_is_use(self, state: RunState) -> str:
        """Node: usefulness check using the [IsUse] reflection token."""

        verdict = self._llm_is_use(state.original_query, state.answer)
        state.trace.append(f"is_use:{verdict}")
        return verdict

    def _node_rewrite_question(self, state: RunState) -> RunState:
        """Node: rewrite question before another retrieval attempt."""

        if state.rewrite_count >= self.max_rewrite:
            state.terminal = "no_answer"
            state.trace.append("no_answer_found:rewrite_exhausted")
            logger.warning("SelfRAG rewrite exhausted after %d attempts.", state.rewrite_count)
            return state

        state.rewrite_count += 1
        rewritten = self._llm_rewrite_question(state.active_query).strip()
        if not rewritten:
            rewritten = state.active_query
        state.active_query = rewritten
        state.trace.append(f"rewrite_question:{state.rewrite_count}->{rewritten[:60]}")
        return state

    def _llm_decide_retrieval(self, query: str) -> str:
        """[Retrieve] token: should we retrieve for this query?"""

        llm_client = self.llm_client
        if llm_client is None:
            lowered = query.strip().lower()
            if any(token in lowered for token in ("hello", "hi", "thanks", "thank you")):
                return RETRIEVE_NO
            return RETRIEVE_YES

        prompt = (
            "You are a retrieval decision module.\n\n"
            f"Question: {query}\n\n"
            "Should retrieve external documents to answer the user query accurately?\n"
            "Reply with exactly one word: YES or NO.\n\n"
            "Your answer:"
        )
        try:
            response = llm_client.complete(prompt).strip().upper()
            words = set(re.findall(r"[A-Z_]+", response))
            if words & {"NO", "NONE", "DIRECT", "SKIP"} and not (words & {"YES", "RETRIEVE"}):
                return RETRIEVE_NO
            if words & {"YES", "VECTOR", "WEB", "HYBRID", "MIXED", "RETRIEVE"}:
                return RETRIEVE_YES

            lowered = query.strip().lower()
            if any(token in lowered for token in ("hello", "hi", "thanks", "thank you")):
                return RETRIEVE_NO
            return RETRIEVE_YES
        except Exception as exc:  # noqa: BLE001
            logger.warning("SelfRAG decide_retrieval LLM error: %s", exc)
            return RETRIEVE_YES

    def _llm_is_relevant(self, query: str, documents: List[Document]) -> str:
        """[IsRel] token: are retrieved docs relevant enough to proceed?"""

        llm_client = self.llm_client
        if llm_client is None:
            if self.evaluator is not None:
                filtered = self.evaluator.evaluate(query, documents)
                if len(filtered) == len(documents):
                    return ISREL_RELEVANT
                if filtered:
                    return ISREL_NOT
                return ISREL_REWRITE

            query_terms = self._tokenize(query)
            matched = 0
            for doc in documents:
                if query_terms & self._tokenize(doc.content):
                    matched += 1

            if matched == 0:
                return ISREL_REWRITE
            if matched >= max(1, len(documents) // 2):
                return ISREL_RELEVANT
            return ISREL_NOT

        snippets = "\n".join(f"[{idx + 1}] {doc.content[:300]}" for idx, doc in enumerate(documents[:5]))
        prompt = (
            "You are a relevance evaluator.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved documents:\n{snippets}\n\n"
            "Reply with exactly one word:\n"
            "  RELEVANT\n"
            "  NOT_RELEVANT\n"
            "  REWRITE\n\n"
            "Your answer:"
        )
        try:
            response = llm_client.complete(prompt).strip().upper()
            words = set(re.findall(r"[A-Z_]+", response))
            if "REWRITE" in words:
                return ISREL_REWRITE
            if "NOT_RELEVANT" in words or "IRRELEVANT" in words or "NOT" in words:
                return ISREL_NOT
            return ISREL_RELEVANT
        except Exception as exc:  # noqa: BLE001
            logger.warning("SelfRAG is_relevant LLM error: %s", exc)
            return ISREL_RELEVANT

    def _llm_is_sup(self, query: str, documents: List[Document], answer: str) -> str:
        """[IsSup] token: is answer supported by retrieved documents?"""

        llm_client = self.llm_client
        if llm_client is None:
            if not answer.strip():
                return ISSUP_NOT
            if self.evaluator is not None:
                filtered = self.evaluator.evaluate(query, documents)
                if not filtered:
                    return ISSUP_NOT
            return ISSUP_SUPPORTED

        snippets = "\n".join(f"[{idx + 1}] {doc.content[:300]}" for idx, doc in enumerate(documents[:5]))
        prompt = (
            "You are a faithfulness evaluator.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved documents:\n{snippets}\n\n"
            f"Generated answer: {answer}\n\n"
            "Reply with exactly one word:\n"
            "  SUPPORTED\n"
            "  NOT_SUPPORTED\n"
            "  CONFLICTING\n\n"
            "Your answer:"
        )
        try:
            response = llm_client.complete(prompt).strip().upper()
            words = set(re.findall(r"[A-Z_]+", response))
            if "CONFLICTING" in words or "CONFLICT" in words:
                return ISSUP_CONFLICT
            if "NOT_SUPPORTED" in words or "UNSUPPORTED" in words or "NOT" in words:
                return ISSUP_NOT
            return ISSUP_SUPPORTED
        except Exception as exc:  # noqa: BLE001
            logger.warning("SelfRAG is_sup LLM error: %s", exc)
            return ISSUP_SUPPORTED

    def _llm_revise_answer(self, query: str, documents: List[Document], current_answer: str) -> str:
        """Revise unsupported answer to better match retrieved evidence."""

        llm_client = self.llm_client
        if llm_client is None:
            return current_answer

        snippets = "\n".join(f"[{idx + 1}] {doc.content[:400]}" for idx, doc in enumerate(documents[:5]))
        prompt = (
            "The current answer is not well supported by retrieved documents.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved documents:\n{snippets}\n\n"
            f"Current answer:\n{current_answer}\n\n"
            "Rewrite the answer so every claim is directly supported by the documents.\n"
            "Do not add new facts.\n\n"
            "Revised answer:"
        )
        try:
            revised = llm_client.complete(prompt).strip()
            return revised or current_answer
        except Exception as exc:  # noqa: BLE001
            logger.warning("SelfRAG revise_answer LLM error: %s", exc)
            return current_answer

    def _llm_is_use(self, original_query: str, answer: str) -> str:
        """[IsUse] token: is answer useful, not useful, or needs more info?"""

        llm_client = self.llm_client
        if llm_client is None:
            if len(answer.strip()) > 20:
                return ISUSE_USEFUL
            return ISUSE_NOT

        prompt = (
            "You are a utility evaluator.\n\n"
            f"Original question: {original_query}\n\n"
            f"Answer: {answer}\n\n"
            "Reply with exactly one word:\n"
            "  USEFUL\n"
            "  NOT_USEFUL\n"
            "  NEEDS_MORE\n\n"
            "Your answer:"
        )
        try:
            response = llm_client.complete(prompt).strip().upper()
            words = set(re.findall(r"[A-Z_]+", response))
            if "NEEDS_MORE" in words or "NEEDS" in words or "MORE" in words:
                return ISUSE_MORE
            if "NOT_USEFUL" in words or "USELESS" in words or "NOT" in words:
                return ISUSE_NOT
            return ISUSE_USEFUL
        except Exception as exc:  # noqa: BLE001
            logger.warning("SelfRAG is_use LLM error: %s", exc)
            return ISUSE_USEFUL

    def _llm_rewrite_question(self, query: str) -> str:
        """Rewrite active query for a better next retrieval step."""

        llm_client = self.llm_client
        if llm_client is not None:
            prompt = (
                "The query did not retrieve useful documents.\n"
                "Rewrite it as a more specific search query.\n"
                "Return only the rewritten query - no explanation.\n\n"
                f"Original query: {query}\n\n"
                "Rewritten query:"
            )
            try:
                rewritten = llm_client.complete(prompt).strip()
                if 5 <= len(rewritten) <= 300:
                    return rewritten
            except Exception as exc:  # noqa: BLE001
                logger.warning("SelfRAG rewrite_question LLM error: %s", exc)

        if self.query_rewriter is not None:
            rewritten = self.query_rewriter.rewrite(query).strip()
            if rewritten:
                return rewritten

        if self.refiner is not None:
            rewritten = self.refiner.refine(query, []).strip()
            if rewritten:
                return rewritten

        return query

    def _finalize(self, state: RunState) -> GenerationResult:
        """Build final GenerationResult for all terminal paths."""

        state.trace.append("__end__")

        if state.terminal == "no_answer":
            answer = (
                "I was unable to find a well-supported answer with the available documents. "
                "Please rephrase the question or provide more context."
            )
            result = GenerationResult(
                answer=answer,
                sources=state.documents,
                reasoning_trace=state.trace,
            )
            return self.post_generate(result)

        if state.terminal == "direct":
            result = GenerationResult(
                answer=state.answer,
                sources=[],
                reasoning_trace=state.trace,
            )
            return self.post_generate(result)

        result = GenerationResult(
            answer=state.answer,
            sources=state.documents,
            reasoning_trace=state.trace,
        )
        return self.post_generate(result)

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text into lowercase alphanumeric terms."""

        return set(term.lower() for term in text.split() if term.strip())
