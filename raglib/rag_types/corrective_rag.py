"""Corrective RAG (CRAG, Yan et al. 2024): grade retrieval, then refine, search, or both."""

from __future__ import annotations

import logging
import re
from typing import Any, List, Tuple

from raglib.components.grader import LABEL_CORRECT, LABEL_INCORRECT, RelevanceGrader
from raglib.components.query_rewriter import clean_llm_query
from raglib.core.base import BaseRAG, PipelineContext
from raglib.core.text import token_set
from raglib.schemas import Document

logger = logging.getLogger(__name__)

VERDICT_CORRECT = "correct"
VERDICT_AMBIGUOUS = "ambiguous"
VERDICT_INCORRECT = "incorrect"


class CorrectiveRAG(BaseRAG):
    """Three-path CRAG pipeline.

    * CORRECT   (at least one document graded correct)  -> knowledge refinement of internal docs.
    * INCORRECT (every document graded incorrect)       -> knowledge search (web, or corpus retry).
    * AMBIGUOUS (otherwise)                             -> refinement + search, combined.

    Cost: one batched grading call, one batched strip-filter call, plus a query rewrite
    and web search only on the INCORRECT/AMBIGUOUS paths. Web search runs only when a
    ``web_retriever`` is configured, so private queries are never sent out by default.
    """

    pipeline_name = "corrective"
    uses_decision_engine = False

    def __init__(self, refine_strips: bool = True, max_strips: int = 40, **kwargs: Any):
        """Initialize CRAG."""

        super().__init__(**kwargs)
        self.refine_strips = refine_strips
        self.max_strips = max_strips

    # ------------------------------------------------------------------ pipeline
    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Retrieve, grade, and route through the CRAG action paths."""

        if self.retriever is None:
            raise ValueError("CorrectiveRAG requires a retriever")
        ctx.trace.append("corrective_started")
        documents = self.retrieve_corpus(ctx, ctx.search_query)

        with ctx.tracer.stage("grade") as meta:
            verdict, relevant = self._evaluate(ctx.search_query, documents)
            meta.update({"verdict": verdict, "kept": len(relevant), "total": len(documents)})
        ctx.trace.append(f"evaluation:{verdict}")
        ctx.metadata["crag_verdict"] = verdict

        internal: List[Document] = []
        external: List[Document] = []
        if verdict in {VERDICT_CORRECT, VERDICT_AMBIGUOUS}:
            internal = self._knowledge_refinement(ctx.search_query, relevant)
        if verdict in {VERDICT_AMBIGUOUS, VERDICT_INCORRECT}:
            external = self._knowledge_search(ctx, documents)
        ctx.trace.append(
            {
                VERDICT_CORRECT: "path:knowledge_refinement_only",
                VERDICT_AMBIGUOUS: "path:refinement_plus_search",
                VERDICT_INCORRECT: "path:search_only",
            }[verdict]
        )

        merged: List[Document] = []
        seen = set()
        for doc in internal + external:
            if doc.id not in seen:
                seen.add(doc.id)
                merged.append(doc)
        return merged

    # ------------------------------------------------------------------ evaluator
    def _grader(self) -> RelevanceGrader:
        """Return the configured grader, building a default one lazily."""

        if self.grader is None:
            self.grader = RelevanceGrader(llm_client=self.llm_client, evaluator=self.evaluator)
        return self.grader

    def _evaluate(self, query: str, documents: List[Document]) -> Tuple[str, List[Document]]:
        """Grade documents and compute the CRAG verdict; returns (verdict, non-incorrect docs)."""

        if not documents:
            return VERDICT_INCORRECT, []
        labels = self._grader().grade(query, documents)
        correct = [doc for doc, label in zip(documents, labels) if label == LABEL_CORRECT]
        ambiguous = [doc for doc, label in zip(documents, labels) if label not in {LABEL_CORRECT, LABEL_INCORRECT}]
        if correct:
            return VERDICT_CORRECT, correct + ambiguous
        if ambiguous:
            return VERDICT_AMBIGUOUS, ambiguous
        return VERDICT_INCORRECT, []

    # ------------------------------------------------------------------ refinement
    def _knowledge_refinement(self, query: str, documents: List[Document]) -> List[Document]:
        """Decompose documents into sentence strips and keep only the relevant ones."""

        if not documents or not self.refine_strips:
            return documents
        strips: List[Tuple[int, str]] = []
        for doc_index, document in enumerate(documents):
            for strip in self._decompose_into_strips(document.content):
                strips.append((doc_index, strip))
        if not strips or len(strips) > self.max_strips:
            return documents

        keep = self._select_strips(query, [text for _, text in strips])
        refined: List[Document] = []
        for doc_index, document in enumerate(documents):
            kept = [text for (owner, text), flag in zip(strips, keep) if owner == doc_index and flag]
            if not kept:
                continue
            refined.append(
                Document(
                    id=f"{document.id}_refined",
                    content=" ".join(kept),
                    metadata={**document.metadata, "refined": True, "refined_from": document.id},
                    score=document.score,
                    source=document.source,
                )
            )
        return refined or documents

    @staticmethod
    def _decompose_into_strips(text: str) -> List[str]:
        """Split text into sentence strips."""

        sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
        return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 10]

    def _select_strips(self, query: str, strips: List[str]) -> List[bool]:
        """Return a keep-flag per strip (one LLM call for all strips)."""

        if self.llm_client is None:
            query_terms = token_set(query, remove_stopwords=True)
            flags = [bool(query_terms & token_set(strip)) for strip in strips]
            return flags if any(flags) else [True] * len(strips)

        numbered = "\n".join(f"{index}. {strip}" for index, strip in enumerate(strips, start=1))
        prompt = (
            f"Query: {query}\n\n"
            f"Knowledge strips:\n{numbered}\n\n"
            "Which strip numbers contain information useful for answering the query? "
            "Reply with only the numbers separated by commas (e.g. 1,3,5). "
            "Reply ALL if all are useful, NONE if none are."
        )
        try:
            response = self.llm_client.complete(prompt).strip().upper()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CRAG strip filter failed: %s - keeping all strips", exc)
            return [True] * len(strips)
        if re.search(r"\bALL\b", response):
            return [True] * len(strips)
        if re.search(r"\bNONE\b", response):
            return [False] * len(strips)
        chosen = {int(number) - 1 for number in re.findall(r"\d+", response)}
        flags = [index in chosen for index in range(len(strips))]
        return flags if any(flags) else [True] * len(strips)

    # ------------------------------------------------------------------ search
    def _knowledge_search(self, ctx: PipelineContext, failed_docs: List[Document]) -> List[Document]:
        """Search external knowledge (web) or, without web, retry the corpus with a refined query."""

        if self.web_retriever is not None:
            search_query = self._rewrite_for_web_search(ctx.search_query)
            ctx.trace.append(f"web_search_query:{search_query[:60]}")
            with ctx.tracer.stage("web_search") as meta:
                try:
                    web_documents = self.web_retriever.retrieve(search_query)
                except Exception as exc:  # noqa: BLE001
                    logger.error("CRAG web search failed: %s", exc)
                    ctx.trace.append("web_search_failed")
                    web_documents = []
                meta["docs"] = len(web_documents)
            ctx.trace.append(f"web_results:{len(web_documents)}")
            return web_documents

        if self.refiner is None or self.retriever is None:
            ctx.trace.append("search_unavailable")
            return []
        query = ctx.search_query
        known_ids = {doc.id for doc in failed_docs}
        for attempt in range(1, self.max_retries + 1):
            refined = self.refiner.refine(query, failed_docs)
            if not refined or refined == query:
                break
            query = refined
            ctx.trace.append(f"corpus_retry_{attempt}:{query[:60]}")
            candidates = [doc for doc in self.retrieve_corpus(ctx, query) if doc.id not in known_ids]
            verdict, relevant = self._evaluate(ctx.search_query, candidates)
            if relevant:
                return relevant
            failed_docs = candidates or failed_docs
        return []

    def _rewrite_for_web_search(self, query: str) -> str:
        """Rewrite the question into a web-search query."""

        if self.llm_client is None:
            return query
        prompt = (
            "Rewrite the question as a concise web search query of keywords.\n"
            f"Question: {query}\n"
            "Return only the search query."
        )
        try:
            return clean_llm_query(self.llm_client.complete(prompt), fallback=query, max_length=200)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CRAG query rewrite failed: %s", exc)
            return query
