"""Second-stage rerankers: cross-encoder, Cohere, listwise-LLM, custom, or pass-through."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from raglib.core.prompt_safety import data_rule, delimiter_for, wrap
from raglib.core.runtime import report_degraded
from raglib.core.tracing import trace_stage, trace_tag
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)

RERANKER_BACKENDS = ("none", "cross_encoder", "cohere", "llm", "custom")
DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-base"
DEFAULT_COHERE_MODEL = "rerank-v3.5"

ScoreFn = Callable[[str, Sequence[str]], Sequence[float]]

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


class Reranker:
    """Reorders candidate documents by query relevance and keeps the top_k.

    Backends:
      * ``"none"`` (default) - keep the first-stage order (RRF/vector), just cut to top_k.
      * ``"cross_encoder"`` - local sentence-transformers CrossEncoder (``pip install raglib-py[rerank]``).
      * ``"cohere"`` - Cohere Rerank API (``pip install cohere``; needs ``api_key`` or COHERE_API_KEY).
      * ``"llm"`` - one listwise LLM call that returns the passage order as JSON.
      * ``"custom"`` - your own ``score_fn(query, texts) -> scores``.
    """

    def __init__(
        self,
        top_k: int = 5,
        backend: str = "none",
        model_name: Optional[str] = None,
        llm_client: Optional[BaseLLMClient] = None,
        api_key: Optional[str] = None,
        score_fn: Optional[ScoreFn] = None,
        max_chars_per_doc: int = 1500,
        skip_threshold: Optional[float] = None,
        skip_margin: Optional[float] = None,
        timeout: float = 10.0,
        max_retries: int = 2,
    ):
        """Configure the reranking backend.

        Conditional reranking: with ``skip_threshold`` set, the reranker is skipped when the
        first-stage top candidate is already confident - its **dense cosine similarity**
        (``metadata["dense_score"]``) is at least ``skip_threshold`` and, with ``skip_margin``,
        at least ``skip_margin`` above the runner-up's. The fusion score is not used on purpose:
        RRF scores are rank-based (about 1/60) and relative-fusion scores are min-max normalized
        per query, so the top candidate scores about 1.0 whether or not it is relevant. Without a
        dense score (BM25-only retrieval) the reranker always runs.

        Trade-off (latency vs accuracy): skipped queries save the whole rerank stage, but a
        confident-looking wrong top hit is no longer corrected. Cosine scales differ per embedding
        model (relevant hits are ~0.8+ on some models, ~0.4 on others): calibrate the threshold
        on a golden eval set and compare accuracy with and without it before shipping.
        """

        normalized = (backend or "none").strip().lower().replace("-", "_")
        if score_fn is not None:
            normalized = "custom"
        if normalized not in RERANKER_BACKENDS:
            raise ValueError(f"Unknown reranker backend '{backend}'. Supported: {list(RERANKER_BACKENDS)}")
        if normalized == "llm" and llm_client is None:
            raise ValueError("reranker backend 'llm' requires llm_client")
        self.top_k = top_k
        self.backend = normalized
        self.model_name = model_name
        self.llm_client = llm_client
        self.api_key = api_key
        self.score_fn = score_fn
        self.max_chars_per_doc = max_chars_per_doc
        self.skip_threshold = skip_threshold
        self.skip_margin = skip_margin
        # API rerankers (Cohere): per-request timeout and retries for transient errors. When they are
        # exhausted the query keeps the first-stage order and the failure is listed in
        # result.metadata["degraded"] (or raised with strict=True).
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    @staticmethod
    def confidence(documents: Sequence[Document]) -> Optional[Tuple[float, Optional[float]]]:
        """``(top dense similarity, margin over the next candidate)`` or None without a dense score."""

        scores = [doc.metadata.get("dense_score") for doc in documents]
        if not scores or scores[0] is None:
            return None
        top = float(scores[0])
        others = [float(score) for score in scores[1:] if score is not None]
        return top, (top - max(others)) if others else None

    def should_skip(self, documents: Sequence[Document]) -> bool:
        """True when conditional reranking applies to these first-stage candidates."""

        if self.skip_threshold is None:
            return False
        confidence = self.confidence(documents)
        if confidence is None:
            return False
        top, margin = confidence
        if top < self.skip_threshold:
            return False
        return self.skip_margin is None or margin is None or margin >= self.skip_margin

    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """Return the top-k documents in reranked order."""

        k = top_k or self.top_k
        if not documents:
            return []
        if self.backend == "none" or len(documents) == 1:
            return list(documents[:k])
        if self.should_skip(documents):
            top, margin = self.confidence(documents) or (0.0, None)
            trace_tag("rerank_skipped", True)
            trace_tag("rerank_confidence", round(top, 4))
            return [
                replace(doc, metadata={**doc.metadata, "rerank_skipped": True}) for doc in documents[:k]
            ]
        trace_tag("reranker_used", True)
        with trace_stage("rerank", backend=self.backend, candidates=len(documents), top_k=k) as meta:
            reranked = self._rerank_scored(query, documents, k)
            meta["degraded"] = reranked is None
        return list(documents[:k]) if reranked is None else reranked

    def _rerank_scored(self, query: str, documents: List[Document], k: int) -> Optional[List[Document]]:
        """Score and sort ``documents``; ``None`` when the backend failed (caller keeps the first-stage order)."""

        try:
            if self.backend == "llm":
                return self._rerank_llm(query, documents)[:k]
            texts = [doc.content[: self.max_chars_per_doc] for doc in documents]
            scores = list(self._score(query, texts))
        except Exception as exc:  # noqa: BLE001 - reranking must degrade to first-stage order
            report_degraded(f"reranker:{self.backend}", exc, logger)
            return None
        rescored = [
            replace(doc, score=float(score), metadata={**doc.metadata, "rerank_score": float(score), "first_stage_score": doc.score})
            for doc, score in zip(documents, scores)
        ]
        rescored.sort(key=lambda doc: doc.score, reverse=True)
        return rescored[:k]

    def _score(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        """Dispatch pointwise scoring to the configured backend."""

        if self.backend == "custom":
            assert self.score_fn is not None
            return self.score_fn(query, texts)
        if self.backend == "cross_encoder":
            model = self._cross_encoder()
            return [float(value) for value in model.predict([(query, text) for text in texts])]
        if self.backend == "cohere":
            return self._score_cohere(query, texts)
        raise ValueError(f"Unsupported backend {self.backend}")

    def _cross_encoder(self) -> Any:
        """Load (once per process) and return a sentence-transformers CrossEncoder."""

        name = self.model_name or DEFAULT_CROSS_ENCODER
        with _MODEL_LOCK:
            if name not in _MODEL_CACHE:
                try:
                    from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise ImportError("cross_encoder reranker requires: pip install raglib-py[rerank]") from exc
                logger.info("Loading cross-encoder reranker '%s'", name)
                _MODEL_CACHE[name] = CrossEncoder(name)
            return _MODEL_CACHE[name]

    def _score_cohere(self, query: str, texts: Sequence[str]) -> List[float]:
        """Score with the Cohere Rerank API."""

        import os

        try:
            import cohere  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("cohere reranker requires: pip install cohere") from exc
        api_key = self.api_key or os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
        if not api_key:
            raise ValueError("cohere reranker requires api_key or COHERE_API_KEY")
        from raglib.llm.resilience import retry_call

        client = cohere.ClientV2(api_key=api_key, timeout=self.timeout)
        response = retry_call(
            client.rerank,
            model=self.model_name or DEFAULT_COHERE_MODEL,
            query=query,
            documents=list(texts),
            max_retries=self.max_retries,
            label="reranker:cohere",
        )
        from raglib.core.runtime import record_search_usage

        # Cohere bills "search units": one query over up to 100 documents.
        record_search_usage("cohere", max(1, -(-len(texts) // 100)))
        scores = [0.0] * len(texts)
        for item in response.results:
            scores[item.index] = float(item.relevance_score)
        return scores

    def _rerank_llm(self, query: str, documents: List[Document]) -> List[Document]:
        """Listwise LLM reranking in a single call."""

        assert self.llm_client is not None
        texts = [doc.content[:600] for doc in documents]
        delimiter = delimiter_for(query, *texts)
        passages = "\n".join(f"[{index}] {wrap(text, delimiter)}" for index, text in enumerate(texts, start=1))
        prompt = (
            "Rank the passages below by how well they help answer the question.\n"
            f"Question: {query}\n\n"
            f"Passages:\n{passages}\n\n"
            f"{data_rule(delimiter)}\n"
            'Return JSON only: {"ranking": [passage numbers, most relevant first]}'
        )
        data = self.llm_client.complete_json(prompt, system="You are a precise search relevance ranker.", default={})
        ranking = data.get("ranking") if isinstance(data, dict) else data if isinstance(data, list) else None
        order: List[int] = []
        for value in ranking or []:
            try:
                index = int(value) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(documents) and index not in order:
                order.append(index)
        order.extend(index for index in range(len(documents)) if index not in order)
        total = len(order)
        return [
            replace(documents[index], score=round(1.0 - rank / total, 4), metadata={**documents[index].metadata, "rerank": "llm"})
            for rank, index in enumerate(order)
        ]
