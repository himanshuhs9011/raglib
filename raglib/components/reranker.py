"""Second-stage rerankers: cross-encoder, Cohere, listwise-LLM, custom, or pass-through."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence

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
    ):
        """Configure the reranking backend."""

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

    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """Return the top-k documents in reranked order."""

        k = top_k or self.top_k
        if not documents:
            return []
        if self.backend == "none" or len(documents) == 1:
            return list(documents[:k])
        try:
            if self.backend == "llm":
                return self._rerank_llm(query, documents)[:k]
            texts = [doc.content[: self.max_chars_per_doc] for doc in documents]
            scores = list(self._score(query, texts))
        except Exception as exc:  # noqa: BLE001 - reranking must degrade to first-stage order
            logger.warning("Reranker backend=%s failed (%s); keeping retrieval order", self.backend, exc)
            return list(documents[:k])
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
        client = cohere.ClientV2(api_key=api_key)
        response = client.rerank(model=self.model_name or DEFAULT_COHERE_MODEL, query=query, documents=list(texts))
        scores = [0.0] * len(texts)
        for item in response.results:
            scores[item.index] = float(item.relevance_score)
        return scores

    def _rerank_llm(self, query: str, documents: List[Document]) -> List[Document]:
        """Listwise LLM reranking in a single call."""

        assert self.llm_client is not None
        passages = "\n".join(
            f"[{index}] {doc.content[:600].strip()}" for index, doc in enumerate(documents, start=1)
        )
        prompt = (
            "Rank the passages below by how well they help answer the question.\n"
            f"Question: {query}\n\n"
            f"Passages:\n{passages}\n\n"
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
