"""Facade extras: the offline local search provider, small helpers, stats() and evaluate()."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence

from raglib.core.text import token_set
from raglib.facade.chunking import unwrap_embedding
from raglib.llm.base_client import (
    BaseLLMClient,
)
from raglib.providers.base_provider import BaseSearchProvider
from raglib.schemas import Document
from raglib.vectorstores.base_store import MetadataFilter, matches_filter


class _LocalSearchProvider(BaseSearchProvider):
    """Offline web-style provider over the loaded corpus (for tests and demos only)."""

    PROVIDER_NAME = "local_web"
    searches_corpus = True  # its "web" results are corpus chunks, so filters / access control apply
    supports_filters = True

    def __init__(self, documents_fn: Callable[[], List[Document]]):
        """Initialize with a callable returning the current corpus."""

        self._documents_fn = documents_fn

    @property
    def name(self) -> str:
        """Return provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5, filters: Optional[MetadataFilter] = None) -> List[Document]:
        """Lexical overlap search over the corpus (only chunks passing ``filters``)."""

        query_terms = token_set(query, remove_stopwords=True) or token_set(query)
        if not query_terms:
            return []
        scored: List[Document] = []
        for document in self._documents_fn():
            if filters and not matches_filter(
                {**document.metadata, "source": document.source, "doc_id": document.id}, filters
            ):
                continue
            overlap = len(query_terms & token_set(document.content))
            if overlap:
                scored.append(
                    replace(
                        document,
                        score=overlap / len(query_terms),
                        source="web",
                        metadata={**document.metadata, "provider": self.name},
                    )
                )
        scored.sort(key=lambda doc: doc.score, reverse=True)
        return scored[:num_results]


def _accepts_keyword(method: Any, name: str) -> bool:
    """True when ``method`` takes the keyword ``name`` (or ``**kwargs``)."""

    import inspect

    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


def _env_flag(name: str) -> bool:
    """Return True when an environment flag is set to a truthy value."""

    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class ExtrasMixin:
    """Facade extras: the offline local search provider, small helpers, stats() and evaluate() (mixin of ``RAG``)."""

    def evaluate(self, dataset: Sequence[Dict[str, Any]], judge_llm: Optional[BaseLLMClient] = None, k: int = 5):
        """Evaluate this pipeline on a labelled dataset (see ``raglib.eval.evaluate``)."""

        from raglib.eval import evaluate

        return evaluate(self, dataset, judge_llm=judge_llm, k=k)

    @property
    def _documents(self) -> List[Document]:
        """Indexed chunks (kept for backward compatibility)."""

        return self._retriever.documents

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""

        info: Dict[str, Any] = {
            "rag_type": self._rag_type,
            "documents": len(self._parents),
            "chunks": len(self._retriever.documents),
            "retrieval_mode": self._retriever.mode,
            "embedding": type(unwrap_embedding(self._embedding)).__name__ if self._embedding else None,
            "llm": type(self._llm_raw).__name__,
            "web_provider": self._web_provider.name if self._web_provider else None,
            "reranker": self._reranker.backend,
            "preset": self._preset,
            "tiers": ["faq" if self._answer_cache else None, "cheap" if self._cheap_rag else None, "full"]
            if self._tier_router
            else None,
            "answer_cache": self._answer_cache.stats if self._answer_cache is not None else None,
            "parent_sections": len(self._sections) if self._parent_retrieval else None,
        }
        if self._graph_indexer is not None:
            info["graph"] = {
                **self._graph_indexer.store.stats(),
                "failed_extractions": len(self._graph_indexer.failed_chunk_ids),
                "skipped_extractions": len(self._graph_indexer.skipped_chunk_ids),
                "build": self.graph_status()["state"],
            }
        return info

    @staticmethod
    def _can_import(module: str) -> bool:
        """Return True if a module can be imported."""

        import importlib.util

        return importlib.util.find_spec(module) is not None

    @staticmethod
    def _resolve_env_value(keys: Sequence[str]) -> Optional[str]:
        """Return the first non-empty environment variable value from keys."""

        for key in keys:
            value = os.getenv(key)
            if value and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _first_non_empty(*values: Optional[str]) -> Optional[str]:
        """Return the first non-empty string from the given values."""

        for value in values:
            if value and str(value).strip():
                return str(value).strip()
        return None
