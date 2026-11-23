"""Cohere timeout/retries, embedding retries/timeouts, the LLM cache default TTL (no network)."""

from __future__ import annotations

import sys
import types
from typing import List

import pytest

from raglib import RAG
from raglib.components.reranker import Reranker
from raglib.core.runtime import run_scope
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.schemas import Document

DOCS = [Document(id=f"d{i}", content=f"passage number {i}") for i in range(4)]


class Transient(Exception):
    status_code = 503


def _install_cohere(monkeypatch: pytest.MonkeyPatch, failures: int) -> dict:
    state = {"calls": 0, "timeout": None}

    class FakeCohere:
        def __init__(self, api_key: str, timeout: float = None) -> None:
            state["timeout"] = timeout

        def rerank(self, model: str, query: str, documents: List[str]):
            state["calls"] += 1
            if state["calls"] <= failures:
                raise Transient("service unavailable")
            return types.SimpleNamespace(
                results=[types.SimpleNamespace(index=i, relevance_score=float(i)) for i in range(len(documents))]
            )

    monkeypatch.setitem(sys.modules, "cohere", types.SimpleNamespace(ClientV2=FakeCohere))
    monkeypatch.setattr("raglib.llm.resilience.time.sleep", lambda seconds: None)  # no real backoff in tests
    return state


def test_cohere_uses_a_timeout_and_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_cohere(monkeypatch, failures=1)
    reranked = Reranker(top_k=2, backend="cohere", api_key="test").rerank("q", DOCS)
    assert state["timeout"] == 10.0 and state["calls"] == 2  # one retry, then success
    assert [doc.id for doc in reranked] == ["d3", "d2"]


def test_cohere_failure_falls_back_to_first_stage_order_with_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_cohere(monkeypatch, failures=99)
    with run_scope() as run:
        reranked = Reranker(top_k=2, backend="cohere", api_key="test", timeout=3.0, max_retries=1).rerank("q", DOCS)
    assert state["timeout"] == 3.0 and state["calls"] == 2
    assert [doc.id for doc in reranked] == ["d0", "d1"]  # first-stage order kept
    assert run.degraded and run.degraded[0]["component"] == "reranker:cohere"


def test_rag_passes_reranker_timeout_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", reranker="cohere", reranker_api_key="k", reranker_timeout=4.0,
              llm_max_retries=1, token_counter="units")
    assert (rag._reranker.timeout, rag._reranker.max_retries) == (4.0, 1)


class OwnEmbedding(BaseEmbedding):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]

    @property
    def dimension(self):
        return 2


def test_user_embedding_instances_respect_llm_max_retries() -> None:
    assert RAG(chat_llm="mock", embedding_llm=OwnEmbedding(), llm_max_retries=0, token_counter="units")._embedding.max_retries == 0
    assert RAG(chat_llm="mock", embedding_llm=OwnEmbedding(), token_counter="units")._embedding.max_retries == 2


def test_embedding_timeout_reaches_the_http_client() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="ollama", embedding_timeout=5.0, token_counter="units")
    assert rag._embedding.inner.timeout == 5.0
    assert RAG(chat_llm="mock", embedding_llm="ollama", token_counter="units")._embedding.inner.timeout == 120.0


def test_llm_answer_cache_expires_after_a_day_by_default() -> None:
    assert RAG(offline=True, token_counter="units")._llm.ttl == 24 * 3600.0
    assert RAG(offline=True, token_counter="units", llm_cache_ttl=None)._llm.ttl is None
