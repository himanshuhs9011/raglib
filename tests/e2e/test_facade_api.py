"""End-to-end tests for the high-level RAG facade API."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from raglib import RAG
from raglib.components.web_retriever import WebRetriever
from raglib.providers.base_provider import BaseSearchProvider
from raglib.schemas import Document
from raglib.schemas import GenerationResult

pytestmark = pytest.mark.e2e


class _FailingSearchProvider(BaseSearchProvider):
    """Test provider that always fails to verify fail-safe behavior."""

    @property
    def name(self) -> str:
        return "failing"

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        raise RuntimeError("simulated web outage")


class _StaticSearchProvider(BaseSearchProvider):
    """Test provider that always returns deterministic web-like rows."""

    @property
    def name(self) -> str:
        return "static"

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        return [
            Document(
                id="static-1",
                content="Static snippet",
                metadata={"title": "Static title", "url": "https://example.com"},
                source=self.name,
                score=0.9,
            )
        ]


def test_facade_rag_minimal_init_and_query_with_raw_text() -> None:
    """Verify RAG facade answers a query from raw text with minimal setup."""

    rag = RAG("RAG combines retrieval and generation.")
    result = rag.query("What does this text describe?")

    assert isinstance(result, GenerationResult)
    assert result.answer.strip() != ""


def test_facade_rag_loads_folder_of_text_files(tmp_path: Path) -> None:
    """Verify RAG facade loads text files from a folder path source."""

    (tmp_path / "a.txt").write_text("Python is a programming language.", encoding="utf-8")
    (tmp_path / "b.md").write_text("RAG can improve grounded responses.", encoding="utf-8")

    rag = RAG(str(tmp_path), top_k=4)
    result = rag.query("What is mentioned about RAG?")

    assert isinstance(result, GenerationResult)
    assert len(result.sources) >= 1


def test_facade_rag_add_ingests_additional_source() -> None:
    """Verify RAG facade add method ingests more content after initialization."""

    rag = RAG("Initial content about systems.")
    before = len(rag.retriever._documents)

    rag.add("Additional content about architecture.")
    after = len(rag.retriever._documents)

    assert after > before


def test_facade_supports_recursive_chunking_strategy() -> None:
    """Verify facade accepts recursive chunking and tags chunk metadata."""

    source = (
        "First paragraph explains retrieval basics with several useful words.\n\n"
        "Second paragraph covers ranking and filtering for better answers.\n\n"
        "Third paragraph closes with generation grounded in context."
    )
    rag = RAG(
        source,
        chunking_strategy="recursive",
        chunk_size=18,
        chunk_overlap=2,
        top_k=3,
    )

    assert rag._splitter.strategy == "recursive"
    assert len(rag.retriever._documents) >= 2
    assert all(
        doc.metadata.get("chunking_strategy") == "recursive"
        for doc in rag.retriever._documents
    )


def test_facade_supports_llm_chunking_strategy() -> None:
    """Verify facade wires chat LLM into semantic chunking strategy."""

    sentences = " ".join(
        f"Facade sentence {index} describes modular RAG pipeline design."
        for index in range(1, 12)
    )
    rag = RAG(
        sentences,
        chunking_strategy="llm",
        chunk_size=35,
        chunk_overlap=5,
        top_k=3,
    )

    assert rag._splitter.strategy == "llm"
    assert rag._splitter._llm_client is rag._llm
    assert len(rag.retriever._documents) >= 1
    result = rag.query("What does the text describe?")
    assert isinstance(result, GenerationResult)
    assert result.answer.strip() != ""


def test_facade_rejects_unknown_chunking_strategy() -> None:
    """Verify invalid chunking_strategy values fail fast."""

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        RAG("text", chunking_strategy="not-a-real-strategy")


def test_facade_rag_supports_rag_type_override() -> None:
    """Verify RAG facade supports selecting a non-default rag_type."""

    rag = RAG("Retrieval quality can be refined.", rag_type="corrective")
    result = rag.query("How can retrieval be improved?")

    assert isinstance(result, GenerationResult)


def test_facade_web_provider_selection_duckduckgo() -> None:
    """Verify facade accepts explicit free web provider selection."""

    rag = RAG("Local corpus for fallback.", web_search_provider="duckduckgo")
    assert rag._web_provider.name == "duckduckgo"


def test_facade_web_provider_defaults_to_duckduckgo() -> None:
    """Verify facade defaults to DuckDuckGo when provider is not specified."""

    rag = RAG("Local corpus text only.")
    assert rag._web_provider.name == "duckduckgo"


def test_facade_requires_chat_key_for_cloud_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify explicit cloud chat provider requires key input or env value."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="chat_api_key"):
        RAG("text", chat_llm="openai")


def test_facade_requires_embedding_key_for_cloud_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify explicit cloud embedding provider requires key input or env value."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="embedding_api_key"):
        RAG("text", embedding_llm="openai")


def test_facade_requires_web_key_for_authenticated_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify authenticated web providers require explicit key input or env value."""

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError, match="web_search_api_key"):
        RAG("text", web_search_provider="tavily")


def test_web_retriever_returns_empty_on_provider_failure() -> None:
    """Verify web retriever fails safely and returns an empty result list."""

    retriever = WebRetriever(
        provider=_FailingSearchProvider(),
        top_k=3,
        fail_silently=True,
        enable_duckduckgo_fallback=False,
    )
    assert retriever.retrieve("latest ai news") == []


def test_web_retriever_fallback_provider_is_used_on_failure() -> None:
    """Verify fallback provider is used when the selected provider fails."""

    retriever = WebRetriever(
        provider=_FailingSearchProvider(),
        fallback_provider=_StaticSearchProvider(),
        top_k=3,
        fail_silently=False,
    )
    docs = retriever.retrieve("latest ai news")

    assert len(docs) == 1
    assert docs[0].metadata.get("provider") == "static"
    assert docs[0].metadata.get("url") == "https://example.com"
