"""Unit tests for fixed, recursive, and LLM chunking strategies."""

from __future__ import annotations

import pytest

from raglib.loaders.chunk_splitter import ChunkSplitter
from raglib.llm.mock_client import MockLLMClient
from raglib.schemas import Document


def _doc(text: str, doc_id: str = "doc-1") -> Document:
    return Document(id=doc_id, content=text, source="test")


def _long_paragraph_text() -> str:
    sentences = [
        f"Sentence number {index} discusses retrieval quality and grounded generation."
        for index in range(1, 21)
    ]
    paragraph_a = " ".join(sentences[:7])
    paragraph_b = " ".join(sentences[7:14])
    paragraph_c = " ".join(sentences[14:])
    return f"{paragraph_a}\n\n{paragraph_b}\n\n{paragraph_c}"


def test_fixed_strategy_is_default_and_tags_metadata() -> None:
    splitter = ChunkSplitter(chunk_size=40, overlap=5)
    chunks = splitter.split([_doc(_long_paragraph_text())])

    assert splitter.strategy == "fixed"
    assert len(chunks) >= 2
    assert all(chunk.metadata["chunking_strategy"] == "fixed" for chunk in chunks)
    assert chunks[0].metadata["parent_id"] == "doc-1"
    assert chunks[0].id.startswith("doc-1_chunk_")


def test_recursive_strategy_prefers_paragraph_boundaries() -> None:
    text = (
        "Alpha paragraph stays together with several words about systems.\n\n"
        "Beta paragraph covers another topic with enough words for a split.\n\n"
        "Gamma paragraph finishes the document with more descriptive content."
    )
    splitter = ChunkSplitter(chunk_size=20, overlap=2, strategy="recursive", min_tokens=3)
    chunks = splitter.split([_doc(text)])

    assert splitter.strategy == "recursive"
    assert len(chunks) >= 2
    assert all(chunk.metadata["chunking_strategy"] == "recursive" for chunk in chunks)
    # First chunk should not jump across all paragraphs when size is small.
    assert "Gamma paragraph" not in chunks[0].content


def test_recursive_strategy_hard_splits_oversized_unstructured_text() -> None:
    words = " ".join(f"token{index}" for index in range(120))
    splitter = ChunkSplitter(chunk_size=25, overlap=5, strategy="recursive", min_tokens=5)
    chunks = splitter.split([_doc(words)])

    assert len(chunks) >= 4
    assert all(len(chunk.content.split()) <= 30 for chunk in chunks)


def test_llm_strategy_uses_mock_client_ranges() -> None:
    sentences = " ".join(
        f"This is semantic sentence {index} about retrieval systems."
        for index in range(1, 10)
    )
    splitter = ChunkSplitter(
        chunk_size=40,
        overlap=5,
        strategy="llm",
        llm_client=MockLLMClient(),
        min_tokens=5,
    )
    chunks = splitter.split([_doc(sentences)])

    assert splitter.strategy == "llm"
    assert len(chunks) >= 2
    assert all(chunk.metadata["chunking_strategy"] == "llm" for chunk in chunks)


def test_llm_strategy_falls_back_to_recursive_without_client() -> None:
    splitter = ChunkSplitter(chunk_size=30, overlap=5, strategy="llm", min_tokens=5)
    chunks = splitter.split([_doc(_long_paragraph_text())])

    assert splitter.strategy == "llm"
    assert len(chunks) >= 2


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        ChunkSplitter(strategy="magic")


def test_strategy_aliases() -> None:
    assert ChunkSplitter(strategy="semantic", llm_client=MockLLMClient()).strategy == "llm"
    assert ChunkSplitter(strategy="recursive_character").strategy == "recursive"
    assert ChunkSplitter(strategy="fixed_size").strategy == "fixed"


def test_empty_document_returns_no_chunks() -> None:
    splitter = ChunkSplitter(strategy="fixed")
    assert splitter.split([_doc("   ")]) == []


def test_preprocessing_reexports_same_splitter() -> None:
    from raglib.preprocessing import ChunkSplitter as PreprocessingChunkSplitter

    assert PreprocessingChunkSplitter is ChunkSplitter
