"""Chunks never exceed the embedding model's input limit (fake tokenizer, small fake limit; no network)."""

from __future__ import annotations

import types
import warnings
from typing import List, Optional

import pytest

from raglib import RAG, EmbeddingLimitError, EmbeddingLimitWarning
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.limits import known_limit
from raglib.schemas import Document

WORDS = " ".join(f"word{i}." if i % 7 == 6 else f"word{i}" for i in range(120))  # sentences of 7 words


class TokenizedEmbedding(BaseEmbedding):
    """A fake model whose tokenizer makes 2 tokens per word and that reads at most ``limit`` tokens."""

    def __init__(self, limit: int = 20, exact: bool = True, name: str = "fake-model") -> None:
        self.limit, self.exact, self._model_name = limit, exact, name
        self.seen: List[str] = []

    @property
    def max_input_tokens(self) -> Optional[int]:
        return self.limit

    def count_input_tokens(self, text: str) -> Optional[int]:
        return 2 * len(text.split()) if self.exact else None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.seen.extend(texts)
        return [[float(len(t) % 7), float(len(t.split()) % 5), 1.0] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        return 3


def _rag(embedding: BaseEmbedding, **kwargs) -> RAG:
    options = dict(chat_llm="mock", embedding_llm=embedding, token_counter="units", chunk_size=50, chunk_overlap=5)
    options.update(kwargs)
    return RAG(**options)


def test_known_limits_table() -> None:
    assert known_limit("BAAI/bge-base-en-v1.5") == 512 and known_limit("bge-large-en") == 512
    assert known_limit("BAAI/bge-m3") == 8192 and known_limit("sentence-transformers/all-MiniLM-L6-v2") == 256
    assert known_limit("text-embedding-3-small") == 8191 and known_limit("nomic-embed-text") == 2048
    assert known_limit("gemini-embedding-001") == 2048 and known_limit("my-custom-model") is None


def test_startup_warns_when_chunk_size_cannot_fit_and_suggests_a_size() -> None:
    with pytest.warns(EmbeddingLimitWarning, match=r"chunk_size<=10 to avoid it"):
        _rag(TokenizedEmbedding(limit=20))  # 50 words = 100 model tokens > 20
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _rag(TokenizedEmbedding(limit=200))  # fits: silent


def test_oversized_chunks_are_split_never_truncated() -> None:
    embedding = TokenizedEmbedding(limit=20)
    with pytest.warns(EmbeddingLimitWarning):
        rag = _rag(embedding)
        rag.add(Document(id="doc", content=WORDS))
    chunks = rag.retriever.documents
    assert chunks and all(2 * len(c.content.split()) <= 20 for c in chunks)
    assert {w for c in chunks for w in c.content.split()} == set(WORDS.split())  # every word is embedded somewhere
    assert all(WORDS[c.metadata["start_char"] : c.metadata["end_char"]] == c.content for c in chunks)
    assert rag.ingest_usage()["oversized_chunks_split"] > 0
    assert all(2 * len(text.split()) <= 20 for text in embedding.seen)  # nothing longer reached the model
    assert rag.chunking_settings()["embedding_max_tokens"] == 20


def test_raise_policy_and_production_preset() -> None:
    with pytest.warns(EmbeddingLimitWarning):
        rag = _rag(TokenizedEmbedding(limit=20), oversized_chunks="raise")
    with pytest.raises(EmbeddingLimitError, match="at most 20"):
        rag.add(Document(id="doc", content=WORDS))
    from raglib.presets import resolve_preset

    assert resolve_preset("production")["oversized_chunks"] == "raise"
    with pytest.raises(ValueError):
        _rag(TokenizedEmbedding(limit=200), oversized_chunks="truncate")


def test_override_and_estimated_counts_use_a_safety_margin() -> None:
    with pytest.warns(EmbeddingLimitWarning, match="estimated"):
        rag = _rag(TokenizedEmbedding(limit=10_000, exact=False), embedding_max_tokens=40, chunk_size=200)
    rag.add(Document(id="doc", content=WORDS))
    from raglib.core.text import count_tokens

    assert all(count_tokens(c.content) <= 36 for c in rag.retriever.documents)  # 90% of 40 without a real tokenizer


def test_semantic_windows_fit_the_model() -> None:
    embedding = TokenizedEmbedding(limit=24)
    with pytest.warns(EmbeddingLimitWarning):
        rag = _rag(embedding, chunking_strategy="embedding", chunk_size=60, semantic_min_tokens=3)
        rag.add(Document(id="doc", content=WORDS))
    assert embedding.seen and all(2 * len(text.split()) <= 24 for text in embedding.seen)


def test_huggingface_uses_the_model_tokenizer_and_max_seq_length() -> None:
    from raglib.embedding.huggingface_embedding import HuggingFaceEmbedding

    model = types.SimpleNamespace(max_seq_length=512, tokenizer=types.SimpleNamespace(encode=lambda text, add_special_tokens: [0] * (len(text) + 2)))
    hf = object.__new__(HuggingFaceEmbedding)
    hf._model, hf._model_name = model, "BAAI/bge-base-en-v1.5"
    assert hf.max_input_tokens == 512 and hf.count_input_tokens("abcd") == 6


def test_openai_rejects_oversized_input_before_calling_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from raglib.embedding.openai_embedding import OpenAIEmbedding
    from raglib.llm import http

    def never(*_, **__):
        raise AssertionError("the API must not be called")

    monkeypatch.setattr(http, "post_json", never)
    client = OpenAIEmbedding(api_key="test-key", model_name="text-embedding-3-small")
    if client.count_input_tokens("hello world") is None:
        pytest.skip("tiktoken not installed: no exact OpenAI token count")
    with pytest.raises(EmbeddingLimitError, match="at most 8191"):
        client.embed_documents(["token " * 9000])
