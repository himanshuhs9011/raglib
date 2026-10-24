"""Rate limiting, retry and fallback behaviour (offline)."""

import time

import pytest

from raglib import RAG
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.embedding.mock_embedding import MockEmbedding
from raglib.llm.base_client import BaseLLMClient, FallbackLLMClient
from raglib.llm.resilience import (
    QuotaExceededError,
    RateLimiter,
    retry_after_seconds,
    retry_call,
    shared_limiter,
)


class FakeRateLimit(Exception):
    def __init__(self, message, body=None):
        super().__init__(message)
        self.body = body


def test_rate_limiter_blocks_after_budget():
    limiter = RateLimiter(rpm=3, window=0.5)
    start = time.monotonic()
    for _ in range(4):
        limiter.acquire()
    assert time.monotonic() - start >= 0.4


def test_retry_after_from_openrouter_body():
    exc = FakeRateLimit("429", body={"error": {"metadata": {"retry_after_seconds": 2}}})
    assert retry_after_seconds(exc) == 2.0
    exc = FakeRateLimit("429", body={"error": {"metadata": {"headers": {"Retry-After": "3"}}}})
    assert retry_after_seconds(exc) == 3.0


def test_transient_error_is_retried_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeRateLimit("Error code: 429 rate limit", body={"error": {"metadata": {"retry_after_seconds": 0}}})
        return "ok"

    assert retry_call(flaky, max_retries=3, base_delay=0.01) == "ok"
    assert calls["n"] == 3


def test_daily_quota_fails_fast_without_retrying():
    calls = {"n": 0}

    def exhausted():
        calls["n"] += 1
        raise FakeRateLimit("Error code: 429 - Rate limit exceeded: free-models-per-day")

    with pytest.raises(QuotaExceededError):
        retry_call(exhausted, max_retries=5, base_delay=0.01)
    assert calls["n"] == 1


def test_long_server_wait_is_reported_as_quota():
    def long_wait():
        raise FakeRateLimit("429 rate limit", body={"error": {"metadata": {"retry_after_seconds": 3600}}})

    with pytest.raises(QuotaExceededError):
        retry_call(long_wait, max_retries=3, base_delay=0.01)


def test_non_retryable_error_is_raised_immediately():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise ValueError("Error code: 400 invalid model")

    with pytest.raises(ValueError):
        retry_call(bad_request, max_retries=3, base_delay=0.01)
    assert calls["n"] == 1


class Broken(BaseLLMClient):
    model_name = "broken"

    def complete(self, prompt, system=""):
        raise RuntimeError("Error code: 503 unavailable")

    def chat(self, messages):
        raise RuntimeError("down")


class Works(BaseLLMClient):
    model_name = "works"

    def complete(self, prompt, system=""):
        return "answer"

    def chat(self, messages):
        return "chat answer"


def test_fallback_client_uses_next_model():
    client = FallbackLLMClient([Broken(), Works()])
    assert client.complete("q") == "answer"
    assert client.chat([{"role": "user", "content": "q"}]) == "chat answer"
    assert "".join(client.stream("q")) == "answer"


def test_cached_embedding_retries_transient_errors():
    class Flaky(MockEmbedding):
        calls = 0

        def embed_documents(self, texts):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("Error code: 502 upstream unavailable")
            return super().embed_documents(texts)

    embedding = CachedEmbedding(Flaky(), max_retries=2)
    assert len(embedding.embed_documents(["a", "b"])) == 2


def test_openrouter_free_models_share_a_20_rpm_limiter():
    rag = RAG(
        llm="openrouter/vendor/some-model:free",
        chat_api_key="sk-or-test-key",
        embedding_llm="openrouter/vendor/some-embed:free",
    )
    chat_limiter = rag._llm_raw.rate_limiter
    assert chat_limiter is not None and chat_limiter.rpm == 20
    assert rag._embedding.rate_limiter is chat_limiter  # one budget for chat + embeddings
    assert shared_limiter("openrouter", "sk-or-test-key", 50) is chat_limiter


def test_paid_models_have_no_limiter_unless_requested():
    rag = RAG(llm="openrouter/openai/gpt-4.1-mini", chat_api_key="sk-or-other", embedding_llm="none")
    assert rag._llm_raw.rate_limiter is None
    rag = RAG(llm="openrouter/openai/gpt-4.1-mini", chat_api_key="sk-or-other2", embedding_llm="none", llm_rpm=5)
    assert rag._llm_raw.rate_limiter.rpm == 5


def test_llm_fallbacks_build_a_fallback_client():
    rag = RAG(
        llm="openrouter/a/model-one:free",
        chat_api_key="sk-or-fb",
        embedding_llm="none",
        llm_fallbacks=["openrouter/b/model-two:free"],
    )
    assert isinstance(rag._llm_raw, FallbackLLMClient)
    assert [c.model_name for c in rag._llm_raw.clients] == ["a/model-one:free", "b/model-two:free"]


def test_add_accepts_document_objects_and_citations_use_file_names():
    from raglib.components.generator import source_label
    from raglib.schemas import Document

    rag = RAG(offline=True)
    rag.add([Document(id="d1", content="Voltline Ltd services the generator.", metadata={"source": "facilities.md"})])
    assert rag.stats()["chunks"] == 1
    doc = Document(id="x", content="t", metadata={"file": r"C:\Users\someone\secret\report.pdf", "page": 3}, source="pdf")
    assert source_label(doc) == "report.pdf p.3"  # never the full local path


def test_context_window_semantic_seeding_finds_paraphrases():
    from raglib.embedding.base_embedding import BaseEmbedding
    from raglib.graph_context.engine import GraphContextWindow
    from raglib.schemas import Document

    concepts = [("reykjavik", "iceland"), ("generator", "power"), ("parking", "car")]

    class ConceptEmbedding(BaseEmbedding):
        def _vec(self, text):
            low = text.lower()
            return [float(any(word in low for word in group)) for group in concepts] + [0.01]

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

        @property
        def dimension(self):
            return len(concepts) + 1

    text = (
        "## Travel\n\nConference trips to Reykjavik are reimbursed up to 1,840 euros.\n\n"
        "## Facilities\n\nThe generator in Building C is serviced monthly.\n\n"
        "## Parking\n\nStaff parking permits are issued by facilities."
    )
    question = "How much is refunded for visiting Iceland?"
    plain = GraphContextWindow(chunking_strategy="recursive", chunk_size=40, chunk_overlap=0)
    plain.add_documents([Document(id="h", content=text)])
    semantic = GraphContextWindow(chunking_strategy="recursive", chunk_size=40, chunk_overlap=0, embedding=ConceptEmbedding())
    semantic.add_documents([Document(id="h", content=text)])
    result = semantic.build_context_window(question)
    assert result.documents and "Reykjavik" in result.documents[0].content
    assert any(line.startswith("Seed chunk") for line in result.traversal_trace)
    # keyword-only seeding has no term in common with the paraphrase, so it finds no seed
    assert not any(line.startswith("Seed") for line in plain.build_context_window(question).traversal_trace)
