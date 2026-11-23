"""Phase 4: cache TTL + invalidation, public-tier caching, answer cache, conditional rerank, prompt caching."""

from __future__ import annotations

from typing import List

import pytest

from raglib import RAG, AnswerCache, LatencyStats, TTLCache, access_tags
from raglib.cache import cache_bypass, cache_dependencies, normalize_question
from raglib.components.reranker import Reranker
from raglib.core.runtime import usage_scope
from raglib.llm.base_client import CachedLLMClient
from raglib.llm.clients import AnthropicClient, OpenAICompatibleClient
from raglib.llm.mock_client import MockLLMClient
from raglib.schemas import Document

POLICY_V1 = "The refund policy gives customers fourteen days to return a product with its receipt."
POLICY_V2 = "The refund policy gives customers thirty days to return a product with its receipt."
QUESTION = "How many days do customers have to return a product?"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class CountingLLM(MockLLMClient):
    """Mock LLM that counts the calls that reach it (i.e. cache misses)."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: List[str] = []

    def complete(self, prompt: str, system: str = "") -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return super().complete(prompt, system)


# ----------------------------------------------------------------------------- TTLCache


def test_ttl_cache_expires_entries_and_evicts_lru() -> None:
    clock = FakeClock()
    cache = TTLCache(max_entries=2, ttl=10, clock=clock)
    cache.set("a", 1)
    clock.now = 9.9
    assert cache.get("a") == 1
    clock.now = 10.0
    assert cache.get("a") is None and cache.expired == 1
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # refresh a: b is now least recently used
    cache.set("c", 3)
    assert "b" not in cache and "a" in cache and "c" in cache
    with pytest.raises(ValueError):
        TTLCache(ttl=0)


def test_ttl_cache_invalidates_by_document_dependency() -> None:
    cache = TTLCache()
    with cache_dependencies(["doc-a"]):
        cache.set("answer-1", "x")
        with cache_dependencies(["doc-b"]):
            cache.set("answer-2", "y")  # depends on doc-a and doc-b
    cache.set("answer-3", "z", deps=["doc-b"])
    cache.set("unrelated", "w")
    assert cache.invalidate(["doc-b"]) == 2
    assert "answer-1" in cache and "unrelated" in cache
    assert cache.invalidate(["doc-a"]) == 1 and len(cache) == 1


def test_cached_llm_client_ttl_invalidate_and_bypass() -> None:
    inner = CountingLLM()
    client = CachedLLMClient(inner, ttl=60)
    clock = FakeClock()
    client._cache._clock = clock
    with cache_dependencies(["policy"]):
        client.complete("prompt about the policy")
    client.complete("prompt about the policy")
    assert inner.calls == 1 and client.hits == 1
    clock.now = 61
    client.complete("prompt about the policy")
    assert inner.calls == 2  # expired
    with cache_dependencies(["policy"]):
        client.complete("another policy prompt")
    assert client.invalidate(["policy"]) >= 1
    client.complete("another policy prompt")
    assert inner.calls == 4
    with cache_bypass():
        client.complete("another policy prompt")
        client.complete("another policy prompt")
    assert inner.calls == 6  # no reads and no writes inside the bypass


# ------------------------------------------------------------------ invalidation in RAG


def _rag(**kwargs) -> RAG:
    llm = kwargs.pop("chat_llm", None) or CountingLLM()
    rag = RAG(chat_llm=llm, embedding_llm="mock", **kwargs)
    rag.add(Document(id="policy", content=POLICY_V1))
    return rag


def test_reingesting_a_changed_document_invalidates_its_cached_answers() -> None:
    rag = _rag()
    llm = rag._llm_raw
    assert "fourteen" in rag.query(QUESTION).answer
    rag.query(QUESTION)
    calls = llm.calls
    version = rag.document_version("policy")
    rag.add(Document(id="policy", content=POLICY_V2))
    assert rag.document_version("policy") != version
    answer = rag.query(QUESTION).answer
    assert "thirty" in answer and "fourteen" not in answer
    assert llm.calls > calls


def test_on_document_update_hook_and_manual_invalidation() -> None:
    rag = _rag()
    rag.query(QUESTION)
    assert len(rag._llm._cache) > 0
    report = rag.on_document_update(source=Document(id="policy", content=POLICY_V2))
    assert report["reindexed_chunks"] == 1 and report["invalidated"] >= 1  # the stale answer was dropped
    rag.query(QUESTION)
    assert rag.on_document_update(doc_ids=["policy"])["invalidated"] >= 1
    rag.query(QUESTION)
    rag.invalidate_all()
    assert len(rag._llm._cache) == 0 and len(rag._embedding._cache) == 0


def test_embedding_cache_ttl_and_text_invalidation() -> None:
    rag = _rag(embedding_cache_ttl=3600)
    assert rag._embedding.ttl == 3600
    vectors = len(rag._embedding._cache)
    assert rag._embedding.invalidate_texts([POLICY_V1]) == 1
    assert len(rag._embedding._cache) == vectors - 1
    with pytest.raises(ValueError):
        rag._embedding.ttl = -1


# ------------------------------------------------------------------ public tier only


def test_personalized_queries_are_never_cached_unless_enabled() -> None:
    rag = _rag(access_control=True)
    rag.add(Document(id="policy", content=POLICY_V1), access=access_tags("public"))
    llm = rag._llm_raw
    user = {"user_id": "u1", "max_access_level": "internal"}
    rag.query(QUESTION, user=user)
    calls = llm.calls
    rag.query(QUESTION, user=user)
    assert llm.calls == calls * 2  # same personalized question: full price again
    rag.query(QUESTION)  # anonymous = public tier
    calls = llm.calls
    rag.query(QUESTION)
    assert llm.calls == calls  # cached

    shared = _rag(access_control=True, cache_personalized=True)
    shared.add(Document(id="policy", content=POLICY_V1), access=access_tags("public"))
    shared.query(QUESTION, user=user)
    calls = shared._llm_raw.calls
    shared.query(QUESTION, user=user)
    assert shared._llm_raw.calls == calls


def test_caller_filters_make_a_query_personalized() -> None:
    rag = _rag()
    rag.query(QUESTION, filters={"chunk_index": 1})
    calls = rag._llm_raw.calls
    rag.query(QUESTION, filters={"chunk_index": 1})
    assert rag._llm_raw.calls == calls * 2


# ------------------------------------------------------------------------ answer cache


def test_answer_cache_serves_repeated_questions_as_faq_tier() -> None:
    stats = LatencyStats()
    rag = _rag(answer_cache=True, on_event=stats)
    first = rag.query(QUESTION)
    assert first.metadata["tier"] == "full" and first.metadata["answer_cache_hit"] is False
    calls = rag._llm_raw.calls
    second = rag.query("  how many days do customers have to RETURN a product ")  # normalized match
    assert second.metadata["tier"] == "faq" and second.metadata["answer_cache_hit"] is True
    assert rag._llm_raw.calls == calls and second.answer == first.answer
    assert [e.stage for e in second.trace_events] == ["answer_cache", "total"]
    totals = [e for e in first.trace_events + second.trace_events if e.stage == "total"]
    assert [e.meta.get("tier") for e in totals] == ["full", "faq"]
    assert len(stats.samples("total")) == 2
    rag.add(Document(id="policy", content=POLICY_V2))  # the source changed: the cached answer is dropped
    third = rag.query(QUESTION)
    assert third.metadata["tier"] == "full" and "thirty" in third.answer
    assert rag.stats()["answer_cache"]["entries"] == 1


def test_answer_cache_checks_document_versions_and_ttl() -> None:
    clock = FakeClock()
    cache = AnswerCache(ttl=100, clock=clock)
    versions = {"policy": "v1"}
    cache.version_of = versions.get
    result = type("R", (), {"sources": [Document(id="policy_chunk_1", content="x", metadata={"parent_id": "policy"})]})()
    assert cache.put("Q?", result)
    assert cache.get("q") is not None
    versions["policy"] = "v2"  # changed without an invalidation event: the version check catches it
    assert cache.get("q") is None
    versions["policy"] = "v2"
    cache.put("Q?", result)
    clock.now = 100
    assert cache.get("q") is None  # expired
    assert normalize_question("  What IS this?? ") == "what is this"


def test_permission_change_drops_cached_public_answers() -> None:
    rag = _rag(access_control=True, answer_cache=True)
    rag.add(Document(id="policy", content=POLICY_V1), access=access_tags("public"))
    assert rag.query(QUESTION).metadata["tier"] == "full"
    assert rag.query(QUESTION).metadata["tier"] == "faq"
    rag.add(Document(id="policy", content=POLICY_V1), access=access_tags("confidential"))
    result = rag.query(QUESTION)
    assert result.metadata["answer_cache_hit"] is False
    assert all("fourteen" not in doc.content for doc in result.sources) and "fourteen" not in result.answer


def test_answer_cache_is_bypassed_for_personalized_queries() -> None:
    rag = _rag(access_control=True, answer_cache=True)
    rag.add(Document(id="policy", content=POLICY_V1), access=access_tags("public"))
    user = {"user_id": "u1", "max_access_level": "internal"}
    rag.query(QUESTION, user=user)
    assert rag.query(QUESTION, user=user).metadata["answer_cache_hit"] is False
    assert len(rag._answer_cache) == 0


# ----------------------------------------------------------------- conditional rerank


def _candidates(*dense_scores) -> List[Document]:
    return [
        Document(id=f"d{i}", content=f"text {i}", metadata={} if score is None else {"dense_score": score})
        for i, score in enumerate(dense_scores)
    ]


def test_conditional_rerank_uses_the_dense_similarity_of_the_top_hit() -> None:
    calls: List[int] = []

    def score_fn(query, texts):
        calls.append(len(texts))
        return list(range(len(texts)))  # reverses the order

    reranker = Reranker(top_k=2, score_fn=score_fn, skip_threshold=0.9)
    confident = reranker.rerank("q", _candidates(0.95, 0.5, 0.4))
    assert calls == [] and [d.id for d in confident] == ["d0", "d1"] and confident[0].metadata["rerank_skipped"]
    unsure = reranker.rerank("q", _candidates(0.6, 0.5, 0.4))
    assert calls == [3] and unsure[0].id == "d2"
    reranker.rerank("q", _candidates(None, None))  # BM25-only: no dense score, always rerank
    assert calls == [3, 2]
    margin = Reranker(top_k=2, score_fn=score_fn, skip_threshold=0.9, skip_margin=0.1)
    margin.rerank("q", _candidates(0.95, 0.92))  # confident but not separated from the runner-up
    assert calls == [3, 2, 2]


def test_skipped_rerank_is_visible_in_the_trace() -> None:
    stats = LatencyStats()
    rag = RAG(
        [POLICY_V1, "Shipping takes five business days for orders inside the country and abroad."],
        offline=True, reranker="llm", rerank_skip_threshold=-1.0, on_event=stats,
    )
    result = rag.query(QUESTION)
    assert "rerank" not in [e.stage for e in result.trace_events]
    total = [e for e in result.trace_events if e.stage == "total"][0]
    assert total.meta["rerank_skipped"] is True and total.meta["reranker_used"] is False


# --------------------------------------------------------------------- prompt caching


def test_anthropic_prompt_caching_marks_the_system_prompt() -> None:
    client = AnthropicClient("claude-opus-5-5", api_key="test-key", provider_name="anthropic")
    _, _, payload = client.build_request([{"role": "user", "content": "hi"}], "Long stable instructions", False, False)
    assert payload["system"] == "Long stable instructions"
    client.prompt_caching = True
    _, _, payload = client.build_request([{"role": "user", "content": "hi"}], "Long stable instructions", False, False)
    assert payload["system"] == [{"type": "text", "text": "Long stable instructions", "cache_control": {"type": "ephemeral"}}]
    body = {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 50, "output_tokens": 10, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 200},
    }
    text, usage = client.parse_response(body)
    assert text == "ok" and usage == (1250, 10, 1000, 200)  # prompt total includes cached tokens
    with usage_scope() as meter:
        client._record_usage(usage, "", "")
    row = meter.breakdown()["llm"]["anthropic/claude-opus-5-5"]
    assert row["cache_read_tokens"] == 1000 and row["cache_write_tokens"] == 200 and row["prompt_tokens"] == 1250


def test_prompt_caching_option_is_a_no_op_for_other_providers() -> None:
    anthropic = AnthropicClient("claude-opus-5-5", api_key="test-key", provider_name="anthropic")
    RAG(chat_llm=anthropic, embedding_llm="mock", prompt_caching=True)
    assert anthropic.prompt_caching is True
    other = OpenAICompatibleClient("gpt-4.1-mini", api_key="test-key", provider_name="openai")
    RAG(chat_llm=other, embedding_llm="mock", prompt_caching=True)
    assert not hasattr(type(other), "prompt_caching") or not other.prompt_caching
