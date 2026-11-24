"""Cost and usage accuracy: early-closed streams, parallel guard usage before the trace ends, answer-cache hits
with guard calls, unchanged documents skip chunking, failed add() keeps its cost (offline, no network)."""

from __future__ import annotations

import time
from typing import List

import pytest

from raglib import RAG, PriceTable
from raglib.core.runtime import record_usage, usage_scope
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.guards import Guard, InjectionClassifier, MaxOutputTokens, allow, block
from raglib.llm.clients import OpenAICompatibleClient
from raglib.llm.mock_client import MockLLMClient
from raglib.schemas import Document

CORPUS = [
    "Acme Corp offices close on Friday afternoons in summer and the cafeteria opens at eight.",
    "The Orion launch window opens in March and closes in April for the whole launch team.",
]


# ------------------------------------------------------------------ #10 usage of a stream closed early


def _sse_client(monkeypatch: pytest.MonkeyPatch, words: List[str]) -> OpenAICompatibleClient:
    def fake_sse(url, payload, headers, timeout=None):
        for word in words:
            yield {"data": {"choices": [{"delta": {"content": word + " "}}]}}

    monkeypatch.setattr("raglib.llm.clients.post_sse", fake_sse)
    return OpenAICompatibleClient(model="gpt-test", api_key="sk-test", base_url="https://example.invalid/v1",
                                  provider_name="custom", stream_usage=False)


def test_a_stream_closed_early_still_records_its_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _sse_client(monkeypatch, [f"word{i}" for i in range(50)])
    with usage_scope() as meter:
        stream = client.stream("A long prompt about office hours " * 20)
        received = [next(stream) for _ in range(3)]
        stream.close()  # the caller stopped reading (a guard truncated the answer)
    assert len(received) == 3
    assert meter.llm_calls == 1 and meter.prompt_tokens > 0 and meter.completion_tokens > 0


def test_truncated_streaming_answers_are_priced(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _sse_client(monkeypatch, [f"word{i}" for i in range(300)])
    rag = RAG(CORPUS, chat_llm=client, embedding_llm="mock", token_counter="units", llm_cache=False,
              prices=PriceTable(llm={"*": (1.0, 2.0)}, embedding={"*": 0.0}),
              output_guards=[MaxOutputTokens(5, token_counter=lambda t: len(t.split()))])
    stream = rag.stream("When do Acme offices close?")
    while True:
        try:
            next(stream)
        except StopIteration as stop:
            result = stop.value
            break
    assert result.answer.endswith(" …")
    assert result.usage["llm_calls"] == 1 and result.usage["total_prompt_tokens"] > 0 and result.usage["cost_usd"] > 0


# ------------------------------------------------------------------ #11 every guard's usage before the trace ends


class FastBlock(Guard):
    name, tier, parallel = "fast_block", 1, True

    def check(self, value, context):
        return block("Blocked.", reason="test")


class SlowModelGuard(Guard):
    name, tier, parallel = "slow_model", 1, True

    def check(self, value, context):
        time.sleep(0.2)
        record_usage(1000, 10, model="mock/guard")  # a paid classifier call
        return allow()


def test_a_parallel_block_waits_for_the_other_guards_before_finishing_the_trace() -> None:
    events = []
    prices = PriceTable(llm={"mock/guard": (1.0, 1.0)}, embedding={"*": 0.0})
    rag = RAG(CORPUS, chat_llm="mock", embedding_llm="mock", token_counter="units", prices=prices,
              input_guards=[FastBlock(), SlowModelGuard()], on_event=events.append)
    result = rag.query("When do Acme offices close?")
    assert result.metadata["blocked_by"] == "fast_block"
    stages = [event.stage for event in events]
    assert stages.index("guard.slow_model") < stages.index("total")  # no guard event after the run ended
    assert "guard.slow_model" in [event.stage for event in result.trace_events]
    assert result.usage["total_prompt_tokens"] == 1000 and result.usage["cost_usd"] > 0


def test_answer_cache_hits_report_the_guard_calls_they_made() -> None:
    class Cheap(MockLLMClient):
        def complete(self, prompt: str, system: str = "") -> str:
            record_usage(400, 10, model="mock/cheap")
            return '{"injection": false, "confidence": 0.95}'

    prices = PriceTable(llm={"mock/cheap": (1.0, 1.0), "mock/mock": (1.0, 1.0)}, embedding={"*": 0.0})
    rag = RAG(CORPUS, chat_llm="mock", embedding_llm="mock", token_counter="units", answer_cache=True,
              guard_cache_ttl=None, prices=prices, input_guards=[InjectionClassifier(llm=Cheap(), use_judge=False)])
    rag.query("When do Acme offices close?")
    hit = rag.query("When do Acme offices close?")
    assert hit.metadata["answer_cache_hit"] is True
    assert hit.usage["llm_calls"] == 1 and hit.usage["cost_usd"] > 0  # the classifier call, no answer call
    assert "guard.injection_classifier" in [event.stage for event in hit.trace_events]
    assert hit.trace_events[-1].stage == "total"


# ------------------------------------------------------------------ #16 unchanged documents skip chunking


class Counting(BaseEmbedding):
    def __init__(self) -> None:
        self.texts = 0

    def embed_documents(self, texts):
        self.texts += len(texts)
        return [[float(len(t) % 5) + 1.0, float(t.count(" ") % 3), 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    @property
    def dimension(self):
        return 3

    @property
    def max_input_tokens(self):
        return 8192


DOCS = [Document(id=f"d{i}", content=" ".join(f"Sentence {j} of document {i} talks about topic {j % 4}." for j in range(30)))
        for i in range(3)]


def _semantic(embedding: Counting) -> RAG:
    return RAG(chat_llm="mock", embedding_llm=embedding, token_counter="units", chunking_strategy="embedding",
               chunk_size=120, llm_cache=False)


def test_re_adding_unchanged_documents_embeds_nothing_even_in_a_fresh_process(tmp_path) -> None:
    embedding = Counting()
    rag = _semantic(embedding)
    assert rag.add(DOCS) > 0
    embedding.texts = 0
    assert rag.add(DOCS) == 0 and embedding.texts == 0  # same process
    rag.save(str(tmp_path / "index"))
    fresh_embedding = Counting()
    fresh = _semantic(fresh_embedding)
    fresh.load(str(tmp_path / "index"))
    fresh_embedding.texts = 0
    assert fresh.add(DOCS) == 0 and fresh_embedding.texts == 0  # fresh process after load
    changed = [Document(id="d0", content=DOCS[0].content + " A new closing sentence about budgets.")]
    assert fresh.add(changed) > 0 and fresh_embedding.texts > 0  # changed text is re-chunked


# ------------------------------------------------------------------ failed add() keeps its cost


def test_a_failed_add_keeps_the_cost_of_the_batches_already_embedded() -> None:
    class FailsLater(Counting):
        def embed_documents(self, texts):
            if self.texts >= 3:
                raise RuntimeError("embedding service down")
            return super().embed_documents(texts)

    rag = RAG(chat_llm="mock", embedding_llm=FailsLater(), token_counter="units", llm_max_retries=0)
    docs = [Document(id=f"x{i}", content=f"Short document number {i} about offices.") for i in range(10)]
    with pytest.raises(Exception):
        rag.add(docs, batch_size=2)
    assert rag.ingest_usage()["embedding_tokens"] > 0
