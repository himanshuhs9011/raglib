"""2.2.2: truncated model output is never an empty success (R2); a retired model says so (R1)."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from raglib import RAG, LLMOutputTruncatedError
from raglib.core.runtime import run_scope
from raglib.llm.clients import AnthropicClient, GeminiClient, OpenAICompatibleClient
from raglib.llm.http import ProviderHTTPError
from raglib.llm.providers import PROVIDERS
from tests.unit.test_native_clients import FakeProvider, _client


@pytest.fixture()
def server():
    fake = FakeProvider()
    yield fake
    fake.close()


def _openai(content: str, finish: str) -> Dict[str, Any]:
    return {"json": {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
                     "usage": {"prompt_tokens": 50, "completion_tokens": 1024}}}


def test_empty_answer_at_the_output_limit_raises(server: FakeProvider) -> None:
    server.reply(**_openai("", "length"))
    with pytest.raises(LLMOutputTruncatedError, match="llm_max_tokens"):
        _client(server, max_tokens=1024).complete("Question?")


def test_a_cut_off_answer_is_returned_and_flagged(server: FakeProvider) -> None:
    server.reply(**_openai("Paris is the capital of", "length"))
    with run_scope() as state:
        assert _client(server, max_tokens=8).complete("Question?") == "Paris is the capital of"
    assert [d["component"] for d in state.degraded] == ["llm_output_truncated:openrouter:m-1"]


def test_a_normal_empty_answer_is_unchanged(server: FakeProvider) -> None:
    server.reply(**_openai("", "stop"))
    with run_scope() as state:
        assert _client(server).complete("Question?") == ""
    assert state.degraded == []


def test_a_think_block_cut_off_by_the_limit_raises(server: FakeProvider) -> None:
    server.reply(**_openai("<think>long reasoning that never ends", "length"))
    with pytest.raises(LLMOutputTruncatedError):
        _client(server).complete("Question?")  # an unclosed think block is all the model wrote


def test_anthropic_and_gemini_limits_are_detected(server: FakeProvider) -> None:
    server.reply(json={"content": [{"type": "thinking", "thinking": "..."}], "stop_reason": "max_tokens",
                       "usage": {"input_tokens": 9, "output_tokens": 1024}})
    with pytest.raises(LLMOutputTruncatedError):
        _client(server, cls=AnthropicClient, provider_name="anthropic", base_url=server.url).complete("Question?")
    server.reply(json={"candidates": [{"content": {"parts": [{"text": "...", "thought": True}]}, "finishReason": "MAX_TOKENS"}]})
    with pytest.raises(LLMOutputTruncatedError):
        _client(server, cls=GeminiClient, provider_name="gemini", base_url=server.url).complete("Question?")


def test_a_stream_that_ends_at_the_limit_without_text_raises(server: FakeProvider) -> None:
    server.reply(sse=[
        {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking..."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
        "[DONE]",
    ])
    with pytest.raises(LLMOutputTruncatedError):
        list(_client(server).stream("Question?"))


def test_a_stream_cut_off_after_some_text_is_flagged(server: FakeProvider) -> None:
    server.reply(sse=[
        {"choices": [{"index": 0, "delta": {"content": "Paris is"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
        "[DONE]",
    ])
    with run_scope() as state:
        assert "".join(_client(server).stream("Question?")) == "Paris is"
    assert len(state.degraded) == 1


def test_rag_query_raises_instead_of_an_empty_success(server: FakeProvider) -> None:
    for _ in range(3):
        server.reply(**_openai("", "length"))
    rag = RAG(["The office closes at 5pm on Fridays."], chat_llm=_client(server), embedding_llm="mock")
    with pytest.raises(LLMOutputTruncatedError):  # before 2.2.2: answer "" and no error
        rag.query("When does the office close on Fridays?")


# ----------------------------------------------------------------- R1: retired models


def test_the_nvidia_default_embedding_model_is_current() -> None:
    assert PROVIDERS["nvidia"].default_embedding_model == "nvidia/nemotron-3.5-embed-1b"  # 3-embed-1b: end of life


def test_a_retired_model_gives_a_clear_error_and_is_not_retried(server: FakeProvider) -> None:
    server.reply(status=410, json={"title": "Gone", "status": 410,
                                   "detail": "The model 'nvidia/nemotron-3-embed-1b' has reached its end of life."})
    with pytest.raises(ProviderHTTPError, match="no longer serves this model") as info:
        _client(server, cls=OpenAICompatibleClient).complete("Question?")
    assert info.value.status_code == 410 and "end of life" in str(info.value)
    assert len(server.requests) == 1  # not retried
