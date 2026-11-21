"""Native HTTP clients against a real local HTTP server (no SDKs, no LangChain, no internet)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import pytest

from raglib import RAG
from raglib.core.runtime import usage_scope
from raglib.embedding.google_embedding import GoogleEmbedding
from raglib.embedding.ollama_embedding import OllamaEmbedding
from raglib.embedding.openai_embedding import OpenAIEmbedding
from raglib.llm.clients import (
    AnthropicClient,
    GeminiClient,
    OpenAICompatibleClient,
    build_chat_client,
)
from raglib.llm.http import ProviderConnectionError, ProviderHTTPError
from raglib.llm.resilience import QuotaExceededError
from raglib.vision.anthropic_vision import AnthropicVisionClient


class FakeProvider:
    """A local HTTP server that records requests and replays scripted responses."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.responses: List[Dict[str, Any]] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"null")
                headers = {key.lower(): value for key, value in self.headers.items()}  # HTTP header names are case-insensitive
                provider.requests.append({"path": self.path, "headers": headers, "json": body})
                spec = provider.responses.pop(0) if provider.responses else {"json": {}}
                status = spec.get("status", 200)
                if "sse" in spec:
                    self.send_response(status)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for event in spec["sse"]:
                        name, data = event if isinstance(event, tuple) else (None, event)
                        chunk = (f"event: {name}\n" if name else "") + f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                    return
                payload = json.dumps(spec.get("json", {})).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                for key, value in spec.get("headers", {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:  # silence test output
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def reply(self, **spec: Any) -> "FakeProvider":
        self.responses.append(spec)
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def last(self) -> Dict[str, Any]:
        return self.requests[-1]


@pytest.fixture()
def fake():
    server = FakeProvider()
    yield server
    server.close()


def _openai_reply(text: str, prompt_tokens: int = 11, completion_tokens: int = 3) -> Dict[str, Any]:
    return {
        "json": {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    }


def _client(fake: FakeProvider, cls=OpenAICompatibleClient, **kwargs: Any):
    defaults: Dict[str, Any] = dict(model="m-1", api_key="key-123", base_url=fake.url + "/v1", provider_name="openrouter", retry_base_delay=0.01)
    defaults.update(kwargs)
    return cls(**defaults)


# ----------------------------------------------------------------------------- OpenAI protocol


def test_openai_complete_sends_the_documented_request(fake: FakeProvider) -> None:
    fake.reply(**_openai_reply("Paris [1]."))
    client = _client(fake, extra_headers={"X-Title": "raglib"}, max_tokens=50)
    with usage_scope() as meter:
        assert client.complete("Capital of France?", system="Be brief.") == "Paris [1]."
    request = fake.last
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer key-123"
    assert request["headers"]["x-title"] == "raglib"
    assert request["json"] == {
        "model": "m-1",
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Capital of France?"}],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    summary = meter.as_dict()
    assert summary["total_prompt_tokens"] == 11 and summary["total_completion_tokens"] == 3
    assert summary["usage_estimated"] is False


def test_openai_reasoning_models_get_max_completion_tokens(fake: FakeProvider) -> None:
    fake.reply(**_openai_reply("ok"))
    _client(fake, provider_name="openai", max_tokens=9).complete("hi")
    assert fake.last["json"]["max_completion_tokens"] == 9 and "max_tokens" not in fake.last["json"]


def test_chat_keeps_roles_and_hoists_system(fake: FakeProvider) -> None:
    fake.reply(**_openai_reply("fine"))
    _client(fake).chat([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
        {"role": "ai", "content": "hello"},
        {"role": "human", "content": "how are you?"},
    ])
    assert fake.last["json"]["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you?"},
    ]


def test_streaming_yields_deltas_and_records_usage(fake: FakeProvider) -> None:
    fake.reply(sse=[
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hel"}}]},
        ": keep-alive comment is ignored",
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
        "[DONE]",
    ])
    with usage_scope() as meter:
        assert list(_client(fake).stream("Say hello")) == ["Hel", "lo"]
    assert fake.last["json"]["stream"] is True
    assert fake.last["json"]["stream_options"] == {"include_usage": True}
    assert meter.as_dict()["total_prompt_tokens"] == 7


def test_json_mode_is_used_and_remembered_when_rejected(fake: FakeProvider) -> None:
    fake.reply(**_openai_reply('{"verdict": "CORRECT"}'))
    client = _client(fake)
    assert client.complete_json("Grade it") == {"verdict": "CORRECT"}
    assert fake.last["json"]["response_format"] == {"type": "json_object"}

    fake.reply(status=400, json={"error": {"message": "response_format is not supported by this model", "type": "invalid_request_error"}})
    fake.reply(**_openai_reply('Sure: {"verdict": "AMBIGUOUS"}'))
    fake.reply(**_openai_reply('{"verdict": "INCORRECT"}'))
    other = _client(fake)
    assert other.complete_json("Grade it") == {"verdict": "AMBIGUOUS"}
    assert "response_format" not in fake.last["json"]
    assert other.complete_json("Grade again") == {"verdict": "INCORRECT"}
    assert "response_format" not in fake.last["json"]  # not retried once rejected


def test_429_with_retry_after_is_retried(fake: FakeProvider) -> None:
    fake.reply(status=429, json={"error": {"message": "Rate limit reached"}}, headers={"Retry-After": "0"})
    fake.reply(**_openai_reply("after retry"))
    assert _client(fake, max_retries=2).complete("q") == "after retry"
    assert len(fake.requests) == 2


def test_400_is_not_retried_and_keeps_status(fake: FakeProvider) -> None:
    fake.reply(status=400, json={"error": {"message": "bad model id", "type": "invalid_request_error"}})
    with pytest.raises(ProviderHTTPError) as info:
        _client(fake, max_retries=3).complete("q")
    assert info.value.status_code == 400 and "bad model id" in str(info.value)
    assert len(fake.requests) == 1


def test_quota_errors_fail_fast(fake: FakeProvider) -> None:
    fake.reply(status=402, json={"error": {"message": "Insufficient credits", "code": 402}})
    with pytest.raises(QuotaExceededError):
        _client(fake, max_retries=3).complete("q")
    assert len(fake.requests) == 1


def test_connection_errors_are_retryable_and_typed() -> None:
    client = OpenAICompatibleClient(model="m", base_url="http://127.0.0.1:9/v1", provider_name="vllm", max_retries=1, retry_base_delay=0.01)
    with pytest.raises(ProviderConnectionError, match="127.0.0.1:9"):
        client.complete("q")


def test_azure_routing(fake: FakeProvider) -> None:
    fake.reply(**_openai_reply("azure ok"))
    client = OpenAICompatibleClient(model="my-deploy", api_key="az", base_url=fake.url, provider_name="azure", azure=True, api_version="2024-10-21")
    assert client.complete("q") == "azure ok"
    assert fake.last["path"] == "/openai/deployments/my-deploy/chat/completions?api-version=2024-10-21"
    assert fake.last["headers"]["api-key"] == "az"


# ----------------------------------------------------------------------------- Anthropic


def test_anthropic_messages_request_and_usage(fake: FakeProvider) -> None:
    fake.reply(json={
        "content": [{"type": "text", "text": "Claude says hi"}],
        "usage": {"input_tokens": 20, "output_tokens": 4},
    })
    client = _client(fake, AnthropicClient, base_url=fake.url, provider_name="anthropic")
    with usage_scope() as meter:
        assert client.complete("Hello", system="Sys") == "Claude says hi"
    request = fake.last
    assert request["path"] == "/v1/messages"  # host-only base URL gets /v1 (SDK convention)
    assert request["headers"]["x-api-key"] == "key-123"
    assert request["headers"]["anthropic-version"] == "2023-06-01"
    assert request["json"]["system"] == "Sys"
    assert request["json"]["max_tokens"] == 4096
    assert request["json"]["messages"] == [{"role": "user", "content": "Hello"}]
    assert meter.as_dict()["total_prompt_tokens"] == 20


def test_anthropic_chat_merges_turns_and_starts_with_user(fake: FakeProvider) -> None:
    fake.reply(json={"content": [{"type": "text", "text": "ok"}]})
    _client(fake, AnthropicClient, base_url=fake.url + "/v1").chat([
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ])
    assert fake.last["json"]["messages"] == [
        {"role": "user", "content": "(conversation start)"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "a\n\nb"},
    ]


def test_anthropic_streaming_events(fake: FakeProvider) -> None:
    fake.reply(sse=[
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 9, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("ping", {"type": "ping"}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi "}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "there"}}),
        ("message_delta", {"type": "message_delta", "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ])
    with usage_scope() as meter:
        assert "".join(_client(fake, AnthropicClient, base_url=fake.url).stream("hi")) == "Hi there"
    summary = meter.as_dict()
    assert summary["total_prompt_tokens"] == 9 and summary["total_completion_tokens"] == 5


def test_anthropic_overloaded_stream_error_is_retried(fake: FakeProvider) -> None:
    fake.reply(sse=[("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})])
    fake.reply(sse=[("content_block_delta", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}})])
    assert list(_client(fake, AnthropicClient, base_url=fake.url, max_retries=1).stream("hi")) == ["ok"]


def test_anthropic_vision_sends_base64_image(fake: FakeProvider, tmp_path) -> None:
    image = tmp_path / "page.jpg"
    image.write_bytes(b"\xff\xd8fake-jpeg")
    fake.reply(json={"content": [{"type": "text", "text": "  Scanned text  "}]})
    client = AnthropicVisionClient(api_key="k", base_url=fake.url)
    assert client.extract_text_from_image(str(image)) == "Scanned text"
    block = fake.last["json"]["messages"][0]["content"][0]
    assert block["type"] == "image" and block["source"]["media_type"] == "image/jpeg"


# ----------------------------------------------------------------------------- Gemini


def test_gemini_generate_content(fake: FakeProvider) -> None:
    fake.reply(json={
        "candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "Answer"}]}}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 2},
    })
    client = _client(fake, GeminiClient, model="gemini-2.5-flash", base_url=fake.url + "/v1beta", max_tokens=64)
    assert client.complete("Q", system="S") == "Answer"
    request = fake.last
    assert request["path"] == "/v1beta/models/gemini-2.5-flash:generateContent"
    assert request["headers"]["x-goog-api-key"] == "key-123"
    assert request["json"]["systemInstruction"] == {"parts": [{"text": "S"}]}
    assert request["json"]["contents"] == [{"role": "user", "parts": [{"text": "Q"}]}]
    assert request["json"]["generationConfig"] == {"temperature": 0.0, "maxOutputTokens": 64}


def test_gemini_json_mode_and_stream(fake: FakeProvider) -> None:
    fake.reply(json={"candidates": [{"content": {"parts": [{"text": '{"a": 1}'}]}}]})
    client = _client(fake, GeminiClient, base_url=fake.url)
    assert client.complete_json("json please") == {"a": 1}
    assert fake.last["json"]["generationConfig"]["responseMimeType"] == "application/json"

    fake.reply(sse=[
        {"candidates": [{"content": {"parts": [{"text": "Str"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "eam"}]}}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}},
    ])
    assert "".join(client.stream("go")) == "Stream"
    assert fake.last["path"].endswith(":streamGenerateContent?alt=sse")


# ----------------------------------------------------------------------------- embeddings


def test_openai_embeddings(fake: FakeProvider) -> None:
    fake.reply(json={"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]})
    embedding = OpenAIEmbedding(api_key="k", model_name="text-embedding-3-small", base_url=fake.url + "/v1")
    assert embedding.embed_documents(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]  # reordered by index
    assert fake.last["json"] == {"input": ["a", "b"], "model": "text-embedding-3-small", "encoding_format": "float"}
    assert embedding.dimension == 2


def test_gemini_embeddings_use_task_types(fake: FakeProvider) -> None:
    fake.reply(json={"embeddings": [{"values": [0.5, 0.5, 0.0]}]})
    fake.reply(json={"embeddings": [{"values": [0.1, 0.2, 0.3]}]})
    embedding = GoogleEmbedding(api_key="g", model_name="gemini-embedding-001", base_url=fake.url)
    embedding.embed_documents(["doc"])
    assert fake.last["path"] == "/models/gemini-embedding-001:batchEmbedContents"
    assert fake.last["json"]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert embedding.embed_query("q") == [0.1, 0.2, 0.3]
    assert fake.last["json"]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


def test_ollama_embeddings(fake: FakeProvider) -> None:
    fake.reply(json={"embeddings": [[1.0, 2.0], [3.0, 4.0]]})
    embedding = OllamaEmbedding(model_name="nomic-embed-text", base_url=fake.url + "/v1")
    assert embedding.embed_documents(["x", "y"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert fake.last["path"] == "/api/embed"


# ----------------------------------------------------------------------------- end to end


def test_full_rag_pipeline_over_http(fake: FakeProvider) -> None:
    """RAG(...) with a provider name talks to the endpoint and returns a cited answer."""

    fake.reply(**_openai_reply("The launch is on 3 March [1].", 40, 8))
    rag = RAG(
        ["The product launch is on 3 March.", "The budget is 2 million."],
        llm="vllm/my-model",
        chat_base_url=fake.url + "/v1",
        embedding_llm="none",
        llm_max_retries=0,
    )
    result = rag.query("When is the launch?")
    assert result.answer.startswith("The launch is on 3 March")
    assert result.citations and "3 March" in result.citations[0].snippet
    assert result.usage["total_prompt_tokens"] == 40
    assert "<source" in fake.last["json"]["messages"][-1]["content"]


def test_build_chat_client_routes_every_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu:11434")
    assert isinstance(build_chat_client("anthropic", api_key="a"), AnthropicClient)
    assert isinstance(build_chat_client("google", api_key="g"), GeminiClient)
    groq = build_chat_client("groq", api_key="gsk_x")
    assert isinstance(groq, OpenAICompatibleClient) and groq.base_url == "https://api.groq.com/openai/v1"
    assert build_chat_client("ollama").base_url == "http://gpu:11434/v1"
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        build_chat_client("anthropic")
    assert "key" not in repr(build_chat_client("openai", api_key="sk-secret")).replace("api_key", "")


def test_no_langchain_is_imported() -> None:
    """Building and using provider clients must not import LangChain."""

    code = (
        "import sys\n"
        "from raglib import RAG\n"
        "from raglib.llm.clients import build_chat_client\n"
        "from raglib.embedding.openai_embedding import OpenAIEmbedding\n"
        "for p in ('openai', 'anthropic', 'google', 'groq', 'ollama', 'openrouter'):\n"
        "    build_chat_client(p, api_key='k')\n"
        "OpenAIEmbedding(api_key='k')\n"
        "RAG('text', llm='openrouter', chat_api_key='k', embedding_llm='none')\n"
        "bad = sorted(m for m in sys.modules if m.startswith(('langchain', 'openai', 'anthropic', 'google.genai')))\n"
        "print(','.join(bad))\n"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert output == ""
