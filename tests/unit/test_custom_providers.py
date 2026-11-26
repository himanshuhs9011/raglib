"""Custom / third-party providers: register_provider, providers files, model flags. Local HTTP only."""

from __future__ import annotations

import json

import pytest

from raglib import RAG, list_models, load_providers, register_provider, unregister_provider
from raglib.cli import main as cli_main
from raglib.llm import providers as provider_registry
from raglib.llm.base_client import unwrap_llm
from raglib.llm.clients import AnthropicClient, GeminiClient, OpenAICompatibleClient, build_chat_client
from raglib.vision.chat_vision import ChatVisionClient
from tests.unit.test_native_clients import FakeProvider, _openai_reply


@pytest.fixture()
def fake():
    server = FakeProvider()
    yield server
    server.close()


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch):
    """Every test starts and ends with only the built-in providers."""

    monkeypatch.delenv(provider_registry.PROVIDERS_FILE_ENV_VAR, raising=False)
    monkeypatch.setattr(provider_registry, "_ENV_FILE_LOADED", None)
    yield
    for name in [n for n, spec in provider_registry.PROVIDERS.items() if spec.custom]:
        unregister_provider(name)


def _register(fake: FakeProvider, **kwargs):
    options = dict(
        name="My AI Provider",
        base_url=fake.url + "/v1",
        api_key="{env:MYPROVIDER_API_KEY}",
        models=[{"id": "model-id", "name": "Display Name"}],
    )
    options.update(kwargs)
    return register_provider("myprovider", **options)


def test_custom_provider_end_to_end(fake: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """The form from the request: id, name, OpenAI API, base URL, key, models."""

    monkeypatch.setenv("MYPROVIDER_API_KEY", "secret-123")
    monkeypatch.setenv("TEAM_ID", "search-team")
    _register(fake, headers={"X-Team": "{env:TEAM_ID}"})
    fake.reply(**_openai_reply("Refunds are allowed within 30 days [1].", 50, 9))

    rag = RAG(["Refunds are allowed within 30 days.", "Shipping is free."], llm="myprovider/model-id", embedding_llm="none")
    result = rag.query("When are refunds allowed?")

    assert result.answer.startswith("Refunds are allowed within 30 days")
    assert result.citations and result.usage["total_prompt_tokens"] == 50
    request = fake.last
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer secret-123"
    assert request["headers"]["x-team"] == "search-team"
    assert request["json"]["model"] == "model-id"
    assert request["json"]["temperature"] == 0.0


def test_default_model_is_the_first_listed(fake: FakeProvider) -> None:
    _register(fake, api_key=None, models=["first-model", "second-model"])
    assert unwrap_llm(RAG(llm="myprovider", embedding_llm="none")._llm).model_name == "first-model"
    assert [model.id for model in list_models("myprovider")] == ["first-model", "second-model"]


def test_auth_by_headers_only(fake: FakeProvider) -> None:
    """API key is optional: leave it empty when auth is done via headers."""

    _register(fake, api_key=None, headers={"X-Api-Token": "tok"})
    fake.reply(**_openai_reply("ok"))
    client = build_chat_client("myprovider")
    assert client.complete("hi") == "ok"
    assert "authorization" not in fake.last["headers"]
    assert fake.last["headers"]["x-api-token"] == "tok"


def test_missing_key_is_reported(fake: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MYPROVIDER_API_KEY", raising=False)
    _register(fake)
    with pytest.raises(ValueError, match="MYPROVIDER_API_KEY"):
        build_chat_client("myprovider")


def test_reasoning_model_sends_no_temperature_and_hides_thinking(fake: FakeProvider) -> None:
    _register(fake, api_key=None, models=[{"id": "r1", "reasoning": True}])
    fake.reply(**_openai_reply("<think>The user wants the date. Source 1 has it.</think>\n\nThe launch is 3 March [1]."))
    rag = RAG(["The launch is on 3 March."], llm="myprovider/r1", embedding_llm="none")
    result = rag.query("When is the launch?")
    assert "temperature" not in fake.last["json"]
    assert result.answer == "The launch is 3 March [1]."
    assert result.citations

    fake.reply(sse=[
        {"choices": [{"delta": {"content": "<think>hmm"}}]},
        {"choices": [{"delta": {"content": " still thinking</thi"}}]},
        {"choices": [{"delta": {"content": "nk>\n"}}]},
        {"choices": [{"delta": {"content": "Answer"}}]},
    ])
    assert "".join(unwrap_llm(rag._llm).stream("q")) == "Answer"


@pytest.mark.parametrize("api, client_cls, path", [
    ("anthropic", AnthropicClient, "/v1/messages"),
    ("gemini", GeminiClient, "/v1/models/model-id:generateContent"),
    ("openai-compatible", OpenAICompatibleClient, "/v1/chat/completions"),
])
def test_every_protocol_can_be_used(fake: FakeProvider, api, client_cls, path) -> None:
    _register(fake, api=api, api_key="k")
    replies = {
        "anthropic": {"json": {"content": [{"type": "text", "text": "hi"}]}},
        "gemini": {"json": {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}},
        "openai-compatible": _openai_reply("hi"),
    }
    fake.reply(**replies[api])
    client = build_chat_client("myprovider")
    assert isinstance(client, client_cls)
    assert client.complete("hello") == "hi"
    assert fake.last["path"] == path


@pytest.mark.parametrize("kwargs, message", [
    ({"provider_id": "My Provider"}, "Invalid provider id"),
    ({"provider_id": "openai"}, "built-in"),
    ({"provider_id": "claude"}, "built-in"),  # alias of a built-in
    ({"api": "cohere"}, "Unknown provider api"),
    ({"base_url": "api.example.com/v1"}, "http"),
    ({"default_model": "nope"}, "not in the models list"),
    ({"models": [{"id": "m", "vision": True}]}, "Unknown option"),
    ({"models": ["m", "m"]}, "Duplicate"),
])
def test_validation(kwargs, message) -> None:
    options = {"provider_id": "myprovider", "base_url": "https://api.example.com/v1", "models": ["m"]}
    options.update(kwargs)
    provider_id = options.pop("provider_id")
    with pytest.raises(ValueError, match=message):
        register_provider(provider_id, **options)


def test_duplicate_needs_override(fake: FakeProvider) -> None:
    _register(fake)
    with pytest.raises(ValueError, match="already registered"):
        _register(fake)
    assert _register(fake, name="Renamed", override=True).description == "Renamed"
    with pytest.raises(ValueError, match="cannot be removed"):
        unregister_provider("openai")


def _write_config(tmp_path, base_url: str):
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({
        "providers": {
            "myprovider": {
                "name": "My AI Provider",
                "api": "openai",
                "base_url": base_url,
                "api_key": "{env:MYPROVIDER_API_KEY}",
                "models": {
                    "model-id": {"name": "Display Name", "reasoning": False, "image": False},
                    "vision-model": {"name": "Vision", "image": True},
                },
                "embedding_model": "embed-1",
            }
        }
    }), encoding="utf-8")
    return path


def test_load_providers_file(fake: FakeProvider, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYPROVIDER_API_KEY", "k")
    [spec] = load_providers(_write_config(tmp_path, fake.url + "/v1"))
    assert spec.description == "My AI Provider" and spec.default_chat_model == "model-id"
    assert spec.model("vision-model").image and spec.supports_vision


def test_providers_file_from_environment(fake: FakeProvider, tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("MYPROVIDER_API_KEY", "k")
    monkeypatch.setenv(provider_registry.PROVIDERS_FILE_ENV_VAR, str(_write_config(tmp_path, fake.url + "/v1")))
    rag = RAG(llm="myprovider/model-id", embedding_llm="none")  # found without calling load_providers()
    assert unwrap_llm(rag._llm).base_url == fake.url + "/v1"
    assert cli_main(["providers"]) == 0
    output = capsys.readouterr().out
    assert "My AI Provider" in output and "vision-model (Vision)  [image]" in output


def test_cli_providers_flag(fake: FakeProvider, tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("MYPROVIDER_API_KEY", "k")
    config = _write_config(tmp_path, fake.url + "/v1")
    doc = tmp_path / "notes.md"
    doc.write_text("The launch date is 3 March.", encoding="utf-8")
    fake.reply(**_openai_reply("3 March [1]."))
    code = cli_main(["ask", str(doc), "When is the launch?", "--providers", str(config),
                     "--llm", "myprovider/model-id", "--embedding", "none"])
    assert code == 0 and "3 March" in capsys.readouterr().out
    assert cli_main(["providers", "--providers", str(tmp_path / "missing.json")]) == 2


def test_bad_providers_file_is_explained(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"providers": {"x": {"api": "openai"}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="needs a base_url"):
        load_providers(path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_providers(path)


def test_custom_embeddings_send_headers(fake: FakeProvider) -> None:
    _register(fake, api_key="k", headers={"X-Team": "t"}, embedding_model="embed-1")
    fake.reply(json={"data": [{"index": 0, "embedding": [1.0, 0.0]}, {"index": 1, "embedding": [0.0, 1.0]}]})
    rag = RAG(["alpha text", "beta text"], llm="myprovider/model-id")
    assert rag.stats()["retrieval_mode"] == "hybrid"
    request = fake.requests[0]
    assert request["path"] == "/v1/embeddings" and request["json"]["model"] == "embed-1"
    assert request["headers"]["x-team"] == "t" and request["headers"]["authorization"] == "Bearer k"


def test_anthropic_protocol_has_no_embeddings(fake: FakeProvider) -> None:
    _register(fake, api="anthropic", api_key="k", embedding_model="x")
    with pytest.raises(ValueError, match="no embeddings endpoint"):
        RAG(llm="myprovider/model-id", embedding_llm="myprovider")


def test_image_models_do_ocr(fake: FakeProvider, tmp_path) -> None:
    _register(fake, api_key=None, models=[{"id": "chat"}, {"id": "eyes", "image": True}])

    rag = RAG(llm="myprovider/eyes", embedding_llm="none")  # chat model accepts images: reused for OCR
    assert isinstance(rag._vision, ChatVisionClient)
    assert RAG(llm="myprovider/chat", embedding_llm="none")._vision is None  # text-only model: no OCR

    rag = RAG(llm="myprovider/chat", embedding_llm="none", vision_llm="myprovider")  # first image model
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG fake")
    fake.reply(**_openai_reply("Invoice #42"))
    assert rag._vision.extract_text_from_image(str(image)) == "Invoice #42"
    assert fake.last["json"]["model"] == "eyes"
    assert fake.last["json"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    with pytest.raises(ValueError, match="not marked image=True"):
        RAG(llm="myprovider/chat", embedding_llm="none", vision_llm="myprovider/chat")


def test_chat_key_is_not_sent_to_a_different_custom_vision_provider(fake: FakeProvider) -> None:
    register_provider("eyes-only", base_url=fake.url + "/v1", models=[{"id": "v", "image": True}])
    rag = RAG(llm="openrouter", chat_api_key="sk-or-v1-secret", embedding_llm="none", vision_llm="eyes-only")
    assert rag._vision._client.api_key is None


# ----------------------------------------------------------------------------- NVIDIA NIM (built-in)


def test_nvidia_is_built_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from raglib.llm.providers import detect_provider_from_key, parse_model_string

    assert detect_provider_from_key("nvapi-abc") == "nvidia"
    assert parse_model_string("nim/meta/llama-3.3-70b-instruct") == ("nvidia", "meta/llama-3.3-70b-instruct")
    client = build_chat_client("nvidia", api_key="nvapi-x")
    assert client.base_url == "https://integrate.api.nvidia.com/v1"
    assert client.model_name == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert client.stream_usage is True
    monkeypatch.setenv("NVIDIA_BASE_URL", "http://my-nim:8000/v1")  # self-hosted NIM
    assert build_chat_client("nvidia", api_key="x").base_url == "http://my-nim:8000/v1"


def test_nvidia_embeddings_send_input_type_and_truncate(fake: FakeProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """Asymmetric NIM retrievers reject requests without input_type (found in live testing)."""

    monkeypatch.setenv("NVIDIA_BASE_URL", fake.url + "/v1")
    fake.reply(json={"data": [{"index": 0, "embedding": [1.0, 0.0]}, {"index": 1, "embedding": [0.0, 1.0]}]})
    rag = RAG(["alpha", "beta"], llm="nvidia", chat_api_key="nvapi-x")
    documents_request = fake.requests[0]["json"]
    assert documents_request["model"] == "nvidia/nemotron-3.5-embed-1b"
    assert documents_request["input_type"] == "passage" and documents_request["truncate"] == "END"

    fake.reply(json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    fake.reply(**_openai_reply("alpha [1]"))
    rag.query("alpha?")
    assert fake.requests[1]["json"]["input_type"] == "query"


def test_nvidia_vision_uses_the_vl_model(monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = RAG(llm="openai", chat_api_key="sk-x", embedding_llm="none", vision_llm="nvidia", vision_api_key="nvapi-v")
    assert isinstance(explicit._vision, ChatVisionClient)
    assert explicit._vision._client.model_name == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert explicit._vision._client.api_key == "nvapi-v"
    automatic = RAG(llm="nvidia", chat_api_key="nvapi-x", embedding_llm="none")  # like openai/anthropic/google
    assert isinstance(automatic._vision, ChatVisionClient)


def test_extra_body_is_merged_into_requests(fake: FakeProvider) -> None:
    register_provider(
        "thinking-off", base_url=fake.url + "/v1", models=["m"], stream_usage=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20},
    )
    client = build_chat_client("thinking-off")
    fake.reply(**_openai_reply('{"a": 1}'))
    assert client.complete_json("json") == {"a": 1}
    body = fake.last["json"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["top_k"] == 20
    assert body["response_format"] == {"type": "json_object"}  # core fields kept
    fake.reply(sse=[{"choices": [{"delta": {"content": "x"}}]}])
    list(client.stream("s"))
    assert fake.last["json"]["stream_options"] == {"include_usage": True}


def test_json_mode_can_be_turned_off_per_model(fake: FakeProvider) -> None:
    """Some small models loop in JSON mode; json_mode=False parses JSON from the plain reply instead."""

    register_provider(
        "loopy", base_url=fake.url + "/v1",
        models=[{"id": "small", "json_mode": False}, {"id": "big"}],
    )
    small = build_chat_client("loopy", model_name="small")
    fake.reply(**_openai_reply('Here you go:\n```json\n{"entities": [1]}\n```'))
    assert small.complete_json("extract") == {"entities": [1]}
    assert "response_format" not in fake.last["json"]
    assert len(fake.requests) == 1  # one call, no JSON-mode attempt first

    big = build_chat_client("loopy", model_name="big")
    fake.reply(**_openai_reply('{"ok": true}'))
    assert big.complete_json("x") == {"ok": True}
    assert fake.last["json"]["response_format"] == {"type": "json_object"}  # default unchanged
    assert provider_registry.PROVIDERS["loopy"].model("small").json_mode is False


def test_custom_embedding_input_types(fake: FakeProvider) -> None:
    with pytest.raises(ValueError, match="two values"):
        register_provider("bad", base_url=fake.url, embedding_input_types=["passage"])
    register_provider("myretriever", base_url=fake.url + "/v1", models=["m"], embedding_model="e",
                      embedding_input_types=["search_document", "search_query"])
    fake.reply(json={"data": [{"index": 0, "embedding": [1.0]}]})
    RAG(["doc"], llm="myretriever")
    assert fake.last["json"]["input_type"] == "search_document"
