"""Provider registry, OpenAI-compatible providers (OpenRouter & co.), simple APIs. No network calls."""

from __future__ import annotations

import pytest

from raglib import RAG, ask
from raglib.cli import main as cli_main
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.llm.base_client import unwrap_llm
from raglib.llm.providers import PROVIDERS, detect_provider_from_key, parse_model_string

langchain_openai = pytest.importorskip("langchain_openai")


def _chat_model(rag: RAG):
    """Return the underlying LangChain chat model of a RAG."""

    return unwrap_llm(rag._llm)._model


@pytest.mark.parametrize(
    "value, expected",
    [
        ("openrouter", ("openrouter", None)),
        ("openrouter/anthropic/claude-sonnet-4", ("openrouter", "anthropic/claude-sonnet-4")),
        ("OpenRouter:meta-llama/llama-3.3-70b-instruct", ("openrouter", "meta-llama/llama-3.3-70b-instruct")),
        ("ollama/llama3.1", ("ollama", "llama3.1")),
        ("gemini/gemini-2.5-pro", ("google", "gemini-2.5-pro")),
        ("meta-llama/llama-3", (None, None)),
    ],
)
def test_parse_model_string(value, expected) -> None:
    """'provider/model' parsing keeps OpenRouter ids (which contain '/') intact."""

    assert parse_model_string(value) == expected


@pytest.mark.parametrize(
    "key, provider",
    [
        ("sk-or-v1-abc", "openrouter"),  # v1 bug: detected as OpenAI
        ("sk-proj-abc", "openai"),
        ("sk-abc", "openai"),
        ("sk-ant-api03-abc", "anthropic"),
        ("gsk_abc", "groq"),
        ("xai-abc", "xai"),
        ("pplx-abc", "perplexity"),
        ("unknown-key", None),
    ],
)
def test_key_prefix_detection(key, provider) -> None:
    """Longest prefix wins."""

    assert detect_provider_from_key(key) == provider


@pytest.mark.parametrize(
    "name",
    [n for n, spec in PROVIDERS.items() if spec.kind == "openai_compatible" and spec.default_chat_model],
)
def test_openai_compatible_providers_build_with_correct_endpoint(name: str) -> None:
    """Every hosted OpenAI-compatible provider points ChatOpenAI at its own base URL."""

    rag = RAG("hello world", llm=name, chat_api_key="test-key", embedding_llm="none")
    model = _chat_model(rag)
    spec = PROVIDERS[name]
    assert isinstance(model, langchain_openai.ChatOpenAI)
    assert model.openai_api_base == spec.base_url
    assert model.model_name == spec.default_chat_model


def test_openrouter_with_model_string_and_headers() -> None:
    """OpenRouter model ids work and the attribution header is sent."""

    rag = RAG("hello", llm="openrouter/deepseek/deepseek-chat", chat_api_key="sk-or-v1-x", embedding_llm="none")
    model = _chat_model(rag)
    assert model.model_name == "deepseek/deepseek-chat"
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.default_headers.get("X-Title") == "raglib"
    assert rag.stats()["retrieval_mode"] == "sparse"


def test_openrouter_key_alone_selects_openrouter() -> None:
    """Passing only an OpenRouter key must not build an OpenAI client."""

    rag = RAG("hello", llm_key="sk-or-v1-abc", embedding_llm="none")
    assert _chat_model(rag).openai_api_base == "https://openrouter.ai/api/v1"


def test_provider_auto_detected_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG('docs') just works when a provider key is in the environment."""

    monkeypatch.delenv("RAGLIB_OFFLINE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-env")
    rag = RAG("hello", embedding_llm="none")
    assert _chat_model(rag).openai_api_base == "https://openrouter.ai/api/v1"


def test_offline_flag_beats_environment_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit offline mode never calls a paid API even if keys exist."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-env")
    assert RAG("hello").stats()["llm"] == "MockLLMClient"


def test_chat_key_is_not_sent_to_a_different_embedding_provider() -> None:
    """An OpenRouter chat key must never be used as an OpenAI embedding key."""

    with pytest.raises(ValueError, match="embedding_api_key"):
        RAG("hello", llm="openrouter", chat_api_key="sk-or-v1-x", embedding_llm="openai")


def test_mistral_uses_its_own_embeddings() -> None:
    """Providers with an embeddings API get semantic search automatically."""

    rag = RAG(llm="mistral", chat_api_key="m-key")  # no source: nothing is embedded, no network
    embedding = rag._embedding
    assert isinstance(embedding, CachedEmbedding)
    inner = embedding.inner._model
    assert inner.model == "mistral-embed"
    assert inner.openai_api_base == "https://api.mistral.ai/v1"
    assert inner.check_embedding_ctx_length is False
    assert rag.stats()["retrieval_mode"] == "hybrid"


def test_embedding_provider_model_string() -> None:
    """embedding_llm also accepts 'provider/model'."""

    rag = RAG(llm="openrouter", chat_api_key="sk-or-v1-x", embedding_llm="mistral/mistral-embed", embedding_api_key="m")
    assert rag._embedding.inner._model.model == "mistral-embed"


def test_local_servers_need_model_but_no_key() -> None:
    """LM Studio / vLLM work without keys; a model name is required."""

    with pytest.raises(ValueError, match="no default model"):
        RAG("hello", llm="lmstudio", embedding_llm="none")
    rag = RAG("hello", llm="vllm/Qwen/Qwen2.5-7B-Instruct", chat_base_url="http://gpu-box:8000/v1", embedding_llm="none")
    model = _chat_model(rag)
    assert model.openai_api_base == "http://gpu-box:8000/v1" and model.model_name == "Qwen/Qwen2.5-7B-Instruct"


def test_missing_key_error_names_the_env_var() -> None:
    """Helpful errors for missing keys."""

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        RAG("hello", llm="openrouter", embedding_llm="none")


def test_unknown_provider_lists_supported_ones() -> None:
    """Typos produce the provider list."""

    with pytest.raises(ValueError, match="openrouter"):
        RAG("hello", llm="opnrouter/some-model")


def test_llm_and_chat_llm_are_exclusive() -> None:
    """Ambiguous configuration fails fast."""

    with pytest.raises(ValueError, match="either llm"):
        RAG("hello", llm="openrouter", chat_llm="openai")


def test_ask_one_liner() -> None:
    """ask() indexes and answers in one call."""

    result = ask("Photosynthesis converts light energy into chemical energy.", "What does photosynthesis convert?")
    assert result.citations
    results = ask("Photosynthesis converts light.", ["What is converted?", "Is light converted?"])
    assert len(results) == 2


def test_cli_ask_and_providers(tmp_path, capsys) -> None:
    """The raglib CLI answers questions about files."""

    doc = tmp_path / "notes.md"
    doc.write_text("The launch date is 3 March. The budget is 2 million.", encoding="utf-8")
    assert cli_main(["ask", str(doc), "When is the launch date?", "--offline"]) == 0
    output = capsys.readouterr().out
    assert "launch date" in output.lower() and "notes.md" in output
    assert cli_main(["providers"]) == 0
    assert "openrouter" in capsys.readouterr().out
    assert cli_main(["ask", "only-a-question"]) == 2
