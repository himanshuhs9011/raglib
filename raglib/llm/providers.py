"""Registry of chat/embedding providers raglib can talk to.

Most hosted LLM APIs (OpenRouter, DeepSeek, Mistral, Together, Fireworks, xAI, Perplexity,
Cerebras, LM Studio, vLLM, ...) speak the OpenAI Chat Completions protocol, so one adapter
with a different ``base_url`` covers all of them. Adding a provider = adding one entry here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

KIND_NATIVE = "native"  # dedicated LangChain integration (openai, anthropic, groq, google, ollama)
KIND_OPENAI_COMPATIBLE = "openai_compatible"
KIND_AZURE = "azure"


@dataclass(frozen=True)
class ProviderSpec:
    """How to reach one provider."""

    name: str
    kind: str
    default_chat_model: Optional[str]
    env_keys: Sequence[str] = ()
    base_url: Optional[str] = None
    base_url_env: Sequence[str] = ()
    requires_key: bool = True
    key_prefixes: Sequence[str] = ()
    default_embedding_model: Optional[str] = None
    supports_vision: bool = False
    extra_headers: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    def resolve_api_key(self, explicit: Optional[str] = None) -> Optional[str]:
        """Explicit key first, then the provider's environment variables."""

        if explicit and explicit.strip():
            return explicit.strip()
        for key in self.env_keys:
            value = os.getenv(key)
            if value and value.strip():
                return value.strip()
        return None

    def resolve_base_url(self, explicit: Optional[str] = None) -> Optional[str]:
        """Explicit base_url first, then env, then the provider default."""

        if explicit and explicit.strip():
            return explicit.strip()
        for key in self.base_url_env:
            value = os.getenv(key)
            if value and value.strip():
                return value.strip()
        return self.base_url


PROVIDERS: Dict[str, ProviderSpec] = {
    spec.name: spec
    for spec in (
        ProviderSpec("openai", KIND_NATIVE, "gpt-4.1-mini", ("OPENAI_API_KEY",), base_url_env=("OPENAI_BASE_URL",),
                     key_prefixes=("sk-proj-", "sk-"), default_embedding_model="text-embedding-3-small",
                     supports_vision=True, description="OpenAI"),
        ProviderSpec("anthropic", KIND_NATIVE, "claude-haiku-5-5", ("ANTHROPIC_API_KEY",),
                     key_prefixes=("sk-ant-",), supports_vision=True, description="Anthropic Claude"),
        ProviderSpec("google", KIND_NATIVE, "gemini-2.5-flash", ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
                     key_prefixes=("AIza",), default_embedding_model="models/gemini-embedding-001",
                     supports_vision=True, description="Google Gemini"),
        ProviderSpec("groq", KIND_NATIVE, "llama-3.3-70b-versatile", ("GROQ_API_KEY",),
                     key_prefixes=("gsk_",), description="Groq"),
        ProviderSpec("ollama", KIND_NATIVE, "llama3.1", base_url="http://localhost:11434",
                     base_url_env=("OLLAMA_BASE_URL", "OLLAMA_HOST"), requires_key=False,
                     default_embedding_model="nomic-embed-text", description="Ollama (local)"),
        ProviderSpec("openrouter", KIND_OPENAI_COMPATIBLE, "openai/gpt-4.1-mini", ("OPENROUTER_API_KEY",),
                     base_url="https://openrouter.ai/api/v1", key_prefixes=("sk-or-",), supports_vision=True,
                     extra_headers={"X-Title": "raglib"},
                     description="OpenRouter (one key for 300+ models: GPT, Claude, Gemini, Llama, DeepSeek, Qwen...)"),
        ProviderSpec("deepseek", KIND_OPENAI_COMPATIBLE, "deepseek-chat", ("DEEPSEEK_API_KEY",),
                     base_url="https://api.deepseek.com/v1", description="DeepSeek"),
        ProviderSpec("mistral", KIND_OPENAI_COMPATIBLE, "mistral-small-latest", ("MISTRAL_API_KEY",),
                     base_url="https://api.mistral.ai/v1", default_embedding_model="mistral-embed",
                     description="Mistral AI"),
        ProviderSpec("together", KIND_OPENAI_COMPATIBLE, "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                     ("TOGETHER_API_KEY",), base_url="https://api.together.xyz/v1", description="Together AI"),
        ProviderSpec("fireworks", KIND_OPENAI_COMPATIBLE, "accounts/fireworks/models/llama-v3p3-70b-instruct",
                     ("FIREWORKS_API_KEY",), base_url="https://api.fireworks.ai/inference/v1",
                     key_prefixes=("fw_",), description="Fireworks AI"),
        ProviderSpec("xai", KIND_OPENAI_COMPATIBLE, "grok-3-mini", ("XAI_API_KEY",), base_url="https://api.x.ai/v1",
                     key_prefixes=("xai-",), description="xAI Grok"),
        ProviderSpec("perplexity", KIND_OPENAI_COMPATIBLE, "sonar", ("PERPLEXITY_API_KEY", "PPLX_API_KEY"),
                     base_url="https://api.perplexity.ai", key_prefixes=("pplx-",), description="Perplexity"),
        ProviderSpec("cerebras", KIND_OPENAI_COMPATIBLE, "llama-3.3-70b", ("CEREBRAS_API_KEY",),
                     base_url="https://api.cerebras.ai/v1", key_prefixes=("csk-",), description="Cerebras"),
        ProviderSpec("lmstudio", KIND_OPENAI_COMPATIBLE, None, ("LMSTUDIO_API_KEY",),
                     base_url="http://localhost:1234/v1", base_url_env=("LMSTUDIO_BASE_URL",), requires_key=False,
                     description="LM Studio (local)"),
        ProviderSpec("vllm", KIND_OPENAI_COMPATIBLE, None, ("VLLM_API_KEY",), base_url="http://localhost:8000/v1",
                     base_url_env=("VLLM_BASE_URL",), requires_key=False, description="vLLM server (self-hosted)"),
        ProviderSpec("openai_compatible", KIND_OPENAI_COMPATIBLE, None, ("OPENAI_COMPATIBLE_API_KEY",),
                     base_url_env=("OPENAI_COMPATIBLE_BASE_URL",), requires_key=False,
                     description="Any other OpenAI-compatible endpoint (pass base_url + model)"),
        ProviderSpec("azure", KIND_AZURE, None, ("AZURE_OPENAI_API_KEY",), base_url_env=("AZURE_OPENAI_ENDPOINT",),
                     supports_vision=True, description="Azure OpenAI (model = deployment name)"),
    )
}

PROVIDER_ALIASES: Dict[str, str] = {
    "or": "openrouter",
    "open_router": "openrouter",
    "open-router": "openrouter",
    "gemini": "google",
    "claude": "anthropic",
    "grok": "xai",
    "x-ai": "xai",
    "lm_studio": "lmstudio",
    "lm-studio": "lmstudio",
    "azure_openai": "azure",
    "azure-openai": "azure",
    "openai-compatible": "openai_compatible",
    "custom": "openai_compatible",
    "pplx": "perplexity",
    "togetherai": "together",
}

ENV_DETECTION_ORDER: Tuple[str, ...] = (
    "openai", "anthropic", "openrouter", "google", "groq", "mistral", "deepseek",
    "together", "fireworks", "xai", "cerebras", "perplexity", "azure",
)


def normalize_provider(name: Optional[str]) -> Optional[str]:
    """Return the canonical provider name, or None if unknown."""

    if not name:
        return None
    key = name.strip().lower()
    key = PROVIDER_ALIASES.get(key, key)
    return key if key in PROVIDERS else None


def get_provider(name: str) -> ProviderSpec:
    """Return a provider spec or raise a helpful error."""

    canonical = normalize_provider(name)
    if canonical is None:
        raise ValueError(f"Unknown LLM provider '{name}'. Supported: {', '.join(sorted(PROVIDERS))}")
    return PROVIDERS[canonical]


def parse_model_string(value: str) -> Tuple[Optional[str], Optional[str]]:
    """Split ``"provider/model"`` or ``"provider:model"`` into (provider, model).

    Only the first separator counts, so OpenRouter ids work:
    ``"openrouter/anthropic/claude-sonnet-4"`` -> ("openrouter", "anthropic/claude-sonnet-4").
    A bare provider name returns (provider, None); anything else returns (None, None).
    """

    text = (value or "").strip()
    canonical = normalize_provider(text)
    if canonical:
        return canonical, None
    for separator in ("/", ":"):
        if separator in text:
            head, tail = text.split(separator, 1)
            canonical = normalize_provider(head)
            if canonical and tail.strip():
                return canonical, tail.strip()
    return None, None


def detect_provider_from_key(api_key: str) -> Optional[str]:
    """Guess the provider from an API key prefix (longest prefix wins, so sk-or- beats sk-)."""

    candidates = [
        (prefix, spec.name)
        for spec in PROVIDERS.values()
        for prefix in spec.key_prefixes
        if api_key.startswith(prefix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: len(item[0]))[1]


def detect_provider_from_env() -> Optional[str]:
    """Return the first provider whose API key environment variable is set."""

    for name in ENV_DETECTION_ORDER:
        if PROVIDERS[name].resolve_api_key() and (name != "azure" or PROVIDERS[name].resolve_base_url()):
            return name
    return None


def provider_table() -> str:
    """Human-readable list of supported providers (used in error messages and docs)."""

    rows = []
    for spec in PROVIDERS.values():
        key = spec.env_keys[0] if spec.env_keys and spec.requires_key else "(no key)"
        rows.append(f"  {spec.name:<18} {key:<26} {spec.description}")
    return "\n".join(rows)
