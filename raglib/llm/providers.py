"""Registry of chat/embedding providers raglib can talk to.

Most hosted LLM APIs (OpenRouter, DeepSeek, Mistral, Together, Fireworks, xAI, Perplexity,
Cerebras, LM Studio, vLLM, ...) speak the OpenAI Chat Completions protocol, so one adapter
with a different ``base_url`` covers all of them (``raglib.llm.clients``). Adding a provider = adding one entry here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

KIND_NATIVE = "native"  # first-party provider with its own default endpoint (see raglib.llm.clients)
KIND_OPENAI_COMPATIBLE = "openai_compatible"
KIND_AZURE = "azure"
KIND_CUSTOM = "custom"  # registered at runtime with register_provider() / load_providers()

# Wire protocols understood by raglib.llm.clients.
PROTOCOL_OPENAI = "openai"
PROTOCOL_ANTHROPIC = "anthropic"
PROTOCOL_GEMINI = "gemini"
_PROTOCOL_ALIASES = {
    "openai": PROTOCOL_OPENAI,
    "openai-compatible": PROTOCOL_OPENAI,
    "openai_compatible": PROTOCOL_OPENAI,
    "chat-completions": PROTOCOL_OPENAI,
    "anthropic": PROTOCOL_ANTHROPIC,
    "anthropic-messages": PROTOCOL_ANTHROPIC,
    "gemini": PROTOCOL_GEMINI,
    "google": PROTOCOL_GEMINI,
}

PROVIDERS_FILE_ENV_VAR = "RAGLIB_PROVIDERS_FILE"
_PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ENV_REF_RE = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}|\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Optional[str]) -> Optional[str]:
    """Replace ``{env:NAME}`` / ``${NAME}`` with environment values (missing variables become "")."""

    if value is None:
        return None
    return _ENV_REF_RE.sub(lambda match: os.getenv(match.group(1) or match.group(2), ""), str(value))


@dataclass(frozen=True)
class ModelSpec:
    """One model a provider serves.

    ``reasoning=True``: the model thinks before answering (DeepSeek-R1, QwQ, o-series, ...), so
    raglib does not send ``temperature`` (many reasoning endpoints reject it).
    ``image=True``: the model accepts images, so it can OCR scanned PDF pages (``vision_llm``).
    ``json_mode=False``: never request JSON mode (``response_format``) for this model. Some small
    models accept JSON mode but then loop until max_tokens; without it raglib parses the JSON
    from the plain reply instead.
    """

    id: str
    name: Optional[str] = None
    reasoning: bool = False
    image: bool = False
    json_mode: bool = True

    @property
    def display_name(self) -> str:
        """Human-readable name (falls back to the id)."""

        return self.name or self.id


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
    protocol: str = PROTOCOL_OPENAI
    api_key: Optional[str] = field(default=None, repr=False)  # literal key or "{env:NAME}" (custom providers)
    models: Tuple[ModelSpec, ...] = ()
    # (document, query) values of the "input_type" field required by asymmetric embedding models (NVIDIA NIM)
    embedding_input_types: Optional[Tuple[str, str]] = None
    embedding_params: Dict[str, Any] = field(default_factory=dict)  # extra JSON fields for /embeddings
    extra_body: Dict[str, Any] = field(default_factory=dict)  # extra JSON fields for chat requests
    stream_usage: bool = False  # supports stream_options.include_usage (token counts while streaming)

    def resolve_api_key(self, explicit: Optional[str] = None) -> Optional[str]:
        """Explicit key first, then the configured key, then the provider's environment variables."""

        if explicit and explicit.strip():
            return explicit.strip()
        configured = expand_env(self.api_key)
        if configured and configured.strip():
            return configured.strip()
        for key in self.env_keys:
            value = os.getenv(key)
            if value and value.strip():
                return value.strip()
        return None

    def resolve_headers(self) -> Dict[str, str]:
        """Extra request headers with ``{env:NAME}`` references expanded (empty values are dropped)."""

        headers = {key: expand_env(value) or "" for key, value in self.extra_headers.items()}
        return {key: value for key, value in headers.items() if value}

    def model(self, model_id: Optional[str]) -> Optional[ModelSpec]:
        """The declared model with this id, if any."""

        for spec in self.models:
            if spec.id == model_id:
                return spec
        return None

    @property
    def custom(self) -> bool:
        """True for providers registered at runtime."""

        return self.kind == KIND_CUSTOM

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
        ProviderSpec("anthropic", KIND_NATIVE, "claude-haiku-5-5", ("ANTHROPIC_API_KEY",), base_url_env=("ANTHROPIC_BASE_URL",),
                     key_prefixes=("sk-ant-",), supports_vision=True, description="Anthropic Claude",
                     protocol=PROTOCOL_ANTHROPIC),
        ProviderSpec("google", KIND_NATIVE, "gemini-2.5-flash", ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
                     key_prefixes=("AIza",), default_embedding_model="models/gemini-embedding-001",
                     supports_vision=True, description="Google Gemini", protocol=PROTOCOL_GEMINI),
        ProviderSpec("groq", KIND_NATIVE, "llama-3.3-70b-versatile", ("GROQ_API_KEY",),
                     base_url="https://api.groq.com/openai/v1", base_url_env=("GROQ_BASE_URL",),
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
        ProviderSpec("nvidia", KIND_OPENAI_COMPATIBLE, "nvidia/nemotron-3.5-lightning-30b-a3b", ("NVIDIA_API_KEY",),
                     base_url="https://integrate.api.nvidia.com/v1", base_url_env=("NVIDIA_BASE_URL",),
                     key_prefixes=("nvapi-",), default_embedding_model="nvidia/nemotron-3-embed-1b",
                     supports_vision=True, embedding_input_types=("passage", "query"),
                     embedding_params={"truncate": "END"}, stream_usage=True,
                     description="NVIDIA NIM (build.nvidia.com; self-hosted NIM via NVIDIA_BASE_URL)"),
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
    "nim": "nvidia",
    "nvidia_nim": "nvidia",
    "nvidia-nim": "nvidia",
}

ENV_DETECTION_ORDER: Tuple[str, ...] = (
    "openai", "anthropic", "openrouter", "google", "groq", "mistral", "deepseek",
    "together", "fireworks", "xai", "cerebras", "perplexity", "nvidia", "azure",
)


BUILTIN_PROVIDERS = frozenset(PROVIDERS)
_REGISTRY_LOCK = threading.Lock()
_ENV_FILE_LOADED: Optional[str] = None
_PROVIDER_OPTIONS = {
    "name", "api", "base_url", "api_key", "api_key_env", "headers", "models", "default_model",
    "embedding_model", "key_prefixes", "extra_body", "embedding_input_types", "embedding_params", "stream_usage",
}


def _ensure_env_providers() -> None:
    """Load the file named by ``RAGLIB_PROVIDERS_FILE`` once (again if the variable changes)."""

    global _ENV_FILE_LOADED
    path = os.getenv(PROVIDERS_FILE_ENV_VAR)
    if not path or path == _ENV_FILE_LOADED:
        return
    _ENV_FILE_LOADED = path  # set first: a broken file is reported once, not on every lookup
    load_providers(path, override=True)


def _models_from_config(models: Any) -> Tuple[ModelSpec, ...]:
    """Accept ``[{"id": ...}]``, ``{"model-id": {...}}``, ``["model-id"]`` or ModelSpec objects."""

    if models is None:
        return ()
    items: List[Tuple[str, Mapping[str, Any]]] = []
    if isinstance(models, Mapping):
        items = [(str(model_id), dict(options or {})) for model_id, options in models.items()]
    else:
        for entry in models:
            if isinstance(entry, ModelSpec):
                items.append((entry.id, {"name": entry.name, "reasoning": entry.reasoning, "image": entry.image,
                                         "json_mode": entry.json_mode}))
            elif isinstance(entry, str):
                items.append((entry, {}))
            elif isinstance(entry, Mapping) and entry.get("id"):
                items.append((str(entry["id"]), entry))
            else:
                raise ValueError(f"Each model needs an 'id': {entry!r}")
    specs = []
    for model_id, options in items:
        if not model_id.strip():
            raise ValueError("Model id must not be empty")
        unknown = set(options) - {"id", "name", "reasoning", "image", "json_mode"}
        if unknown:
            raise ValueError(
                f"Unknown option(s) {sorted(unknown)} for model '{model_id}' (use name, reasoning, image, json_mode)"
            )
        specs.append(
            ModelSpec(
                id=model_id.strip(),
                name=options.get("name") or None,
                reasoning=bool(options.get("reasoning", False)),
                image=bool(options.get("image", False)),
                json_mode=bool(options.get("json_mode", True)),
            )
        )
    ids = [spec.id for spec in specs]
    duplicates = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate model ids: {duplicates}")
    return tuple(specs)


def register_provider(
    provider_id: str,
    *,
    base_url: str,
    api: str = PROTOCOL_OPENAI,
    name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: Union[None, str, Sequence[str]] = None,
    headers: Optional[Mapping[str, str]] = None,
    models: Any = None,
    default_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    key_prefixes: Sequence[str] = (),
    extra_body: Optional[Mapping[str, Any]] = None,
    embedding_input_types: Optional[Sequence[str]] = None,
    embedding_params: Optional[Mapping[str, Any]] = None,
    stream_usage: bool = False,
    override: bool = False,
) -> ProviderSpec:
    """Register a third-party or self-hosted provider so ``RAG(llm="<id>/<model>")`` works.

    Args:
        provider_id: lowercase letters, digits, ``-`` or ``_`` (e.g. ``"myprovider"``).
        base_url: API root, e.g. ``"https://api.myprovider.com/v1"``.
        api: wire protocol - ``"openai"`` (Chat Completions, used by most providers),
            ``"anthropic"`` (Messages API) or ``"gemini"`` (generateContent).
        name: display name shown by ``raglib providers``.
        api_key: optional key, or ``"{env:NAME}"`` to read it from the environment. Leave it
            out when auth is done with ``headers``.
        api_key_env: environment variable(s) to read the key from.
        headers: extra request headers; values may use ``{env:NAME}``.
        models: ``[{"id": "model-id", "name": "Display Name", "reasoning": False, "image": False}]``;
            add ``"json_mode": False`` for a model that loops or returns empty replies in JSON mode.
            or ``{"model-id": {...}}``. The first is the default unless ``default_model`` is set.
            Other model ids still work; the list is for defaults, flags and ``raglib providers``.
        embedding_model: model id for embeddings (OpenAI and Gemini protocols), used when
            ``embedding_llm`` is not given.
        extra_body: extra JSON fields merged into every chat request, e.g.
            ``{"chat_template_kwargs": {"enable_thinking": False}}`` for vLLM / NVIDIA NIM.
        embedding_input_types: ``("passage", "query")`` for asymmetric embedding models that need an
            ``input_type`` field (documents get the first value, search queries the second).
        embedding_params: extra JSON fields for embedding requests, e.g. ``{"truncate": "END"}``.
        stream_usage: the server supports ``stream_options.include_usage`` (real token counts when streaming).
        override: allow replacing a custom provider with the same id (built-ins are never replaced).
    """

    provider_id = (provider_id or "").strip()
    if not _PROVIDER_ID_RE.match(provider_id):
        raise ValueError(f"Invalid provider id '{provider_id}': use lowercase letters, numbers, hyphens or underscores")
    if provider_id in BUILTIN_PROVIDERS or provider_id in PROVIDER_ALIASES:
        raise ValueError(f"'{provider_id}' is a built-in provider; choose another id")
    protocol = _PROTOCOL_ALIASES.get((api or "").strip().lower())
    if protocol is None:
        raise ValueError(f"Unknown provider api '{api}'. Use one of: openai, anthropic, gemini")
    url = (base_url or "").strip().rstrip("/")
    if not re.match(r"^https?://[^/\s]+", url):
        raise ValueError(f"base_url must start with http:// or https:// (got '{base_url}')")

    model_specs = _models_from_config(models)
    if default_model and model_specs and not any(spec.id == default_model for spec in model_specs):
        raise ValueError(f"default_model '{default_model}' is not in the models list")
    env_keys = (api_key_env,) if isinstance(api_key_env, str) else tuple(api_key_env or ())
    input_types: Optional[Tuple[str, str]] = None
    if embedding_input_types is not None:
        values = tuple(str(value) for value in embedding_input_types)
        if len(values) != 2 or not all(values):
            raise ValueError('embedding_input_types must be two values: (document, query), e.g. ["passage", "query"]')
        input_types = (values[0], values[1])
    for label, value in (("extra_body", extra_body), ("embedding_params", embedding_params)):
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object / dict")
    spec = ProviderSpec(
        name=provider_id,
        kind=KIND_CUSTOM,
        default_chat_model=default_model or (model_specs[0].id if model_specs else None),
        env_keys=env_keys,
        base_url=url,
        requires_key=bool(api_key or env_keys),
        key_prefixes=tuple(key_prefixes),
        default_embedding_model=embedding_model,
        supports_vision=any(model.image for model in model_specs),
        extra_headers={str(key): str(value) for key, value in (headers or {}).items()},
        description=name or provider_id,
        protocol=protocol,
        api_key=api_key,
        models=model_specs,
        embedding_input_types=input_types,
        embedding_params=dict(embedding_params or {}),
        extra_body=dict(extra_body or {}),
        stream_usage=bool(stream_usage),
    )
    with _REGISTRY_LOCK:
        if provider_id in PROVIDERS and not override:
            raise ValueError(f"Provider '{provider_id}' is already registered (pass override=True to replace it)")
        PROVIDERS[provider_id] = spec
    logger.info("Registered provider '%s' (%s API at %s, %d models)", provider_id, protocol, url, len(model_specs))
    return spec


def unregister_provider(provider_id: str) -> bool:
    """Remove a custom provider; returns False when it was not registered."""

    if provider_id in BUILTIN_PROVIDERS:
        raise ValueError(f"'{provider_id}' is a built-in provider and cannot be removed")
    with _REGISTRY_LOCK:
        return PROVIDERS.pop(provider_id, None) is not None


def load_providers(source: Union[str, Path, Mapping[str, Any]], override: bool = True) -> List[ProviderSpec]:
    """Register every provider in a JSON file (or an already-parsed dict).

    Each provider needs ``base_url``; ``api`` defaults to ``"openai"``::

        {"providers": {"myprovider": {
            "name": "My AI Provider", "api": "openai", "base_url": "https://api.myprovider.com/v1",
            "api_key": "{env:MYPROVIDER_API_KEY}", "headers": {"X-Team": "search"},
            "models": {"model-id": {"name": "Display Name", "reasoning": false, "image": false}}}}}

    Set ``RAGLIB_PROVIDERS_FILE=/path/providers.json`` to load a file automatically.
    """

    if isinstance(source, Mapping):
        config: Any = source
        origin = "<dict>"
    else:
        path = Path(source).expanduser()
        origin = str(path)
        if not path.is_file():
            raise FileNotFoundError(f"Providers file not found: {path}")
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"Providers file {path} is not valid JSON: {exc}") from exc
    providers = config.get("providers", config) if isinstance(config, Mapping) else None
    if not isinstance(providers, Mapping):
        raise ValueError(f'{origin}: expected {{"providers": {{"<id>": {{...}}}}}}')
    registered = []
    for provider_id, options in providers.items():
        if not isinstance(options, Mapping):
            raise ValueError(f"{origin}: provider '{provider_id}' must be an object")
        unknown = set(options) - _PROVIDER_OPTIONS
        if unknown:
            raise ValueError(f"{origin}: unknown option(s) {sorted(unknown)} for provider '{provider_id}'")
        if "base_url" not in options:
            raise ValueError(f"{origin}: provider '{provider_id}' needs a base_url")
        if options.get("api_key") and not _ENV_REF_RE.search(str(options["api_key"])):
            logger.warning("%s: provider '%s' stores a literal api_key; prefer \"{env:NAME}\"", origin, provider_id)
        registered.append(
            register_provider(
                str(provider_id),
                base_url=str(options["base_url"]),
                api=str(options.get("api", PROTOCOL_OPENAI)),
                name=options.get("name"),
                api_key=options.get("api_key"),
                api_key_env=options.get("api_key_env"),
                headers=options.get("headers"),
                models=options.get("models"),
                default_model=options.get("default_model"),
                embedding_model=options.get("embedding_model"),
                key_prefixes=tuple(options.get("key_prefixes") or ()),
                extra_body=options.get("extra_body"),
                embedding_input_types=options.get("embedding_input_types"),
                embedding_params=options.get("embedding_params"),
                stream_usage=bool(options.get("stream_usage", False)),
                override=override,
            )
        )
    return registered


def list_models(provider: str) -> List[ModelSpec]:
    """Models declared for a provider (custom providers; built-ins accept any model id)."""

    return list(get_provider(provider).models)


def normalize_provider(name: Optional[str]) -> Optional[str]:
    """Return the canonical provider name, or None if unknown."""

    _ensure_env_providers()
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
        for spec in list(PROVIDERS.values())  # copy: another thread may be registering a provider
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

    _ensure_env_providers()
    rows = []
    for spec in list(PROVIDERS.values()):
        if spec.custom:
            key = spec.env_keys[0] if spec.env_keys else ("(configured)" if spec.api_key else "(no key)")
            label = f"{spec.description} [custom {spec.protocol} API: {spec.base_url}]"
        else:
            key = spec.env_keys[0] if spec.env_keys and spec.requires_key else "(no key)"
            label = spec.description
        rows.append(f"  {spec.name:<18} {key:<26} {label}")
        for model in spec.models:
            flags = ", ".join(flag for flag, on in (("reasoning", model.reasoning), ("image", model.image)) if on)
            title = f" ({model.name})" if model.name else ""
            rows.append(f"      - {model.id}{title}" + (f"  [{flags}]" if flags else ""))
    return "\n".join(rows)
