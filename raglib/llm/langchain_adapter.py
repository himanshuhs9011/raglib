"""LangChain chat model adapter implementing raglib BaseLLMClient."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

from raglib.llm.base_client import BaseLLMClient
from raglib.llm.constants import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    OLLAMA_DEFAULT_BASE_URL,
    PROVIDER_DEFAULT_CHAT_MODEL,
)

from raglib.llm.resilience import RateLimiter, is_retryable, retry_call

logger = logging.getLogger(__name__)

_is_retryable = is_retryable  # backwards-compatible alias


class LangChainAdapter(BaseLLMClient):
    """Bridge a LangChain chat model into raglib's BaseLLMClient interface."""

    def __init__(
        self,
        langchain_model: Any,
        provider_name: str = "unknown",
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
        retry_base_delay: float = 1.0,
        model_name: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """Store the LangChain chat model plus retry and rate-limit policy."""

        self._model = langchain_model
        self._provider = provider_name
        self._max_retries = max(0, max_retries)
        self._retry_base_delay = retry_base_delay
        self.model_name = model_name or getattr(langchain_model, "model_name", None) or getattr(langchain_model, "model", None)
        self.rate_limiter = rate_limiter

    @property
    def provider(self) -> str:
        """Return the provider name."""

        return self._provider

    def complete(self, prompt: str, system: str = "") -> str:
        """Execute a single prompt completion using LangChain message objects."""

        payload = self._build_payload(prompt=prompt, system=system)
        response = self._invoke_with_retry(payload)
        return self._coerce_content(getattr(response, "content", response))

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Execute a multi-turn chat completion from role/content dictionaries."""

        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            role_map = {
                "user": HumanMessage,
                "human": HumanMessage,
                "assistant": AIMessage,
                "ai": AIMessage,
                "system": SystemMessage,
            }
            payload: Any = [
                role_map.get(message.get("role", "user").lower(), HumanMessage)(
                    content=message.get("content", "")
                )
                for message in messages
            ]
        except ImportError:
            payload = "\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages
            )
        response = self._invoke_with_retry(payload)
        return self._coerce_content(getattr(response, "content", response))

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Stream tokens when the underlying model supports ``.stream``."""

        payload = self._build_payload(prompt=prompt, system=system)
        stream_fn = getattr(self._model, "stream", None)
        if stream_fn is None:
            yield self.complete(prompt=prompt, system=system)
            return
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        for chunk in stream_fn(payload):
            text = self._coerce_content(getattr(chunk, "content", chunk))
            if text:
                yield text

    async def acomplete(self, prompt: str, system: str = "") -> str:
        """Use native ``ainvoke`` when available, else a worker thread."""

        ainvoke = getattr(self._model, "ainvoke", None)
        if ainvoke is None:
            return await super().acomplete(prompt, system)
        payload = self._build_payload(prompt=prompt, system=system)
        if self.rate_limiter is not None:
            import asyncio

            await asyncio.to_thread(self.rate_limiter.acquire)
        response = await ainvoke(payload)
        return self._coerce_content(getattr(response, "content", response))

    def _build_payload(self, prompt: str, system: str) -> Any:
        """Build LangChain messages, or a plain string when langchain-core is missing."""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages: List[Any] = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            return messages
        except ImportError:
            return f"{system}\n\n{prompt}".strip() if system else prompt

    def _invoke_with_retry(self, payload: Any) -> Any:
        """Invoke the model with rate limiting and retries (honours Retry-After; quota errors fail fast)."""

        return retry_call(
            self._model.invoke,
            payload,
            max_retries=self._max_retries,
            base_delay=self._retry_base_delay,
            limiter=self.rate_limiter,
            label=f"{self._provider}:{self.model_name}",
        )

    @classmethod
    def build(
        cls,
        provider: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = DEFAULT_LLM_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
    ) -> "LangChainAdapter":
        """Build a provider-specific LangChain chat model and wrap it with this adapter."""

        from raglib.llm.providers import KIND_AZURE, KIND_OPENAI_COMPATIBLE, get_provider

        spec = get_provider(provider)
        normalized_provider = spec.name
        resolved_model = model_name or spec.default_chat_model or PROVIDER_DEFAULT_CHAT_MODEL.get(normalized_provider)
        if not resolved_model:
            raise ValueError(
                f"Provider '{normalized_provider}' has no default model; pass chat_model=... "
                "(for OpenRouter-style ids use e.g. llm='openrouter/meta-llama/llama-3.3-70b-instruct')"
            )

        if spec.kind == KIND_OPENAI_COMPATIBLE:
            lc_model = cls._build_openai_compatible(
                spec=spec,
                api_key=api_key,
                model=resolved_model,
                base_url=base_url,
                temperature=temperature,
                timeout=timeout,
            )
            logger.info("Built LangChainAdapter provider='%s' model='%s'", normalized_provider, resolved_model)
            return cls(langchain_model=lc_model, provider_name=normalized_provider, max_retries=max_retries, model_name=resolved_model)

        if spec.kind == KIND_AZURE:
            try:
                from langchain_openai import AzureChatOpenAI
            except ImportError as exc:
                raise ImportError("pip install raglib-py[openai]") from exc
            import os

            endpoint = spec.resolve_base_url(base_url)
            if not endpoint:
                raise ValueError("Azure OpenAI requires chat_base_url (your endpoint) or AZURE_OPENAI_ENDPOINT")
            lc_model = AzureChatOpenAI(
                azure_deployment=resolved_model,
                azure_endpoint=endpoint,
                api_key=spec.resolve_api_key(api_key),
                api_version=os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21",
                temperature=temperature,
                timeout=timeout,
                max_retries=0,
            )
            logger.info("Built LangChainAdapter provider='azure' deployment='%s'", resolved_model)
            return cls(langchain_model=lc_model, provider_name="azure", max_retries=max_retries, model_name=resolved_model)

        if normalized_provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise ImportError("pip install raglib-py[openai]") from exc
            lc_model = ChatOpenAI(
                model=resolved_model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                timeout=timeout,
                max_retries=0,
            )
        elif normalized_provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:
                raise ImportError("pip install raglib-py[anthropic]") from exc
            lc_model = ChatAnthropic(
                model=resolved_model,
                api_key=api_key,
                temperature=temperature,
                timeout=timeout,
                max_retries=0,
            )
        elif normalized_provider == "groq":
            try:
                from langchain_groq import ChatGroq
            except ImportError as exc:
                raise ImportError("pip install raglib-py[groq]") from exc
            lc_model = ChatGroq(
                model=resolved_model,
                api_key=api_key,
                temperature=temperature,
                timeout=timeout,
                max_retries=0,
            )
        elif normalized_provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ImportError("pip install raglib-py[google]") from exc
            lc_model = ChatGoogleGenerativeAI(
                model=resolved_model,
                google_api_key=api_key,
                temperature=temperature,
                timeout=timeout,
                max_retries=0,
            )
        elif normalized_provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise ImportError("pip install raglib-py[ollama]") from exc
            lc_model = ChatOllama(
                model=resolved_model,
                base_url=base_url or OLLAMA_DEFAULT_BASE_URL,
                temperature=temperature,
            )
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. Supported: openai, anthropic, groq, google, ollama"
            )

        logger.info("Built LangChainAdapter provider='%s' model='%s'", normalized_provider, resolved_model)
        return cls(langchain_model=lc_model, provider_name=normalized_provider, max_retries=max_retries, model_name=resolved_model)

    @staticmethod
    def _build_openai_compatible(
        spec: Any,
        api_key: Optional[str],
        model: str,
        base_url: Optional[str],
        temperature: float,
        timeout: float,
    ) -> Any:
        """ChatOpenAI pointed at an OpenAI-compatible endpoint (OpenRouter, DeepSeek, vLLM, ...)."""

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(f"Provider '{spec.name}' requires: pip install raglib-py[openai]") from exc
        import os

        resolved_url = spec.resolve_base_url(base_url)
        if not resolved_url:
            raise ValueError(f"Provider '{spec.name}' requires chat_base_url (e.g. http://localhost:8000/v1)")
        key = spec.resolve_api_key(api_key)
        if spec.requires_key and not key:
            raise ValueError(f"Provider '{spec.name}' requires an API key (chat_api_key or {spec.env_keys[0]})")
        headers = dict(spec.extra_headers)
        if spec.name == "openrouter":
            if os.getenv("OPENROUTER_APP_NAME"):
                headers["X-Title"] = os.environ["OPENROUTER_APP_NAME"]
            if os.getenv("OPENROUTER_SITE_URL"):
                headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
        return ChatOpenAI(
            model=model,
            api_key=key or "not-needed",  # local servers ignore the key but the client requires one
            base_url=resolved_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=0,
            default_headers=headers or None,
        )

    @classmethod
    def _coerce_content(cls, content: Any) -> str:
        """Convert varying LangChain response content payloads into plain text."""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text is not None:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            text = content.get("text") or content.get("content")
            if text is not None:
                return str(text)
        return str(content)
