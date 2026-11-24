"""Adapter for LangChain chat models that users pass in themselves.

raglib's own provider clients are native (``raglib.llm.clients``) and do not use LangChain.
This adapter only wraps an existing LangChain ``BaseChatModel`` object, e.g.
``RAG(chat_llm=ChatOpenAI(...))``, so it gets raglib's retries, rate limiting and usage tracking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

from raglib.llm.base_client import BaseLLMClient
from raglib.llm.constants import DEFAULT_LLM_MAX_RETRIES, DEFAULT_LLM_TIMEOUT_SECONDS
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
        self._json_mode: Optional[bool] = None  # None = not yet known for this model

    @property
    def provider(self) -> str:
        """Return the provider name."""

        return self._provider

    def complete(self, prompt: str, system: str = "") -> str:
        """Execute a single prompt completion using LangChain message objects."""

        payload = self._build_payload(prompt=prompt, system=system)
        response = self._invoke_with_retry(payload)
        text = self._coerce_content(getattr(response, "content", response))
        self._record_usage(response, f"{system}\n{prompt}", text)
        return text

    def complete_json(self, prompt: str, system: str = "", default: Any = None) -> Any:
        """Complete in the provider's JSON mode when supported, else parse JSON from free text.

        JSON mode (``response_format={"type": "json_object"}``) is used for OpenAI and
        OpenAI-compatible providers. If the model rejects it, raglib remembers that and
        falls back to plain completion for this client.
        """

        from raglib.core.parsing import parse_json

        if self._json_mode_supported():
            payload = self._build_payload(prompt=prompt, system=system)
            try:
                bound = self._model.bind(response_format={"type": "json_object"})
                response = retry_call(
                    bound.invoke,
                    payload,
                    max_retries=self._max_retries,
                    base_delay=self._retry_base_delay,
                    limiter=self.rate_limiter,
                    label=f"{self._provider}:{self.model_name}:json",
                )
            except Exception as exc:  # noqa: BLE001 - unsupported parameter: remember and fall back
                if is_retryable(exc):
                    logger.warning("JSON-mode call failed (%s); retrying without JSON mode", exc)
                else:
                    logger.info("Model %s rejected JSON mode (%s); using plain completions", self.model_name, exc)
                    self._json_mode = False
            else:
                text = self._coerce_content(getattr(response, "content", response))
                self._record_usage(response, f"{system}\n{prompt}", text)
                parsed = parse_json(text, default=None)
                if parsed is not None:
                    return parsed
        return super().complete_json(prompt, system=system, default=default)

    def _json_mode_supported(self) -> bool:
        """Whether to try the provider's JSON mode for this client."""

        if self._json_mode is not None:
            return self._json_mode
        from raglib.llm.providers import KIND_OPENAI_COMPATIBLE, PROVIDERS

        spec = PROVIDERS.get(self._provider)
        supported = self._provider in {"openai", "azure"} or (spec is not None and spec.kind == KIND_OPENAI_COMPATIBLE)
        return supported and hasattr(self._model, "bind")

    def _record_usage(self, response: Any, prompt_text: str, completion: str) -> None:
        """Report token usage: the provider's counts when present, else an estimate."""

        from raglib.core.runtime import record_usage
        from raglib.core.text import count_tokens

        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict) and (usage.get("input_tokens") or usage.get("output_tokens")):
            record_usage(int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0), estimated=False)
        else:
            record_usage(count_tokens(prompt_text), count_tokens(completion), estimated=True)

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
        text = self._coerce_content(getattr(response, "content", response))
        self._record_usage(response, "\n".join(str(message.get("content", "")) for message in messages), text)
        return text

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Stream tokens when the underlying model supports ``.stream``.

        Opening the stream and receiving the first chunk are retried like ``complete``
        (rate limits, timeouts, 5xx); once tokens are flowing, errors propagate, because
        a retry would repeat text the caller has already received.
        """

        payload = self._build_payload(prompt=prompt, system=system)
        stream_fn = getattr(self._model, "stream", None)
        if stream_fn is None:
            yield self.complete(prompt=prompt, system=system)
            return

        def open_stream():
            iterator = iter(stream_fn(payload))
            return iterator, next(iterator, None)

        iterator, first = retry_call(
            open_stream,
            max_retries=self._max_retries,
            base_delay=self._retry_base_delay,
            limiter=self.rate_limiter,
            label=f"{self._provider}:{self.model_name}:stream",
        )
        parts: List[str] = []
        usage: Optional[Dict[str, int]] = None
        chunk = first
        try:
            while chunk is not None:
                chunk_usage = getattr(chunk, "usage_metadata", None)
                if isinstance(chunk_usage, dict) and (chunk_usage.get("input_tokens") or chunk_usage.get("output_tokens")):
                    usage = {
                        "input_tokens": (usage or {}).get("input_tokens", 0) + int(chunk_usage.get("input_tokens") or 0),
                        "output_tokens": (usage or {}).get("output_tokens", 0) + int(chunk_usage.get("output_tokens") or 0),
                    }
                text = self._coerce_content(getattr(chunk, "content", chunk))
                if text:
                    parts.append(text)
                    yield text
                chunk = next(iterator, None)
        finally:  # also when the caller stops reading early: what was sent and received was paid for

            class _Usage:
                usage_metadata = usage

            self._record_usage(_Usage, f"{system}\n{prompt}", "".join(parts))

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
        text = self._coerce_content(getattr(response, "content", response))
        self._record_usage(response, f"{system}\n{prompt}", text)
        return text

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
        max_tokens: Optional[int] = None,
    ) -> BaseLLMClient:
        """Deprecated: use ``raglib.llm.build_chat_client`` (native clients, no LangChain needed)."""

        import warnings

        from raglib.llm.clients import build_chat_client

        warnings.warn(
            "LangChainAdapter.build() is deprecated; use raglib.llm.build_chat_client(), which needs no LangChain.",
            DeprecationWarning,
            stacklevel=2,
        )
        return build_chat_client(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
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
