"""Native chat clients: OpenAI-compatible, Anthropic and Gemini over plain HTTPS.

All 17 providers raglib supports speak one of three protocols:

* **OpenAI Chat Completions** - OpenAI, Azure OpenAI, Groq, Ollama (``/v1``), OpenRouter,
  DeepSeek, Mistral, Together, Fireworks, xAI, Perplexity, Cerebras, LM Studio, vLLM, ...
* **Anthropic Messages** - Claude.
* **Gemini generateContent** - Google.

Each client is a small ``BaseLLMClient`` with retries, rate limiting, real token usage,
streaming (SSE) and JSON mode. They need no SDK; ``build_chat_client`` picks the right one.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from raglib.llm.base_client import BaseLLMClient
from raglib.llm.constants import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    OLLAMA_DEFAULT_BASE_URL,
    PROVIDER_DEFAULT_CHAT_MODEL,
)
from raglib.llm.http import post_json, post_sse
from raglib.llm.providers import PROTOCOL_ANTHROPIC, PROTOCOL_GEMINI
from raglib.llm.resilience import RateLimiter, is_retryable, retry_call

logger = logging.getLogger(__name__)

Message = Dict[str, str]
Usage = Optional[Tuple[int, int]]

OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
AZURE_DEFAULT_API_VERSION = "2024-10-21"


def azure_api_version() -> str:
    """API version for Azure OpenAI (env override, else a GA version)."""

    return os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or AZURE_DEFAULT_API_VERSION


class LLMOutputTruncatedError(RuntimeError):
    """The model hit its output limit (``max_tokens``) before writing any answer text.

    Reasoning models can spend the whole budget thinking. Raised instead of returning an empty
    answer as if it succeeded; raise ``llm_max_tokens`` (or turn the model's thinking off).
    """


_LENGTH_REASONS = {"length", "max_tokens", "MAX_TOKENS"}


class HTTPChatClient(BaseLLMClient):
    """Shared request / retry / usage / streaming logic; subclasses map one wire protocol."""

    supports_json_mode = False

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_name: str = "unknown",
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = None,
        timeout: float = DEFAULT_LLM_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
        retry_base_delay: float = 1.0,
        extra_headers: Optional[Dict[str, str]] = None,
        rate_limiter: Optional[RateLimiter] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        """Configure the endpoint, sampling options and retry policy.

        ``extra_body`` is merged into every request payload (provider-specific options).
        """

        if not model:
            raise ValueError("model is required")
        self.model_name = model
        self.api_key = api_key
        self.base_url = (base_url or self.default_base_url()).rstrip("/")
        self._provider = provider_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._max_retries = max(0, max_retries)
        self._retry_base_delay = retry_base_delay
        self.extra_headers = dict(extra_headers or {})
        self.rate_limiter = rate_limiter
        self.extra_body = dict(extra_body or {})
        self._json_mode: Optional[bool] = None if self.supports_json_mode else False

    # ------------------------------------------------------------------ protocol hooks

    def default_base_url(self) -> str:
        """Base URL used when none is given."""

        raise NotImplementedError

    def build_request(
        self, messages: List[Message], system: str, stream: bool, json_mode: bool
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Return (url, headers, payload) for one call."""

        raise NotImplementedError

    def _request(self, messages: List[Message], system: str, stream: bool, json_mode: bool):
        """``build_request`` plus ``extra_body`` (merged one level deep into dict values)."""

        url, headers, payload = self.build_request(messages, system, stream, json_mode)
        for key, value in self.extra_body.items():
            if isinstance(value, dict) and isinstance(payload.get(key), dict):
                payload[key] = {**payload[key], **value}
            else:
                payload[key] = value
        return url, headers, payload

    def parse_response(self, body: Any) -> Tuple[str, Usage]:
        """Return (text, (prompt_tokens, completion_tokens) or None) from a JSON response."""

        raise NotImplementedError

    def parse_stream_event(self, event: Dict[str, Any]) -> Tuple[str, Usage]:
        """Return (text delta, usage or None) from one SSE event."""

        raise NotImplementedError

    def finish_reason(self, body: Any) -> Optional[str]:
        """Why the model stopped (from a response body or one stream event), when it says."""

        return None

    def image_content(self, prompt: str, image_b64: str, mime_type: str) -> Any:
        """The user-message content that carries an image (protocol specific)."""

        raise NotImplementedError

    # ------------------------------------------------------------------ BaseLLMClient

    @property
    def provider(self) -> str:
        """Return the provider name."""

        return self._provider

    def complete(self, prompt: str, system: str = "") -> str:
        """Single-prompt completion."""

        return self._call([{"role": "user", "content": prompt}], system, prompt_text=f"{system}\n{prompt}")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Multi-turn completion from role/content dictionaries (system messages are hoisted)."""

        system, turns = split_system(messages)
        return self._call(turns, system, prompt_text="\n".join(str(m.get("content", "")) for m in messages))

    def complete_json(self, prompt: str, system: str = "", default: Any = None) -> Any:
        """Use the provider's JSON mode when supported, else parse JSON from free text.

        A model that rejects JSON mode is remembered, and later calls use plain completions.
        """

        from raglib.core.parsing import parse_json

        if self._json_mode is not False:
            try:
                text = self._call(
                    [{"role": "user", "content": prompt}], system, prompt_text=f"{system}\n{prompt}", json_mode=True
                )
            except Exception as exc:  # noqa: BLE001 - unsupported parameter: remember and fall back
                if is_retryable(exc):
                    logger.warning("JSON-mode call failed (%s); retrying without JSON mode", exc)
                else:
                    logger.info("Model %s rejected JSON mode (%s); using plain completions", self.model_name, exc)
                    self._json_mode = False
            else:
                self._json_mode = True
                parsed = parse_json(text, default=None)
                if parsed is not None:
                    return parsed
        return super().complete_json(prompt, system=system, default=default)

    def complete_with_image(self, prompt: str, image_b64: str, mime_type: str = "image/png") -> str:
        """Send one image plus an instruction (used for OCR of scanned pages)."""

        content = self.image_content(prompt, image_b64, mime_type)
        return self._call([{"role": "user", "content": content}], "", prompt_text=prompt)  # type: ignore[dict-item]

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Stream text deltas over SSE.

        Opening the stream and reading the first event are retried like ``complete``; once
        text is flowing, errors propagate, because a retry would repeat text already yielded.
        """

        url, headers, payload = self._request([{"role": "user", "content": prompt}], system, True, False)

        def open_stream():
            # The first event is parsed here too, so an error event at the start of the stream
            # (e.g. Anthropic "overloaded_error") is retried like an HTTP error.
            events = post_sse(url, payload, headers, timeout=self.timeout)
            first = next(events, None)
            return events, first, (None if first is None else self.parse_stream_event(first))

        events, event, parsed = retry_call(
            open_stream,
            max_retries=self._max_retries,
            base_delay=self._retry_base_delay,
            limiter=self.rate_limiter,
            label=f"{self._provider}:{self.model_name}:stream",
        )
        parts: List[str] = []
        prompt_tokens = completion_tokens = 0
        reported = False
        thinking = LeadingThinkFilter()
        finish: Optional[str] = None
        while parsed is not None:
            raw, usage = parsed
            if usage is not None:
                reported = True
                prompt_tokens = max(prompt_tokens, usage[0])
                completion_tokens = max(completion_tokens, usage[1])
            text = thinking.feed(raw) if raw else ""
            if text:
                parts.append(text)
                yield text
            finish = self.finish_reason(event.get("data") if isinstance(event, dict) else None) or finish
            event = next(events, None)
            parsed = None if event is None else self.parse_stream_event(event)
        tail = thinking.flush()
        if tail:
            parts.append(tail)
            yield tail
        self._record_usage((prompt_tokens, completion_tokens) if reported else None, f"{system}\n{prompt}", "".join(parts))
        self._check_truncation("".join(parts), finish)

    # ------------------------------------------------------------------ internals

    def _call(self, messages: List[Message], system: str, prompt_text: str, json_mode: bool = False) -> str:
        """POST one request with rate limiting and retries, then record usage."""

        url, headers, payload = self._request(messages, system, False, json_mode)
        body = retry_call(
            post_json,
            url,
            payload,
            headers,
            timeout=self.timeout,
            max_retries=self._max_retries,
            base_delay=self._retry_base_delay,
            limiter=self.rate_limiter,
            label=f"{self._provider}:{self.model_name}" + (":json" if json_mode else ""),
        )
        text, usage = self.parse_response(body)
        self._record_usage(usage, prompt_text, text)
        answer = strip_leading_think(text)
        self._check_truncation(answer, self.finish_reason(body))
        return answer

    def _check_truncation(self, text: str, finish: Optional[str]) -> None:
        """Raise when the output limit left no answer text; flag (``metadata["degraded"]``) a cut answer."""

        if finish not in _LENGTH_REASONS:
            return
        limit = getattr(self, "max_tokens", None)
        where = f"{self._provider}:{self.model_name}"
        stripped = text.lstrip()
        if stripped.startswith(_THINK_OPEN) and _THINK_CLOSE not in stripped:
            text = ""  # cut off inside an unclosed <think> block: no answer was written
        if not text.strip():
            raise LLMOutputTruncatedError(
                f"{where} reached its output limit (max_tokens={limit}) before writing any answer text; reasoning "
                "models can spend the whole budget thinking. Raise llm_max_tokens (RAG(llm_max_tokens=...))."
            )
        from raglib.core.runtime import report_degraded

        report_degraded(
            f"llm_output_truncated:{where}",
            LLMOutputTruncatedError(f"the answer was cut off at the output limit (max_tokens={limit})"),
            logger,
        )

    @staticmethod
    def _record_usage(usage: Usage, prompt_text: str, completion: str) -> None:
        """Report the provider's token counts, or an estimate when it sent none."""

        from raglib.core.runtime import record_usage
        from raglib.core.text import count_tokens

        if usage is not None and (usage[0] or usage[1]):
            record_usage(int(usage[0]), int(usage[1]), estimated=False)
        else:
            record_usage(count_tokens(prompt_text), count_tokens(completion), estimated=True)

    def __repr__(self) -> str:
        """Readable form that never shows the API key."""

        return f"{type(self).__name__}(provider={self._provider!r}, model={self.model_name!r}, base_url={self.base_url!r})"


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_leading_think(text: str) -> str:
    """Drop a leading ``<think>...</think>`` block (reasoning models served over OpenAI-style APIs).

    Only a block at the very start is removed, so answers that mention the tag are untouched.
    An unclosed block is kept as is rather than losing the whole answer.
    """

    stripped = text.lstrip()
    if not stripped.startswith(_THINK_OPEN):
        return text
    end = stripped.find(_THINK_CLOSE)
    return text if end < 0 else stripped[end + len(_THINK_CLOSE):].lstrip()


class LeadingThinkFilter:
    """Streaming version of ``strip_leading_think``: hides a leading think block, chunk by chunk."""

    def __init__(self) -> None:
        self._state = "start"  # start -> (think ->) pass
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """Return the visible part of ``chunk`` (may be empty while deciding or thinking)."""

        if self._state == "pass":
            return chunk
        if self._state == "after":  # think block closed; skip the whitespace before the answer
            visible = chunk.lstrip()
            if visible:
                self._state = "pass"
            return visible
        self._buffer += chunk
        if self._state == "start":
            head = self._buffer.lstrip()
            if len(head) < len(_THINK_OPEN) and _THINK_OPEN.startswith(head):
                return ""  # could still become "<think>"
            if not head.startswith(_THINK_OPEN):
                return self._release()
            self._state = "think"
        end = self._buffer.find(_THINK_CLOSE)
        if end < 0:
            return ""
        visible = self._buffer[end + len(_THINK_CLOSE):].lstrip()
        self._buffer = ""
        self._state = "pass" if visible else "after"
        return visible

    def _release(self) -> str:
        text, self._buffer, self._state = self._buffer, "", "pass"
        return text

    def flush(self) -> str:
        """Text still held when the stream ends (an unclosed think block is returned unchanged)."""

        if self._state in {"start", "think"}:
            return self._release()
        return ""


def split_system(messages: List[Dict[str, str]]) -> Tuple[str, List[Message]]:
    """Hoist system messages out of a transcript and normalize roles to user/assistant."""

    system_parts: List[str] = []
    turns: List[Message] = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = message.get("content", "")
        if role == "system":
            system_parts.append(str(content))
        else:
            turns.append({"role": "assistant" if role in {"assistant", "ai", "model"} else "user", "content": content})
    return "\n\n".join(part for part in system_parts if part), turns


def merge_consecutive(turns: List[Message]) -> List[Message]:
    """Join consecutive same-role text turns and make the transcript start with a user turn."""

    merged: List[Message] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"] and isinstance(turn["content"], str) and isinstance(merged[-1]["content"], str):
            merged[-1] = {"role": turn["role"], "content": f"{merged[-1]['content']}\n\n{turn['content']}"}
        else:
            merged.append(dict(turn))
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": "(conversation start)"})
    return merged or [{"role": "user", "content": ""}]


def _text_of(value: Any) -> str:
    """Flatten string-or-list-of-parts message content into text."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(part.get("text", "")) for part in value if isinstance(part, dict) and part.get("type", "text") == "text")
    return "" if value is None else str(value)


class OpenAICompatibleClient(HTTPChatClient):
    """OpenAI Chat Completions protocol (OpenAI, Azure, Groq, Ollama, OpenRouter, vLLM, ...).

    ``azure=True`` targets an Azure OpenAI deployment: ``base_url`` is the resource endpoint,
    ``model`` the deployment name, and the key goes in the ``api-key`` header.
    """

    supports_json_mode = True
    _STREAM_USAGE_PROVIDERS = {"openai", "azure", "openrouter", "groq", "deepseek", "together", "fireworks", "vllm"}

    def __init__(
        self,
        *args: Any,
        azure: bool = False,
        api_version: Optional[str] = None,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ):
        """See ``HTTPChatClient``; ``azure`` / ``api_version`` select Azure OpenAI routing.

        ``stream_usage`` sends ``stream_options.include_usage`` (default: on for providers known to accept it).
        """

        self.azure = azure
        self.api_version = api_version or (azure_api_version() if azure else None)
        super().__init__(*args, **kwargs)
        self.stream_usage = self._provider in self._STREAM_USAGE_PROVIDERS if stream_usage is None else stream_usage

    def default_base_url(self) -> str:
        """OpenAI's public endpoint."""

        if self.azure:
            raise ValueError("Azure OpenAI requires base_url (your endpoint) or AZURE_OPENAI_ENDPOINT")
        return OPENAI_BASE_URL

    def build_request(self, messages, system, stream, json_mode):
        """Chat Completions request."""

        body_messages: List[Dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        body_messages.extend(messages)
        payload: Dict[str, Any] = {"model": self.model_name, "messages": body_messages}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens:
            # OpenAI's reasoning models only accept max_completion_tokens; compatible servers expect max_tokens.
            payload["max_completion_tokens" if self._provider in {"openai", "azure"} else "max_tokens"] = self.max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if stream:
            payload["stream"] = True
            if self.stream_usage:
                payload["stream_options"] = {"include_usage": True}

        headers = dict(self.extra_headers)
        if self.azure:
            url = f"{self.base_url}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}"
            if self.api_key:
                headers["api-key"] = self.api_key
        else:
            url = f"{self.base_url}/chat/completions"
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return url, headers, payload

    def parse_response(self, body):
        """``choices[0].message.content`` and ``usage``."""

        choices = (body.get("choices") or []) if isinstance(body, dict) else []
        if not choices:
            raise ValueError(f"{self._provider} returned no choices: {str(body)[:300]}")
        message = choices[0].get("message") or {}
        text = _text_of(message.get("content"))
        return text, self._usage(body)

    def finish_reason(self, body):
        """``choices[0].finish_reason`` ("length" when ``max_tokens`` stopped it)."""

        choices = (body.get("choices") or []) if isinstance(body, dict) else []
        reason = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
        return str(reason) if reason else None

    def parse_stream_event(self, event):
        """``choices[0].delta.content``; the final chunk may carry ``usage``."""

        data = event.get("data")
        if not isinstance(data, dict):
            return "", None
        if data.get("error"):
            from raglib.llm.http import ProviderHTTPError

            error = data["error"]
            status = error.get("code") if isinstance(error, dict) and isinstance(error.get("code"), int) else 500
            raise ProviderHTTPError(status, data)
        text = ""
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            text += _text_of(delta.get("content"))
        return text, self._usage(data)

    @staticmethod
    def _usage(body: Dict[str, Any]) -> Usage:
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict) and (usage.get("prompt_tokens") or usage.get("completion_tokens")):
            return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        return None

    def image_content(self, prompt, image_b64, mime_type):
        """Text part plus a data-URL image part."""

        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        ]


class AnthropicClient(HTTPChatClient):
    """Anthropic Messages API (Claude)."""

    def default_base_url(self) -> str:
        """Anthropic's public endpoint."""

        return ANTHROPIC_BASE_URL

    def build_request(self, messages, system, stream, json_mode):
        """Messages request; ``max_tokens`` is required by the API."""

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
            "messages": merge_consecutive(messages),
        }
        if system:
            payload["system"] = system
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if stream:
            payload["stream"] = True
        headers = {**self.extra_headers, "anthropic-version": ANTHROPIC_VERSION}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        # ANTHROPIC_BASE_URL is a host by SDK convention ("https://api.anthropic.com"); accept both forms.
        root = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        return f"{root}/messages", headers, payload

    def parse_response(self, body):
        """Join the text blocks; ``usage.input_tokens`` / ``output_tokens``."""

        if not isinstance(body, dict):
            raise ValueError(f"anthropic returned an unexpected body: {str(body)[:300]}")
        blocks = body.get("content") or []
        text = "".join(str(block.get("text", "")) for block in blocks if isinstance(block, dict) and block.get("type") == "text")
        usage = body.get("usage") or {}
        counts = (int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))
        return text, counts if any(counts) else None

    def finish_reason(self, body):
        """``stop_reason`` of a message, or of a stream's ``message_delta`` ("max_tokens")."""

        if not isinstance(body, dict):
            return None
        reason = body.get("stop_reason") or (body.get("delta") or {}).get("stop_reason")
        return str(reason) if reason else None

    def parse_stream_event(self, event):
        """``content_block_delta`` text; usage arrives in ``message_start`` and ``message_delta``."""

        data = event.get("data")
        if not isinstance(data, dict):
            return "", None
        kind = data.get("type") or event.get("event")
        if kind == "error":
            from raglib.llm.http import ProviderHTTPError

            error = data.get("error") or {}
            status = 529 if error.get("type") == "overloaded_error" else 500
            raise ProviderHTTPError(status, data)
        if kind == "content_block_delta":
            delta = data.get("delta") or {}
            return (str(delta.get("text", "")) if delta.get("type") == "text_delta" else ""), None
        if kind == "message_start":
            usage = (data.get("message") or {}).get("usage") or {}
            return "", (int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))
        if kind == "message_delta":
            usage = data.get("usage") or {}
            return "", (int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))
        return "", None

    def image_content(self, prompt, image_b64, mime_type):
        """Base64 image block followed by the instruction."""

        return [
            {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}},
            {"type": "text", "text": prompt},
        ]


class GeminiClient(HTTPChatClient):
    """Google Gemini ``generateContent`` API."""

    supports_json_mode = True

    def default_base_url(self) -> str:
        """Gemini's public endpoint."""

        return GEMINI_BASE_URL

    def _model_path(self) -> str:
        return self.model_name if self.model_name.startswith(("models/", "tunedModels/")) else f"models/{self.model_name}"

    def build_request(self, messages, system, stream, json_mode):
        """generateContent (or streamGenerateContent with ``alt=sse``) request."""

        contents = []
        for turn in merge_consecutive(messages):
            content = turn["content"]
            parts = content if isinstance(content, list) else [{"text": content}]
            contents.append({"role": "model" if turn["role"] == "assistant" else "user", "parts": parts})
        payload: Dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        config: Dict[str, Any] = {}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.max_tokens:
            config["maxOutputTokens"] = self.max_tokens
        if json_mode:
            config["responseMimeType"] = "application/json"
        if config:
            payload["generationConfig"] = config
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        headers = dict(self.extra_headers)
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        return f"{self.base_url}/{self._model_path()}:{method}", headers, payload

    def parse_response(self, body):
        """Join the candidate's text parts (thought summaries are skipped)."""

        if not isinstance(body, dict):
            return "", None
        candidates = body.get("candidates") or []
        if not candidates:
            feedback = body.get("promptFeedback") or {}
            if feedback.get("blockReason"):
                raise ValueError(f"Gemini blocked the prompt: {feedback.get('blockReason')}")
            return "", self._usage(body)
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict) and not part.get("thought"))
        return text, self._usage(body)

    def finish_reason(self, body):
        """``candidates[0].finishReason`` ("MAX_TOKENS")."""

        candidates = (body.get("candidates") or []) if isinstance(body, dict) else []
        reason = candidates[0].get("finishReason") if candidates and isinstance(candidates[0], dict) else None
        return str(reason) if reason else None

    def parse_stream_event(self, event):
        """Each SSE event is a partial ``GenerateContentResponse``."""

        data = event.get("data")
        if isinstance(data, dict) and data.get("error"):
            from raglib.llm.http import ProviderHTTPError

            error = data["error"]
            raise ProviderHTTPError(int(error.get("code") or 500) if isinstance(error, dict) else 500, data)
        if not isinstance(data, dict) or not data.get("candidates"):
            return "", self._usage(data) if isinstance(data, dict) else None
        return self.parse_response(data)

    @staticmethod
    def _usage(body: Dict[str, Any]) -> Usage:
        usage = body.get("usageMetadata") or {}
        counts = (int(usage.get("promptTokenCount") or 0), int(usage.get("candidatesTokenCount") or 0))
        return counts if any(counts) else None

    def image_content(self, prompt, image_b64, mime_type):
        """Inline image part plus the instruction."""

        return [{"inline_data": {"mime_type": mime_type, "data": image_b64}}, {"text": prompt}]


def ollama_openai_url(base_url: Optional[str]) -> str:
    """Ollama's OpenAI-compatible endpoint (``<host>/v1``)."""

    root = (base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
    return root if root.endswith("/v1") else f"{root}/v1"


def build_chat_client(
    provider: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    timeout: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_LLM_MAX_RETRIES,
    max_tokens: Optional[int] = None,
) -> HTTPChatClient:
    """Build the native chat client for any supported provider name.

    ``build_chat_client("openrouter", model_name="anthropic/claude-sonnet-4")`` - the key is read
    from the provider's environment variable when ``api_key`` is not given.
    """

    from raglib.llm.providers import KIND_AZURE, KIND_OPENAI_COMPATIBLE, get_provider

    spec = get_provider(provider)
    name = spec.name
    model = model_name or spec.default_chat_model or PROVIDER_DEFAULT_CHAT_MODEL.get(name)
    if not model:
        raise ValueError(
            f"Provider '{name}' has no default model; pass chat_model=... "
            "(for OpenRouter-style ids use e.g. llm='openrouter/meta-llama/llama-3.3-70b-instruct')"
        )
    key = spec.resolve_api_key(api_key)
    if spec.requires_key and not key:
        raise ValueError(f"Provider '{name}' requires an API key ({missing_key_hint(spec)})")
    model_spec = spec.model(model)
    if model_spec is not None and model_spec.reasoning:
        temperature = None  # reasoning endpoints commonly reject a temperature
    common: Dict[str, Any] = dict(
        model=model,
        api_key=key,
        provider_name=name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        extra_headers=spec.resolve_headers(),
        extra_body=dict(spec.extra_body),
    )

    if spec.protocol == PROTOCOL_ANTHROPIC:
        client: HTTPChatClient = AnthropicClient(base_url=spec.resolve_base_url(base_url), **common)
    elif spec.protocol == PROTOCOL_GEMINI:
        client = GeminiClient(base_url=spec.resolve_base_url(base_url), **common)
    elif spec.kind == KIND_AZURE:
        endpoint = spec.resolve_base_url(base_url)
        if not endpoint:
            raise ValueError("Azure OpenAI requires chat_base_url (your endpoint) or AZURE_OPENAI_ENDPOINT")
        client = OpenAICompatibleClient(base_url=endpoint, azure=True, **common)
    elif name == "ollama":
        client = OpenAICompatibleClient(base_url=ollama_openai_url(spec.resolve_base_url(base_url)), **common)
    else:
        resolved_url = spec.resolve_base_url(base_url)
        if spec.kind == KIND_OPENAI_COMPATIBLE and not resolved_url:
            raise ValueError(f"Provider '{name}' requires chat_base_url (e.g. http://localhost:8000/v1)")
        if name == "openrouter":
            headers = common["extra_headers"]
            if os.getenv("OPENROUTER_APP_NAME"):
                headers["X-Title"] = os.environ["OPENROUTER_APP_NAME"]
            if os.getenv("OPENROUTER_SITE_URL"):
                headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
        client = OpenAICompatibleClient(base_url=resolved_url, stream_usage=spec.stream_usage or None, **common)
    if model_spec is not None and not model_spec.json_mode:
        client._json_mode = False  # parse JSON from plain replies; never send response_format
    logger.info("Built %s provider='%s' model='%s'", type(client).__name__, name, model)
    return client


def missing_key_hint(spec: Any) -> str:
    """Where a missing key can be supplied, for error messages."""

    if spec.env_keys:
        return f"chat_api_key or {spec.env_keys[0]}"
    if getattr(spec, "api_key", None):
        return f"chat_api_key, or set the variable referenced by its api_key '{spec.api_key}'"
    return "chat_api_key"
