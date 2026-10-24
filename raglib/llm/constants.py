"""Centralized default model constants for all LLM-related providers.

Defaults are chosen to be cheap, currently-served models. Override any of them with
``chat_model=``, ``embedding_model=`` or ``vision_model=`` on ``RAG``.
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

PROVIDER_DEFAULT_CHAT_MODEL: Dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-5-5",
    "groq": "llama-3.3-70b-versatile",
    "google": "gemini-2.5-flash",
    "ollama": "llama3.1",
}

PROVIDER_DEFAULT_EMBEDDING_MODEL: Dict[str, str] = {
    "openai": "text-embedding-3-small",
    "google": "models/gemini-embedding-001",
    "ollama": "nomic-embed-text",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "mock": "mock-embedding-v1",
}

PROVIDER_DEFAULT_VISION_MODEL: Dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-5-5",
    "google": "gemini-2.5-flash",
}

OLLAMA_DEFAULT_BASE_URL: str = "http://localhost:11434"

DEFAULT_LLM_TIMEOUT_SECONDS: float = 60.0
DEFAULT_LLM_MAX_RETRIES: int = 2
