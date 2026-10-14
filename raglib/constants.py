"""Shared constants for provider support and default facade settings."""

from __future__ import annotations

import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

SUPPORTED_CHAT_PROVIDERS: Set[str] = {"openai", "anthropic", "groq", "google", "ollama"}
SUPPORTED_EMBEDDING_PROVIDERS: Set[str] = {
    "openai",
    "google",
    "ollama",
    "huggingface",
    "free",
    "local",
    "mock",
}
SUPPORTED_VISION_PROVIDERS: Set[str] = {"openai", "anthropic", "google", "mock"}
SUPPORTED_VECTOR_DBS: Set[str] = {"chroma", "chromadb", "memory", "in_memory", "inmemory", "mock"}
SUPPORTED_WEB_PROVIDERS: Set[str] = {
    "local",
    "duckduckgo",
    "tavily",
    "serpapi",
    "brave",
    "bing",
    "google_cse",
    "exa",
    "searxng",
}
WEB_SEARCH_RAG_TYPES: Set[str] = {"web", "hybrid", "routing", "corrective"}
WEB_PROVIDERS_REQUIRING_API_KEY: Set[str] = {
    "tavily",
    "serpapi",
    "brave",
    "bing",
    "google_cse",
    "exa",
    "searxng",
}

LLM_KEY_PREFIX_MAP: Dict[str, str] = {
    "sk-ant-": "anthropic",
    "sk-": "openai",
    "gsk_": "groq",
    "AIza": "google",
}

DEFAULT_RAG_TYPE = "corrective"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_CHUNKING_STRATEGY = "fixed"
SUPPORTED_CHUNKING_STRATEGIES: Set[str] = {"fixed", "recursive", "llm"}
