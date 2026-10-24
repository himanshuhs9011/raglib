"""Shared constants for provider support and default facade settings."""

from __future__ import annotations

import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

from raglib.llm.providers import PROVIDERS as _PROVIDERS  # noqa: E402

SUPPORTED_CHAT_PROVIDERS: Set[str] = set(_PROVIDERS)
SUPPORTED_EMBEDDING_PROVIDERS: Set[str] = {
    "openai",
    "google",
    "ollama",
    "huggingface",
    "mistral",
    "openai_compatible",
    "azure",
    "lmstudio",
    "vllm",
    "free",
    "local",
    "mock",
}
SUPPORTED_VISION_PROVIDERS: Set[str] = {"openai", "anthropic", "google", "mock"}
SUPPORTED_VECTOR_DBS: Set[str] = {"chroma", "chromadb", "memory", "in_memory", "inmemory", "mock"}
SUPPORTED_RETRIEVAL_MODES: Set[str] = {"dense", "sparse", "hybrid"}
SUPPORTED_RERANKERS: Set[str] = {"none", "cross_encoder", "cohere", "llm"}
SUPPORTED_WEB_PROVIDERS: Set[str] = {
    "local",  # offline lexical search over the loaded corpus (tests/demos)
    "duckduckgo",
    "tavily",
    "serpapi",
    "brave",
    "bing",
    "google_cse",
    "exa",
    "searxng",
}
WEB_REQUIRED_RAG_TYPES: Set[str] = {"web", "hybrid_web"}
WEB_OPTIONAL_RAG_TYPES: Set[str] = {"corrective", "routing", "agentic"}
LLM_KEY_PREFIX_MAP: Dict[str, str] = dict(
    sorted(
        ((prefix, spec.name) for spec in _PROVIDERS.values() for prefix in spec.key_prefixes),
        key=lambda item: len(item[0]),
        reverse=True,  # longest prefix first: "sk-or-" (OpenRouter) before "sk-" (OpenAI)
    )
)

DEFAULT_RAG_TYPE = "advanced"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_CHUNKING_STRATEGY = "fixed"
SUPPORTED_CHUNKING_STRATEGIES: Set[str] = {"fixed", "recursive", "llm"}
DEFAULT_MAX_CONTEXT_TOKENS = 3000
OFFLINE_ENV_VAR = "RAGLIB_OFFLINE"
