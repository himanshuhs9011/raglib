"""Production-grade high-level facade for raglib."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from raglib.components.context_reducer import ContextReducer
from raglib.components.decision import DecisionEngine
from raglib.components.deduplicator import Deduplicator
from raglib.components.evaluator import Evaluator
from raglib.components.generator import Generator
from raglib.components.hybrid_retriever import HybridRetriever
from raglib.components.memory import MemoryModule
from raglib.components.planner import Planner
from raglib.components.query_expander import QueryExpander
from raglib.components.query_rewriter import QueryRewriter
from raglib.components.refiner import Refiner
from raglib.components.reflection import ReflectionModule
from raglib.components.reranker import Reranker
from raglib.components.retriever import Retriever
from raglib.components.router_retriever import RouterRetriever
from raglib.components.web_retriever import WebRetriever
from raglib.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_RAG_TYPE,
    DEFAULT_TOP_K,
    SUPPORTED_CHAT_PROVIDERS,
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_VECTOR_DBS,
    SUPPORTED_VISION_PROVIDERS,
    SUPPORTED_WEB_PROVIDERS,
    WEB_PROVIDERS_REQUIRING_API_KEY,
    WEB_SEARCH_RAG_TYPES,
)
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.embedding_factory import EmbeddingFactory
from raglib.llm.base_client import BaseLLMClient
from raglib.llm.provider_detector import LLMProviderDetector
from raglib.loaders.chunk_splitter import ChunkSplitter
from raglib.loaders.document_loader import DocumentLoader, SourceInput
from raglib.providers import (
    BingProvider,
    BraveProvider,
    DuckDuckGoProvider,
    ExaProvider,
    GoogleCSEProvider,
    SearxNGProvider,
    SerpAPIProvider,
    TavilyProvider,
)
from raglib.providers.base_provider import BaseSearchProvider
from raglib.schemas import Document, GenerationResult
from raglib.vectorstores.base_store import BaseVectorStore
from raglib.vision.base_vision import BaseVisionClient
from raglib.vision.vision_factory import VisionFactory

logger = logging.getLogger(__name__)

RAG_TYPE_ALIASES: Dict[str, str] = {
    "naive": "naive",
    "advanced": "advanced",
    "corrective": "corrective",
    "self": "self",
    "self_rag": "self",
    "agentic": "agentic",
    "hybrid": "hybrid",
    "multi_query": "multi_query",
    "multi-query": "multi_query",
    "multiquery": "multi_query",
    "multi_hop": "multi_hop",
    "multi-hop": "multi_hop",
    "multihop": "multi_hop",
    "routing": "routing",
    "memory": "memory",
    "web": "web",
    "tool": "tool",
    "tool_augmented": "tool",
    "tool-augmented": "tool",
    "toolaugmented": "tool",
}

WEB_PROVIDER_ALIASES: Dict[str, str] = {
    "local_web": "local",
    "ddg": "duckduckgo",
    "duckduckgo_search": "duckduckgo",
    "google": "google_cse",
}

CHAT_PROVIDER_ENV_KEYS: Dict[str, Sequence[str]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}

EMBEDDING_PROVIDER_ENV_KEYS: Dict[str, Sequence[str]] = {
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}

WEB_PROVIDER_ENV_KEYS: Dict[str, Sequence[str]] = {
    "tavily": ("TAVILY_API_KEY",),
    "serpapi": ("SERPAPI_API_KEY",),
    "brave": ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"),
    "bing": ("BING_SEARCH_API_KEY", "BING_SEARCH_V7_SUBSCRIPTION_KEY"),
    "google_cse": ("GOOGLE_API_KEY",),
    "exa": ("EXA_API_KEY",),
    "searxng": ("SEARXNG_API_KEY",),
}


class _LocalSearchProvider(BaseSearchProvider):
    """Offline web-style provider over in-memory chunked documents."""

    PROVIDER_NAME = "local_web"

    def __init__(self, documents: Optional[List[Document]] = None):
        """Initialize provider with an optional local document list."""

        self._documents = documents or []

    @property
    def name(self) -> str:
        """Return provider identifier."""

        return self.PROVIDER_NAME

    def update_documents(self, documents: List[Document]) -> None:
        """Replace local search corpus with new chunked documents."""

        self._documents = documents

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Search local documents using lexical overlap scoring."""

        if not self._documents:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored: List[Document] = []
        for document in self._documents:
            doc_terms = self._tokenize(document.content)
            if not doc_terms:
                continue

            overlap = len(query_terms & doc_terms)
            if overlap == 0:
                continue

            score = overlap / max(len(query_terms), 1)
            scored.append(
                replace(
                    document,
                    score=score,
                    source="web",
                    metadata={**document.metadata, "provider": self.name},
                )
            )

        scored.sort(key=lambda doc: doc.score, reverse=True)
        return scored[:num_results]

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize input text into lowercase terms."""

        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


class RAG:
    """Single entry point for raglib with chat, embedding, vision, and orchestration wiring."""

    def __init__(
        self,
        source: SourceInput = None,
        chat_llm: Optional[Union[str, BaseLLMClient, Any]] = None,
        embedding_llm: Optional[str] = None,
        vision_llm: Optional[str] = None,
        llm_key: Optional[str] = None,
        chat_api_key: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        vision_api_key: Optional[str] = None,
        rag_type: str = DEFAULT_RAG_TYPE,
        top_k: int = DEFAULT_TOP_K,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY,
        output_dir: Optional[str] = None,
        chat_model: Optional[str] = None,
        chat_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        vision_model: Optional[str] = None,
        vision_base_url: Optional[str] = None,
        web_search_provider: str = "duckduckgo",
        web_search_api_key: Optional[str] = None,
        web_search_base_url: Optional[str] = None,
        web_search_cse_id: Optional[str] = None,
        web_search_provider_kwargs: Optional[Dict[str, Any]] = None,
        validate_web_search_api_key: bool = False,
        vector_db: Optional[Union[str, BaseVectorStore]] = None,
        vector_db_kwargs: Optional[Dict[str, Any]] = None,
        **rag_type_kwargs: Any,
    ):
        """Initialize the full RAG stack with optional provider and model overrides."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self._rag_type = self._normalize_rag_type(rag_type)
        self._output_dir = output_dir

        explicit_chat_key = chat_api_key
        shared_key = llm_key
        effective_chat_key = explicit_chat_key if explicit_chat_key is not None else shared_key

        if isinstance(chat_llm, str):
            normalized_chat_llm = chat_llm.strip().lower()
            if normalized_chat_llm in SUPPORTED_CHAT_PROVIDERS:
                self._validate_chat_provider_api_key(
                    provider=normalized_chat_llm,
                    explicit_api_key=effective_chat_key,
                )

        chat_input: Optional[Union[str, BaseLLMClient]] = chat_llm if chat_llm is not None else effective_chat_key
        chat_provider_hint: Optional[str] = None
        if isinstance(chat_llm, str) and effective_chat_key:
            normalized_chat_llm = chat_llm.strip().lower()
            if normalized_chat_llm in SUPPORTED_CHAT_PROVIDERS:
                chat_provider_hint = normalized_chat_llm
                chat_input = effective_chat_key

        self._llm = LLMProviderDetector.detect(
            value=chat_input,
            provider_hint=chat_provider_hint,
            model_name=chat_model,
            base_url=chat_base_url,
        )
        inferred_chat_provider = LLMProviderDetector.infer_provider(
            value=chat_input,
            provider_hint=chat_provider_hint,
        )

        self._embedding = self._resolve_embedding(
            embedding_llm=embedding_llm,
            llm_key=shared_key,
            chat_api_key=explicit_chat_key,
            embedding_api_key=embedding_api_key,
            inferred_chat_provider=inferred_chat_provider,
            model_name=embedding_model,
            base_url=embedding_base_url,
        )
        self._vision = self._resolve_vision(
            vision_llm=vision_llm,
            llm_key=shared_key,
            chat_api_key=explicit_chat_key,
            vision_api_key=vision_api_key,
            inferred_chat_provider=inferred_chat_provider,
            model_name=vision_model,
            base_url=vision_base_url,
        )

        if isinstance(vector_db, str):
            normalized_vector_db = vector_db.strip().lower()
            if normalized_vector_db not in SUPPORTED_VECTOR_DBS:
                raise ValueError(
                    f"Unknown vector_db '{vector_db}'. Supported values: {sorted(SUPPORTED_VECTOR_DBS)}"
                )

        self._loader = DocumentLoader(vision_client=self._vision)
        self._chunking_strategy = chunking_strategy
        normalized_chunk_strategy = str(chunking_strategy).strip().lower()
        llm_for_chunking = (
            self._llm
            if normalized_chunk_strategy in {"llm", "semantic", "llm_semantic"}
            else None
        )
        self._splitter = ChunkSplitter(
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            strategy=chunking_strategy,
            llm_client=llm_for_chunking,
        )
        self._documents: List[Document] = []
        if source is not None:
            self._documents.extend(self._ingest(source, return_only=True))

        self._retriever = Retriever(
            documents=self._documents,
            embedding=self._embedding,
            top_k=top_k,
            vector_db=vector_db,
            vector_db_kwargs=vector_db_kwargs,
        )
        self.retriever = self._retriever

        normalized_web_provider = self._normalize_web_provider_name(web_search_provider)
        self._web_provider = self._resolve_web_provider(
            provider_name=normalized_web_provider,
            web_search_api_key=web_search_api_key,
            web_search_base_url=web_search_base_url,
            web_search_cse_id=web_search_cse_id,
            web_search_provider_kwargs=web_search_provider_kwargs,
        )
        if validate_web_search_api_key and normalized_web_provider != "local":
            self._validate_web_provider_connection(self._web_provider)

        if self._rag_type in WEB_SEARCH_RAG_TYPES and normalized_web_provider == "local":
            logger.warning(
                "RAG type '%s' can use web retrieval but web_search_provider='local' is offline. "
                "Set web_search_provider to an internet provider to query live web.",
                self._rag_type,
            )

        self._web_retriever = WebRetriever(
            provider=self._web_provider,
            top_k=top_k,
            fail_silently=True,
        )
        self._hybrid_retriever = HybridRetriever(
            vector_retriever=self._retriever,
            web_retriever=self._web_retriever,
        )
        self._router_retriever = RouterRetriever(
            vector_retriever=self._retriever,
            web_retriever=self._web_retriever,
            hybrid_retriever=self._hybrid_retriever,
            llm_client=self._llm,
        )

        self._reranker = Reranker(top_k=top_k)
        self._evaluator = Evaluator()
        self._refiner = Refiner(llm_client=self._llm)
        self._decision_engine = DecisionEngine(llm_client=self._llm)
        self._reflection_module = ReflectionModule(evaluator=self._evaluator)
        self._memory_module = MemoryModule()
        self._planner = Planner(llm_client=self._llm)
        self._query_rewriter = QueryRewriter(llm_client=self._llm)
        self._query_expander = QueryExpander(llm_client=self._llm)
        self._deduplicator = Deduplicator()
        self._context_reducer = ContextReducer()
        self._generator = Generator(llm_client=self._llm)

        self._rag = self._build_orchestrator(rag_type=self._rag_type, kwargs=rag_type_kwargs)
        logger.info(
            "RAG initialized (rag_type=%s docs=%d llm=%s embedding=%s vision=%s web_provider=%s)",
            self._rag_type,
            len(self._documents),
            type(self._llm).__name__,
            type(self._embedding).__name__,
            type(self._vision).__name__,
            self._web_provider.name,
        )

    def query(self, question: str) -> GenerationResult:
        """Ask a single question and return a GenerationResult."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        result = self._rag.run(question.strip())
        if self._output_dir:
            self._save_result(question.strip(), result)
        return result

    def chat(self) -> None:
        """Open an interactive terminal Q&A loop for this RAG instance."""

        from raglib.session.interactive import InteractiveSession

        session = InteractiveSession(rag=self)
        session.start()

    def add(self, source: SourceInput) -> None:
        """Add new sources to the existing corpus and update retrieval indexes."""

        new_chunks = self._ingest(source, return_only=True)
        if not new_chunks:
            logger.warning("No chunks produced from add(source=%s)", source)
            return

        self._documents.extend(new_chunks)
        self._retriever.add_documents(new_chunks)
        if isinstance(self._web_provider, _LocalSearchProvider):
            self._web_provider.update_documents(self._documents)
        logger.info("Added %d chunks. Total corpus size=%d", len(new_chunks), len(self._documents))

    def _resolve_embedding(
        self,
        embedding_llm: Optional[str],
        llm_key: Optional[str],
        chat_api_key: Optional[str],
        embedding_api_key: Optional[str],
        inferred_chat_provider: Optional[str],
        model_name: Optional[str],
        base_url: Optional[str],
    ) -> BaseEmbedding:
        """Resolve embedding provider and build embedding model instance."""

        if embedding_llm is None:
            if inferred_chat_provider in {"openai", "google", "ollama"}:
                provider = inferred_chat_provider
            else:
                provider = "mock"
        else:
            provider = embedding_llm.strip().lower()

        if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Unknown embedding provider '{provider}'. "
                f"Supported: {sorted(SUPPORTED_EMBEDDING_PROVIDERS)}"
            )

        api_key = None
        if provider in {"openai", "google"}:
            api_key = self._first_non_empty(
                embedding_api_key,
                llm_key,
                chat_api_key,
                self._resolve_env_value(EMBEDDING_PROVIDER_ENV_KEYS.get(provider, ())),
            )
            if not api_key:
                raise ValueError(
                    f"Embedding provider '{provider}' requires embedding_api_key (or llm_key/chat_api_key) "
                    f"or a configured environment variable: {list(EMBEDDING_PROVIDER_ENV_KEYS.get(provider, ()))}"
                )

        embedding = EmbeddingFactory.build(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )
        logger.info("Embedding model resolved provider=%s", provider)
        return embedding

    def _resolve_vision(
        self,
        vision_llm: Optional[str],
        llm_key: Optional[str],
        chat_api_key: Optional[str],
        vision_api_key: Optional[str],
        inferred_chat_provider: Optional[str],
        model_name: Optional[str],
        base_url: Optional[str],
    ) -> BaseVisionClient:
        """Resolve vision provider and build vision model instance."""

        effective_key = vision_api_key or llm_key or chat_api_key

        if vision_llm is None:
            if inferred_chat_provider in {"openai", "anthropic", "google"} and effective_key:
                provider = inferred_chat_provider
            else:
                provider = "mock"
        else:
            provider = vision_llm.strip().lower()

        if provider not in SUPPORTED_VISION_PROVIDERS:
            raise ValueError(
                f"Unknown vision provider '{provider}'. "
                f"Supported: {sorted(SUPPORTED_VISION_PROVIDERS)}"
            )

        api_key = effective_key if provider in {"openai", "anthropic", "google"} else None
        vision_client = VisionFactory.build(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )
        logger.info("Vision model resolved provider=%s", provider)
        return vision_client

    def _validate_chat_provider_api_key(
        self,
        provider: str,
        explicit_api_key: Optional[str],
    ) -> None:
        """Require API keys for cloud chat providers unless env vars are configured."""

        if provider not in {"openai", "anthropic", "groq", "google"}:
            return

        env_key = self._resolve_env_value(CHAT_PROVIDER_ENV_KEYS.get(provider, ()))
        if explicit_api_key or env_key:
            return

        raise ValueError(
            f"chat_llm='{provider}' requires chat_api_key (or llm_key) "
            f"or one of environment variables: {list(CHAT_PROVIDER_ENV_KEYS.get(provider, ()))}"
        )

    def _normalize_web_provider_name(self, provider_name: str) -> str:
        """Normalize web provider aliases to canonical provider names."""

        normalized = (provider_name or "duckduckgo").strip().lower()
        normalized = WEB_PROVIDER_ALIASES.get(normalized, normalized)
        if normalized not in SUPPORTED_WEB_PROVIDERS:
            raise ValueError(
                f"Unknown web_search_provider '{provider_name}'. "
                f"Supported: {sorted(SUPPORTED_WEB_PROVIDERS)}"
            )
        return normalized

    def _resolve_web_provider(
        self,
        provider_name: str,
        web_search_api_key: Optional[str],
        web_search_base_url: Optional[str],
        web_search_cse_id: Optional[str],
        web_search_provider_kwargs: Optional[Dict[str, Any]],
    ) -> BaseSearchProvider:
        """Build a web search provider from constructor inputs."""

        kwargs = dict(web_search_provider_kwargs or {})

        if provider_name == "local":
            if kwargs:
                raise ValueError(
                    "web_search_provider='local' does not accept web_search_provider_kwargs"
                )
            return _LocalSearchProvider(documents=self._documents)

        resolved_api_key = self._first_non_empty(
            web_search_api_key,
            self._resolve_env_value(WEB_PROVIDER_ENV_KEYS.get(provider_name, ())),
        )

        if provider_name in WEB_PROVIDERS_REQUIRING_API_KEY and not resolved_api_key:
            raise ValueError(
                f"web_search_provider='{provider_name}' requires web_search_api_key "
                f"or one of environment variables: {list(WEB_PROVIDER_ENV_KEYS.get(provider_name, ()))}"
            )

        if provider_name == "duckduckgo":
            provider = DuckDuckGoProvider(
                region=str(kwargs.pop("region", "wt-wt")),
                safesearch=str(kwargs.pop("safesearch", "moderate")),
                timelimit=kwargs.pop("timelimit", None),
                backend=str(kwargs.pop("backend", "auto")),
            )
        elif provider_name == "tavily":
            provider = TavilyProvider(
                api_key=resolved_api_key,
                topic=str(kwargs.pop("topic", "general")),
                search_depth=str(kwargs.pop("search_depth", "basic")),
                include_answer=bool(kwargs.pop("include_answer", False)),
                timeout=int(kwargs.pop("timeout", 10)),
            )
        elif provider_name == "serpapi":
            provider = SerpAPIProvider(
                api_key=resolved_api_key,
                engine=str(kwargs.pop("engine", "google")),
                location=str(kwargs.pop("location", "United States")),
                timeout=int(kwargs.pop("timeout", 10)),
            )
        elif provider_name == "brave":
            provider = BraveProvider(
                api_key=resolved_api_key,
                country=str(kwargs.pop("country", "us")),
                search_lang=str(kwargs.pop("search_lang", "en")),
                timeout=int(kwargs.pop("timeout", 10)),
            )
        elif provider_name == "bing":
            provider = BingProvider(
                api_key=resolved_api_key,
                market=str(kwargs.pop("market", "en-US")),
                timeout=int(kwargs.pop("timeout", 10)),
            )
        elif provider_name == "google_cse":
            cse_id = self._first_non_empty(
                web_search_cse_id,
                kwargs.pop("cse_id", None),
                self._resolve_env_value(("GOOGLE_CSE_ID", "GOOGLE_CSE_ENGINE_ID")),
            )
            if not cse_id:
                raise ValueError(
                    "web_search_provider='google_cse' requires web_search_cse_id "
                    "or environment variable GOOGLE_CSE_ID"
                )
            provider = GoogleCSEProvider(
                api_key=resolved_api_key,
                cse_id=cse_id,
                safe=str(kwargs.pop("safe", "off")),
                timeout=int(kwargs.pop("timeout", 10)),
            )
        elif provider_name == "exa":
            provider = ExaProvider(
                api_key=resolved_api_key,
                use_autoprompt=bool(kwargs.pop("use_autoprompt", True)),
                timeout=int(kwargs.pop("timeout", 10)),
            )
        elif provider_name == "searxng":
            base_url = self._first_non_empty(
                web_search_base_url,
                kwargs.pop("base_url", None),
                os.getenv("SEARXNG_BASE_URL"),
            )
            if not base_url:
                raise ValueError(
                    "web_search_provider='searxng' requires web_search_base_url "
                    "or environment variable SEARXNG_BASE_URL"
                )
            provider = SearxNGProvider(
                api_key=resolved_api_key,
                base_url=base_url,
                categories=str(kwargs.pop("categories", "general")),
                language=str(kwargs.pop("language", "en")),
                timeout=int(kwargs.pop("timeout", 10)),
            )
        else:
            raise ValueError(
                f"Unsupported web_search_provider '{provider_name}'. "
                f"Supported: {sorted(SUPPORTED_WEB_PROVIDERS)}"
            )

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise ValueError(
                f"Unknown web_search_provider_kwargs for provider '{provider_name}': {unknown}"
            )

        logger.info("Web provider resolved provider=%s", provider.name)
        return provider

    def _validate_web_provider_connection(self, provider: BaseSearchProvider) -> None:
        """Perform a lightweight web provider credential check."""

        try:
            provider.search(query="raglib api key validation", num_results=1)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"web_search_provider='{provider.name}' failed validation. "
                "Check internet access and web_search_api_key."
            ) from exc

    @staticmethod
    def _resolve_env_value(keys: Sequence[str]) -> Optional[str]:
        """Return the first non-empty environment variable value from keys."""

        for key in keys:
            value = os.getenv(key)
            if value and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _first_non_empty(*values: Optional[str]) -> Optional[str]:
        """Return the first non-empty string from the given values."""

        for value in values:
            if value and str(value).strip():
                return str(value).strip()
        return None

    def _ingest(self, source: SourceInput, return_only: bool = False) -> List[Document]:
        """Load and chunk source inputs into retrievable documents."""

        loaded_documents = self._loader.load(source)
        chunked_documents = self._splitter.split(loaded_documents)
        logger.info(
            "Ingestion complete: loaded_docs=%d chunked_docs=%d",
            len(loaded_documents),
            len(chunked_documents),
        )

        if return_only:
            return chunked_documents

        self._documents.extend(chunked_documents)
        return chunked_documents

    def _build_orchestrator(self, rag_type: str, kwargs: Dict[str, Any]) -> Any:
        """Import and instantiate the selected RAG orchestrator."""

        orchestrator_map = {
            "naive": "raglib.rag_types.naive_rag.NaiveRAG",
            "advanced": "raglib.rag_types.advanced_rag.AdvancedRAG",
            "corrective": "raglib.rag_types.corrective_rag.CorrectiveRAG",
            "self": "raglib.rag_types.self_rag.SelfRAG",
            "agentic": "raglib.rag_types.agentic_rag.AgenticRAG",
            "hybrid": "raglib.rag_types.hybrid_rag.HybridRAG",
            "multi_query": "raglib.rag_types.multi_query_rag.MultiQueryRAG",
            "multi_hop": "raglib.rag_types.multi_hop_rag.MultiHopRAG",
            "routing": "raglib.rag_types.routing_rag.RoutingRAG",
            "memory": "raglib.rag_types.memory_rag.MemoryRAG",
            "web": "raglib.rag_types.web_rag.WebRAG",
            "tool": "raglib.rag_types.tool_augmented_rag.ToolAugmentedRAG",
        }

        if rag_type not in orchestrator_map:
            raise ValueError(f"Unknown rag_type '{rag_type}'. Options: {list(orchestrator_map)}")

        module_path, class_name = orchestrator_map[rag_type].rsplit(".", 1)
        module = importlib.import_module(module_path)
        rag_class = getattr(module, class_name)

        shared_components: Dict[str, Any] = {
            "retriever": self._retriever,
            "web_retriever": self._web_retriever,
            "hybrid_retriever": self._hybrid_retriever,
            "router_retriever": self._router_retriever,
            "reranker": self._reranker,
            "evaluator": self._evaluator,
            "refiner": self._refiner,
            "decision_engine": self._decision_engine,
            "reflection_module": self._reflection_module,
            "memory_module": self._memory_module,
            "planner": self._planner,
            "query_rewriter": self._query_rewriter,
            "query_expander": self._query_expander,
            "deduplicator": self._deduplicator,
            "context_reducer": self._context_reducer,
            "generator": self._generator,
            "llm_client": self._llm,
        }
        shared_components.update(kwargs)
        return rag_class(**shared_components)

    def _normalize_rag_type(self, rag_type: str) -> str:
        """Normalize rag_type aliases to canonical orchestrator names."""

        normalized = (rag_type or DEFAULT_RAG_TYPE).strip().lower()
        if normalized in RAG_TYPE_ALIASES:
            return RAG_TYPE_ALIASES[normalized]

        raise ValueError(
            f"Unknown rag_type '{rag_type}'. "
            f"Supported: {sorted(set(RAG_TYPE_ALIASES.values()))}"
        )

    def _save_result(self, question: str, result: GenerationResult) -> None:
        """Persist query output as JSON in output_dir."""

        if not self._output_dir:
            return

        output_path = Path(self._output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rag_type": self._rag_type,
            "question": question,
            "answer": result.answer,
            "reasoning_trace": result.reasoning_trace,
            "sources": [
                {
                    "id": document.id,
                    "score": document.score,
                    "source": document.source,
                    "metadata": document.metadata,
                }
                for document in result.sources
            ],
        }

        filename = f"result_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.json"
        target = output_path / filename
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved query result to %s", target)
