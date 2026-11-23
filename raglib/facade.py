"""High-level facade: one ``RAG`` object that wires models, ingestion, retrieval and strategies."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import threading
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from raglib.access import ACCESS_TAG_KEYS, AccessControl, UserContext, merge_filters
from raglib.cache import AnswerCache, cache_bypass
from raglib.components.decision import DecisionEngine
from raglib.components.deduplicator import Deduplicator
from raglib.components.evaluator import Evaluator
from raglib.components.generator import Generator
from raglib.components.grader import RelevanceGrader
from raglib.components.hybrid_retriever import HybridRetriever
from raglib.components.memory import ConversationStore
from raglib.components.planner import Planner
from raglib.components.query_expander import QueryExpander
from raglib.components.query_rewriter import QueryCondenser, QueryRewriter
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
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RAG_TYPE,
    DEFAULT_TOP_K,
    OFFLINE_ENV_VAR,
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_VECTOR_DBS,
    SUPPORTED_VISION_PROVIDERS,
    WEB_REQUIRED_RAG_TYPES,
)
from raglib.core.runtime import UsageMeter, usage_scope
from raglib.core.text import token_set
from raglib.core.tracing import OnEvent, Tracer, run_tags
from raglib.costs import PriceTable
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.embedding.embedding_factory import EmbeddingFactory
from raglib.llm.base_client import (
    BaseLLMClient,
    CachedLLMClient,
    FallbackLLMClient,
    unwrap_llm,
)
from raglib.llm.constants import PROVIDER_DEFAULT_VISION_MODEL
from raglib.llm.mock_client import MockLLMClient
from raglib.llm.provider_detector import LLMProviderDetector
from raglib.llm.providers import (
    PROVIDERS,
    detect_provider_from_env,
    detect_provider_from_key,
    get_provider,
    normalize_provider,
    parse_model_string,
    provider_table,
)
from raglib.llm.resilience import shared_limiter
from raglib.loaders.chunk_splitter import ChunkSplitter
from raglib.loaders.document_loader import DocumentLoader, SourceInput
from raglib.loaders.sections import PARENT_MODES, build_sections
from raglib.presets import apply_presets
from raglib.providers.base_provider import BaseSearchProvider
from raglib.providers.factory import build_search_provider, normalize_provider_name
from raglib.routing import TIER_CHEAP, TIER_FAQ, TIER_FULL, HeuristicTierRouter, TierRouter, decide
from raglib.schemas import Document, GenerationResult, TraceEvent
from raglib.vectorstores.base_store import BaseVectorStore, MetadataFilter, VectorRecord, matches_filter
from raglib.vectorstores.in_memory_store import InMemoryVectorStore
from raglib.vision.base_vision import BaseVisionClient
from raglib.vision.vision_factory import VisionFactory

logger = logging.getLogger(__name__)

OPENROUTER_FREE_RPM = 20  # OpenRouter free-model limit (requests/minute)

RAG_TYPE_ALIASES: Dict[str, str] = {
    "naive": "naive",
    "advanced": "advanced",
    "corrective": "corrective",
    "crag": "corrective",
    "self": "self",
    "self_rag": "self",
    "agentic": "agentic",
    "agent": "agentic",
    "hybrid": "hybrid",
    "hybrid_web": "hybrid_web",
    "web_hybrid": "hybrid_web",
    "multi_query": "multi_query",
    "multi-query": "multi_query",
    "multiquery": "multi_query",
    "multi_hop": "multi_hop",
    "multi-hop": "multi_hop",
    "multihop": "multi_hop",
    "routing": "routing",
    "memory": "memory",
    "conversational": "memory",
    "web": "web",
    "tool": "tool",
    "tool_augmented": "tool",
    "tool-augmented": "tool",
    "toolaugmented": "tool",
    "graph": "graph",
    "graphrag": "graph",
    "graph_rag": "graph",
    "knowledge_graph": "graph",
}

ORCHESTRATORS: Dict[str, str] = {
    "naive": "raglib.rag_types.naive_rag.NaiveRAG",
    "advanced": "raglib.rag_types.advanced_rag.AdvancedRAG",
    "corrective": "raglib.rag_types.corrective_rag.CorrectiveRAG",
    "self": "raglib.rag_types.self_rag.SelfRAG",
    "agentic": "raglib.rag_types.agentic_rag.AgenticRAG",
    "hybrid": "raglib.rag_types.hybrid_rag.HybridRAG",
    "hybrid_web": "raglib.rag_types.hybrid_rag.HybridRAG",
    "multi_query": "raglib.rag_types.multi_query_rag.MultiQueryRAG",
    "multi_hop": "raglib.rag_types.multi_hop_rag.MultiHopRAG",
    "routing": "raglib.rag_types.routing_rag.RoutingRAG",
    "memory": "raglib.rag_types.memory_rag.MemoryRAG",
    "web": "raglib.rag_types.web_rag.WebRAG",
    "tool": "raglib.rag_types.tool_augmented_rag.ToolAugmentedRAG",
    "graph": "raglib.rag_types.graph_rag.GraphRAG",
}

GRAPH_INDEXER_OPTIONS = {
    "entity_types", "max_workers", "gleanings", "max_levels", "max_community_size",
    "community_rebuild_threshold", "resolve_with_embeddings", "resolution_threshold", "min_llm_report_size",
    "max_extraction_attempts",
}
GRAPH_LOCAL_OPTIONS = {"top_k_entities", "hops", "max_neighbors", "max_relations", "max_reports"}
GRAPH_GLOBAL_OPTIONS = {"top_communities", "max_points"}


class _LocalSearchProvider(BaseSearchProvider):
    """Offline web-style provider over the loaded corpus (for tests and demos only)."""

    PROVIDER_NAME = "local_web"
    searches_corpus = True  # its "web" results are corpus chunks, so filters / access control apply
    supports_filters = True

    def __init__(self, documents_fn: Callable[[], List[Document]]):
        """Initialize with a callable returning the current corpus."""

        self._documents_fn = documents_fn

    @property
    def name(self) -> str:
        """Return provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5, filters: Optional[MetadataFilter] = None) -> List[Document]:
        """Lexical overlap search over the corpus (only chunks passing ``filters``)."""

        query_terms = token_set(query, remove_stopwords=True) or token_set(query)
        if not query_terms:
            return []
        scored: List[Document] = []
        for document in self._documents_fn():
            if filters and not matches_filter(
                {**document.metadata, "source": document.source, "doc_id": document.id}, filters
            ):
                continue
            overlap = len(query_terms & token_set(document.content))
            if overlap:
                scored.append(
                    replace(
                        document,
                        score=overlap / len(query_terms),
                        source="web",
                        metadata={**document.metadata, "provider": self.name},
                    )
                )
        scored.sort(key=lambda doc: doc.score, reverse=True)
        return scored[:num_results]


def _env_flag(name: str) -> bool:
    """Return True when an environment flag is set to a truthy value."""

    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class RAG:
    """Single entry point for raglib.

    Minimal::

        rag = RAG("docs/", chat_llm="openai")          # key from OPENAI_API_KEY
        print(rag.query("What is our refund policy?").answer)

    Offline (tests / demos, no API keys - answers are templated, not real)::

        rag = RAG("docs/", offline=True)
    """

    @apply_presets
    def __init__(
        self,
        source: SourceInput = None,
        chat_llm: Optional[Union[str, BaseLLMClient, Any]] = None,
        embedding_llm: Optional[Union[str, BaseEmbedding]] = None,
        vision_llm: Optional[Union[str, BaseVisionClient]] = None,
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
        web_search_provider: Optional[str] = None,
        web_search_api_key: Optional[str] = None,
        web_search_base_url: Optional[str] = None,
        web_search_cse_id: Optional[str] = None,
        web_search_provider_kwargs: Optional[Dict[str, Any]] = None,
        validate_web_search_api_key: bool = False,
        vector_db: Optional[Union[str, BaseVectorStore]] = None,
        vector_db_kwargs: Optional[Dict[str, Any]] = None,
        offline: Optional[bool] = None,
        retrieval_mode: Optional[str] = None,
        fusion: str = "relative",
        hybrid_alpha: float = 0.5,
        reranker: str = "none",
        reranker_model: Optional[str] = None,
        reranker_api_key: Optional[str] = None,
        fetch_k: Optional[int] = None,
        mmr_lambda: Optional[float] = None,
        neighbor_window: int = 0,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        memory: Optional[bool] = None,
        max_history_turns: int = 10,
        max_history_tokens: int = 1200,
        rewrite_query: bool = False,
        enable_graph: bool = False,
        graph_store: Optional[Any] = None,
        graph_kwargs: Optional[Dict[str, Any]] = None,
        graph_llm: Optional[Union[str, BaseLLMClient, Any]] = None,
        graph_build: str = "sync",
        tools: Optional[Sequence[Any]] = None,
        on_event: OnEvent = None,
        llm_cache: bool = True,
        llm_timeout: float = 60.0,
        llm_max_retries: int = 2,
        llm_rpm: Optional[float] = None,
        llm_fallbacks: Optional[Sequence[str]] = None,
        llm: Optional[Union[str, BaseLLMClient, Any]] = None,
        allowed_dirs: Optional[Sequence[str]] = None,
        allow_private_urls: bool = False,
        llm_temperature: float = 0.0,
        llm_max_tokens: Optional[int] = None,
        conversation_store: Optional[ConversationStore] = None,
        strict: bool = False,
        access_control: Union[bool, AccessControl, Mapping[str, Any], None] = None,
        preset: Optional[str] = None,
        rerank_candidates: Optional[int] = None,
        rerank_skip_threshold: Optional[float] = None,
        rerank_skip_margin: Optional[float] = None,
        llm_cache_ttl: Optional[float] = None,
        embedding_cache_ttl: Optional[float] = None,
        cache_personalized: bool = False,
        answer_cache: Union[bool, AnswerCache, None] = None,
        prompt_caching: bool = False,
        cheap_llm: Optional[Union[str, BaseLLMClient, Any]] = None,
        tier_router: Optional[TierRouter] = None,
        parent_retrieval: Optional[str] = None,
        parent_chunk_size: int = 2000,
        semantic_breakpoint: str = "percentile",
        semantic_threshold: Optional[float] = None,
        prices: Union[PriceTable, Mapping[str, Any], None] = None,
        semantic_min_tokens: Optional[int] = None,
        token_counter: Union[str, Callable[[str], int]] = "units",
        on_embedding_failure: str = "fallback",
        abbreviations: Optional[Sequence[str]] = None,
        **rag_type_kwargs: Any,
    ):
        """Build the full RAG stack. See README for every option.

        ``llm`` is shorthand for ``chat_llm`` and accepts ``"provider/model"``,
        e.g. ``RAG("docs/", llm="openrouter/google/gemini-2.5-flash")``.

        ``llm_rpm`` caps requests per minute to the provider (chat + embeddings share the budget;
        OpenRouter ``:free`` models default to 20, ``0`` disables). ``llm_fallbacks`` lists
        ``"provider/model"`` strings tried in order when the primary model fails.

        ``llm_temperature`` / ``llm_max_tokens`` apply to models raglib builds from a provider name.
        ``graph_llm`` builds the knowledge graph (entity extraction, community reports, global-search
        map step) with a different model than the answers, e.g. a cheap fast model for the thousands
        of indexing calls: ``RAG(..., rag_type="graph", llm="openai/gpt-4.1", graph_llm="openai/gpt-4.1-nano")``.
        Its key comes from ``chat_api_key`` only when it is the same provider, otherwise from its env var.

        ``conversation_store`` replaces the in-memory session history (e.g.
        ``SQLiteConversationStore("chats.db")``). ``strict=True`` raises when an optional step
        (grader, router, reranker, query rewrite, web search, ...) fails, instead of degrading
        and listing the failure in ``result.metadata["degraded"]``.

        ``access_control=True`` (or an ``AccessControl``) turns on document permissions: tag
        documents at ingest (``add(..., access={"access_level": "confidential", ...})``) and pass
        ``user=UserContext(...)`` to ``query``. Deny by default: untagged documents are never
        returned, and a query without a user sees only the lowest level. See ``raglib.access``.

        ``preset="production"`` fills in tested production settings for every argument you do not
        pass yourself (see ``raglib.presets``); ``RAG.production(...)`` is the shortcut.
        ``rerank_candidates`` is how many chunks are retrieved for the reranker to cut to ``top_k``.

        Caching and cost (each option names its trade-off in the README):

        * ``rerank_skip_threshold`` / ``rerank_skip_margin``: skip the reranker when the top hit's
          dense cosine similarity is already this high (see ``Reranker``).
        * ``llm_cache_ttl`` / ``embedding_cache_ttl``: seconds before cached LLM responses / vectors expire.
        * ``cache_personalized=False``: LLM and answer caches are skipped for queries with a ``user``
          or caller ``filters`` (only the public tier is cached).
        * ``answer_cache``: question-level answer cache (FAQ tier); on by default with tier routing.
        * ``prompt_caching``: Anthropic ``cache_control`` on the system prompt (no-op elsewhere).
        * ``cheap_llm`` + ``tier_router``: route easy questions to a cheaper model (``raglib.routing``).
        * ``parent_retrieval="heading" | "fixed"``: search small chunks, answer from their parent
          sections of up to ``parent_chunk_size`` tokens.
        * ``chunking_strategy="embedding"``: semantic chunking with ``semantic_breakpoint`` /
          ``semantic_threshold`` / ``semantic_min_tokens``; ``on_embedding_failure="raise"`` makes a
          failed embedding call fail ingestion instead of falling back (see ``ChunkSplitter``).
        * ``token_counter``: how chunk sizes are measured - ``"units"`` (words; default),
          ``"tiktoken"``, ``"auto"`` or a callable; the embedding cost estimate uses the same one.
        * ``prices``: a ``PriceTable``; ``result.usage`` then reports ``cost_usd`` per stage.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if llm_max_tokens is not None and llm_max_tokens <= 0:
            raise ValueError("llm_max_tokens must be greater than zero")
        self._llm_temperature = llm_temperature
        self._llm_max_tokens = llm_max_tokens
        if llm is not None:
            if chat_llm is not None:
                raise ValueError("Pass either llm=... or chat_llm=..., not both")
            chat_llm = llm
        self._rag_type = self._normalize_rag_type(rag_type)
        self._output_dir = output_dir
        self._offline = bool(offline) if offline is not None else _env_flag(OFFLINE_ENV_VAR)
        self._top_k = top_k
        self._preset = preset
        self._access: Optional[AccessControl] = AccessControl.coerce(access_control)
        self._parent_acl: Dict[str, str] = {}  # parent document id -> access-tag signature

        effective_chat_key = chat_api_key if chat_api_key is not None else llm_key
        self._llm_rpm = llm_rpm
        self._llm_max_retries = llm_max_retries
        self._llm_timeout = llm_timeout
        self._llm_raw, chat_provider = self._resolve_chat_llm(
            chat_llm=chat_llm,
            api_key=effective_chat_key,
            model_name=chat_model,
            base_url=chat_base_url,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
        )
        if llm_fallbacks:
            clients = [self._llm_raw]
            for fallback in llm_fallbacks:
                fallback_provider = parse_model_string(fallback)[0] if isinstance(fallback, str) else None
                clients.append(
                    self._resolve_chat_llm(
                        chat_llm=fallback,
                        api_key=effective_chat_key if fallback_provider == chat_provider else None,
                        model_name=None,
                        base_url=None,
                        timeout=llm_timeout,
                        max_retries=llm_max_retries,
                    )[0]
                )
            self._llm_raw = FallbackLLMClient(clients)
        if prompt_caching:
            _enable_prompt_caching(self._llm_raw)
        self._llm: BaseLLMClient = CachedLLMClient(self._llm_raw, ttl=llm_cache_ttl) if llm_cache else self._llm_raw
        self._chat_provider = chat_provider
        self._cache_personalized = cache_personalized
        self._prices: Optional[PriceTable] = PriceTable.coerce(prices)
        self._ingest_meter = UsageMeter()

        self._embedding = self._resolve_embedding(
            embedding_llm=embedding_llm,
            api_key=embedding_api_key,
            chat_api_key=self._first_non_empty(chat_api_key, llm_key),
            chat_provider=chat_provider,
            model_name=embedding_model,
            base_url=embedding_base_url,
        )
        if embedding_cache_ttl is not None and isinstance(self._embedding, CachedEmbedding):
            self._embedding.ttl = embedding_cache_ttl
        self._vision = self._resolve_vision(
            vision_llm=vision_llm,
            api_key=self._first_non_empty(vision_api_key, llm_key, chat_api_key),
            chat_provider=chat_provider,
            model_name=vision_model,
            base_url=vision_base_url,
            explicit_key=self._first_non_empty(vision_api_key, llm_key),
            chat_api_key=chat_api_key,
        )

        if isinstance(vector_db, str) and vector_db.strip().lower() not in SUPPORTED_VECTOR_DBS:
            raise ValueError(f"Unknown vector_db '{vector_db}'. Supported values: {sorted(SUPPORTED_VECTOR_DBS)}")
        self._loader = DocumentLoader(
            vision_client=self._vision,
            allow_private_urls=allow_private_urls,
            allowed_dirs=allowed_dirs,
        )
        self._chunking_strategy = chunking_strategy
        uses_llm_chunking = str(chunking_strategy).strip().lower() in {"llm", "semantic", "llm_semantic"}
        self._splitter = ChunkSplitter(
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            strategy=chunking_strategy,
            llm_client=self._llm if uses_llm_chunking else None,
            embedding=self._embedding,
            semantic_breakpoint=semantic_breakpoint,
            semantic_threshold=semantic_threshold,
            semantic_min_tokens=semantic_min_tokens,
            token_counter=token_counter,
            on_embedding_failure=on_embedding_failure,
            abbreviations=abbreviations,
            on_fallback=self._chunking_fallback,
        )
        self._chunking_fallbacks: List[Dict[str, str]] = []
        if isinstance(self._embedding, CachedEmbedding) and self._splitter.token_counter_name != "units":
            # One counter for chunk sizes and embedding cost estimates (words are not model tokens,
            # so with "units" the cost estimate keeps the token estimator).
            self._embedding.token_counter = self._splitter.token_counter
        if parent_retrieval is not None and parent_retrieval not in PARENT_MODES:
            raise ValueError(f"parent_retrieval must be one of {list(PARENT_MODES)} or None")
        if parent_chunk_size <= 0:
            raise ValueError("parent_chunk_size must be greater than zero")
        self._parent_retrieval = parent_retrieval
        self._parent_chunk_size = parent_chunk_size
        self._sections: Dict[str, Document] = {}  # section id -> parent section (small-to-big)
        self._parent_sections: Dict[str, List[str]] = {}  # document id -> its section ids
        self._staged_sections: Dict[str, List[Document]] = {}
        self._doc_versions: Dict[str, str] = {}  # document id -> content + permission hash
        self._invalidated_entries = 0  # cache entries dropped by document updates (all time)
        self._retriever = Retriever(
            documents=[],
            embedding=self._embedding,
            top_k=top_k,
            vector_db=vector_db,
            vector_db_kwargs=vector_db_kwargs,
            mode=retrieval_mode,
            fusion=fusion,
            hybrid_alpha=hybrid_alpha,
            fetch_k=fetch_k,
            mmr_lambda=mmr_lambda,
            neighbor_window=neighbor_window,
        )
        self.retriever = self._retriever
        if parent_retrieval is not None:
            self._retriever.parent_lookup = self._sections.get
        self._parents: Dict[str, List[str]] = {}
        self._ingest_lock = threading.RLock()  # add/update/delete are serialized; queries are not blocked

        self._web_provider = self._resolve_web_provider(
            provider_name=web_search_provider,
            api_key=web_search_api_key,
            base_url=web_search_base_url,
            cse_id=web_search_cse_id,
            provider_kwargs=web_search_provider_kwargs,
        )
        if validate_web_search_api_key and self._web_provider is not None:
            self._validate_web_provider_connection(self._web_provider)
        self._web_retriever = WebRetriever(provider=self._web_provider, top_k=top_k) if self._web_provider else None
        self._hybrid_retriever = (
            HybridRetriever(vector_retriever=self._retriever, web_retriever=self._web_retriever)
            if self._web_retriever is not None
            else None
        )

        self._graph_indexer: Optional[Any] = None
        self._graph_retriever: Optional[Any] = None
        self._graph_builder: Optional[Any] = None
        if graph_build not in {"sync", "background"}:
            raise ValueError("graph_build must be 'sync' or 'background'")
        self._graph_build = graph_build
        if self._rag_type == "graph" or enable_graph:
            self._graph_llm = self._resolve_graph_llm(
                graph_llm, chat_provider, effective_chat_key, llm_cache,
                max_tokens=int((graph_kwargs or {}).get("extraction_max_tokens", 4096)),
            )
            self._build_graph(graph_store=graph_store, graph_kwargs=dict(graph_kwargs or {}))
        elif graph_llm is not None:
            raise ValueError("graph_llm needs rag_type='graph' or enable_graph=True")
        elif graph_build != "sync":
            raise ValueError("graph_build needs rag_type='graph' or enable_graph=True")

        self._evaluator = Evaluator()
        self._reranker = Reranker(
            top_k=top_k,
            backend=reranker,
            model_name=reranker_model,
            llm_client=self._llm if reranker == "llm" else None,
            api_key=reranker_api_key,
            skip_threshold=rerank_skip_threshold,
            skip_margin=rerank_skip_margin,
        )
        self._conversations = conversation_store or ConversationStore(
            max_turns=max_history_turns, max_tokens=max_history_tokens
        )
        self._generator = Generator(llm_client=self._llm, max_context_tokens=max_context_tokens)
        self._router_retriever = RouterRetriever(
            vector_retriever=self._retriever,
            web_retriever=self._web_retriever,
            hybrid_retriever=self._hybrid_retriever,
            llm_client=self._llm,
            graph_retriever=self._graph_retriever,
        )
        self._components: Dict[str, Any] = {
            "retriever": self._retriever,
            "web_retriever": self._web_retriever,
            "hybrid_retriever": self._hybrid_retriever,
            "router_retriever": self._router_retriever,
            "graph_retriever": self._graph_retriever,
            "reranker": self._reranker,
            "evaluator": self._evaluator,
            "refiner": Refiner(llm_client=self._llm),
            "decision_engine": DecisionEngine(llm_client=self._llm, allow_web=self._web_retriever is not None),
            "reflection_module": ReflectionModule(evaluator=self._evaluator),
            "planner": Planner(llm_client=self._llm),
            "query_rewriter": QueryRewriter(llm_client=self._llm) if rewrite_query else None,
            "query_expander": QueryExpander(llm_client=self._llm),
            "deduplicator": Deduplicator(),
            "generator": self._generator,
            "grader": RelevanceGrader(llm_client=self._llm, evaluator=self._evaluator),
            "llm_client": self._llm,
            "conversation_store": self._conversations,
            "condenser": QueryCondenser(llm_client=self._llm),
            "use_memory": bool(memory) if memory is not None else self._rag_type == "memory",
            "on_event": on_event,
            "strict": strict,
            "rerank_candidates": rerank_candidates,
            "price_table": self._prices,
        }
        if self._rag_type == "hybrid_web":
            if self._hybrid_retriever is None:
                raise ValueError("rag_type='hybrid_web' requires a web_search_provider")
            rag_type_kwargs.setdefault("include_web", True)
        if tools and self._rag_type == "tool" and "tool_call_hook" not in rag_type_kwargs:
            # rag_type='tool' + tools=[...]: let the LLM choose among the tools (agentic tool loop).
            self._rag_type = "agentic"
        if tools and self._rag_type not in {"agentic"}:
            raise ValueError("tools=[...] is supported with rag_type='agentic' or 'tool'")
        if tools:
            rag_type_kwargs["tools"] = list(tools)
        self._rag = self._build_orchestrator(self._rag_type, rag_type_kwargs)

        # Tier routing: FAQ (answer cache) -> cheap model -> full pipeline.
        self._cheap_rag: Optional[Any] = None
        if cheap_llm is not None:
            cheap_provider = parse_model_string(cheap_llm)[0] if isinstance(cheap_llm, str) else None
            cheap_raw = self._resolve_chat_llm(
                chat_llm=cheap_llm,
                api_key=effective_chat_key if cheap_provider == chat_provider else None,
                model_name=None,
                base_url=None,
                timeout=llm_timeout,
                max_retries=llm_max_retries,
            )[0]
            if prompt_caching:
                _enable_prompt_caching(cheap_raw)
            cheap = CachedLLMClient(cheap_raw, ttl=llm_cache_ttl) if llm_cache else cheap_raw
            self._cheap_rag = self._build_orchestrator(
                "advanced",
                {
                    "llm_client": cheap,
                    "generator": Generator(llm_client=cheap, max_context_tokens=max_context_tokens),
                    "reranker": Reranker(top_k=top_k, backend="none"),
                    "query_rewriter": None,
                },
            )
        if isinstance(answer_cache, AnswerCache):
            self._answer_cache: Optional[AnswerCache] = answer_cache
        elif answer_cache or (answer_cache is None and (cheap_llm is not None or tier_router is not None)):
            self._answer_cache = AnswerCache()
        else:
            self._answer_cache = None
        if self._answer_cache is not None:
            self._answer_cache.version_of = self.document_version
        routing = self._answer_cache is not None or self._cheap_rag is not None or tier_router is not None
        self._tier_router: Optional[TierRouter] = (tier_router or HeuristicTierRouter()) if routing else None

        if source is not None:
            self.add(source)
        logger.info(
            "RAG ready (rag_type=%s chunks=%d llm=%s embedding=%s web=%s graph=%s)",
            self._rag_type,
            len(self._retriever.documents),
            type(self._llm_raw).__name__,
            type(self._embedding).__name__ if self._embedding else "none(BM25)",
            self._web_provider.name if self._web_provider else "off",
            "on" if self._graph_indexer else "off",
        )

    @classmethod
    def production(cls, source: SourceInput = None, **kwargs: Any) -> "RAG":
        """Shortcut for ``RAG(source, preset="production", **kwargs)``; your kwargs override the preset."""

        kwargs.setdefault("preset", "production")
        return cls(source, **kwargs)

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
        user: Union[UserContext, Mapping[str, Any], None] = None,
    ) -> GenerationResult:
        """Answer one question. ``session_id`` isolates conversation memory per user.

        With ``access_control`` on, ``user`` (a ``UserContext`` or a dict with ``user_id``,
        ``roles``, ``departments``, ``max_access_level``) decides which documents are visible;
        ``filters`` can only narrow that further.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        question = question.strip()
        effective_filters, session_key = self._scope(filters, user, session_id)
        personalized = self._personalized(filters, user)
        with cache_bypass(personalized and not self._cache_personalized):
            result = self._answer(question, session_key, effective_filters, personalized)
        result.metadata["session_id"] = session_id
        if self._output_dir:
            self._save_result(question, result)
        return result

    def _personalized(self, filters: Optional[MetadataFilter], user: Any) -> bool:
        """True when the answer may depend on who asks (a user context or caller filters)."""

        return user is not None or bool(filters)

    def _answer(
        self, question: str, session_key: Optional[str], filters: Optional[MetadataFilter], personalized: bool
    ) -> GenerationResult:
        """Route to the FAQ / cheap / full tier and run it."""

        if self._tier_router is None:
            return self._rag.run(question, session_id=session_key, filters=filters)
        cache = self._answer_cache
        # Answers that depend on who asks, or on the conversation so far, are never served from cache.
        cacheable = cache is not None and (self._cache_personalized or not personalized) and not self._rag.use_memory
        scope = json.dumps(filters, sort_keys=True, default=str) if filters else None
        cached = cache.get(question, scope, self._rag_type) if cacheable and cache is not None else None
        context = {
            "answer_cached": cached is not None,
            "cheap_available": self._cheap_rag is not None,
            "personalized": personalized,
        }
        decision = decide(self._tier_router, question, context)
        logger.info("tier=%s (%s)", decision.tier, decision.reason)
        if decision.tier == TIER_FAQ and cached is not None:
            return self._faq_result(cached, decision.reason)
        tier = TIER_CHEAP if decision.tier in (TIER_FAQ, TIER_CHEAP) and self._cheap_rag is not None else TIER_FULL
        pipeline = self._cheap_rag if tier == TIER_CHEAP else self._rag
        with run_tags(tier=tier):
            result = pipeline.run(question, session_id=session_key, filters=filters)
        result.metadata.update({"tier": tier, "tier_reason": decision.reason, "answer_cache_hit": False})
        if cacheable and cache is not None:
            cache.put(question, result, scope, self._rag_type)
        return result

    def _faq_result(self, cached: GenerationResult, reason: str) -> GenerationResult:
        """Return a cached answer, traced as its own short run (tier ``faq``)."""

        tracer = Tracer(on_event=self._rag.on_event)
        tracer.tag("tier", TIER_FAQ)
        with tracer.stage("answer_cache"):
            pass
        total = tracer.finish(pipeline=self._rag_type, cache_hit={"answer": True}, cost_usd=0.0)
        cached.metadata.update({"tier": TIER_FAQ, "tier_reason": reason, "answer_cache_hit": True})
        cached.trace_events = list(tracer.events)
        cached.usage = {"llm_calls": 0, "total_tokens": 0, "latency_ms": tracer.total_latency_ms(), "wall_latency_ms": total.latency_ms}
        if self._prices is not None:
            cached.usage["cost_usd"] = 0.0
        return cached

    async def aquery(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
        user: Union[UserContext, Mapping[str, Any], None] = None,
    ) -> GenerationResult:
        """Async ``query`` (runs the pipeline in a worker thread)."""

        import asyncio

        return await asyncio.to_thread(self.query, question, session_id, filters, user)

    def stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
        user: Union[UserContext, Mapping[str, Any], None] = None,
    ) -> Iterator[str]:
        """Stream answer tokens: ``for token in rag.stream(q): print(token, end="")``."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        effective_filters, session_key = self._scope(filters, user, session_id)
        # Streaming always runs the full pipeline (tier routing and the answer cache are query() features).
        with cache_bypass(self._personalized(filters, user) and not self._cache_personalized):
            result = yield from self._rag.stream(question.strip(), session_id=session_key, filters=effective_filters)
        if result is not None:
            result.metadata["session_id"] = session_id
        if self._output_dir and result is not None:
            self._save_result(question.strip(), result)
        return result

    @property
    def access_control(self) -> Optional[AccessControl]:
        """The access-control policy (``None`` when access control is off)."""

        return self._access

    def permission_filter(self, user: Union[UserContext, Mapping[str, Any], None] = None) -> Optional[MetadataFilter]:
        """The metadata filter access control applies for ``user`` (``None`` when it is off)."""

        if self._access is None:
            return None
        return self._access.filter_for(UserContext.coerce(user))

    def _scope(
        self,
        filters: Optional[MetadataFilter],
        user: Union[UserContext, Mapping[str, Any], None],
        session_id: Optional[str],
    ) -> Tuple[Optional[MetadataFilter], Optional[str]]:
        """Merge the permission filter into ``filters`` and namespace the memory session by user."""

        context = UserContext.coerce(user)
        if self._access is None:
            if context is not None:
                raise ValueError("user=... needs access_control (RAG(..., access_control=True))")
            return filters, session_id
        acl = self._access.filter_for(context)
        # Memory is per user: two users never read each other's history, even with the same (or no)
        # session_id. Without a user_id the owner is the permission scope itself.
        if context is not None and context.user_id:
            owner = f"user:{context.user_id}"
        else:
            owner = "acl:" + hashlib.sha256(json.dumps(acl, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return merge_filters(acl, filters), f"{owner}::{session_id or 'default'}"

    def chat(self, session_id: str = "interactive") -> None:
        """Open an interactive terminal Q&A loop with conversation memory."""

        from raglib.session.interactive import InteractiveSession

        previous = self._rag.use_memory
        self._rag.use_memory = True
        try:
            InteractiveSession(rag=self, session_id=session_id).start()
        finally:
            self._rag.use_memory = previous  # later rag.query() calls keep their own setting

    def add(
        self,
        source: SourceInput,
        batch_size: int = 256,
        on_progress: Optional[Callable[[Dict[str, int]], None]] = None,
        *,
        access: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Add or update sources. Unchanged documents are skipped; changed ones are replaced.

        ``metadata`` is merged into every loaded document. ``access`` sets the access tags
        (``access_level``, ``department``, ``allowed_roles``; see ``raglib.access.access_tags``)
        on every loaded document; tags already in a ``Document``'s metadata work too. Re-adding
        unchanged text with different tags updates its permissions.

        Large inputs are processed in batches of ~``batch_size`` chunks: each batch is embedded
        and indexed before the next file is read, so memory stays bounded, progress is reported
        via ``on_progress({"files": .., "chunks_indexed": .., "skipped": ..})``, and if the run
        fails half-way, already-indexed files are kept (re-running skips them).
        Returns the number of new chunks indexed.
        """

        if access is not None:
            unknown = set(access) - set(ACCESS_TAG_KEYS)
            if unknown:
                raise ValueError(f"Unknown access tags: {sorted(unknown)}. Supported: {list(ACCESS_TAG_KEYS)}")
        extra = {**dict(metadata or {}), **dict(access or {})}
        stats = {"files": 0, "chunks_indexed": 0, "skipped": 0}
        pending: List[Document] = []

        def flush() -> None:
            if pending:
                stats["chunks_indexed"] += self._index_chunks(pending)
                pending.clear()
            if on_progress is not None:
                on_progress(dict(stats))

        with usage_scope() as meter:  # ingest cost: embeddings, LLM chunking, graph extraction
            for loaded in self._loader.iter_load(source):
                stats["files"] += 1
                if not loaded:
                    stats["skipped"] += 1
                    continue
                if extra:
                    loaded = [replace(document, metadata={**document.metadata, **extra}) for document in loaded]
                pending.extend(self._chunk(loaded))
                if len(pending) >= batch_size:
                    flush()
            flush()
        self._ingest_meter.merge(meter)
        logger.info(
            "Ingested %d new chunks from %d sources (corpus=%d)",
            stats["chunks_indexed"],
            stats["files"],
            len(self._retriever.documents),
        )
        return stats["chunks_indexed"]

    def _chunking_fallback(self, document: Document, error: Exception) -> None:
        """Record an embedding-chunking fallback: counted in ``ingest_usage()`` and sent to ``on_event``."""

        from raglib.core.tracing import _callbacks

        self._chunking_fallbacks.append({"document_id": document.id, "error": f"{type(error).__name__}: {error}"[:300]})
        rag = getattr(self, "_rag", None)
        event = TraceEvent(
            stage="chunking_fallback",
            latency_ms=0.0,
            meta={"event": "warning", "document_id": document.id, "fallback": "recursive"},
            error=f"{type(error).__name__}: {error}"[:300],
        )
        for callback in _callbacks(getattr(rag, "on_event", None)):
            try:
                callback(event)
            except Exception as exc:  # noqa: BLE001 - observers must never break ingestion
                logger.warning("on_event callback failed: %s", exc)

    def _tag(self, document: Document) -> Document:
        """Apply access-control tags to a chunk or section (no-op without access control)."""

        return replace(document, metadata=self._access.tag(document.metadata)) if self._access is not None else document

    def _chunk(self, documents: List[Document]) -> List[Document]:
        """Split loaded documents into chunks (children of parent sections with parent_retrieval)."""

        if self._parent_retrieval is None:
            return [self._tag(chunk) for chunk in self._splitter.split(documents)]
        chunks: List[Document] = []
        for document in documents:
            sections = [self._tag(section) for section in build_sections(document, self._parent_retrieval, self._parent_chunk_size)]
            with self._ingest_lock:
                self._staged_sections.setdefault(document.id, []).extend(sections)
            for child in self._splitter.split(sections):
                # Children belong to the document (replace/delete semantics) and point to their section.
                metadata = {**child.metadata, "parent_id": document.id, "section_id": child.metadata["parent_id"]}
                chunks.append(self._tag(replace(child, metadata=metadata)))
        return chunks

    def _register_sections(self, parent_id: str, version: Optional[str] = None) -> None:
        """Make the staged sections of ``parent_id`` current (replacing its old ones)."""

        staged = self._staged_sections.pop(parent_id, None)
        if staged is None:
            return
        for section_id in self._parent_sections.pop(parent_id, []):
            self._sections.pop(section_id, None)
        for section in staged:
            if version is not None:
                section.metadata["document_version"] = version
            self._sections[section.id] = section
        self._parent_sections[parent_id] = [section.id for section in staged]

    def document_version(self, parent_id: str) -> Optional[str]:
        """Content + permission hash of an indexed document (``None`` when not indexed)."""

        return self._doc_versions.get(parent_id)

    def _invalidate_documents(self, parent_ids: Sequence[str], old_texts: Sequence[str] = ()) -> int:
        """Drop cached answers built from ``parent_ids`` and the cached vectors of ``old_texts``."""

        dropped = 0
        if isinstance(self._llm, CachedLLMClient):
            dropped += self._llm.invalidate(parent_ids)
        if self._answer_cache is not None:
            dropped += self._answer_cache.invalidate(parent_ids)
        if old_texts and isinstance(self._embedding, CachedEmbedding):
            dropped += self._embedding.invalidate_texts(list(old_texts))
        self._invalidated_entries += dropped
        return dropped

    def invalidate(self, source: SourceInput) -> int:
        """Drop every cached answer (LLM cache and answer cache) that used these documents.

        ``source`` is a document id, a file path, or a list of them. Returns the number of cache
        entries dropped. Use it when a document changed outside raglib; ``add``/``update`` of a
        changed document and ``on_document_update`` invalidate automatically.
        """

        items = source if isinstance(source, (list, tuple, set)) else [source]
        parent_ids = self._known_parents_for(source) | {str(item) for item in items if isinstance(item, str)}
        texts = [doc.content for pid in parent_ids for doc in (self._retriever.get(cid) for cid in self._parents.get(pid, [])) if doc]
        return self._invalidate_documents(sorted(parent_ids), texts)

    def invalidate_all(self) -> None:
        """Clear the LLM, embedding and answer caches."""

        self._purge_caches()
        if self._answer_cache is not None:
            self._answer_cache.clear()

    def on_document_update(self, source: SourceInput = None, doc_ids: Optional[Sequence[str]] = None) -> Dict[str, int]:
        """Hook for external update jobs (e.g. the nightly corpus sync): event-driven invalidation.

        ``source``: files/folders/URLs/texts that changed - they are re-ingested (``add``), which
        replaces changed documents and drops the cached answers that used them. ``doc_ids``:
        documents changed elsewhere - their cached answers are dropped. Returns
        ``{"reindexed_chunks": n, "invalidated": m}``. Cache freshness is a correctness issue: call
        this from the job that updates the corpus instead of relying on TTLs alone.
        """

        before = self._invalidated_entries
        reindexed = self.add(source) if source is not None else 0
        if doc_ids:
            self.invalidate(list(doc_ids))
        return {"reindexed_chunks": reindexed, "invalidated": self._invalidated_entries - before}

    def ingest_usage(self) -> Dict[str, Any]:
        """Tokens (and USD, with ``prices``) spent on ingestion so far: embeddings, LLM chunking, graph."""

        info: Dict[str, Any] = {
            **self._ingest_meter.as_dict(),
            "usage_by_model": self._ingest_meter.breakdown(),
            "chunking_fallbacks": len(self._chunking_fallbacks),
            "chunking_fallback_documents": [item["document_id"] for item in self._chunking_fallbacks[-50:]],
        }
        if self._prices is not None:
            cost = self._prices.cost(self._ingest_meter.breakdown())
            info.update({"cost_usd": round(cost["total"], 8), "cost_by_model": cost["by_model"], "cost_unpriced": cost["unpriced"]})
        return info

    def _index_chunks(self, chunks: List[Document]) -> int:
        """Upsert one batch of chunks (per-document replace semantics)."""

        with self._ingest_lock:
            by_parent: Dict[str, List[Document]] = {}
            for chunk in chunks:
                by_parent.setdefault(str(chunk.metadata.get("parent_id", chunk.id)), []).append(chunk)
            to_add: List[Document] = []
            changed: Dict[str, List[str]] = {}
            retagged: List[Document] = []
            signatures: Dict[str, str] = {}
            for parent_id, parent_chunks in by_parent.items():
                new_ids = [chunk.id for chunk in parent_chunks]
                old_ids = self._parents.get(parent_id)
                signature = AccessControl.signature(parent_chunks[0].metadata)
                version = hashlib.sha256(json.dumps([new_ids, signature]).encode("utf-8")).hexdigest()[:16]
                if old_ids == new_ids:
                    if self._parent_acl.get(parent_id, signature) != signature:
                        _stamp_version(parent_chunks, version)
                        retagged.extend(parent_chunks)  # same text, new permissions
                        signatures[parent_id] = signature
                        self._register_sections(parent_id, version)
                        self._doc_versions[parent_id] = version
                        # A cached public answer must not outlive a permission change.
                        self._invalidate_documents([parent_id])
                    else:
                        self._staged_sections.pop(parent_id, None)
                    continue
                signatures[parent_id] = signature
                if old_ids:
                    old_texts = [doc.content for doc in (self._retriever.get(cid) for cid in old_ids) if doc]
                    self._delete_chunk_ids(old_ids)
                    self._parents.pop(parent_id, None)
                    self._invalidate_documents([parent_id], old_texts)
                    logger.info("Document %s changed; replaced %d old chunks", parent_id, len(old_ids))
                # Chunks carry their document's version, so stale vectors can be found or filtered
                # in the vector store ({"document_version": rag.document_version(doc_id)}).
                _stamp_version(parent_chunks, version)
                self._register_sections(parent_id, version)
                self._doc_versions[parent_id] = version
                changed[parent_id] = new_ids
                to_add.extend(parent_chunks)
            if retagged:
                # Chunk ids are unchanged, so the graph keeps pointing at them and picks up the new
                # tags on lookup; only the retriever indexes (which store metadata) are rewritten.
                self._retriever.delete_documents([chunk.id for chunk in retagged])
                self._retriever.add_documents(retagged)
                logger.info("Updated access tags of %d chunks", len(retagged))
            added = self._retriever.add_documents(to_add)
            if self._graph_indexer is not None and to_add:
                self._add_to_graph(to_add)  # skips chunks it already extracted
            # Record documents only after they are really indexed, so a failed batch is retried next run.
            self._parents.update(changed)
            self._parent_acl.update(signatures)
            return len(added)

    def update(self, source: SourceInput) -> int:
        """Re-ingest sources, replacing documents whose content changed (alias of ``add``)."""

        return self.add(source)

    def delete(self, source: SourceInput) -> int:
        """Remove every chunk that came from ``source`` (file path, document id, raw text, or a list).

        Returns the number of chunks removed. Works even if the file no longer exists on disk.
        """

        with self._ingest_lock:
            parent_ids = self._known_parents_for(source)
            if not parent_ids:
                parent_ids = {document.id for document in self._loader.load(source)}
            chunk_ids: List[str] = []
            for parent_id in parent_ids:
                chunk_ids.extend(self._parents.pop(parent_id, []))
                self._parent_acl.pop(parent_id, None)
                self._doc_versions.pop(parent_id, None)
                for section_id in self._parent_sections.pop(parent_id, []):
                    self._sections.pop(section_id, None)
            removed = self._delete_chunk_ids(chunk_ids)
            if removed:
                # Deleted text must not survive anywhere, including caches keyed on its prompt.
                self.invalidate_all()
            return removed

    def reset_memory(
        self, session_id: Optional[str] = None, user: Union[UserContext, Mapping[str, Any], None] = None
    ) -> None:
        """Clear conversation memory for one session (or all sessions).

        With access control on, sessions are kept per user: pass the same ``user`` as to ``query``.
        """

        if self._access is not None and (session_id is not None or user is not None):
            session_id = self._scope(None, user, session_id)[1]
        self._conversations.clear(session_id)

    def retry_graph_extraction(self, include_skipped: bool = False) -> Dict[str, int]:
        """Extract again the chunks whose graph extraction failed (e.g. after rate limits).

        Failed chunks are never cached as empty, and ``stats()["graph"]["failed_extractions"]``
        counts them. Loading a saved index also retries them, up to ``max_extraction_attempts``
        (graph_kwargs, default 3) in total; after that a chunk is skipped
        (``stats()["graph"]["skipped_extractions"]``). ``include_skipped=True`` retries those too.
        """

        if self._graph_indexer is None:
            raise ValueError("Graph is not enabled (use rag_type='graph' or enable_graph=True)")
        return self._graph_indexer.retry_failed(self._retriever.get, include_skipped=include_skipped)

    def rebuild_graph_communities(self) -> int:
        """Re-cluster the knowledge graph and refresh changed community reports."""

        if self._graph_indexer is None:
            raise ValueError("Graph is not enabled (use rag_type='graph' or enable_graph=True)")
        return len(self._graph_indexer.rebuild_communities())

    def visualize_graph(self, path: str = "raglib_graph.html", max_nodes: int = 400, rebuild_communities: bool = True) -> str:
        """Write an interactive HTML view of the knowledge graph (rag_type='graph' / enable_graph=True).

        Open the file in any browser: nodes are entities (size = number of relations, colour = type
        or community), edges are relations. Returns the file path.
        """

        if self._graph_indexer is None:
            raise ValueError("Graph is not enabled (use rag_type='graph' or enable_graph=True)")
        if rebuild_communities and self._graph_indexer.is_dirty:
            try:
                self._graph_indexer.rebuild_communities()
            except ImportError:
                logger.info("networkx not installed; visualizing without communities")
        from raglib.visualize import graphrag_to_html

        graphrag_to_html(self._graph_indexer.store, path=path, max_nodes=max_nodes)
        return path

    def evaluate(self, dataset: Sequence[Dict[str, Any]], judge_llm: Optional[BaseLLMClient] = None, k: int = 5):
        """Evaluate this pipeline on a labelled dataset (see ``raglib.eval.evaluate``)."""

        from raglib.eval import evaluate

        return evaluate(self, dataset, judge_llm=judge_llm, k=k)

    def save(self, directory: str) -> None:
        """Persist the chunk corpus, in-memory vectors and graph to ``directory``."""

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        chunks = [asdict(document) for document in self._retriever.documents]
        store = self._retriever.vector_store
        payload = {
            "version": 3,
            "index": self._index_fingerprint(),
            "parents": self._parents,
            "acl": self._parent_acl,
            "versions": self._doc_versions,
            "sections": {section_id: asdict(section) for section_id, section in self._sections.items()},
            "parent_sections": self._parent_sections,
            "chunks": chunks,
        }
        # Write each file atomically, so a crash mid-save never leaves a half-written index.
        if isinstance(store, InMemoryVectorStore):
            store.dump(str(path), "vectors")  # binary float32 + metadata: loads ~50x faster than JSON
            legacy = path / "vectors.json"
            if legacy.exists():
                legacy.unlink()
        _atomic_write(path / "chunks.json", json.dumps(payload, ensure_ascii=False))
        if self._graph_indexer is not None:
            self._graph_indexer.save(str(path / "graph"))
        logger.info("Saved RAG index to %s", path)

    def load(self, directory: str, allow_model_change: bool = False) -> "RAG":
        """Load an index written by ``save`` (embeddings are reused, not recomputed).

        Raises ``ValueError`` when the index was built with a different embedding model:
        its stored vectors are not comparable with this model's query vectors. Pass
        ``allow_model_change=True`` to load the chunks anyway and re-embed them.
        """

        path = Path(directory)
        payload = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        saved = payload.get("index") or {}
        current = self._index_fingerprint()
        changed = [
            key
            for key in ("embedding_model", "dimension")
            if saved.get(key) is not None and current.get(key) is not None and saved[key] != current[key]
        ]
        reembed = False
        if changed:
            detail = ", ".join(f"{key}: saved={saved[key]!r} current={current[key]!r}" for key in changed)
            if not allow_model_change:
                raise ValueError(
                    f"Index in {directory} was built with a different embedding model ({detail}). "
                    "Use the same embedding_llm/embedding_model, or load(..., allow_model_change=True) to re-embed."
                )
            logger.warning("Embedding model changed (%s); re-embedding the loaded chunks", detail)
            reembed = True
        store = self._retriever.vector_store
        vectors_path = path / "vectors.json"
        if isinstance(store, InMemoryVectorStore) and not reembed:
            if InMemoryVectorStore.has_dump(str(path), "vectors"):
                store.restore(str(path), "vectors")
            elif vectors_path.exists():  # indexes saved before raglib 2.2
                store.upsert([VectorRecord(**record) for record in json.loads(vectors_path.read_text(encoding="utf-8"))])
        documents = [Document(**item) for item in payload.get("chunks", [])]
        if reembed and store is not None and not isinstance(store, InMemoryVectorStore):
            store.delete([document.id for document in documents])  # drop vectors from the old model
        self._retriever.add_documents(documents)
        self._parents.update({key: list(value) for key, value in payload.get("parents", {}).items()})
        self._parent_acl.update({key: str(value) for key, value in payload.get("acl", {}).items()})
        self._doc_versions.update({key: str(value) for key, value in payload.get("versions", {}).items()})
        self._sections.update({key: Document(**value) for key, value in payload.get("sections", {}).items()})
        self._parent_sections.update({key: list(value) for key, value in payload.get("parent_sections", {}).items()})
        if self._graph_indexer is not None and (path / "graph").exists():
            self._graph_indexer.load(str(path / "graph"))
            missing = [doc for doc in documents if doc.id not in self._graph_indexer.indexed_chunk_ids]
            if missing:
                self._add_to_graph(missing)
        elif self._graph_indexer is not None:
            self._add_to_graph(documents)
        logger.info("Loaded %d chunks from %s", len(documents), path)
        return self

    @property
    def rag_type(self) -> str:
        """Return the canonical rag_type."""

        return self._rag_type

    @property
    def pipeline(self) -> Any:
        """Return the underlying orchestrator (BaseRAG subclass)."""

        return self._rag

    @property
    def graph(self) -> Any:
        """Return the GraphIndexer (None unless the graph is enabled)."""

        return self._graph_indexer

    @property
    def _documents(self) -> List[Document]:
        """Indexed chunks (kept for backward compatibility)."""

        return self._retriever.documents

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""

        info: Dict[str, Any] = {
            "rag_type": self._rag_type,
            "documents": len(self._parents),
            "chunks": len(self._retriever.documents),
            "retrieval_mode": self._retriever.mode,
            "embedding": type(unwrap_embedding(self._embedding)).__name__ if self._embedding else None,
            "llm": type(self._llm_raw).__name__,
            "web_provider": self._web_provider.name if self._web_provider else None,
            "reranker": self._reranker.backend,
            "preset": self._preset,
            "tiers": ["faq" if self._answer_cache else None, "cheap" if self._cheap_rag else None, "full"]
            if self._tier_router
            else None,
            "answer_cache": self._answer_cache.stats if self._answer_cache is not None else None,
            "parent_sections": len(self._sections) if self._parent_retrieval else None,
        }
        if self._graph_indexer is not None:
            info["graph"] = {
                **self._graph_indexer.store.stats(),
                "failed_extractions": len(self._graph_indexer.failed_chunk_ids),
                "skipped_extractions": len(self._graph_indexer.skipped_chunk_ids),
                "build": self.graph_status()["state"],
            }
        return info

    def _resolve_chat_llm(
        self,
        chat_llm: Optional[Union[str, BaseLLMClient, Any]],
        api_key: Optional[str],
        model_name: Optional[str],
        base_url: Optional[str],
        timeout: float,
        max_retries: int,
    ):
        """Return (client, provider_name). Never silently falls back to a mock model.

        Accepts a provider name ("openrouter"), "provider/model" ("openrouter/deepseek/deepseek-chat"),
        an API key (provider detected from its prefix), a BaseLLMClient, or a LangChain chat model.
        With nothing configured, the first provider whose API key env var is set is used.
        """

        if isinstance(chat_llm, BaseLLMClient):
            provider = "mock" if isinstance(unwrap_llm(chat_llm), MockLLMClient) else None
            return chat_llm, provider
        if chat_llm is not None and not isinstance(chat_llm, str):
            return LLMProviderDetector.detect(value=chat_llm), None

        provider: Optional[str] = None
        text = chat_llm.strip() if isinstance(chat_llm, str) else ""
        if text.lower() == "mock":
            return MockLLMClient(), "mock"
        if text:
            provider, model_from_string = parse_model_string(text)
            if provider is None:
                provider = detect_provider_from_key(text)
                if provider is None:
                    raise ValueError(
                        f"Unknown chat_llm '{text}'. Use a provider name, 'provider/model', or a BaseLLMClient. "
                        f"Providers:\n{provider_table()}"
                    )
                api_key = api_key or text
            elif model_from_string and not model_name:
                model_name = model_from_string
        elif api_key:
            provider = detect_provider_from_key(api_key)
            if provider is None:
                raise ValueError(
                    "Could not detect the provider from the API key; also pass chat_llm='<provider>' "
                    f"(e.g. 'deepseek', 'openrouter'). Providers:\n{provider_table()}"
                )
        elif self._offline:
            logger.warning("offline mode: using MockLLMClient (answers are templated, not generated)")
            return MockLLMClient(), "mock"
        else:
            provider = detect_provider_from_env()
            if provider is None:
                raise ValueError(
                    "No chat model configured. Pass llm='openrouter/<model>' (or chat_llm='openai', ...), set a "
                    "provider API key environment variable, pass a BaseLLMClient / LangChain chat model, "
                    f"or use offline=True (or {OFFLINE_ENV_VAR}=1) for the offline mock.\nProviders:\n{provider_table()}"
                )
            logger.info("No chat_llm given; using '%s' (API key found in environment)", provider)

        spec = get_provider(provider)
        key = spec.resolve_api_key(api_key)
        if spec.requires_key and not key:
            raise ValueError(
                f"chat_llm='{spec.name}' requires chat_api_key (or llm_key) or one of environment variables: "
                f"{list(spec.env_keys)}"
            )
        from raglib.llm.clients import build_chat_client

        client = build_chat_client(
            provider=spec.name,
            api_key=key,
            model_name=model_name,
            base_url=base_url,
            temperature=getattr(self, "_llm_temperature", 0.0),
            max_tokens=getattr(self, "_llm_max_tokens", None),
            timeout=timeout,
            max_retries=max_retries,
        )
        rpm = self._effective_rpm(spec.name, client.model_name)
        if rpm:
            client.rate_limiter = shared_limiter(spec.name, key, rpm)
        return client, spec.name

    def _effective_rpm(self, provider: str, model_name: Optional[str]) -> Optional[float]:
        """Explicit ``llm_rpm`` wins; OpenRouter free models default to their 20 requests/minute cap."""

        explicit = getattr(self, "_llm_rpm", None)
        if explicit is not None:
            return explicit or None
        model = (model_name or "").lower()
        if provider == "openrouter" and (model.endswith(":free") or model == "openrouter/free"):
            return OPENROUTER_FREE_RPM
        return None

    def _resolve_embedding(
        self,
        embedding_llm: Optional[Union[str, BaseEmbedding]],
        api_key: Optional[str],
        chat_provider: Optional[str],
        model_name: Optional[str],
        base_url: Optional[str],
        chat_api_key: Optional[str] = None,
    ) -> Optional[BaseEmbedding]:
        """Resolve the embedding model; ``None`` means BM25-only retrieval. Never uses hash vectors silently.

        ``api_key`` is an explicit embedding key; ``chat_api_key`` is reused only when the
        embedding provider is the same as the chat provider.
        """

        if isinstance(embedding_llm, BaseEmbedding):
            return embedding_llm if isinstance(embedding_llm, CachedEmbedding) else CachedEmbedding(embedding_llm)

        if embedding_llm is None:
            chat_spec = PROVIDERS.get(chat_provider or "")
            if chat_provider == "mock":
                provider = "mock"
            elif chat_spec is not None and chat_spec.default_embedding_model:
                provider = chat_spec.name
            elif self._can_import("sentence_transformers"):
                logger.info("No embedding_llm given; using local HuggingFace embeddings")
                provider = "huggingface"
            else:
                logger.warning(
                    "No embedding model available for chat provider '%s'; using BM25 keyword retrieval only. "
                    "Pass embedding_llm='openai' | 'mistral' | 'ollama' | 'huggingface' | 'provider/model' "
                    "for semantic search.",
                    chat_provider,
                )
                return None
        else:
            provider = embedding_llm.strip()
            for separator in ("/", ":"):
                head, _, tail = provider.partition(separator)
                if tail and (head.lower() in SUPPORTED_EMBEDDING_PROVIDERS or normalize_provider(head)):
                    provider, model_name = head, model_name or tail
                    break
            provider = provider.lower()

        provider = normalize_provider(provider) or provider
        if provider in {"none", "bm25", "sparse"}:
            return None
        if provider in {"mock", "huggingface", "free", "local"}:
            return CachedEmbedding(EmbeddingFactory.build(provider=provider, model_name=model_name, base_url=base_url))
        if provider in PROVIDERS and PROVIDERS[provider].custom and PROVIDERS[provider].protocol == "anthropic":
            raise ValueError(
                f"Provider '{provider}' uses the Anthropic API, which has no embeddings endpoint; "
                "pass embedding_llm='huggingface' | 'openai' | 'ollama' | ... or 'none'"
            )
        if provider not in PROVIDERS or provider in {"anthropic", "groq"}:
            raise ValueError(
                f"Unknown embedding provider '{provider}'. Supported: "
                f"{sorted((SUPPORTED_EMBEDDING_PROVIDERS | set(PROVIDERS) | {'none'}) - {'groq', 'anthropic'})}"
            )

        spec = PROVIDERS[provider]
        key = spec.resolve_api_key(api_key or (chat_api_key if provider == chat_provider else None))
        if spec.requires_key and not key:
            raise ValueError(
                f"Embedding provider '{provider}' requires embedding_api_key (or llm_key/chat_api_key) "
                f"or a configured environment variable: {list(spec.env_keys)}"
            )
        resolved_model = model_name or spec.default_embedding_model
        rpm = self._effective_rpm(provider, resolved_model)
        limiter = shared_limiter(provider, key, rpm) if rpm else None
        if spec.custom:
            if not resolved_model:
                raise ValueError(
                    f"Custom provider '{provider}' has no embedding_model; set it in register_provider()/the "
                    f"providers file, or pass embedding_llm='{provider}/<model>'"
                )
            embedding = EmbeddingFactory.build(
                provider="google" if spec.protocol == "gemini" else "openai_compatible",
                api_key=key,
                model_name=resolved_model,
                base_url=spec.resolve_base_url(base_url),
                headers=spec.resolve_headers(),
                input_types=spec.embedding_input_types,
                extra_params=spec.embedding_params,
            )
        elif provider in {"openai", "google", "ollama"}:
            embedding = EmbeddingFactory.build(provider=provider, api_key=key, model_name=resolved_model, base_url=base_url)
        else:
            if not resolved_model:
                raise ValueError(
                    f"Provider '{provider}' has no default embedding model; pass embedding_model=... "
                    "or embedding_llm='provider/model'"
                )
            embedding = EmbeddingFactory.build(
                provider="azure" if provider == "azure" else "openai_compatible",
                api_key=key,
                model_name=resolved_model,
                base_url=spec.resolve_base_url(base_url),
                input_types=spec.embedding_input_types,
                extra_params=spec.embedding_params,
            )
        return CachedEmbedding(embedding, max_retries=getattr(self, "_llm_max_retries", 2), rate_limiter=limiter)

    def _resolve_vision(
        self,
        vision_llm: Optional[Union[str, BaseVisionClient]],
        api_key: Optional[str],
        chat_provider: Optional[str],
        model_name: Optional[str],
        base_url: Optional[str],
        explicit_key: Optional[str] = None,
        chat_api_key: Optional[str] = None,
    ) -> Optional[BaseVisionClient]:
        """Resolve OCR/vision for scanned pages; None disables it (pages are skipped with a warning).

        ``explicit_key`` is vision_api_key / llm_key; ``chat_api_key`` is only reused for a custom
        provider when it is the chat provider too (a key is never sent to a different host).
        """

        if isinstance(vision_llm, BaseVisionClient):
            return vision_llm
        if vision_llm is None:
            chat_spec = PROVIDERS.get(chat_provider or "")
            chat_model = getattr(unwrap_llm(getattr(self, "_llm_raw", None)), "model_name", None)
            model_spec = chat_spec.model(chat_model) if chat_spec is not None and chat_spec.custom else None
            if model_spec is not None and model_spec.image:
                # the custom chat model accepts images: reuse it (same provider, same key) for OCR
                return self._chat_vision(chat_spec, model_name or chat_model, explicit_key, chat_api_key, base_url)
            if chat_provider in {"openai", "anthropic", "google"} and api_key:
                provider = chat_provider
            elif chat_spec is not None and chat_provider in PROVIDER_DEFAULT_VISION_MODEL and api_key:
                return self._chat_vision(chat_spec, model_name, explicit_key, chat_api_key, base_url)  # e.g. nvidia
            else:
                return None
        else:
            head, _, tail = vision_llm.strip().partition("/")
            spec = PROVIDERS.get(normalize_provider(head) or "")
            provider = head.lower()
            model_name = model_name or tail or None
            if spec is not None and (spec.custom or (spec.supports_vision and provider not in SUPPORTED_VISION_PROVIDERS)):
                same = spec.name == chat_provider
                return self._chat_vision(spec, model_name, explicit_key, chat_api_key if same else None, base_url)
        if provider not in SUPPORTED_VISION_PROVIDERS:
            raise ValueError(f"Unknown vision provider '{provider}'. Supported: {sorted(SUPPORTED_VISION_PROVIDERS)}")
        return VisionFactory.build(
            provider=provider,
            api_key=api_key if provider != "mock" else None,
            model_name=model_name,
            base_url=base_url,
        )

    def _chat_vision(
        self,
        spec: Any,
        model: Optional[str],
        explicit_key: Optional[str],
        chat_api_key: Optional[str],
        base_url: Optional[str],
    ) -> BaseVisionClient:
        """OCR through a chat client: custom providers' image models, or built-ins such as NVIDIA."""

        from raglib.llm.clients import build_chat_client
        from raglib.vision.chat_vision import ChatVisionClient

        image_models = [item.id for item in spec.models if item.image]
        model = model or (image_models[0] if image_models else None) or PROVIDER_DEFAULT_VISION_MODEL.get(spec.name)
        if not model:
            hint = "declares no model with image=True" if spec.custom else "has no default vision model"
            raise ValueError(f"Provider '{spec.name}' {hint}; pass vision_llm='{spec.name}/<model>'")
        declared = spec.model(model)
        if declared is not None and not declared.image:
            raise ValueError(
                f"Model '{model}' of provider '{spec.name}' is not marked image=True, so it cannot read scanned pages"
                + (f"; image models: {image_models}" if image_models else "")
            )
        client = build_chat_client(
            spec.name,
            api_key=explicit_key or chat_api_key,
            model_name=model,
            base_url=base_url,
            max_tokens=8192,  # room for reasoning tokens before the transcription
            timeout=getattr(self, "_llm_timeout", 60.0),
            max_retries=getattr(self, "_llm_max_retries", 2),
        )
        return ChatVisionClient(client)

    def _resolve_web_provider(
        self,
        provider_name: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        cse_id: Optional[str],
        provider_kwargs: Optional[Dict[str, Any]],
    ) -> Optional[BaseSearchProvider]:
        """Build the web provider. Web search is OFF unless configured (or required by rag_type)."""

        if provider_name is None:
            if self._rag_type not in WEB_REQUIRED_RAG_TYPES:
                return None
            logger.info("rag_type='%s' needs web search; defaulting to DuckDuckGo", self._rag_type)
            provider_name = "duckduckgo"
        name = normalize_provider_name(provider_name)
        if name == "local":
            if provider_kwargs:
                raise ValueError("web_search_provider='local' does not accept web_search_provider_kwargs")
            return _LocalSearchProvider(documents_fn=lambda: self._retriever.documents)
        return build_search_provider(
            provider_name=name,
            api_key=api_key,
            base_url=base_url,
            cse_id=cse_id,
            provider_kwargs=provider_kwargs,
        )

    def _validate_web_provider_connection(self, provider: BaseSearchProvider) -> None:
        """Perform a lightweight web provider credential check."""

        try:
            provider.search(query="raglib api key validation", num_results=1)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"web_search_provider='{provider.name}' failed validation. Check internet access and web_search_api_key."
            ) from exc

    def _resolve_graph_llm(
        self,
        graph_llm: Optional[Union[str, BaseLLMClient, Any]],
        chat_provider: Optional[str],
        chat_key: Optional[str],
        llm_cache: bool,
        max_tokens: Optional[int] = 4096,
    ) -> Optional[BaseLLMClient]:
        """The model that builds the knowledge graph (None: use the answer model).

        ``max_tokens`` caps each extraction reply: small models in JSON mode sometimes loop
        (repeating entities or whitespace) and would otherwise hold a local GPU for minutes.
        """

        if graph_llm is None:
            return None
        same_provider = False
        if isinstance(graph_llm, str):
            provider, _ = parse_model_string(graph_llm)
            same_provider = provider is not None and provider == chat_provider
        saved_max_tokens = getattr(self, "_llm_max_tokens", None)
        self._llm_max_tokens = max_tokens  # applies to models built from a provider string
        try:
            client, _ = self._resolve_chat_llm(
                chat_llm=graph_llm,
                api_key=chat_key if same_provider else None,
                model_name=None,
                base_url=None,
                timeout=self._llm_timeout,
                max_retries=self._llm_max_retries,
            )
        finally:
            self._llm_max_tokens = saved_max_tokens
        logger.info("GraphRAG indexing model: %s", getattr(unwrap_llm(client), "model_name", type(client).__name__))
        return CachedLLMClient(client) if llm_cache else client

    def _build_graph(self, graph_store: Optional[Any], graph_kwargs: Dict[str, Any]) -> None:
        """Create the GraphRAG indexer and retriever."""

        from raglib.graphrag import (
            GlobalSearch,
            GraphIndexer,
            GraphRetriever,
            LocalSearch,
            build_graph_store,
        )

        unknown = set(graph_kwargs) - GRAPH_INDEXER_OPTIONS - GRAPH_LOCAL_OPTIONS - GRAPH_GLOBAL_OPTIONS - {
            "store_options", "global_level", "default_mode", "extraction_max_tokens", "build_batch_size",
        }
        if unknown:
            raise ValueError(f"Unknown graph_kwargs: {sorted(unknown)}")
        store = build_graph_store(graph_store, graph_kwargs.get("store_options"))
        graph_llm = getattr(self, "_graph_llm", None) or self._llm
        self._graph_indexer = GraphIndexer(
            llm_client=graph_llm,
            embedding=self._embedding,
            store=store,
            **{key: graph_kwargs[key] for key in GRAPH_INDEXER_OPTIONS if key in graph_kwargs},
        )
        if self._graph_build == "background":
            from raglib.graphrag.background import BackgroundGraphBuilder

            self._graph_builder = BackgroundGraphBuilder(
                self._graph_indexer, self._retriever.get, batch_size=int(graph_kwargs.get("build_batch_size", 64))
            )
        local = LocalSearch(
            indexer=self._graph_indexer,
            chunk_lookup=self._retriever.get,
            **{key: graph_kwargs[key] for key in GRAPH_LOCAL_OPTIONS if key in graph_kwargs},
        )
        global_search = GlobalSearch(
            indexer=self._graph_indexer,
            llm_client=graph_llm,
            chunk_lookup=self._retriever.get,
            level=int(graph_kwargs.get("global_level", 0)),
            max_workers=int(graph_kwargs.get("max_workers", 8)),
            **{key: graph_kwargs[key] for key in GRAPH_GLOBAL_OPTIONS if key in graph_kwargs},
        )
        self._graph_retriever = GraphRetriever(
            local=local,
            global_search=global_search,
            default_mode=str(graph_kwargs.get("default_mode", "auto")),
            top_k=self._top_k,
        )

    def _build_orchestrator(self, rag_type: str, kwargs: Dict[str, Any]) -> Any:
        """Import and instantiate the selected orchestrator."""

        module_path, class_name = ORCHESTRATORS[rag_type].rsplit(".", 1)
        rag_class = getattr(importlib.import_module(module_path), class_name)
        components = dict(self._components)
        components.update(kwargs)
        return rag_class(**components)

    def _index_fingerprint(self) -> Dict[str, Any]:
        """Describe how the index was built (embedding model + vector size), stored by ``save``."""

        info: Dict[str, Any] = {"embedding_class": None, "embedding_model": None, "dimension": None}
        if self._embedding is not None:
            info.update(self._embedding.fingerprint())
            try:
                info["dimension"] = int(self._embedding.dimension) or None  # 0 = not known until first call
            except Exception:  # noqa: BLE001 - custom embeddings may not know their size
                info["dimension"] = None
        store = self._retriever.vector_store
        if isinstance(store, InMemoryVectorStore) and store.dimension:
            info["dimension"] = store.dimension
        return info

    def _purge_caches(self) -> None:
        """Deleted text must not survive in caches (LLM responses, embeddings) - privacy requirement."""

        if isinstance(self._llm, CachedLLMClient):
            self._llm.clear()
        if isinstance(self._embedding, CachedEmbedding):
            self._embedding.clear()

    def _known_parents_for(self, source: SourceInput) -> set:
        """Match a source (document id or file path, or a list of them) against indexed documents.

        This lets ``delete()`` work even after the file was removed from disk.
        """

        items = source if isinstance(source, (list, tuple, set)) else [source]
        wanted = {os.fspath(item) for item in items if isinstance(item, (str, os.PathLike))}
        wanted |= {os.path.abspath(item) for item in wanted}
        matches = {parent_id for parent_id in self._parents if parent_id in wanted}
        for document in self._retriever.documents:
            file_path = document.metadata.get("file")
            if file_path and (file_path in wanted or os.path.abspath(str(file_path)) in wanted):
                matches.add(str(document.metadata.get("parent_id", document.id)))
        return matches

    def _delete_chunk_ids(self, chunk_ids: Sequence[str]) -> int:
        """Delete chunks from the retriever and graph."""

        removed = self._retriever.delete_documents(chunk_ids)
        if self._graph_builder is not None:
            self._graph_builder.discard(chunk_ids)
        if self._graph_indexer is not None:
            self._graph_indexer.remove_chunks(chunk_ids)
        return removed

    def _add_to_graph(self, chunks: List[Document]) -> None:
        """Extract chunks into the graph now, or queue them for the background builder."""

        if self._graph_builder is not None:
            self._graph_builder.submit(chunks)
        else:
            self._graph_indexer.add_chunks(chunks)

    def graph_status(self) -> Dict[str, Any]:
        """Progress of the knowledge-graph build.

        ``state`` is ``"building"`` while chunks are queued, extracted or communities are being
        summarized (``graph_build="background"``), ``"ready"`` when the graph is complete, and
        ``"empty"`` before any chunk was indexed. Queries work in every state.
        """

        if self._graph_indexer is None:
            raise ValueError("Graph is not enabled (use rag_type='graph' or enable_graph=True)")
        if self._graph_builder is not None:
            return self._graph_builder.status()
        indexer = self._graph_indexer
        return {
            "state": "ready" if indexer.indexed_chunk_ids else "empty",
            "phase": "idle",
            "pending_chunks": 0,
            "failed_extractions": len(indexer.failed_chunk_ids),
            "skipped_extractions": len(indexer.skipped_chunk_ids),
        }

    def wait_for_graph(self, timeout: Optional[float] = None) -> bool:
        """Block until a background graph build has finished (True), or ``timeout`` seconds passed (False)."""

        if self._graph_builder is None:
            return True
        return self._graph_builder.wait(timeout)

    def _normalize_rag_type(self, rag_type: str) -> str:
        """Normalize rag_type aliases to canonical orchestrator names."""

        normalized = (rag_type or DEFAULT_RAG_TYPE).strip().lower()
        if normalized in RAG_TYPE_ALIASES:
            return RAG_TYPE_ALIASES[normalized]
        raise ValueError(f"Unknown rag_type '{rag_type}'. Supported: {sorted(set(RAG_TYPE_ALIASES.values()))}")

    @staticmethod
    def _can_import(module: str) -> bool:
        """Return True if a module can be imported."""

        import importlib.util

        return importlib.util.find_spec(module) is not None

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

    def _save_result(self, question: str, result: GenerationResult) -> None:
        """Persist query output as JSON in output_dir."""

        output_path = Path(self._output_dir or ".")
        output_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rag_type": self._rag_type,
            "question": question,
            "answer": result.answer,
            "citations": [asdict(citation) for citation in result.citations],
            "reasoning_trace": result.reasoning_trace,
            "usage": result.usage,
            "sources": [
                {"id": document.id, "score": document.score, "source": document.source, "metadata": document.metadata}
                for document in result.sources
            ],
        }
        target = output_path / f"result_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _atomic_write(target: Path, text: str) -> None:
    """Write ``text`` to ``target`` via a temp file + rename (never leaves a partial file)."""

    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def unwrap_embedding(embedding: Optional[BaseEmbedding]) -> Optional[BaseEmbedding]:
    """Return the innermost embedding behind caching wrappers."""

    while isinstance(embedding, CachedEmbedding):
        embedding = embedding.inner
    return embedding


def _enable_prompt_caching(client: Any) -> None:
    """Turn on Anthropic prompt caching on ``client`` and its fallbacks (no-op for other providers)."""

    from raglib.llm.clients import AnthropicClient

    seen = 0
    for candidate in [client, *getattr(client, "clients", [])]:
        inner = unwrap_llm(candidate)
        if isinstance(inner, AnthropicClient):
            inner.prompt_caching = True
            seen += 1
    if not seen:
        logger.info("prompt_caching=True has no effect: the chat model is not an Anthropic model")


def _stamp_version(chunks: List[Document], version: str) -> None:
    """Write ``document_version`` into each chunk's metadata (not part of the chunk id or ACL signature)."""

    for chunk in chunks:
        chunk.metadata["document_version"] = version
