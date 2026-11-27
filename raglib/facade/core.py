"""High-level facade: one ``RAG`` object that wires models, ingestion, retrieval and strategies."""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from raglib.access import AccessControl
from raglib.cache import AnswerCache
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
    SUPPORTED_VECTOR_DBS,
)
from raglib.core.runtime import UsageMeter
from raglib.core.tracing import OnEvent
from raglib.costs import PriceTable
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.facade.chunking import ChunkingMixin
from raglib.facade.extras import ExtrasMixin, _env_flag
from raglib.facade.graph import GraphMixin
from raglib.facade.guarding import GuardingMixin
from raglib.facade.ingest import IngestMixin
from raglib.facade.permissions import PermissionsMixin
from raglib.facade.persistence import PersistenceMixin
from raglib.facade.providers import ProvidersMixin, _enable_prompt_caching
from raglib.facade.query import QueryMixin
from raglib.guards import (
    GuardConfig,
    GuardContext,
    GuardSpec,
    GuardStore,
    InMemoryGuardStore,
    ToolPolicy,
)
from raglib.guards.defaults import guard_tip, secure_defaults_options, with_secure_defaults
from raglib.llm.base_client import (
    BaseLLMClient,
    CachedLLMClient,
    FallbackLLMClient,
    GatedLLMClient,
)
from raglib.llm.providers import (
    parse_model_string,
)
from raglib.loaders.chunk_splitter import ChunkSplitter
from raglib.loaders.document_loader import DocumentLoader, SourceInput
from raglib.loaders.sections import PARENT_MODES
from raglib.presets import apply_presets
from raglib.routing import HeuristicTierRouter, TierRouter
from raglib.schemas import Document
from raglib.vectorstores.base_store import BaseVectorStore
from raglib.vision.base_vision import BaseVisionClient

logger = logging.getLogger("raglib.facade")  # the name it had before the split


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


class RAG(PersistenceMixin, GraphMixin, ChunkingMixin, ExtrasMixin, PermissionsMixin, IngestMixin, GuardingMixin, QueryMixin, ProvidersMixin):
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
        max_file_bytes: Optional[int] = 100 * 1024 * 1024,
        max_pages: Optional[int] = 5000,
        llm_temperature: float = 0.0,
        llm_max_tokens: Optional[int] = None,
        conversation_store: Optional[ConversationStore] = None,
        strict: bool = False,
        access_control: Union[bool, AccessControl, Mapping[str, Any], None] = None,
        preset: Optional[str] = None,
        rerank_candidates: Optional[int] = None,
        rerank_skip_threshold: Optional[float] = None,
        rerank_skip_margin: Optional[float] = None,
        llm_cache_ttl: Optional[float] = 24 * 3600.0,
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
        token_counter: Union[str, Callable[[str], int], None] = None,
        on_embedding_failure: str = "fallback",
        abbreviations: Optional[Sequence[str]] = None,
        semantic_buffer_size: int = 2,
        semantic_number_of_chunks: Optional[int] = None,
        sentence_splitter: Any = None,
        semantic_overlap_sentences: int = 0,
        allow_web_after_private: bool = False,
        embedding_max_tokens: Optional[int] = None,
        oversized_chunks: Optional[str] = None,
        embedding_timeout: Optional[float] = None,
        reranker_timeout: float = 10.0,
        input_guards: GuardSpec = None,
        retrieval_guards: GuardSpec = None,
        output_guards: GuardSpec = None,
        tool_policy: Union[ToolPolicy, str, None] = None,
        guard_cache_ttl: Optional[float] = 3600.0,
        rate_limit_key: Optional[Callable[[GuardContext], Optional[str]]] = None,
        guard_store: Optional[GuardStore] = None,
        stream_holdback_chars: int = 64,
        secure_defaults: Union[bool, Mapping[str, bool]] = True,
        guard_models: Optional[Mapping[str, Any]] = None,
        judge_model: Optional[Union[str, BaseLLMClient, Any]] = None,
        judge_budget: Optional[Any] = None,
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
        * ``llm_cache_ttl`` (default 24 h) / ``embedding_cache_ttl``: seconds before cached LLM responses /
          vectors expire. All caches, and their invalidation, are per process for now.
        * ``embedding_timeout``: seconds per embedding API request (models raglib builds);
          ``reranker_timeout``: seconds per Cohere rerank request (default 10).
        * ``cache_personalized=False``: LLM and answer caches are skipped for queries with a ``user``
          or caller ``filters`` (only the public tier is cached).
        * ``answer_cache``: question-level answer cache (FAQ tier); on by default with tier routing.
        * ``prompt_caching``: Anthropic ``cache_control`` on the system prompt (no-op elsewhere).
        * ``cheap_llm`` + ``tier_router``: route easy questions to a cheaper model (``raglib.routing``).
        * ``parent_retrieval="heading" | "fixed"``: search small chunks, answer from their parent
          sections of up to ``parent_chunk_size`` tokens.
        * ``chunking_strategy="embedding"``: semantic chunking with ``semantic_breakpoint`` /
          ``semantic_threshold`` / ``semantic_min_tokens``; ``on_embedding_failure="raise"`` makes a
          failed embedding call fail ingestion instead of falling back; ``semantic_buffer_size``,
          ``semantic_number_of_chunks``, ``sentence_splitter`` and ``semantic_overlap_sentences``
          tune it (see ``ChunkSplitter``).
        * ``token_counter``: how chunk sizes are measured - ``"units"`` (words; default, becomes
          ``"auto"`` in the next major version), ``"tiktoken"``, ``"auto"`` or a callable; the embedding
          cost estimate uses the same one. The chunking settings are recorded in the index; opening or
          extending an index with different ones raises an ``IndexSettingsWarning``.
        * ``prices``: a ``PriceTable``; ``result.usage`` then reports ``cost_usd`` per stage.
        * ``max_file_bytes`` (100 MB) / ``max_pages`` (5000): per-file limits checked before parsing; a
          zip-based file (DOCX, PPTX, XLSX, EPUB) may unpack to 10 x ``max_file_bytes``. A file over a
          limit, not what its extension says, or unparsable raises ``DocumentLoadError``. ``None``
          turns a limit off.

        Guardrails (``raglib.guards``; cheapest first, stop at the first block):

        * ``input_guards`` run on the question before anything else (PII masking, length, rate
          limit, budget, injection keywords); ``retrieval_guards`` on the retrieved documents
          before the answer LLM call (``NothingRelevant`` answers "couldn't find this" for free);
          ``output_guards`` on the answer (leak check, citation access, JSON schema, length).
          Each takes a list of ``Guard`` objects, guard names, or ``"default"``.
        * ``tool_policy``: a ``ToolPolicy`` for ``rag_type="agentic"`` (tools per role, calls per
          question, argument limits); ``"default"`` caps calls and argument length.
        * ``guard_cache_ttl``: seconds a cacheable guard decision is reused for the same input
          (``None`` disables the decision cache).
        * ``rate_limit_key``: ``callable(GuardContext) -> str | None`` naming the caller for rate
          limits and budgets (e.g. ``lambda c: c.request.get("api_key_id")``); default: the user id
          from ``query(..., user_id=...)`` or ``UserContext``.
        * ``guard_store``: where rate-limit / budget counters live (``GuardStore``); default in
          process memory, so each worker process counts on its own.
        * ``stream_holdback_chars``: how far (in characters) streamed text may trail the model while
          streaming-safe output guards check it (default 64, minimum 40; see ``raglib.guards.streaming``).
        * ``secure_defaults`` (on unless ``False``): protections added to every ``RAG`` that never
          change a normal answer - ``"citation_access"`` (``CitationAccessGuard`` with access control),
          ``"tool_limit"`` (agent: at most 10 tool calls per question), and flag-only
          ``"injection_flags"`` (question) / ``"secret_flags"`` (secrets in the answer), reported in
          ``result.guard_flags``. Turn one off with e.g. ``secure_defaults={"injection_flags": False}``.
          A guard of the same kind that you configure yourself replaces the default one.
        * ``guard_models``: local models for model-based guards, e.g.
          ``{"injection": "protectai/deberta-v3-base-prompt-injection-v2", "nli": ...}`` (a model id that
          is already downloaded, or a callable). Optional: without one, a guard uses your cheapest
          configured API model, or is skipped with a one-time tip. Nothing is ever downloaded for you.
        * ``judge_model``: the tier 2 LLM judge tier 1 guards escalate to (default: the cheapest
          configured chat model); ``judge_budget``: a ``JudgeBudget`` (or dict) capping judge calls /
          cost per minute and per day - past it the tier 1 verdict stands (see ``raglib.guards.judge``).
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
        self._fingerprints: Dict[str, str] = {}  # parent document id -> content/settings/tags hash
        self._pending_fingerprints: Dict[str, str] = {}

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
        # Every model call passes the guard gate, whatever the client and cache setting.
        self._llm: BaseLLMClient = GatedLLMClient(
            CachedLLMClient(self._llm_raw, ttl=llm_cache_ttl) if llm_cache else self._llm_raw
        )
        self._chat_provider = chat_provider
        self._cache_personalized = cache_personalized
        self._prices: Optional[PriceTable] = PriceTable.coerce(prices)
        self._ingest_meter = UsageMeter()

        self._embedding_timeout = embedding_timeout
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
            max_file_bytes=max_file_bytes,
            max_pages=max_pages,
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
            buffer_size=semantic_buffer_size,
            number_of_chunks=semantic_number_of_chunks,
            sentence_splitter=sentence_splitter,
            semantic_overlap_sentences=semantic_overlap_sentences,
        )
        self._chunking_fallbacks: List[Dict[str, str]] = []
        self._oversized_chunks: List[Dict[str, Any]] = []
        self._semantic_min_tokens_setting = semantic_min_tokens
        self._configure_embedding_limit(embedding_max_tokens, oversized_chunks)
        self._index_signatures: set = set()  # chunking signatures of the chunks already in this index
        self._settings_warned = False
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
        if self._web_retriever is not None and self._access is not None:
            self._web_retriever.public_access_level = self._access.levels[0]  # public web results are public
            self._web_retriever.access = self._access  # internal providers' results are checked by their tags
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
            timeout=reranker_timeout,
            max_retries=llm_max_retries,
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
            "private_document": self._private_document if self._access is not None else None,
            "allow_web_after_private": allow_web_after_private,
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
        from raglib.guards.streaming import check_holdback

        self._stream_holdback = check_holdback(stream_holdback_chars)
        self._rate_limit_key = rate_limit_key
        self._guard_store: GuardStore = guard_store or InMemoryGuardStore()
        self._secure_defaults = secure_defaults_options(secure_defaults)
        inputs, retrievals, outputs, tool_policy = with_secure_defaults(
            self._secure_defaults, access_control=self._access is not None,
            input_guards=input_guards, retrieval_guards=retrieval_guards, output_guards=output_guards,
            tool_policy=tool_policy,
        )
        self._guards: Optional[GuardConfig] = GuardConfig.build(
            inputs, retrievals, outputs, tool_policy, cache_ttl=guard_cache_ttl
        )
        if preset is None and input_guards is None and retrieval_guards is None and output_guards is None:
            guard_tip()
        self._guard_signature: Optional[str] = None
        self._guard_models: Dict[str, Any] = dict(guard_models or {})

        # Tier routing: FAQ (answer cache) -> cheap model -> full pipeline.
        self._cheap_rag: Optional[Any] = None
        self._cheap_llm: Optional[Any] = None
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
            cheap = GatedLLMClient(CachedLLMClient(cheap_raw, ttl=llm_cache_ttl) if llm_cache else cheap_raw)
            self._cheap_llm = cheap
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

        # Bound last: model-based guards pick their model (local -> cheapest API -> skipped) from
        # everything configured above, including cheap_llm; the tier 2 judge likewise.
        self._judge: Optional[Any] = None
        if self._guards is not None and any(hasattr(g, "use_judge") for g in self._guards.all()):
            from raglib.guards.judge import build_judge

            judge_client = judge_model
            if isinstance(judge_model, str):
                judge_provider = parse_model_string(judge_model)[0]
                judge_client = self._resolve_chat_llm(
                    chat_llm=judge_model,
                    api_key=effective_chat_key if judge_provider == chat_provider else None,
                    model_name=None, base_url=None, timeout=llm_timeout, max_retries=llm_max_retries,
                )[0]
            self._judge = build_judge(self, judge_client, judge_budget, cache_ttl=guard_cache_ttl)
        if self._guards is not None:
            self._guards.bind(self)
            self._guard_signature = self._guards.signature()

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


    @property
    def rag_type(self) -> str:
        """Return the canonical rag_type."""

        return self._rag_type

    @property
    def pipeline(self) -> Any:
        """Return the underlying orchestrator (BaseRAG subclass)."""

        return self._rag


    def _build_orchestrator(self, rag_type: str, kwargs: Dict[str, Any]) -> Any:
        """Import and instantiate the selected orchestrator."""

        module_path, class_name = ORCHESTRATORS[rag_type].rsplit(".", 1)
        rag_class = getattr(importlib.import_module(module_path), class_name)
        components = dict(self._components)
        components.update(kwargs)
        return rag_class(**components)


    def _normalize_rag_type(self, rag_type: str) -> str:
        """Normalize rag_type aliases to canonical orchestrator names."""

        normalized = (rag_type or DEFAULT_RAG_TYPE).strip().lower()
        if normalized in RAG_TYPE_ALIASES:
            return RAG_TYPE_ALIASES[normalized]
        raise ValueError(f"Unknown rag_type '{rag_type}'. Supported: {sorted(set(RAG_TYPE_ALIASES.values()))}")


