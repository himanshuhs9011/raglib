"""High-level facade: one ``RAG`` object that wires models, ingestion, retrieval and strategies."""

from __future__ import annotations

import importlib
import threading
import json
import logging
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

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
from raglib.core.text import token_set
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.embedding.embedding_factory import EmbeddingFactory
from raglib.llm.base_client import BaseLLMClient, CachedLLMClient, FallbackLLMClient, unwrap_llm
from raglib.llm.mock_client import MockLLMClient
from raglib.llm.resilience import shared_limiter
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
from raglib.loaders.chunk_splitter import ChunkSplitter
from raglib.loaders.document_loader import DocumentLoader, SourceInput
from raglib.providers.base_provider import BaseSearchProvider
from raglib.providers.factory import build_search_provider, normalize_provider_name
from raglib.schemas import Document, GenerationResult, TraceEvent
from raglib.vectorstores.base_store import BaseVectorStore, MetadataFilter, VectorRecord
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
}
GRAPH_LOCAL_OPTIONS = {"top_k_entities", "hops", "max_neighbors", "max_relations", "max_reports"}
GRAPH_GLOBAL_OPTIONS = {"top_communities", "max_points"}


class _LocalSearchProvider(BaseSearchProvider):
    """Offline web-style provider over the loaded corpus (for tests and demos only)."""

    PROVIDER_NAME = "local_web"

    def __init__(self, documents_fn: Callable[[], List[Document]]):
        """Initialize with a callable returning the current corpus."""

        self._documents_fn = documents_fn

    @property
    def name(self) -> str:
        """Return provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Lexical overlap search over the corpus."""

        query_terms = token_set(query, remove_stopwords=True) or token_set(query)
        if not query_terms:
            return []
        scored: List[Document] = []
        for document in self._documents_fn():
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
        tools: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[TraceEvent], None]] = None,
        llm_cache: bool = True,
        llm_timeout: float = 60.0,
        llm_max_retries: int = 2,
        llm_rpm: Optional[float] = None,
        llm_fallbacks: Optional[Sequence[str]] = None,
        llm: Optional[Union[str, BaseLLMClient, Any]] = None,
        allowed_dirs: Optional[Sequence[str]] = None,
        allow_private_urls: bool = False,
        **rag_type_kwargs: Any,
    ):
        """Build the full RAG stack. See README for every option.

        ``llm`` is shorthand for ``chat_llm`` and accepts ``"provider/model"``,
        e.g. ``RAG("docs/", llm="openrouter/google/gemini-2.5-flash")``.

        ``llm_rpm`` caps requests per minute to the provider (chat + embeddings share the budget;
        OpenRouter ``:free`` models default to 20, ``0`` disables). ``llm_fallbacks`` lists
        ``"provider/model"`` strings tried in order when the primary model fails.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if llm is not None:
            if chat_llm is not None:
                raise ValueError("Pass either llm=... or chat_llm=..., not both")
            chat_llm = llm
        self._rag_type = self._normalize_rag_type(rag_type)
        self._output_dir = output_dir
        self._offline = bool(offline) if offline is not None else _env_flag(OFFLINE_ENV_VAR)
        self._top_k = top_k

        effective_chat_key = chat_api_key if chat_api_key is not None else llm_key
        self._llm_rpm = llm_rpm
        self._llm_max_retries = llm_max_retries
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
        self._llm: BaseLLMClient = CachedLLMClient(self._llm_raw) if llm_cache else self._llm_raw
        self._chat_provider = chat_provider

        self._embedding = self._resolve_embedding(
            embedding_llm=embedding_llm,
            api_key=embedding_api_key,
            chat_api_key=self._first_non_empty(chat_api_key, llm_key),
            chat_provider=chat_provider,
            model_name=embedding_model,
            base_url=embedding_base_url,
        )
        self._vision = self._resolve_vision(
            vision_llm=vision_llm,
            api_key=self._first_non_empty(vision_api_key, llm_key, chat_api_key),
            chat_provider=chat_provider,
            model_name=vision_model,
            base_url=vision_base_url,
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
        )
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

        self._graph_indexer = None
        self._graph_retriever = None
        if self._rag_type == "graph" or enable_graph:
            self._build_graph(graph_store=graph_store, graph_kwargs=dict(graph_kwargs or {}))

        self._evaluator = Evaluator()
        self._reranker = Reranker(
            top_k=top_k,
            backend=reranker,
            model_name=reranker_model,
            llm_client=self._llm if reranker == "llm" else None,
            api_key=reranker_api_key,
        )
        self._conversations = ConversationStore(max_turns=max_history_turns, max_tokens=max_history_tokens)
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

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> GenerationResult:
        """Answer one question. ``session_id`` isolates conversation memory per user."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        result = self._rag.run(question.strip(), session_id=session_id, filters=filters)
        if self._output_dir:
            self._save_result(question.strip(), result)
        return result

    async def aquery(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> GenerationResult:
        """Async ``query`` (runs the pipeline in a worker thread)."""

        import asyncio

        return await asyncio.to_thread(self.query, question, session_id, filters)

    def stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> Iterator[str]:
        """Stream answer tokens: ``for token in rag.stream(q): print(token, end="")``."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        result = yield from self._rag.stream(question.strip(), session_id=session_id, filters=filters)
        if self._output_dir and result is not None:
            self._save_result(question.strip(), result)
        return result

    def chat(self, session_id: str = "interactive") -> None:
        """Open an interactive terminal Q&A loop with conversation memory."""

        from raglib.session.interactive import InteractiveSession

        self._rag.use_memory = True
        InteractiveSession(rag=self, session_id=session_id).start()

    def add(
        self,
        source: SourceInput,
        batch_size: int = 256,
        on_progress: Optional[Callable[[Dict[str, int]], None]] = None,
    ) -> int:
        """Add or update sources. Unchanged documents are skipped; changed ones are replaced.

        Large inputs are processed in batches of ~``batch_size`` chunks: each batch is embedded
        and indexed before the next file is read, so memory stays bounded, progress is reported
        via ``on_progress({"files": .., "chunks_indexed": .., "skipped": ..})``, and if the run
        fails half-way, already-indexed files are kept (re-running skips them).
        Returns the number of new chunks indexed.
        """

        stats = {"files": 0, "chunks_indexed": 0, "skipped": 0}
        pending: List[Document] = []

        def flush() -> None:
            if pending:
                stats["chunks_indexed"] += self._index_chunks(pending)
                pending.clear()
            if on_progress is not None:
                on_progress(dict(stats))

        for loaded in self._loader.iter_load(source):
            stats["files"] += 1
            if not loaded:
                stats["skipped"] += 1
                continue
            pending.extend(self._splitter.split(loaded))
            if len(pending) >= batch_size:
                flush()
        flush()
        logger.info(
            "Ingested %d new chunks from %d sources (corpus=%d)",
            stats["chunks_indexed"],
            stats["files"],
            len(self._retriever.documents),
        )
        return stats["chunks_indexed"]

    def _index_chunks(self, chunks: List[Document]) -> int:
        """Upsert one batch of chunks (per-document replace semantics)."""

        with self._ingest_lock:
            by_parent: Dict[str, List[Document]] = {}
            for chunk in chunks:
                by_parent.setdefault(str(chunk.metadata.get("parent_id", chunk.id)), []).append(chunk)
            to_add: List[Document] = []
            changed: Dict[str, List[str]] = {}
            for parent_id, parent_chunks in by_parent.items():
                new_ids = [chunk.id for chunk in parent_chunks]
                old_ids = self._parents.get(parent_id)
                if old_ids == new_ids:
                    continue
                if old_ids:
                    self._delete_chunk_ids(old_ids)
                    self._parents.pop(parent_id, None)
                    logger.info("Document %s changed; replaced %d old chunks", parent_id, len(old_ids))
                changed[parent_id] = new_ids
                to_add.extend(parent_chunks)
            added = self._retriever.add_documents(to_add)
            if self._graph_indexer is not None and to_add:
                self._graph_indexer.add_chunks(to_add)  # skips chunks it already extracted
            # Record documents only after they are really indexed, so a failed batch is retried next run.
            self._parents.update(changed)
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
            removed = self._delete_chunk_ids(chunk_ids)
            if removed:
                self._purge_caches()
            return removed

    def reset_memory(self, session_id: Optional[str] = None) -> None:
        """Clear conversation memory for one session (or all sessions)."""

        self._conversations.clear(session_id)

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
        (path / "chunks.json").write_text(
            json.dumps({"version": 2, "parents": self._parents, "chunks": chunks}, ensure_ascii=False),
            encoding="utf-8",
        )
        store = self._retriever.vector_store
        if isinstance(store, InMemoryVectorStore):
            records = [asdict(record) for record in store.records()]
            (path / "vectors.json").write_text(json.dumps(records), encoding="utf-8")
        if self._graph_indexer is not None:
            self._graph_indexer.save(str(path / "graph"))
        logger.info("Saved RAG index to %s", path)

    def load(self, directory: str) -> "RAG":
        """Load an index written by ``save`` (embeddings are reused, not recomputed)."""

        path = Path(directory)
        payload = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        store = self._retriever.vector_store
        vectors_path = path / "vectors.json"
        if isinstance(store, InMemoryVectorStore) and vectors_path.exists():
            store.upsert([VectorRecord(**record) for record in json.loads(vectors_path.read_text(encoding="utf-8"))])
        documents = [Document(**item) for item in payload.get("chunks", [])]
        self._retriever.add_documents(documents)
        self._parents.update({key: list(value) for key, value in payload.get("parents", {}).items()})
        if self._graph_indexer is not None and (path / "graph").exists():
            self._graph_indexer.load(str(path / "graph"))
            missing = [doc for doc in documents if doc.id not in self._graph_indexer.indexed_chunk_ids]
            if missing:
                self._graph_indexer.add_chunks(missing)
        elif self._graph_indexer is not None:
            self._graph_indexer.add_chunks(documents)
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
        }
        if self._graph_indexer is not None:
            info["graph"] = self._graph_indexer.store.stats()
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
        from raglib.llm.langchain_adapter import LangChainAdapter

        client = LangChainAdapter.build(
            provider=spec.name,
            api_key=key,
            model_name=model_name,
            base_url=base_url,
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
            elif self._can_import("sentence_transformers") and self._can_import("langchain_huggingface"):
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
        if provider in {"openai", "google", "ollama"}:
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
            )
        return CachedEmbedding(embedding, max_retries=getattr(self, "_llm_max_retries", 2), rate_limiter=limiter)

    def _resolve_vision(
        self,
        vision_llm: Optional[Union[str, BaseVisionClient]],
        api_key: Optional[str],
        chat_provider: Optional[str],
        model_name: Optional[str],
        base_url: Optional[str],
    ) -> Optional[BaseVisionClient]:
        """Resolve OCR/vision for scanned pages; None disables it (pages are skipped with a warning)."""

        if isinstance(vision_llm, BaseVisionClient):
            return vision_llm
        if vision_llm is None:
            if chat_provider in {"openai", "anthropic", "google"} and api_key:
                provider = chat_provider
            else:
                return None
        else:
            provider = vision_llm.strip().lower()
        if provider not in SUPPORTED_VISION_PROVIDERS:
            raise ValueError(f"Unknown vision provider '{provider}'. Supported: {sorted(SUPPORTED_VISION_PROVIDERS)}")
        return VisionFactory.build(
            provider=provider,
            api_key=api_key if provider != "mock" else None,
            model_name=model_name,
            base_url=base_url,
        )

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

    def _build_graph(self, graph_store: Optional[Any], graph_kwargs: Dict[str, Any]) -> None:
        """Create the GraphRAG indexer and retriever."""

        from raglib.graphrag import GlobalSearch, GraphIndexer, GraphRetriever, LocalSearch, build_graph_store

        unknown = set(graph_kwargs) - GRAPH_INDEXER_OPTIONS - GRAPH_LOCAL_OPTIONS - GRAPH_GLOBAL_OPTIONS - {
            "store_options", "global_level", "default_mode",
        }
        if unknown:
            raise ValueError(f"Unknown graph_kwargs: {sorted(unknown)}")
        store = build_graph_store(graph_store, graph_kwargs.get("store_options"))
        self._graph_indexer = GraphIndexer(
            llm_client=self._llm,
            embedding=self._embedding,
            store=store,
            **{key: graph_kwargs[key] for key in GRAPH_INDEXER_OPTIONS if key in graph_kwargs},
        )
        local = LocalSearch(
            indexer=self._graph_indexer,
            chunk_lookup=self._retriever.get,
            **{key: graph_kwargs[key] for key in GRAPH_LOCAL_OPTIONS if key in graph_kwargs},
        )
        global_search = GlobalSearch(
            indexer=self._graph_indexer,
            llm_client=self._llm,
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
        if self._graph_indexer is not None:
            self._graph_indexer.remove_chunks(chunk_ids)
        return removed

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


def unwrap_embedding(embedding: Optional[BaseEmbedding]) -> Optional[BaseEmbedding]:
    """Return the innermost embedding behind caching wrappers."""

    while isinstance(embedding, CachedEmbedding):
        embedding = embedding.inner
    return embedding
