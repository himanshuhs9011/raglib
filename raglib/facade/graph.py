"""GraphRAG on the facade: building the graph and its LLM, adding chunks, status, waiting, rebuilding communities, retrying extraction and visualizing."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from raglib.llm.base_client import (
    BaseLLMClient,
    CachedLLMClient,
    GatedLLMClient,
    unwrap_llm,
)
from raglib.llm.providers import (
    parse_model_string,
)
from raglib.schemas import Document

logger = logging.getLogger("raglib.facade")  # one logger name for the whole facade, as before the split


GRAPH_INDEXER_OPTIONS = {
    "entity_types", "max_workers", "gleanings", "max_levels", "max_community_size",
    "community_rebuild_threshold", "resolve_with_embeddings", "resolution_threshold", "min_llm_report_size",
    "max_extraction_attempts",
}


GRAPH_LOCAL_OPTIONS = {"top_k_entities", "hops", "max_neighbors", "max_relations", "max_reports"}


GRAPH_GLOBAL_OPTIONS = {"top_communities", "max_points"}


class GraphMixin:
    """GraphRAG on the facade: building the graph and its LLM, adding chunks, status, waiting, rebuilding communities, retrying extraction and visualizing (mixin of ``RAG``)."""

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

    @property
    def graph(self) -> Any:
        """Return the GraphIndexer (None unless the graph is enabled)."""

        return self._graph_indexer

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
        return GatedLLMClient(CachedLLMClient(client) if llm_cache else client)

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
        # With access control, every summary document (entities, relations, community reports, global
        # points) inherits the strictest access tags of the chunks it was built from.
        local.access = global_search.access = self._access
        self._graph_retriever = GraphRetriever(
            local=local,
            global_search=global_search,
            default_mode=str(graph_kwargs.get("default_mode", "auto")),
            top_k=self._top_k,
        )

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
