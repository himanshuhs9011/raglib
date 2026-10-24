"""GraphRAG: answer from a knowledge graph (local entity search or global community search)."""

from __future__ import annotations

import logging
from typing import Any, List

from raglib.core.base import BaseRAG, PipelineContext
from raglib.core.fusion import reciprocal_rank_fusion
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class GraphRAG(BaseRAG):
    """Microsoft-style GraphRAG.

    * ``local`` - entity-neighbourhood search; graph chunks are fused (RRF) with normal
      hybrid chunk retrieval, plus entity/relationship/community context. No extra LLM calls.
    * ``global`` - map over the most relevant community reports, reduce in the generator.
    * ``auto`` (default) - global for corpus-wide questions ("main themes", "overall"), else local.
    """

    pipeline_name = "graph"
    uses_decision_engine = False

    def __init__(self, search_mode: str = "auto", fuse_with_chunks: bool = True, **kwargs: Any):
        """Initialize GraphRAG."""

        super().__init__(**kwargs)
        if search_mode not in {"auto", "local", "global"}:
            raise ValueError("search_mode must be 'auto', 'local' or 'global'")
        if self.graph_retriever is None:
            raise ValueError("GraphRAG requires graph_retriever (use RAG(rag_type='graph'))")
        self.search_mode = search_mode
        self.fuse_with_chunks = fuse_with_chunks

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Run local or global graph search."""

        mode = self.graph_retriever.resolve_mode(ctx.search_query, self.search_mode)
        ctx.trace.append(f"graph_mode:{mode}")
        ctx.metadata["graph_mode"] = mode
        graph_docs = self.graph_retriever.retrieve(ctx.search_query, mode=mode)
        if mode == "global":
            ctx.trace.append(f"global_points:{len(graph_docs)}")
            if graph_docs or self.retriever is None:
                return graph_docs
            ctx.trace.append("global_empty:fallback_local")
            graph_docs = self.graph_retriever.retrieve(ctx.search_query, mode="local")

        context_docs = [doc for doc in graph_docs if doc.metadata.get("graph") != "chunk"]
        graph_chunks = [doc for doc in graph_docs if doc.metadata.get("graph") == "chunk"]
        chunks = graph_chunks
        if self.fuse_with_chunks and self.retriever is not None:
            corpus = self.retrieve_corpus(ctx, ctx.search_query)
            chunks = reciprocal_rank_fusion([graph_chunks, corpus], list_names=["graph", "corpus"])
        ctx.trace.append(f"graph_context:{len(context_docs)} chunks:{len(chunks)}")
        ctx.metadata["graph_context_docs"] = context_docs
        return chunks

    def process_documents(self, ctx: PipelineContext, documents: List[Document]) -> List[Document]:
        """Rerank chunks, then put compact graph context (entities/relations/reports) first."""

        if ctx.metadata.get("graph_mode") == "global":
            return documents
        chunks = super().process_documents(ctx, documents)
        return list(ctx.metadata.get("graph_context_docs", [])) + chunks
