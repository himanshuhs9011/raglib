"""Public package exports for raglib."""

from __future__ import annotations

from raglib.access import AccessControl, FilterNotSupportedError, UserContext, access_tags
from raglib.cache import AnswerCache, TTLCache
from raglib.components import (
    BM25Index,
    ContextReducer,
    ConversationStore,
    DecisionEngine,
    Deduplicator,
    Evaluator,
    Generator,
    HybridRetriever,
    MemoryModule,
    MultiHopRetriever,
    MultiQueryRetriever,
    Planner,
    QueryCondenser,
    QueryExpander,
    QueryRewriter,
    Refiner,
    ReflectionModule,
    RelevanceGrader,
    Reranker,
    Retriever,
    RouterRetriever,
    SQLiteConversationStore,
    SQLTool,
    Tool,
    WebRetriever,
)
from raglib.core.base import BaseRAG
from raglib.costs import CostReport, ModelPrice, PriceTable
from raglib.embedding.limits import EmbeddingLimitError, EmbeddingLimitWarning
from raglib.exporters import OpenTelemetryExporter, PrometheusExporter
from raglib.facade import RAG
from raglib.graph_context import (
    ConnectedNodeRef,
    GraphContextResult,
    GraphContextWindow,
    GraphNode,
)
from raglib.latency import LatencySample, LatencyStats, benchmark
from raglib.llm.base_client import BaseLLMClient, CachedLLMClient
from raglib.llm.clients import LLMOutputTruncatedError, build_chat_client
from raglib.llm.mock_client import MockLLMClient
from raglib.llm.providers import ModelSpec, list_models, load_providers, register_provider, unregister_provider
from raglib.loaders.chunk_splitter import ChunkSplitter, EmbeddingChunkingError, IndexSettingsWarning
from raglib.quick import ask
from raglib.rag_types import (
    AdvancedRAG,
    AgenticRAG,
    CorrectiveRAG,
    GraphRAG,
    HybridRAG,
    MemoryRAG,
    MultiHopRAG,
    MultiQueryRAG,
    NaiveRAG,
    RoutingRAG,
    SelfRAG,
    ToolAugmentedRAG,
    WebRAG,
)
from raglib.routing import HeuristicTierRouter, TierDecision
from raglib.schemas import (
    Citation,
    Document,
    GenerationResult,
    QueryResult,
    RetrievalContext,
    TraceEvent,
)

__all__ = [
    "EmbeddingLimitError",
    "EmbeddingLimitWarning",
    "IndexSettingsWarning",
    "EmbeddingChunkingError",
    "AnswerCache",
    "TTLCache",
    "PriceTable",
    "ModelPrice",
    "CostReport",
    "HeuristicTierRouter",
    "TierDecision",
    "LatencyStats",
    "LatencySample",
    "benchmark",
    "PrometheusExporter",
    "OpenTelemetryExporter",
    "AccessControl",
    "UserContext",
    "FilterNotSupportedError",
    "LLMOutputTruncatedError",
    "access_tags",
    "RAG",
    "ask",
    "BaseRAG",
    "Document",
    "QueryResult",
    "RetrievalContext",
    "GenerationResult",
    "Citation",
    "TraceEvent",
    "BaseLLMClient",
    "CachedLLMClient",
    "MockLLMClient",
    "build_chat_client",
    "register_provider",
    "unregister_provider",
    "load_providers",
    "list_models",
    "ModelSpec",
    "ChunkSplitter",
    "BM25Index",
    "Retriever",
    "WebRetriever",
    "HybridRetriever",
    "RouterRetriever",
    "MultiQueryRetriever",
    "MultiHopRetriever",
    "Reranker",
    "Evaluator",
    "RelevanceGrader",
    "Refiner",
    "DecisionEngine",
    "ReflectionModule",
    "MemoryModule",
    "ConversationStore",
    "Planner",
    "QueryExpander",
    "QueryRewriter",
    "QueryCondenser",
    "Deduplicator",
    "ContextReducer",
    "Generator",
    "Tool",
    "SQLTool",
    "SQLiteConversationStore",
    "NaiveRAG",
    "AdvancedRAG",
    "CorrectiveRAG",
    "SelfRAG",
    "AgenticRAG",
    "HybridRAG",
    "MultiQueryRAG",
    "MultiHopRAG",
    "RoutingRAG",
    "MemoryRAG",
    "WebRAG",
    "ToolAugmentedRAG",
    "GraphRAG",
    "ConnectedNodeRef",
    "GraphNode",
    "GraphContextResult",
    "GraphContextWindow",
]

__version__ = "2.2.0"
