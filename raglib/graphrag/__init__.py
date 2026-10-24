"""GraphRAG: knowledge-graph retrieval with local (entity) and global (community) search."""

from __future__ import annotations

from raglib.graphrag.communities import CommunityReporter, detect_communities
from raglib.graphrag.extractor import DEFAULT_ENTITY_TYPES, GraphExtractor
from raglib.graphrag.indexer import GraphIndexer
from raglib.graphrag.schemas import Community, Entity, ExtractionResult, Relation, normalize_entity_key
from raglib.graphrag.search import GlobalSearch, GraphRetriever, LocalSearch
from raglib.graphrag.store import BaseGraphStore, NetworkXGraphStore, build_graph_store

__all__ = [
    "BaseGraphStore",
    "Community",
    "CommunityReporter",
    "DEFAULT_ENTITY_TYPES",
    "Entity",
    "ExtractionResult",
    "GlobalSearch",
    "GraphExtractor",
    "GraphIndexer",
    "GraphRetriever",
    "LocalSearch",
    "NetworkXGraphStore",
    "Relation",
    "build_graph_store",
    "detect_communities",
    "normalize_entity_key",
]
