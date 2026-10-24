"""Graph store backends for GraphRAG."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from raglib.graphrag.store.base import BaseGraphStore
from raglib.graphrag.store.networkx_store import NetworkXGraphStore

__all__ = ["BaseGraphStore", "NetworkXGraphStore", "Neo4jGraphStore", "build_graph_store"]


def __getattr__(name: str) -> Any:
    """Import the Neo4j store lazily (its driver is an optional dependency)."""

    if name == "Neo4jGraphStore":
        from raglib.graphrag.store.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore
    raise AttributeError(name)


def build_graph_store(
    graph_store: Optional[Union[str, BaseGraphStore]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> BaseGraphStore:
    """Build a graph store from a name ('networkx' | 'neo4j') or return an instance."""

    if isinstance(graph_store, BaseGraphStore):
        return graph_store
    name = (graph_store or "networkx").strip().lower()
    kwargs = dict(options or {})
    if name in {"networkx", "nx", "memory", "in_memory"}:
        return NetworkXGraphStore(path=kwargs.pop("path", None))
    if name == "neo4j":
        from raglib.graphrag.store.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(**kwargs)
    raise ValueError(f"Unknown graph_store '{graph_store}'. Supported: 'networkx', 'neo4j' or a BaseGraphStore")
