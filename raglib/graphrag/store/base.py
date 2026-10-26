"""Graph store contract for GraphRAG backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence

from raglib.graphrag.schemas import Community, Entity, Relation

logger = logging.getLogger(__name__)


class BaseGraphStore(ABC):
    """Storage for entities, relations and communities.

    Implementations must make ``upsert_*`` idempotent and merge repeated mentions
    (union of source chunk ids, merged descriptions, summed relation weights).
    """

    @abstractmethod
    def upsert_entities(self, entities: Sequence[Entity]) -> List[str]:
        """Insert or merge entities; returns ids whose embedding text changed."""

    @abstractmethod
    def upsert_relations(self, relations: Sequence[Relation]) -> None:
        """Insert or merge relations."""

    @abstractmethod
    def remove_chunks(self, chunk_ids: Iterable[str]) -> List[str]:
        """Drop chunk references; delete orphaned entities/relations. Returns removed entity ids."""

    @abstractmethod
    def set_communities(self, communities: Sequence[Community]) -> None:
        """Replace all communities (and the entity -> community assignments)."""

    @abstractmethod
    def get_entities(self, ids: Iterable[str]) -> List[Entity]:
        """Return entities by id (missing ids are skipped)."""

    @abstractmethod
    def all_entities(self) -> List[Entity]:
        """Return every entity."""

    @abstractmethod
    def all_relations(self) -> List[Relation]:
        """Return every relation."""

    @abstractmethod
    def neighbors(self, entity_ids: Iterable[str], hops: int = 1, max_nodes: int = 50) -> Dict[str, int]:
        """Return {entity_id: hop_distance} for entities within ``hops`` (seeds have distance 0)."""

    @abstractmethod
    def relations_for(self, entity_ids: Iterable[str], limit: int = 50) -> List[Relation]:
        """Return relations touching any of the given entities, strongest first."""

    @abstractmethod
    def communities(self, level: Optional[int] = None) -> List[Community]:
        """Return communities, optionally for a single level."""

    @abstractmethod
    def degree(self, entity_id: str) -> int:
        """Return the number of relations touching an entity."""

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Return one entity or None."""

        found = self.get_entities([entity_id])
        return found[0] if found else None

    def get_community(self, community_id: str) -> Optional[Community]:
        """Return one community or None."""

        for community in self.communities():
            if community.id == community_id:
                return community
        return None

    def to_networkx(self) -> Any:
        """Build an undirected weighted networkx graph (for community detection)."""

        try:
            import networkx as nx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("GraphRAG community detection requires: pip install raglib-py[graph]") from exc
        graph = nx.Graph()
        for entity in self.all_entities():
            graph.add_node(entity.id)
        for relation in self.all_relations():
            if relation.source == relation.target:
                continue
            if graph.has_edge(relation.source, relation.target):
                graph[relation.source][relation.target]["weight"] += relation.weight
            else:
                graph.add_edge(relation.source, relation.target, weight=relation.weight)
        return graph

    def stats(self) -> Dict[str, int]:
        """Return entity/relation/community counts."""

        return {
            "entities": len(self.all_entities()),
            "relations": len(self.all_relations()),
            "communities": len(self.communities()),
        }

    def save(self, path: str) -> None:
        """Persist the graph (no-op for database-backed stores)."""

    def load(self, path: str) -> None:
        """Load a persisted graph (no-op for database-backed stores)."""
