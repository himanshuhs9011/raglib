"""In-process graph store (adjacency dicts) with JSON persistence; communities via networkx."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set

from raglib.graphrag.schemas import Community, Entity, Relation, merge_descriptions
from raglib.graphrag.store.base import BaseGraphStore

logger = logging.getLogger(__name__)


class NetworkXGraphStore(BaseGraphStore):
    """Default GraphRAG store: fast in-memory lookups, saved as one JSON file.

    Neighbour expansion is a breadth-first walk over an adjacency index, so local
    search cost depends on the neighbourhood size, not the graph size.
    """

    def __init__(self, path: Optional[str] = None):
        """Create an empty store, loading ``path`` if it exists."""

        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)  # entity id -> relation keys
        self._communities: Dict[str, Community] = {}
        self._lock = threading.RLock()
        self.path = path
        if path and os.path.exists(path):
            self.load(path)

    def upsert_entities(self, entities: Sequence[Entity]) -> List[str]:
        """Insert or merge entities; returns ids whose embedding text changed."""

        changed: List[str] = []
        with self._lock:
            for entity in entities:
                existing = self._entities.get(entity.id)
                if existing is None:
                    self._entities[entity.id] = Entity(
                        id=entity.id,
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        source_chunk_ids=list(dict.fromkeys(entity.source_chunk_ids)),
                    )
                    changed.append(entity.id)
                    continue
                before = existing.text_hash()
                existing.description = merge_descriptions(existing.description, entity.description)
                if existing.type in {"", "CONCEPT"} and entity.type not in {"", "CONCEPT"}:
                    existing.type = entity.type
                for chunk_id in entity.source_chunk_ids:
                    if chunk_id not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(chunk_id)
                if existing.text_hash() != before:
                    changed.append(entity.id)
        return changed

    def upsert_relations(self, relations: Sequence[Relation]) -> None:
        """Insert or merge relations (weights accumulate across mentions)."""

        with self._lock:
            for relation in relations:
                if relation.source not in self._entities or relation.target not in self._entities:
                    continue
                key = relation.key
                existing = self._relations.get(key)
                if existing is None:
                    stored = Relation(
                        source=relation.source,
                        target=relation.target,
                        type=relation.type,
                        description=relation.description,
                        weight=relation.weight,
                        source_chunk_ids=list(dict.fromkeys(relation.source_chunk_ids)),
                    )
                    self._relations[key] = stored
                    self._adjacency[relation.source].add(key)
                    self._adjacency[relation.target].add(key)
                    continue
                new_chunks = [c for c in relation.source_chunk_ids if c not in existing.source_chunk_ids]
                if new_chunks:
                    existing.weight += relation.weight
                    existing.source_chunk_ids.extend(new_chunks)
                existing.description = merge_descriptions(existing.description, relation.description)

    def remove_chunks(self, chunk_ids: Iterable[str]) -> List[str]:
        """Drop chunk references and delete orphaned entities/relations."""

        removed_chunks = set(chunk_ids)
        removed_entities: List[str] = []
        with self._lock:
            for key in list(self._relations):
                relation = self._relations[key]
                remaining = [c for c in relation.source_chunk_ids if c not in removed_chunks]
                if len(remaining) == len(relation.source_chunk_ids):
                    continue
                if not remaining:
                    self._drop_relation(key)
                else:
                    relation.weight = max(1.0, relation.weight * len(remaining) / len(relation.source_chunk_ids))
                    relation.source_chunk_ids = remaining
            for entity_id in list(self._entities):
                entity = self._entities[entity_id]
                remaining = [c for c in entity.source_chunk_ids if c not in removed_chunks]
                if remaining:
                    entity.source_chunk_ids = remaining
                    continue
                for key in list(self._adjacency.get(entity_id, ())):
                    self._drop_relation(key)
                self._adjacency.pop(entity_id, None)
                del self._entities[entity_id]
                removed_entities.append(entity_id)
        return removed_entities

    def _drop_relation(self, key: str) -> None:
        """Remove a relation and its adjacency entries."""

        relation = self._relations.pop(key, None)
        if relation is None:
            return
        self._adjacency[relation.source].discard(key)
        self._adjacency[relation.target].discard(key)

    def set_communities(self, communities: Sequence[Community]) -> None:
        """Replace communities and refresh entity assignments."""

        with self._lock:
            self._communities = {community.id: community for community in communities}
            for entity in self._entities.values():
                entity.community_ids = {}
            for community in communities:
                for entity_id in community.entity_ids:
                    entity = self._entities.get(entity_id)
                    if entity is not None:
                        entity.community_ids[str(community.level)] = community.id

    def get_entities(self, ids: Iterable[str]) -> List[Entity]:
        """Return entities by id."""

        with self._lock:
            return [self._entities[i] for i in ids if i in self._entities]

    def all_entities(self) -> List[Entity]:
        """Return every entity."""

        with self._lock:
            return list(self._entities.values())

    def all_relations(self) -> List[Relation]:
        """Return every relation."""

        with self._lock:
            return list(self._relations.values())

    def degree(self, entity_id: str) -> int:
        """Return the number of relations touching an entity."""

        with self._lock:
            return len(self._adjacency.get(entity_id, ()))

    def neighbors(self, entity_ids: Iterable[str], hops: int = 1, max_nodes: int = 50) -> Dict[str, int]:
        """Breadth-first expansion; strongest edges are followed first."""

        with self._lock:
            return self._neighbors_locked(entity_ids, hops, max_nodes)

    def _neighbors_locked(self, entity_ids: Iterable[str], hops: int, max_nodes: int) -> Dict[str, int]:
        """BFS; caller holds the lock."""

        distances: Dict[str, int] = {}
        frontier = [entity_id for entity_id in entity_ids if entity_id in self._entities]
        for entity_id in frontier:
            distances[entity_id] = 0
        for hop in range(1, hops + 1):
            next_frontier: List[str] = []
            for entity_id in frontier:
                edges = sorted(
                    (self._relations[key] for key in self._adjacency.get(entity_id, ())),
                    key=lambda relation: relation.weight,
                    reverse=True,
                )
                for relation in edges:
                    other = relation.target if relation.source == entity_id else relation.source
                    if other in distances:
                        continue
                    distances[other] = hop
                    next_frontier.append(other)
                    if len(distances) >= max_nodes:
                        return distances
            frontier = next_frontier
            if not frontier:
                break
        return distances

    def relations_for(self, entity_ids: Iterable[str], limit: int = 50) -> List[Relation]:
        """Return relations touching the entities, strongest first."""

        keys: Set[str] = set()
        with self._lock:
            for entity_id in entity_ids:
                keys.update(self._adjacency.get(entity_id, ()))
            relations = [self._relations[key] for key in keys if key in self._relations]
        relations.sort(key=lambda relation: relation.weight, reverse=True)
        return relations[:limit]

    def communities(self, level: Optional[int] = None) -> List[Community]:
        """Return communities, optionally for one level."""

        with self._lock:
            values = list(self._communities.values())
        if level is None:
            return values
        return [community for community in values if community.level == level]

    def get_community(self, community_id: str) -> Optional[Community]:
        """Return one community or None."""

        with self._lock:
            return self._communities.get(community_id)

    def save(self, path: Optional[str] = None) -> None:
        """Write the graph to a JSON file (atomic replace)."""

        target = path or self.path
        if not target:
            raise ValueError("NetworkXGraphStore.save requires a path")
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with self._lock:
            payload = {
                "version": 1,
                "entities": [entity.to_dict() for entity in self._entities.values()],
                "relations": [relation.to_dict() for relation in self._relations.values()],
                "communities": [community.to_dict() for community in self._communities.values()],
            }
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp, target)

    def load(self, path: Optional[str] = None) -> None:
        """Load a graph saved by ``save``."""

        source = path or self.path
        if not source:
            raise ValueError("NetworkXGraphStore.load requires a path")
        with open(source, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        with self._lock:
            self._entities = {item["id"]: Entity.from_dict(item) for item in payload.get("entities", [])}
            self._relations = {}
            self._adjacency = defaultdict(set)
            for item in payload.get("relations", []):
                relation = Relation.from_dict(item)
                self._relations[relation.key] = relation
                self._adjacency[relation.source].add(relation.key)
                self._adjacency[relation.target].add(relation.key)
            self._communities = {item["id"]: Community.from_dict(item) for item in payload.get("communities", [])}
