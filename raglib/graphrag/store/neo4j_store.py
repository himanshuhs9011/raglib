"""Neo4j-backed GraphRAG store (``pip install raglib-py[neo4j]``)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from raglib.graphrag.schemas import Community, Entity, Relation, merge_descriptions
from raglib.graphrag.store.base import BaseGraphStore

logger = logging.getLogger(__name__)


class Neo4jGraphStore(BaseGraphStore):
    """Stores entities as ``(:RagEntity)``, relations as ``[:RAG_REL]``, communities as ``(:RagCommunity)``.

    All writes use batched ``UNWIND ... MERGE``. A ``namespace`` property isolates several
    raglib indexes inside one database.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: Optional[str] = None,
        database: Optional[str] = None,
        namespace: str = "default",
        driver: Any = None,
    ):
        """Connect to Neo4j (or use an injected driver) and ensure constraints exist."""

        if driver is None:
            try:
                from neo4j import GraphDatabase  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError("Neo4j graph store requires: pip install raglib-py[neo4j]") from exc
            driver = GraphDatabase.driver(uri, auth=(user, password) if password else None)
        self._driver = driver
        self._database = database
        self.namespace = namespace
        try:
            self._run(
                "CREATE CONSTRAINT rag_entity_key IF NOT EXISTS "
                "FOR (e:RagEntity) REQUIRE (e.namespace, e.id) IS UNIQUE"
            )
        except Exception as exc:  # noqa: BLE001 - older servers / restricted users
            logger.warning("Could not create Neo4j uniqueness constraint (%s); creating an index instead", exc)
            self._run("CREATE INDEX rag_entity_idx IF NOT EXISTS FOR (e:RagEntity) ON (e.namespace, e.id)")

    def close(self) -> None:
        """Close the driver."""

        self._driver.close()

    def _run(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        """Run a Cypher query and return records as dicts."""

        with self._driver.session(database=self._database) as session:
            return [record.data() for record in session.run(query, namespace=self.namespace, **params)]

    def upsert_entities(self, entities: Sequence[Entity]) -> List[str]:
        """Merge entities; descriptions are merged client-side to keep them bounded."""

        if not entities:
            return []
        existing = {entity.id: entity for entity in self.get_entities(entity.id for entity in entities)}
        rows = []
        changed: List[str] = []
        for entity in entities:
            current = existing.get(entity.id)
            if current is None:
                merged = entity
                changed.append(entity.id)
            else:
                before = current.text_hash()
                merged = Entity(
                    id=entity.id,
                    name=current.name,
                    type=current.type if current.type not in {"", "CONCEPT"} else entity.type,
                    description=merge_descriptions(current.description, entity.description),
                    source_chunk_ids=list(dict.fromkeys(current.source_chunk_ids + entity.source_chunk_ids)),
                    community_ids=current.community_ids,
                )
                if merged.text_hash() != before:
                    changed.append(entity.id)
            rows.append(
                {
                    "id": merged.id,
                    "name": merged.name,
                    "type": merged.type,
                    "description": merged.description,
                    "chunks": merged.source_chunk_ids,
                }
            )
        self._run(
            "UNWIND $rows AS row MERGE (e:RagEntity {namespace: $namespace, id: row.id}) "
            "SET e.name = row.name, e.type = row.type, e.description = row.description, e.chunks = row.chunks",
            rows=rows,
        )
        return changed

    def upsert_relations(self, relations: Sequence[Relation]) -> None:
        """Merge relations, accumulating weight for new source chunks."""

        if not relations:
            return
        rows = [
            {
                "key": relation.key,
                "source": relation.source,
                "target": relation.target,
                "type": relation.type,
                "description": relation.description,
                "weight": relation.weight,
                "chunks": relation.source_chunk_ids,
            }
            for relation in relations
        ]
        self._run(
            "UNWIND $rows AS row "
            "MATCH (a:RagEntity {namespace: $namespace, id: row.source}) "
            "MATCH (b:RagEntity {namespace: $namespace, id: row.target}) "
            "MERGE (a)-[r:RAG_REL {key: row.key}]->(b) "
            "ON CREATE SET r.type = row.type, r.description = row.description, r.weight = row.weight, r.chunks = row.chunks "
            "ON MATCH SET r.weight = r.weight + CASE WHEN any(c IN row.chunks WHERE NOT c IN r.chunks) THEN row.weight ELSE 0 END, "
            "r.chunks = r.chunks + [c IN row.chunks WHERE NOT c IN r.chunks]",
            rows=rows,
        )

    def remove_chunks(self, chunk_ids: Iterable[str]) -> List[str]:
        """Remove chunk references; delete orphaned relations and entities."""

        ids = list(chunk_ids)
        if not ids:
            return []
        self._run(
            "MATCH (:RagEntity {namespace: $namespace})-[r:RAG_REL]->() "
            "WHERE any(c IN r.chunks WHERE c IN $ids) "
            "SET r.chunks = [c IN r.chunks WHERE NOT c IN $ids] "
            "WITH r WHERE size(r.chunks) = 0 DELETE r",
            ids=ids,
        )
        removed = self._run(
            "MATCH (e:RagEntity {namespace: $namespace}) WHERE any(c IN e.chunks WHERE c IN $ids) "
            "SET e.chunks = [c IN e.chunks WHERE NOT c IN $ids] "
            "WITH e WHERE size(e.chunks) = 0 WITH e, e.id AS id DETACH DELETE e RETURN id",
            ids=ids,
        )
        return [row["id"] for row in removed]

    def set_communities(self, communities: Sequence[Community]) -> None:
        """Replace communities and IN_COMMUNITY assignments."""

        self._run("MATCH (c:RagCommunity {namespace: $namespace}) DETACH DELETE c")
        rows = [
            {
                "id": community.id,
                "level": community.level,
                "payload": json.dumps(community.to_dict(), ensure_ascii=False),
                "members": community.entity_ids,
            }
            for community in communities
        ]
        if rows:
            self._run(
                "UNWIND $rows AS row CREATE (c:RagCommunity {namespace: $namespace, id: row.id, level: row.level, payload: row.payload}) "
                "WITH c, row UNWIND row.members AS member "
                "MATCH (e:RagEntity {namespace: $namespace, id: member}) MERGE (e)-[:IN_COMMUNITY]->(c)",
                rows=rows,
            )

    @staticmethod
    def _entity(row: Dict[str, Any]) -> Entity:
        """Build an Entity from a Cypher row."""

        return Entity(
            id=row["id"],
            name=row.get("name") or row["id"],
            type=row.get("type") or "CONCEPT",
            description=row.get("description") or "",
            source_chunk_ids=list(row.get("chunks") or []),
            community_ids=dict(row.get("communities") or {}),
        )

    def get_entities(self, ids: Iterable[str]) -> List[Entity]:
        """Return entities by id."""

        id_list = list(ids)
        if not id_list:
            return []
        rows = self._run(
            "MATCH (e:RagEntity {namespace: $namespace}) WHERE e.id IN $ids "
            "OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(c:RagCommunity) "
            "RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description, e.chunks AS chunks, "
            "collect([toString(c.level), c.id]) AS pairs",
            ids=id_list,
        )
        entities = []
        for row in rows:
            row["communities"] = {pair[0]: pair[1] for pair in row.pop("pairs", []) if pair and pair[1]}
            entities.append(self._entity(row))
        return entities

    def all_entities(self) -> List[Entity]:
        """Return every entity."""

        rows = self._run(
            "MATCH (e:RagEntity {namespace: $namespace}) "
            "RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description, e.chunks AS chunks"
        )
        return [self._entity(row) for row in rows]

    def all_relations(self) -> List[Relation]:
        """Return every relation."""

        rows = self._run(
            "MATCH (a:RagEntity {namespace: $namespace})-[r:RAG_REL]->(b:RagEntity) "
            "RETURN a.id AS source, b.id AS target, r.type AS type, r.description AS description, "
            "r.weight AS weight, r.chunks AS chunks"
        )
        return [
            Relation(
                source=row["source"],
                target=row["target"],
                type=row.get("type") or "RELATED_TO",
                description=row.get("description") or "",
                weight=float(row.get("weight") or 1.0),
                source_chunk_ids=list(row.get("chunks") or []),
            )
            for row in rows
        ]

    def degree(self, entity_id: str) -> int:
        """Return the number of relations touching an entity."""

        rows = self._run(
            "MATCH (e:RagEntity {namespace: $namespace, id: $id})-[r:RAG_REL]-() RETURN count(r) AS degree",
            id=entity_id,
        )
        return int(rows[0]["degree"]) if rows else 0

    def neighbors(self, entity_ids: Iterable[str], hops: int = 1, max_nodes: int = 50) -> Dict[str, int]:
        """Variable-length expansion up to ``hops``."""

        id_list = list(entity_ids)
        if not id_list:
            return {}
        hops = max(1, min(int(hops), 3))
        rows = self._run(
            f"MATCH (s:RagEntity {{namespace: $namespace}}) WHERE s.id IN $ids "
            f"OPTIONAL MATCH p = (s)-[:RAG_REL*1..{hops}]-(n:RagEntity) "
            "WITH n, min(length(p)) AS distance WHERE n IS NOT NULL "
            "RETURN n.id AS id, distance ORDER BY distance LIMIT $limit",
            ids=id_list,
            limit=max_nodes,
        )
        distances = {entity_id: 0 for entity_id in id_list}
        for row in rows:
            distances.setdefault(row["id"], int(row["distance"]))
        return distances

    def relations_for(self, entity_ids: Iterable[str], limit: int = 50) -> List[Relation]:
        """Return relations touching the entities, strongest first."""

        rows = self._run(
            "MATCH (a:RagEntity {namespace: $namespace})-[r:RAG_REL]-(b:RagEntity) WHERE a.id IN $ids "
            "RETURN DISTINCT startNode(r).id AS source, endNode(r).id AS target, r.type AS type, "
            "r.description AS description, r.weight AS weight, r.chunks AS chunks ORDER BY weight DESC LIMIT $limit",
            ids=list(entity_ids),
            limit=limit,
        )
        return [
            Relation(
                source=row["source"],
                target=row["target"],
                type=row.get("type") or "RELATED_TO",
                description=row.get("description") or "",
                weight=float(row.get("weight") or 1.0),
                source_chunk_ids=list(row.get("chunks") or []),
            )
            for row in rows
        ]

    def communities(self, level: Optional[int] = None) -> List[Community]:
        """Return communities, optionally for one level."""

        if level is None:
            rows = self._run("MATCH (c:RagCommunity {namespace: $namespace}) RETURN c.payload AS payload")
        else:
            rows = self._run(
                "MATCH (c:RagCommunity {namespace: $namespace, level: $level}) RETURN c.payload AS payload",
                level=level,
            )
        return [Community.from_dict(json.loads(row["payload"])) for row in rows]
