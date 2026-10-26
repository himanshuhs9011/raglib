"""GraphRAG indexing: extract -> resolve -> upsert -> embed entities -> (lazy) communities."""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from raglib.core.text import token_set
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.graphrag.communities import CommunityReporter, detect_communities
from raglib.graphrag.extractor import DEFAULT_ENTITY_TYPES, GraphExtractor
from raglib.graphrag.schemas import Community, Entity, Relation
from raglib.graphrag.store import BaseGraphStore, NetworkXGraphStore
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document
from raglib.vectorstores.base_store import BaseVectorStore, VectorRecord
from raglib.vectorstores.in_memory_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


class GraphIndexer:
    """Builds and maintains the knowledge graph incrementally.

    Cost model (why this stays fast):
      * ingestion = one extraction call per *new* chunk, run in parallel, cached by content;
      * entity embeddings are recomputed only for entities whose description changed;
      * communities + reports are built lazily (first global query, ``rebuild_communities()``,
        or after ``community_rebuild_threshold`` new chunks), and only changed communities
        are re-summarized.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding: Optional[BaseEmbedding] = None,
        store: Optional[BaseGraphStore] = None,
        entity_vector_store: Optional[BaseVectorStore] = None,
        entity_types: Sequence[str] = DEFAULT_ENTITY_TYPES,
        max_workers: int = 8,
        gleanings: int = 0,
        max_levels: int = 2,
        max_community_size: int = 25,
        community_rebuild_threshold: Optional[int] = None,
        resolve_with_embeddings: bool = False,
        resolution_threshold: float = 0.92,
        min_llm_report_size: int = 3,
    ):
        """Configure the indexer."""

        self.llm_client = llm_client
        self.embedding = embedding
        self.store = store or NetworkXGraphStore()
        self.entity_vectors = entity_vector_store or (InMemoryVectorStore() if embedding is not None else None)
        self.extractor = GraphExtractor(
            llm_client=llm_client, entity_types=entity_types, max_workers=max_workers, gleanings=gleanings
        )
        self.reporter = CommunityReporter(
            llm_client=llm_client, embedding=embedding, max_workers=max_workers, min_llm_report_size=min_llm_report_size
        )
        self.max_levels = max_levels
        self.max_community_size = max_community_size
        self.community_rebuild_threshold = community_rebuild_threshold
        self.resolve_with_embeddings = resolve_with_embeddings and embedding is not None
        self.resolution_threshold = resolution_threshold
        self.indexed_chunk_ids: set = set()
        self._chunks_since_rebuild = 0
        self._dirty = False
        self._lock = threading.RLock()

    def add_chunks(self, chunks: Sequence[Document]) -> Dict[str, int]:
        """Extract and merge graph fragments for chunks not yet indexed."""

        new_chunks = [chunk for chunk in chunks if chunk.id not in self.indexed_chunk_ids and chunk.content.strip()]
        if not new_chunks:
            return {"chunks": 0, "entities": 0, "relations": 0}
        extractions = self.extractor.extract(new_chunks)

        entities: List[Entity] = []
        relations: List[Relation] = []
        for result in extractions.values():
            entities.extend(result.entities)
            relations.extend(result.relations)

        with self._lock:
            if self.resolve_with_embeddings:
                entities, relations = self._resolve(entities, relations)
            changed = self.store.upsert_entities(entities)
            self.store.upsert_relations(relations)
            self._embed_entities(changed)
            self.indexed_chunk_ids.update(chunk.id for chunk in new_chunks)
            self._chunks_since_rebuild += len(new_chunks)
            self._dirty = True

        if self.community_rebuild_threshold and self._chunks_since_rebuild >= self.community_rebuild_threshold:
            self.rebuild_communities()
        logger.info(
            "GraphIndexer added %d chunks: %d entity mentions, %d relation mentions",
            len(new_chunks),
            len(entities),
            len(relations),
        )
        return {"chunks": len(new_chunks), "entities": len(entities), "relations": len(relations)}

    def remove_chunks(self, chunk_ids: Iterable[str]) -> None:
        """Remove chunks from the graph (orphaned entities and relations are deleted)."""

        ids = [chunk_id for chunk_id in chunk_ids if chunk_id in self.indexed_chunk_ids]
        if not ids:
            return
        with self._lock:
            removed = self.store.remove_chunks(ids)
            self.extractor.forget(ids)
            if self.entity_vectors is not None and removed:
                self.entity_vectors.delete(removed)
            self.indexed_chunk_ids.difference_update(ids)
            self._dirty = True

    def _resolve(self, entities: List[Entity], relations: List[Relation]) -> Tuple[List[Entity], List[Relation]]:
        """Map new entity keys onto existing near-duplicate entities (embedding similarity)."""

        assert self.embedding is not None and self.entity_vectors is not None
        known = {entity.id for entity in self.store.get_entities(entity.id for entity in entities)}
        candidates = [entity for entity in entities if entity.id not in known]
        if not candidates or self.entity_vectors.count() == 0:
            return entities, relations
        unique = {entity.id: entity for entity in candidates}
        vectors = self.embedding.embed_documents([entity.embedding_text() for entity in unique.values()])
        mapping: Dict[str, str] = {}
        for entity, vector in zip(unique.values(), vectors):
            hits = self.entity_vectors.query(vector, top_k=1)
            if hits and hits[0].score >= self.resolution_threshold and hits[0].metadata.get("type") == entity.type:
                mapping[entity.id] = hits[0].document_id
        if not mapping:
            return entities, relations
        for entity in entities:
            if entity.id in mapping:
                entity.id = mapping[entity.id]
        for relation in relations:
            relation.source = mapping.get(relation.source, relation.source)
            relation.target = mapping.get(relation.target, relation.target)
        return entities, [relation for relation in relations if relation.source != relation.target]

    def _embed_entities(self, entity_ids: List[str]) -> None:
        """Embed entities whose text changed."""

        if self.embedding is None or self.entity_vectors is None or not entity_ids:
            return
        entities = self.store.get_entities(entity_ids)
        vectors = self.embedding.embed_documents([entity.embedding_text() for entity in entities])
        self.entity_vectors.upsert(
            [
                VectorRecord(
                    document_id=entity.id,
                    embedding=list(vector),
                    document_text=entity.embedding_text(),
                    metadata={"name": entity.name, "type": entity.type},
                )
                for entity, vector in zip(entities, vectors)
            ]
        )

    @property
    def is_dirty(self) -> bool:
        """True when the graph changed since communities were last built."""

        return self._dirty

    def ensure_communities(self) -> List[Community]:
        """Build communities/reports if the graph changed (lazy path for global search)."""

        if self._dirty or not self.store.communities():
            self.rebuild_communities()
        return self.store.communities()

    def rebuild_communities(self) -> List[Community]:
        """Re-cluster the graph and (re)generate reports for changed communities only."""

        with self._lock:
            previous = {community.id: community for community in self.store.communities()}
            communities = detect_communities(
                self.store, max_levels=self.max_levels, max_community_size=self.max_community_size
            )
            communities = self.reporter.report(communities, self.store, previous=previous)
            self.store.set_communities(communities)
            self._dirty = False
            self._chunks_since_rebuild = 0
            return communities

    def entity_search(self, query: str, top_k: int = 8) -> List[Tuple[Entity, float]]:
        """Find the entities most related to the query (vector search, else lexical match)."""

        if self.embedding is not None and self.entity_vectors is not None and self.entity_vectors.count() > 0:
            hits = self.entity_vectors.query(self.embedding.embed_query(query), top_k=top_k)
            entities = {entity.id: entity for entity in self.store.get_entities(hit.document_id for hit in hits)}
            return [(entities[hit.document_id], float(hit.score)) for hit in hits if hit.document_id in entities]

        query_terms = token_set(query, remove_stopwords=True)
        if not query_terms:
            return []
        scored: List[Tuple[Entity, float]] = []
        for entity in self.store.all_entities():
            name_terms = token_set(entity.name)
            overlap = len(query_terms & name_terms)
            if overlap:
                scored.append((entity, overlap / max(len(name_terms), 1)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def save(self, directory: str) -> None:
        """Persist graph, extraction cache, indexed chunk ids and entity vectors."""

        os.makedirs(directory, exist_ok=True)
        if isinstance(self.store, NetworkXGraphStore):
            self.store.save(os.path.join(directory, "graph.json"))
        state: Dict[str, Any] = {
            "indexed_chunk_ids": sorted(self.indexed_chunk_ids),
            "extraction_cache": self.extractor.cache,
            "chunk_cache_keys": self.extractor.chunk_cache_keys,
            "dirty": self._dirty,
        }
        if isinstance(self.entity_vectors, InMemoryVectorStore):
            state["entity_vectors"] = [
                {"id": r.document_id, "embedding": r.embedding, "text": r.document_text, "metadata": r.metadata}
                for r in self.entity_vectors.records()
            ]
        with open(os.path.join(directory, "graph_state.json"), "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)

    def load(self, directory: str) -> None:
        """Load state written by ``save``."""

        graph_path = os.path.join(directory, "graph.json")
        if isinstance(self.store, NetworkXGraphStore) and os.path.exists(graph_path):
            self.store.load(graph_path)
        state_path = os.path.join(directory, "graph_state.json")
        if not os.path.exists(state_path):
            return
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        self.indexed_chunk_ids = set(state.get("indexed_chunk_ids", []))
        self.extractor.cache.update(state.get("extraction_cache", {}))
        self.extractor.chunk_cache_keys.update(state.get("chunk_cache_keys", {}))
        self._dirty = bool(state.get("dirty", False))
        if isinstance(self.entity_vectors, InMemoryVectorStore) and state.get("entity_vectors"):
            self.entity_vectors.upsert(
                [
                    VectorRecord(document_id=r["id"], embedding=r["embedding"], document_text=r["text"], metadata=r["metadata"])
                    for r in state["entity_vectors"]
                ]
            )
