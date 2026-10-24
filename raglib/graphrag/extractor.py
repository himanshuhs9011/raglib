"""Parallel, cached LLM extraction of entities and relations from chunks."""

from __future__ import annotations

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from raglib.graphrag.schemas import Entity, ExtractionResult, Relation, normalize_entity_key
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = (
    "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT", "TECHNOLOGY", "CONCEPT", "DATE",
)

EXTRACTION_SYSTEM = "You are a precise information extraction system that outputs valid JSON only."


class GraphExtractor:
    """Extracts a knowledge graph fragment from each chunk with one LLM call per chunk.

    * Runs chunks concurrently (``max_workers``) - extraction is I/O bound.
    * Caches results by chunk content hash, so re-ingesting unchanged text is free.
    * ``gleanings`` > 0 adds "anything missed?" passes (higher recall, more cost).
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        entity_types: Sequence[str] = DEFAULT_ENTITY_TYPES,
        max_workers: int = 8,
        gleanings: int = 0,
        max_chars: int = 6000,
        max_entities_per_chunk: int = 20,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Configure extraction."""

        self.llm_client = llm_client
        self.entity_types = [entity_type.upper() for entity_type in entity_types]
        self.max_workers = max(1, max_workers)
        self.gleanings = max(0, gleanings)
        self.max_chars = max_chars
        self.max_entities_per_chunk = max_entities_per_chunk
        self.cache: Dict[str, Dict[str, Any]] = cache if cache is not None else {}
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.chunk_cache_keys: Dict[str, str] = {}  # chunk id -> cache key (so deletion can purge the cache)

    def extract(self, chunks: Sequence[Document]) -> Dict[str, ExtractionResult]:
        """Return {chunk_id: ExtractionResult} for every chunk (in parallel)."""

        if not chunks:
            return {}
        if self.max_workers == 1 or len(chunks) == 1:
            return {chunk.id: self.extract_one(chunk) for chunk in chunks}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as pool:
            results = list(pool.map(self.extract_one, chunks))
        return {chunk.id: result for chunk, result in zip(chunks, results)}

    def extract_one(self, chunk: Document) -> ExtractionResult:
        """Extract (or load from cache) the graph fragment of one chunk."""

        text = chunk.content[: self.max_chars]
        cache_key = hashlib.sha256(f"{','.join(self.entity_types)}\x00{text}".encode("utf-8")).hexdigest()
        with self._lock:
            self.chunk_cache_keys[chunk.id] = cache_key
            cached = self.cache.get(cache_key)
        if cached is not None:
            return self._attach_chunk(ExtractionResult.from_dict(cached), chunk.id)

        raw_entities: List[Dict[str, Any]] = []
        raw_relations: List[Dict[str, Any]] = []
        data = self._call(self._prompt(text))
        if isinstance(data, dict):
            raw_entities.extend(data.get("entities") or [])
            raw_relations.extend(data.get("relations") or [])
        for _ in range(self.gleanings):
            known = ", ".join(str(item.get("name", "")) for item in raw_entities if isinstance(item, dict))
            more = self._call(self._prompt(text, already_found=known))
            if not isinstance(more, dict) or not (more.get("entities") or more.get("relations")):
                break
            raw_entities.extend(more.get("entities") or [])
            raw_relations.extend(more.get("relations") or [])

        result = self._parse(raw_entities, raw_relations)
        with self._lock:
            self.cache[cache_key] = result.to_dict()
        return self._attach_chunk(result, chunk.id)

    # ------------------------------------------------------------------ internals
    def forget(self, chunk_ids) -> None:
        """Remove cached extractions for deleted chunks (unless another live chunk has the same text)."""

        with self._lock:
            removed = {self.chunk_cache_keys.pop(chunk_id, None) for chunk_id in chunk_ids}
            still_used = set(self.chunk_cache_keys.values())
            for key in removed - still_used - {None}:
                self.cache.pop(key, None)

    def _prompt(self, text: str, already_found: str = "") -> str:
        """Build the extraction prompt."""

        extra = (
            f"These entities were already extracted: {already_found}. Return ONLY entities and relations that were missed.\n"
            if already_found
            else ""
        )
        return (
            "Task: knowledge graph extraction.\n"
            f"Extract the important entities (types: {', '.join(self.entity_types)}) and the relations between them "
            "from the text. Use the exact entity names as written in the text. Each relation must connect two "
            "extracted entities. Describe each entity and relation in one sentence, based only on the text.\n"
            f"{extra}"
            'Return JSON: {"entities": [{"name": "...", "type": "...", "description": "..."}], '
            '"relations": [{"source": "...", "target": "...", "type": "SHORT_VERB_PHRASE", '
            '"description": "...", "weight": 1-10}]}\n\n'
            f"Text:\n{text}"
        )

    def _call(self, prompt: str) -> Any:
        """Call the LLM and parse JSON (None on failure)."""

        with self._lock:
            self.llm_calls += 1
        return self.llm_client.complete_json(prompt, system=EXTRACTION_SYSTEM, default=None)

    def _parse(self, raw_entities: List[Any], raw_relations: List[Any]) -> ExtractionResult:
        """Validate and normalize raw LLM output."""

        entities: Dict[str, Entity] = {}
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            key = normalize_entity_key(name)
            if not key or len(key) < 2:
                continue
            entity_type = str(item.get("type") or "CONCEPT").strip().upper().replace(" ", "_") or "CONCEPT"
            description = str(item.get("description") or "").strip()
            if key in entities:
                if description and description not in entities[key].description:
                    entities[key].description = f"{entities[key].description} | {description}".strip(" |")
                continue
            if len(entities) >= self.max_entities_per_chunk:
                continue
            entities[key] = Entity(id=key, name=name, type=entity_type, description=description)

        relations: Dict[str, Relation] = {}
        for item in raw_relations:
            if not isinstance(item, dict):
                continue
            source = normalize_entity_key(str(item.get("source") or ""))
            target = normalize_entity_key(str(item.get("target") or ""))
            if not source or not target or source == target:
                continue
            # Relations may mention entities the model forgot to list; add them as concepts.
            for key, raw_name in ((source, item.get("source")), (target, item.get("target"))):
                if key not in entities:
                    entities[key] = Entity(id=key, name=str(raw_name).strip(), type="CONCEPT")
            try:
                weight = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            relation = Relation(
                source=source,
                target=target,
                type=str(item.get("type") or "RELATED_TO").strip().upper().replace(" ", "_")[:60] or "RELATED_TO",
                description=str(item.get("description") or "").strip(),
                weight=max(0.1, min(weight, 10.0)),
            )
            relations.setdefault(relation.key, relation)
        return ExtractionResult(entities=list(entities.values()), relations=list(relations.values()))

    @staticmethod
    def _attach_chunk(result: ExtractionResult, chunk_id: str) -> ExtractionResult:
        """Return a copy of the result whose items reference ``chunk_id``."""

        return ExtractionResult(
            entities=[
                Entity(id=e.id, name=e.name, type=e.type, description=e.description, source_chunk_ids=[chunk_id])
                for e in result.entities
            ],
            relations=[
                Relation(
                    source=r.source,
                    target=r.target,
                    type=r.type,
                    description=r.description,
                    weight=r.weight,
                    source_chunk_ids=[chunk_id],
                )
                for r in result.relations
            ],
        )
