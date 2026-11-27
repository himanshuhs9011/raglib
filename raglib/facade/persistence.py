"""Saving and loading an index: save(), load(), the index fingerprint and atomic file writes."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from raglib.schemas import Document
from raglib.vectorstores.base_store import VectorRecord
from raglib.vectorstores.in_memory_store import InMemoryVectorStore

if TYPE_CHECKING:
    from raglib.facade.core import RAG

logger = logging.getLogger("raglib.facade")  # one logger name for the whole facade, as before the split


def _atomic_write(target: Path, text: str) -> None:
    """Write ``text`` to ``target`` via a temp file + rename (never leaves a partial file)."""

    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


class PersistenceMixin:
    """Saving and loading an index: save(), load(), the index fingerprint and atomic file writes (mixin of ``RAG``)."""

    def save(self, directory: str) -> None:
        """Persist the chunk corpus, in-memory vectors and graph to ``directory``."""

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        chunks = [asdict(document) for document in self._retriever.documents]
        store = self._retriever.vector_store
        payload = {
            "version": 3,
            "index": self._index_fingerprint(),
            "parents": self._parents,
            "acl": self._parent_acl,
            "fingerprints": self._fingerprints,
            "versions": self._doc_versions,
            "sections": {section_id: asdict(section) for section_id, section in self._sections.items()},
            "parent_sections": self._parent_sections,
            "chunks": chunks,
        }
        # Write each file atomically, so a crash mid-save never leaves a half-written index.
        if isinstance(store, InMemoryVectorStore):
            store.dump(str(path), "vectors")  # binary float32 + metadata: loads ~50x faster than JSON
            legacy = path / "vectors.json"
            if legacy.exists():
                legacy.unlink()
        _atomic_write(path / "chunks.json", json.dumps(payload, ensure_ascii=False))
        if self._graph_indexer is not None:
            self._graph_indexer.save(str(path / "graph"))
        logger.info("Saved RAG index to %s", path)

    def load(self, directory: str, allow_model_change: bool = False) -> "RAG":
        """Load an index written by ``save`` (embeddings are reused, not recomputed).

        Raises ``ValueError`` when the index was built with a different embedding model:
        its stored vectors are not comparable with this model's query vectors. Pass
        ``allow_model_change=True`` to load the chunks anyway and re-embed them.
        """

        path = Path(directory)
        payload = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        saved = payload.get("index") or {}
        current = self._index_fingerprint()
        changed = [
            key
            for key in ("embedding_model", "dimension")
            if saved.get(key) is not None and current.get(key) is not None and saved[key] != current[key]
        ]
        reembed = False
        if changed:
            detail = ", ".join(f"{key}: saved={saved[key]!r} current={current[key]!r}" for key in changed)
            if not allow_model_change:
                raise ValueError(
                    f"Index in {directory} was built with a different embedding model ({detail}). "
                    "Use the same embedding_llm/embedding_model, or load(..., allow_model_change=True) to re-embed."
                )
            logger.warning("Embedding model changed (%s); re-embedding the loaded chunks", detail)
            reembed = True
        store = self._retriever.vector_store
        vectors_path = path / "vectors.json"
        if isinstance(store, InMemoryVectorStore) and not reembed:
            if InMemoryVectorStore.has_dump(str(path), "vectors"):
                store.restore(str(path), "vectors")
            elif vectors_path.exists():  # indexes saved before raglib 2.2
                store.upsert([VectorRecord(**record) for record in json.loads(vectors_path.read_text(encoding="utf-8"))])
        documents = [Document(**item) for item in payload.get("chunks", [])]
        self._index_signatures.update(str(doc.metadata.get("chunking_signature", "")) for doc in documents)
        if saved.get("chunking_signature"):
            self._index_signatures.add(str(saved["chunking_signature"]))
        self._check_chunking_settings("opened", saved=saved.get("chunking"), where=f"The index in {directory}")
        if reembed and store is not None and not isinstance(store, InMemoryVectorStore):
            store.delete([document.id for document in documents])  # drop vectors from the old model
        self._retriever.add_documents(documents)
        self._parents.update({key: list(value) for key, value in payload.get("parents", {}).items()})
        self._parent_acl.update({key: str(value) for key, value in payload.get("acl", {}).items()})
        self._fingerprints.update({key: str(value) for key, value in payload.get("fingerprints", {}).items()})
        self._doc_versions.update({key: str(value) for key, value in payload.get("versions", {}).items()})
        self._sections.update({key: Document(**value) for key, value in payload.get("sections", {}).items()})
        self._parent_sections.update({key: list(value) for key, value in payload.get("parent_sections", {}).items()})
        if self._graph_indexer is not None and (path / "graph").exists():
            self._graph_indexer.load(str(path / "graph"))
            missing = [doc for doc in documents if doc.id not in self._graph_indexer.indexed_chunk_ids]
            if missing:
                self._add_to_graph(missing)
        elif self._graph_indexer is not None:
            self._add_to_graph(documents)
        logger.info("Loaded %d chunks from %s", len(documents), path)
        return self

    def _index_fingerprint(self) -> Dict[str, Any]:
        """Describe how the index was built (embedding model + vector size), stored by ``save``."""

        info: Dict[str, Any] = {"embedding_class": None, "embedding_model": None, "dimension": None}
        info["chunking"] = self.chunking_settings()
        info["chunking_signature"] = self.chunking_signature()
        if self._embedding is not None:
            info.update(self._embedding.fingerprint())
            try:
                info["dimension"] = int(self._embedding.dimension) or None  # 0 = not known until first call
            except Exception:  # noqa: BLE001 - custom embeddings may not know their size
                info["dimension"] = None
        store = self._retriever.vector_store
        if isinstance(store, InMemoryVectorStore) and store.dimension:
            info["dimension"] = store.dimension
        return info
