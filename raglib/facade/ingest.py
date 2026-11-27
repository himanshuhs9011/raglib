"""Ingestion on the facade: add, update and delete sources, document versions and fingerprints, invalidation, indexing chunks and purging caches."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from raglib.access import ACCESS_TAG_KEYS, AccessControl
from raglib.core.runtime import usage_scope
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.facade.permissions import _UNTAGGED_SIGNATURE
from raglib.llm.base_client import (
    CachedLLMClient,
    llm_cache_of,
)
from raglib.loaders.document_loader import SourceInput
from raglib.schemas import Document

logger = logging.getLogger("raglib.facade")  # one logger name for the whole facade, as before the split



def _stamp_version(chunks: List[Document], version: str) -> None:
    """Write ``document_version`` into each chunk's metadata (not part of the chunk id or ACL signature)."""

    for chunk in chunks:
        chunk.metadata["document_version"] = version


class IngestMixin:
    """Ingestion on the facade: add, update and delete sources, document versions and fingerprints, invalidation, indexing chunks and purging caches (mixin of ``RAG``)."""

    def add(
        self,
        source: SourceInput,
        batch_size: int = 256,
        on_progress: Optional[Callable[[Dict[str, int]], None]] = None,
        *,
        access: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Add or update sources. Unchanged documents are skipped; changed ones are replaced.

        ``metadata`` is merged into every loaded document. ``access`` sets the access tags
        (``access_level``, ``department``, ``allowed_roles``; see ``raglib.access.access_tags``)
        on every loaded document; tags already in a ``Document``'s metadata work too. Re-adding
        unchanged text with different tags updates its permissions.

        Large inputs are processed in batches of ~``batch_size`` chunks: each batch is embedded
        and indexed before the next file is read, so memory stays bounded, progress is reported
        via ``on_progress({"files": .., "chunks_indexed": .., "skipped": ..})``, and if the run
        fails half-way, already-indexed files are kept (re-running skips them).
        Returns the number of new chunks indexed.
        """

        if access is not None:
            unknown = set(access) - set(ACCESS_TAG_KEYS)
            if unknown:
                raise ValueError(f"Unknown access tags: {sorted(unknown)}. Supported: {list(ACCESS_TAG_KEYS)}")
        extra = {**dict(metadata or {}), **dict(access or {})}
        self._check_chunking_settings("added to")
        stats = {"files": 0, "chunks_indexed": 0, "skipped": 0}
        pending: List[Document] = []

        def flush() -> None:
            if pending:
                stats["chunks_indexed"] += self._index_chunks(pending)
                pending.clear()
            if on_progress is not None:
                on_progress(dict(stats))

        with usage_scope() as meter:  # ingest cost: embeddings, LLM chunking, graph extraction
            try:
                for loaded in self._loader.iter_load(source):
                    stats["files"] += 1
                    if not loaded:
                        stats["skipped"] += 1
                        continue
                    if extra:
                        loaded = [replace(document, metadata={**document.metadata, **extra}) for document in loaded]
                    changed_documents = [document for document in loaded if not self._unchanged(document)]
                    if not changed_documents:
                        continue  # unchanged: not even re-chunked (the embedding chunker would re-pay its windows)
                    pending.extend(self._chunk(changed_documents))
                    if len(pending) >= batch_size:
                        flush()
                flush()
            finally:
                self._ingest_meter.merge(meter)  # batches embedded before a failure were paid for too
        logger.info(
            "Ingested %d new chunks from %d sources (corpus=%d)",
            stats["chunks_indexed"],
            stats["files"],
            len(self._retriever.documents),
        )
        return stats["chunks_indexed"]

    def _document_fingerprint(self, document: Document) -> str:
        """Content + chunking settings + oversized-chunk policy + access tags: equal means re-adding
        changes nothing (so switching ``oversized_chunks`` and re-adding really re-chunks)."""

        policy = self._oversized_policy() if getattr(self._splitter, "_embed_limit", None) is not None else None
        raw = json.dumps([document.content, self.chunking_signature(), policy, AccessControl.signature(document.metadata)])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _unchanged(self, document: Document) -> bool:
        """True when this document is indexed with the same text, chunking settings and tags."""

        fingerprint = self._document_fingerprint(document)
        with self._ingest_lock:
            self._pending_fingerprints[document.id] = fingerprint
            return document.id in self._parents and self._fingerprints.get(document.id) == fingerprint

    def document_version(self, parent_id: str) -> Optional[str]:
        """Content + permission hash of an indexed document (``None`` when not indexed)."""

        return self._doc_versions.get(parent_id)

    def _invalidate_documents(self, parent_ids: Sequence[str], old_texts: Sequence[str] = ()) -> int:
        """Drop cached answers built from ``parent_ids`` and the cached vectors of ``old_texts``."""

        dropped = 0
        for llm_cache in self._llm_caches():
            dropped += llm_cache.invalidate(parent_ids)
        if self._answer_cache is not None:
            dropped += self._answer_cache.invalidate(parent_ids)
        if old_texts and isinstance(self._embedding, CachedEmbedding):
            dropped += self._embedding.invalidate_texts(list(old_texts))
        self._invalidated_entries += dropped
        return dropped

    def invalidate(self, source: SourceInput) -> int:
        """Drop every cached answer (LLM cache and answer cache) that used these documents.

        ``source`` is a document id, a file path, or a list of them. Returns the number of cache
        entries dropped. Use it when a document changed outside raglib; ``add``/``update`` of a
        changed document and ``on_document_update`` invalidate automatically.
        """

        items = source if isinstance(source, (list, tuple, set)) else [source]
        parent_ids = self._known_parents_for(source) | {str(item) for item in items if isinstance(item, str)}
        texts = [doc.content for pid in parent_ids for doc in (self._retriever.get(cid) for cid in self._parents.get(pid, [])) if doc]
        return self._invalidate_documents(sorted(parent_ids), texts)

    def invalidate_all(self) -> None:
        """Clear every cache: the response caches of all models, the embedding, answer and judge caches."""

        self._purge_caches()

    def on_document_update(self, source: SourceInput = None, doc_ids: Optional[Sequence[str]] = None) -> Dict[str, int]:
        """Hook for external update jobs (e.g. the nightly corpus sync): event-driven invalidation.

        ``source``: files/folders/URLs/texts that changed - they are re-ingested (``add``), which
        replaces changed documents and drops the cached answers that used them. ``doc_ids``:
        documents changed elsewhere - their cached answers are dropped. Returns
        ``{"reindexed_chunks": n, "invalidated": m}``. Cache freshness is a correctness issue: call
        this from the job that updates the corpus instead of relying on TTLs alone.
        """

        before = self._invalidated_entries
        reindexed = self.add(source) if source is not None else 0
        if doc_ids:
            self.invalidate(list(doc_ids))
        return {"reindexed_chunks": reindexed, "invalidated": self._invalidated_entries - before}

    def ingest_usage(self) -> Dict[str, Any]:
        """Tokens (and USD, with ``prices``) spent on ingestion so far: embeddings, LLM chunking, graph."""

        info: Dict[str, Any] = {
            **self._ingest_meter.as_dict(),
            "usage_by_model": self._ingest_meter.breakdown(),
            "chunking_fallbacks": len(self._chunking_fallbacks),
            "oversized_chunks_split": len(self._oversized_chunks),
            "chunking_fallback_documents": [item["document_id"] for item in self._chunking_fallbacks[-50:]],
        }
        if self._prices is not None:
            cost = self._prices.cost(self._ingest_meter.breakdown())
            info.update({"cost_usd": round(cost["total"], 8), "cost_by_model": cost["by_model"], "cost_unpriced": cost["unpriced"]})
        return info

    def _index_chunks(self, chunks: List[Document]) -> int:
        """Upsert one batch of chunks (per-document replace semantics)."""

        with self._ingest_lock:
            by_parent: Dict[str, List[Document]] = {}
            for chunk in chunks:
                by_parent.setdefault(str(chunk.metadata.get("parent_id", chunk.id)), []).append(chunk)
            to_add: List[Document] = []
            changed: Dict[str, List[str]] = {}
            retagged: List[Document] = []
            signatures: Dict[str, str] = {}
            for parent_id, parent_chunks in by_parent.items():
                new_ids = [chunk.id for chunk in parent_chunks]
                old_ids = self._parents.get(parent_id)
                signature = AccessControl.signature(parent_chunks[0].metadata)
                version = hashlib.sha256(json.dumps([new_ids, signature]).encode("utf-8")).hexdigest()[:16]
                if old_ids == new_ids:
                    known = self._parent_acl.get(parent_id)
                    # An index saved before permissions were recorded has no entry: tags on the new
                    # chunks then count as a change (otherwise they would never be applied).
                    if (known is None and signature != _UNTAGGED_SIGNATURE) or (known is not None and known != signature):
                        _stamp_version(parent_chunks, version)
                        retagged.extend(parent_chunks)  # same text, new permissions
                        signatures[parent_id] = signature
                        self._register_sections(parent_id, version)
                        self._doc_versions[parent_id] = version
                        # A cached public answer must not outlive a permission change.
                        self._invalidate_documents([parent_id])
                    else:
                        self._staged_sections.pop(parent_id, None)
                    continue
                signatures[parent_id] = signature
                if old_ids:
                    old_texts = [doc.content for doc in (self._retriever.get(cid) for cid in old_ids) if doc]
                    self._delete_chunk_ids(old_ids)
                    self._parents.pop(parent_id, None)
                    self._invalidate_documents([parent_id], old_texts)
                    logger.info("Document %s changed; replaced %d old chunks", parent_id, len(old_ids))
                # Chunks carry their document's version, so stale vectors can be found or filtered
                # in the vector store ({"document_version": rag.document_version(doc_id)}).
                _stamp_version(parent_chunks, version)
                self._register_sections(parent_id, version)
                self._doc_versions[parent_id] = version
                changed[parent_id] = new_ids
                to_add.extend(parent_chunks)
            if retagged:
                # Chunk ids are unchanged, so the graph keeps pointing at them and picks up the new
                # tags on lookup; only the retriever indexes (which store metadata) are rewritten.
                self._retriever.delete_documents([chunk.id for chunk in retagged])
                self._retriever.add_documents(retagged)
                logger.info("Updated access tags of %d chunks", len(retagged))
            added = self._retriever.add_documents(to_add)
            self._index_signatures.update(str(chunk.metadata.get("chunking_signature", "")) for chunk in added)
            if self._graph_indexer is not None and to_add:
                self._add_to_graph(to_add)  # skips chunks it already extracted
            # Record documents only after they are really indexed, so a failed batch is retried next run.
            if self._answer_cache is not None and any(parent not in self._parents for parent in changed):
                self._answer_cache.new_documents_added()  # a new document can change cached answers
            self._parents.update(changed)
            for parent_id in by_parent:  # indexed now: re-adding the same document can skip chunking
                fingerprint = self._pending_fingerprints.pop(parent_id, None)
                if fingerprint is not None:
                    self._fingerprints[parent_id] = fingerprint
            self._parent_acl.update(signatures)
            return len(added)

    def update(self, source: SourceInput) -> int:
        """Re-ingest sources, replacing documents whose content changed (alias of ``add``)."""

        return self.add(source)

    def delete(self, source: SourceInput) -> int:
        """Remove every chunk that came from ``source`` (file path, document id, raw text, or a list).

        Returns the number of chunks removed. Works even if the file no longer exists on disk.
        """

        with self._ingest_lock:
            parent_ids = self._known_parents_for(source)
            if not parent_ids:
                parent_ids = {document.id for document in self._loader.load(source)}
            chunk_ids: List[str] = []
            for parent_id in parent_ids:
                chunk_ids.extend(self._parents.pop(parent_id, []))
                self._parent_acl.pop(parent_id, None)
                self._fingerprints.pop(parent_id, None)
                self._doc_versions.pop(parent_id, None)
                for section_id in self._parent_sections.pop(parent_id, []):
                    self._sections.pop(section_id, None)
            removed = self._delete_chunk_ids(chunk_ids)
            if removed:
                # Deleted text must not survive anywhere, including caches keyed on its prompt.
                self.invalidate_all()
            return removed

    def _llm_caches(self) -> List[CachedLLMClient]:
        """The response caches of every model this RAG uses (main, cheap, graph), without duplicates."""

        caches: List[CachedLLMClient] = []
        for client in (self._llm, getattr(self, "_cheap_llm", None), getattr(self, "_graph_llm", None)):
            cache = llm_cache_of(client)
            if cache is not None and all(cache is not known for known in caches):
                caches.append(cache)
        return caches

    def _purge_caches(self) -> None:
        """Deleted text must not survive in caches - privacy requirement. Clears the response caches
        of every model (main, cheap, graph), the embedding cache, the answer cache and the judge's
        verdict cache."""

        for llm_cache in self._llm_caches():
            llm_cache.clear()
        if isinstance(self._embedding, CachedEmbedding):
            self._embedding.clear()
        if getattr(self, "_answer_cache", None) is not None:
            self._answer_cache.clear()
        judge = getattr(self, "_judge", None)
        if judge is not None and judge.cache is not None:
            judge.cache.clear()

    def _known_parents_for(self, source: SourceInput) -> set:
        """Match a source (document id or file path, or a list of them) against indexed documents.

        This lets ``delete()`` work even after the file was removed from disk.
        """

        items = source if isinstance(source, (list, tuple, set)) else [source]
        wanted = {os.fspath(item) for item in items if isinstance(item, (str, os.PathLike))}
        wanted |= {os.path.abspath(item) for item in wanted}
        matches = {parent_id for parent_id in self._parents if parent_id in wanted}
        for document in self._retriever.documents:
            file_path = document.metadata.get("file")
            if file_path and (file_path in wanted or os.path.abspath(str(file_path)) in wanted):
                matches.add(str(document.metadata.get("parent_id", document.id)))
        return matches

    def _delete_chunk_ids(self, chunk_ids: Sequence[str]) -> int:
        """Delete chunks from the retriever and graph."""

        removed = self._retriever.delete_documents(chunk_ids)
        if self._graph_builder is not None:
            self._graph_builder.discard(chunk_ids)
        if self._graph_indexer is not None:
            self._graph_indexer.remove_chunks(chunk_ids)
        return removed
