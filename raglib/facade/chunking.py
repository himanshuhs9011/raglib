"""Chunking on the facade: splitting documents, chunking settings and signature, embedding-limit checks and oversized chunks."""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.loaders.chunk_splitter import IndexSettingsWarning
from raglib.loaders.sections import build_sections
from raglib.schemas import Document, TraceEvent

logger = logging.getLogger("raglib.facade")  # one logger name for the whole facade, as before the split


def unwrap_embedding(embedding: Optional[BaseEmbedding]) -> Optional[BaseEmbedding]:
    """Return the innermost embedding behind caching wrappers."""

    while isinstance(embedding, CachedEmbedding):
        embedding = embedding.inner
    return embedding


def _signed(chunk: Document, signature: str) -> Document:
    """Stamp the chunking signature on a chunk (lets mixed-settings chunks be found in the store)."""

    chunk.metadata["chunking_signature"] = signature
    return chunk


class ChunkingMixin:
    """Chunking on the facade: splitting documents, chunking settings and signature, embedding-limit checks and oversized chunks (mixin of ``RAG``)."""

    def _chunking_fallback(self, document: Document, error: Exception) -> None:
        """Record an embedding-chunking fallback: counted in ``ingest_usage()`` and sent to ``on_event``."""

        from raglib.core.tracing import _callbacks

        self._chunking_fallbacks.append({"document_id": document.id, "error": f"{type(error).__name__}: {error}"[:300]})
        rag = getattr(self, "_rag", None)
        event = TraceEvent(
            stage="chunking_fallback",
            latency_ms=0.0,
            meta={"event": "warning", "document_id": document.id, "fallback": "recursive"},
            error=f"{type(error).__name__}: {error}"[:300],
        )
        for callback in _callbacks(getattr(rag, "on_event", None)):
            try:
                callback(event)
            except Exception as exc:  # noqa: BLE001 - observers must never break ingestion
                logger.warning("on_event callback failed: %s", exc)

    def _configure_embedding_limit(self, override: Optional[int], policy: str) -> None:
        """Make the chunker respect the embedding model's input limit, and warn when chunk_size cannot fit."""

        from raglib.core.text import count_tokens
        from raglib.embedding.limits import ESTIMATE_MARGIN, TOKENIZER_SAMPLE, EmbeddingLimitWarning

        if policy not in {None, "split", "raise", "warn"}:
            raise ValueError("oversized_chunks must be 'split', 'raise', 'warn' or None")
        self._oversized_setting = policy
        embedding = self._embedding
        limit = override or (embedding.max_input_tokens if embedding is not None else None)
        self._embedding_limit = limit
        if embedding is not None and not limit:
            # Unknown limit: warn (once per model), never raise - not even with oversized_chunks="raise".
            from raglib.embedding.limits import warn_unknown_limit
            from raglib.embedding.mock_embedding import MockEmbedding

            if not isinstance(unwrap_embedding(embedding), MockEmbedding):  # offline hash vectors never truncate
                warn_unknown_limit(embedding.model_name or type(unwrap_embedding(embedding)).__name__)
        if not limit or embedding is None:
            return
        exact = embedding.count_input_tokens(TOKENIZER_SAMPLE) is not None
        if exact:
            counter: Callable[[str], int] = lambda text: embedding.count_input_tokens(text) or 0  # noqa: E731
        else:
            counter = count_tokens  # tiktoken when installed, else ~4 characters per token
        self._splitter.set_embedding_limit(limit, counter, exact, policy or "split", on_oversized=self._record_oversized)
        # Startup check: how many model tokens does a chunk of chunk_size (in the chunker's units) hold?
        ratio = counter(TOKENIZER_SAMPLE) / max(1, self._splitter._count_tokens(TOKENIZER_SAMPLE))
        usable = limit if exact else int(limit * ESTIMATE_MARGIN)
        expected = self._splitter._chunk_size * ratio
        if expected <= usable:
            return
        safe = max(1, int(usable / ratio))
        model = embedding.model_name or type(unwrap_embedding(embedding)).__name__
        filled = getattr(self, "_preset_filled", frozenset())
        if self._preset == "production" and "chunk_size" in filled:
            # The preset's chunk_size is a target, not a promise: fit it to the model, with a margin.
            fitted = max(16, int(safe * 0.9))
            self._splitter._chunk_size = fitted
            if "chunk_overlap" in filled:
                self._splitter._overlap = fitted // 5
            if self._semantic_min_tokens_setting is None:
                self._splitter._semantic_min_tokens = max(1, min(32, fitted // 8))
            logger.info("preset='production': chunk_size %d -> %d to fit %s (reads at most %d tokens)",
                        expected / ratio, fitted, model, limit)
            return
        message = (
            f"chunk_size={self._splitter._chunk_size} ({self._splitter.token_counter_name}) is about {expected:.0f} "
            f"tokens for {model}, which reads at most {limit}{'' if exact else ' (estimated)'}. Use chunk_size<={safe} to avoid it"
        )
        if self._preset == "production" and policy is None:
            # Your own chunk_size with the production preset and no oversized_chunks choice: fail early.
            from raglib.embedding.limits import EmbeddingLimitError

            raise EmbeddingLimitError(message + ", or pass oversized_chunks='split' to split such chunks at ingest.")
        action = {"warn": "kept whole (and truncated by the model)", "raise": "rejected",
                  None: "split further (kept whole in an index built before raglib 2.3)"}.get(policy, "split further")
        message += f". Chunks over the limit are {action} at ingest."
        warnings.warn(message, EmbeddingLimitWarning, stacklevel=4)
        logger.warning(message)

    def _oversized_policy(self) -> str:
        """``oversized_chunks``, or (default) split in new indexes and keep chunks as they were in an index
        built before chunking signatures existed (raglib 2.2): re-chunking it would re-embed everything."""

        if self._oversized_setting:
            return self._oversized_setting
        return "warn" if "" in self._index_signatures else "split"

    def _record_oversized(self, document: Document, tokens: int, parts: int) -> None:
        """A chunk was split to fit the embedding model (``parts=0``: kept whole); count it and warn."""

        from raglib.embedding.limits import EmbeddingLimitWarning

        self._oversized_chunks.append({"document_id": document.id, "tokens": tokens, "parts": parts})
        if parts == 0:
            if len(self._oversized_chunks) == 1 or len(self._oversized_chunks) % 1000 == 0:
                warnings.warn(
                    f"A chunk of {document.id!r} has {tokens} tokens, over the embedding limit of {self._embedding_limit}; "
                    "it was kept whole because this index was built before raglib 2.3 (the model truncates it). "
                    "Pass oversized_chunks='split' and re-add your documents to split such chunks.",
                    EmbeddingLimitWarning,
                    stacklevel=2,
                )
            return
        if len(self._oversized_chunks) == 1 or len(self._oversized_chunks) % 1000 == 0:
            warnings.warn(
                f"A chunk of {document.id!r} had {tokens} tokens, over the embedding limit of {self._embedding_limit}; "
                f"split into {parts} parts (counted in ingest_usage()['oversized_chunks_split']).",
                EmbeddingLimitWarning,
                stacklevel=2,
            )

    def chunking_settings(self) -> Dict[str, Any]:
        """Settings that decide chunk boundaries (strategy, sizes, token counter, semantic options)."""

        info = dict(self._splitter.settings())
        if getattr(self, "_embedding_limit", None):
            info["embedding_max_tokens"] = self._embedding_limit  # oversized chunks are split at this limit
        if self._parent_retrieval is not None:
            info.update(parent_retrieval=self._parent_retrieval, parent_chunk_size=self._parent_chunk_size)
        return info

    def chunking_signature(self) -> str:
        """Short hash of ``chunking_settings()`` (stamped on every chunk as ``chunking_signature``)."""

        raw = json.dumps(self.chunking_settings(), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def _check_chunking_settings(self, action: str, saved: Optional[Dict[str, Any]] = None, where: str = "this index") -> None:
        """Warn (once) when the index holds chunks made with other chunking settings."""

        current = self.chunking_signature()
        foreign = {sig for sig in self._index_signatures if sig and sig != current}
        if not foreign or self._settings_warned:
            return
        self._settings_warned = True
        mine = self.chunking_settings()
        if saved:
            changes = ", ".join(
                f"{key}: index={saved.get(key)!r} now={mine.get(key)!r}"
                for key in sorted(set(saved) | set(mine))
                if saved.get(key) != mine.get(key)
            )
        else:
            changes = f"chunking signature(s) {sorted(foreign)} vs {current}"
        message = (
            f"{where} was built with different chunking settings than this RAG ({changes}). It is being {action} "
            "with the current settings: new and changed documents get differently sized chunks, and unchanged "
            "documents are re-chunked and re-embedded the next time they are added. To re-index consistently, build "
            "a RAG with the settings you want, add() all your sources again and save() to a new directory - or keep "
            "the index as it is by passing the settings it was built with (see rag.chunking_settings())."
        )
        warnings.warn(message, IndexSettingsWarning, stacklevel=3)
        logger.warning(message)

    def _chunk(self, documents: List[Document]) -> List[Document]:
        """Split loaded documents into chunks (children of parent sections with parent_retrieval)."""

        if getattr(self._splitter, "_embed_limit", None) is not None:
            self._splitter._oversized_policy = self._oversized_policy()
        signature = self.chunking_signature()
        if self._parent_retrieval is None:
            return [self._tag(_signed(chunk, signature)) for chunk in self._splitter.split(documents)]
        chunks: List[Document] = []
        for document in documents:
            sections = [self._tag(section) for section in build_sections(document, self._parent_retrieval, self._parent_chunk_size)]
            with self._ingest_lock:
                self._staged_sections.setdefault(document.id, []).extend(sections)
            for child in self._splitter.split(sections):
                # Children belong to the document (replace/delete semantics) and point to their section.
                metadata = {
                    **child.metadata,
                    "parent_id": document.id,
                    "section_id": child.metadata["parent_id"],
                    "chunking_signature": signature,
                }
                chunks.append(self._tag(replace(child, metadata=metadata)))
        return chunks

    def _register_sections(self, parent_id: str, version: Optional[str] = None) -> None:
        """Make the staged sections of ``parent_id`` current (replacing its old ones)."""

        staged = self._staged_sections.pop(parent_id, None)
        if staged is None:
            return
        for section_id in self._parent_sections.pop(parent_id, []):
            self._sections.pop(section_id, None)
        for section in staged:
            if version is not None:
                section.metadata["document_version"] = version
            self._sections[section.id] = section
        self._parent_sections[parent_id] = [section.id for section in staged]
