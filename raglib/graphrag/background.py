"""Build the knowledge graph in the background while queries are answered."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class BackgroundGraphBuilder:
    """Extracts queued chunks into a ``GraphIndexer`` on one worker thread.

    ``RAG(rag_type="graph", graph_build="background")`` uses it: ``add`` / ``load`` return as soon
    as chunks are searchable (BM25 + vectors), and GraphRAG answers from hybrid search plus however
    much of the graph is built so far. Chunks are extracted in batches so the graph grows steadily
    and each merge holds the graph's write lock only briefly. When the queue is empty, failed chunks
    are retried once (bounded by ``max_extraction_attempts``) and community reports are rebuilt, so
    global search is ready afterwards. A save during the build stores what is done; the rest is
    queued again by the next ``load``.
    """

    def __init__(
        self,
        indexer: Any,
        chunk_lookup: Callable[[str], Optional[Document]],
        batch_size: int = 64,
        rebuild_communities: bool = True,
    ):
        """Wrap ``indexer``; ``chunk_lookup(id)`` returns a chunk (used to retry failed extractions)."""

        self.indexer = indexer
        self.chunk_lookup = chunk_lookup
        self.batch_size = max(1, int(batch_size))
        self.rebuild_communities = rebuild_communities
        self._queue: "OrderedDict[str, Document]" = OrderedDict()
        self._discarded: set = set()
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._in_flight = 0
        self.done_chunks = 0
        self.submitted_chunks = 0
        self.phase = "idle"
        self.last_error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        indexer.defer_community_builds = True

    # ----------------------------------------------------------------- public

    def submit(self, chunks: Sequence[Document]) -> int:
        """Queue chunks for extraction (already indexed or queued ones are skipped); returns how many."""

        added = 0
        with self._cond:
            for chunk in chunks:
                if chunk.id in self.indexer.indexed_chunk_ids or chunk.id in self._queue or not chunk.content.strip():
                    continue
                self._discarded.discard(chunk.id)
                self._queue[chunk.id] = chunk
                added += 1
            if added:
                self.submitted_chunks += added
                self.indexer.building = True
                if self._thread is None or not self._thread.is_alive():
                    self.started_at = self.started_at if self.phase != "idle" else time.time()
                    self.finished_at = None
                    self._thread = threading.Thread(target=self._run, name="raglib-graph-builder", daemon=True)
                    self._thread.start()
                self._cond.notify_all()
        return added

    def discard(self, chunk_ids: Iterable[str]) -> None:
        """Drop deleted chunks from the queue (and from the graph if they are being extracted now)."""

        with self._cond:
            for chunk_id in chunk_ids:
                if self._queue.pop(chunk_id, None) is None:
                    self._discarded.add(chunk_id)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the queue is empty and communities are built; False on timeout."""

        with self._cond:
            return self._cond.wait_for(lambda: not self._queue and self._in_flight == 0 and self.phase == "idle", timeout)

    def status(self) -> Dict[str, Any]:
        """Progress of the build (safe to call from any thread)."""

        with self._cond:
            pending = len(self._queue) + self._in_flight
            building = bool(pending) or self.phase != "idle"
            elapsed = (self.finished_at or time.time()) - self.started_at if self.started_at else 0.0
            return {
                "state": "building" if building else ("ready" if self.indexer.indexed_chunk_ids else "empty"),
                "phase": self.phase,  # idle | extracting | retrying | communities
                "pending_chunks": pending,
                "done_chunks": self.done_chunks,
                "submitted_chunks": self.submitted_chunks,
                "failed_extractions": len(self.indexer.failed_chunk_ids),
                "skipped_extractions": len(self.indexer.skipped_chunk_ids),
                "elapsed_s": round(elapsed, 1),
                "last_error": self.last_error,
            }

    # ----------------------------------------------------------------- worker

    def _take_batch(self) -> Sequence[Document]:
        with self._cond:
            batch = []
            while self._queue and len(batch) < self.batch_size:
                _, chunk = self._queue.popitem(last=False)
                batch.append(chunk)
            self._in_flight = len(batch)
            if batch:
                self.phase = "extracting"
            return batch

    def _run(self) -> None:
        changed = False
        while True:
            batch = self._take_batch()
            if batch:
                changed |= self._extract(batch)
                continue
            # queue drained: one bounded retry of failed chunks, then refresh communities
            failed = sorted(self.indexer.failed_chunk_ids)
            if failed:
                self._set_phase("retrying")
                chunks = [chunk for chunk in (self.chunk_lookup(chunk_id) for chunk_id in failed) if chunk]
                if chunks:
                    changed |= self._extract(chunks)
            with self._cond:
                if self._queue:  # more work arrived meanwhile
                    continue
            if changed and self.rebuild_communities:
                self._set_phase("communities")
                try:
                    self.indexer.rebuild_communities()
                except Exception as exc:  # noqa: BLE001 - keep serving; global search uses older reports
                    self._error(f"community rebuild failed: {exc}")
                changed = False
            with self._cond:
                if self._queue:
                    continue
                self.phase = "idle"
                self.indexer.building = False
                self.finished_at = time.time()
                self._thread = None
                self._cond.notify_all()
                logger.info("Background graph build finished: %s", self.status())
                return

    def _extract(self, batch: Sequence[Document]) -> bool:
        try:
            result = self.indexer.add_chunks(batch)
        except Exception as exc:  # noqa: BLE001 - chunks stay unindexed and are queued again by the next load
            self._error(f"extraction batch failed: {exc}")
            result = {}
        with self._cond:
            deleted = {chunk.id for chunk in batch} & self._discarded
            self._discarded -= deleted
            self.done_chunks += len(batch)
            self._in_flight = 0
            self._cond.notify_all()
        if deleted:  # deleted while being extracted
            self.indexer.remove_chunks(deleted)
        return bool(result.get("chunks"))

    def _set_phase(self, phase: str) -> None:
        with self._cond:
            self.phase = phase

    def _error(self, message: str) -> None:
        logger.warning("Background graph build: %s", message)
        with self._cond:
            self.last_error = message
