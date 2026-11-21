"""graph_build="background": answer immediately from hybrid search while the graph is built."""

from __future__ import annotations

import threading
import time

import pytest

from raglib import RAG
from tests.unit.test_graph_llm import _Recorder

DOCS = [
    "Ryan Petersen returned as CEO of Flexport in 2023.",
    "Travis Kelce plays for the Kansas City Chiefs.",
    "OpenAI released ChatGPT in November 2022.",
]


class _GatedExtractor(_Recorder):
    """Extraction blocks until ``gate`` is set, like a slow provider building a large graph."""

    def __init__(self) -> None:
        super().__init__("graph")
        self.gate = threading.Event()

    def complete(self, prompt: str, system: str = "") -> str:
        if "knowledge graph extraction" in prompt:
            assert self.gate.wait(10), "test gate never opened"
        return super().complete(prompt, system)


def _rag(extractor, docs=DOCS, **kwargs):
    return RAG(docs, chat_llm=_Recorder("answers"), graph_llm=extractor, embedding_llm="mock", rag_type="graph",
               llm_cache=False, graph_build="background", **kwargs)


def test_queries_are_answered_while_the_graph_builds() -> None:
    extractor = _GatedExtractor()
    started = time.time()
    rag = _rag(extractor)
    assert time.time() - started < 5, "adding documents does not wait for graph extraction"
    status = rag.graph_status()
    assert status["state"] == "building" and status["pending_chunks"] > 0

    result = rag.query("Who leads Flexport?")  # hybrid search answers; the graph is still empty
    assert result.answer
    assert any("Flexport" in doc.content for doc in result.sources)

    extractor.gate.set()
    assert rag.wait_for_graph(timeout=20)
    status = rag.graph_status()
    assert status["state"] == "ready" and status["pending_chunks"] == 0 and status["phase"] == "idle"
    assert rag.stats()["graph"]["entities"] > 0
    assert rag.stats()["graph"]["build"] == "ready"


def test_save_during_build_and_load_requeues_the_rest(tmp_path) -> None:
    extractor = _GatedExtractor()
    rag = _rag(extractor)
    rag.save(str(tmp_path / "index"))  # nothing extracted yet
    extractor.gate.set()
    rag.wait_for_graph(timeout=20)

    second = _GatedExtractor()
    second.gate.set()
    loaded = RAG(chat_llm=_Recorder("answers"), graph_llm=second, embedding_llm="mock", rag_type="graph",
                 llm_cache=False, graph_build="background").load(str(tmp_path / "index"))
    assert loaded.wait_for_graph(timeout=20)
    assert loaded.stats()["graph"]["entities"] > 0, "chunks missing from the saved graph were queued again"


def test_deleted_documents_leave_the_queue() -> None:
    extractor = _GatedExtractor()
    rag = _rag(extractor, docs=[])
    rag.add(DOCS[:1])
    rag.add(DOCS[1:])
    rag.delete(DOCS[1])  # deleted before it was extracted
    extractor.gate.set()
    assert rag.wait_for_graph(timeout=20)
    texts = " ".join(c.content for c in rag._retriever.documents)
    assert "Kelce" not in texts
    assert not any("Kelce" in str(rag._retriever.get(i)) for i in rag._graph_indexer.indexed_chunk_ids)


def test_graph_build_option_is_validated() -> None:
    with pytest.raises(ValueError, match="graph_build"):
        RAG(llm="mock", embedding_llm="mock", rag_type="graph", graph_build="later")
    with pytest.raises(ValueError, match="graph_build needs"):
        RAG(llm="mock", embedding_llm="mock", graph_build="background")
    sync = RAG(DOCS[:1], chat_llm=_Recorder("a"), graph_llm=_Recorder("g"), embedding_llm="mock", rag_type="graph",
               llm_cache=False)
    assert sync.graph_status()["state"] == "ready" and sync.wait_for_graph() is True
