"""Binary vector persistence: fast loading, old JSON indexes still load, damaged dumps are rejected."""

from __future__ import annotations

import json
import os

import pytest

from raglib import RAG
from raglib.vectorstores.base_store import VectorRecord
from raglib.vectorstores.in_memory_store import InMemoryVectorStore


def _store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.upsert([
        VectorRecord(document_id="a", embedding=[1.0, 0.0, 0.0], document_text="alpha", metadata={"k": 1}),
        VectorRecord(document_id="b", embedding=[0.0, 1.0, 0.0], document_text="beta", metadata={}),
        VectorRecord(document_id="c", embedding=[0.6, 0.8, 0.0], document_text="gamma", metadata={"x": "y"}),
    ])
    return store


def test_dump_restore_round_trip(tmp_path) -> None:
    store = _store()
    store.dump(str(tmp_path), "vectors")
    assert InMemoryVectorStore.has_dump(str(tmp_path), "vectors")
    restored = InMemoryVectorStore()
    assert restored.restore(str(tmp_path), "vectors") == 3
    assert restored.dimension == 3
    hits = restored.query([1.0, 0.1, 0.0], top_k=2)
    assert [h.document_id for h in hits] == [h.document_id for h in store.query([1.0, 0.1, 0.0], top_k=2)]
    assert hits[0].document_text == "alpha" and hits[0].metadata == {"k": 1}


def test_incomplete_dump_is_rejected(tmp_path) -> None:
    _store().dump(str(tmp_path), "vectors")
    path = os.path.join(str(tmp_path), "vectors.f32")
    with open(path, "rb") as handle:
        data = handle.read()
    with open(path, "wb") as handle:
        handle.write(data[:-4])  # one float missing
    with pytest.raises(ValueError, match="incomplete"):
        InMemoryVectorStore().restore(str(tmp_path), "vectors")


def test_rag_saves_binary_vectors_and_loads_them(tmp_path) -> None:
    rag = RAG(["Flexport is a logistics company.", "Travis Kelce plays for the Chiefs."], llm="mock", embedding_llm="mock")
    rag.save(str(tmp_path / "index"))
    files = set(os.listdir(tmp_path / "index"))
    assert {"vectors.f32", "vectors.meta.json"} <= files and "vectors.json" not in files
    loaded = RAG(llm="mock", embedding_llm="mock").load(str(tmp_path / "index"))
    assert loaded._retriever.vector_store.count() == rag._retriever.vector_store.count()


def test_indexes_saved_as_json_still_load(tmp_path) -> None:
    """Indexes written before binary vectors keep working."""

    rag = RAG(["Flexport is a logistics company."], llm="mock", embedding_llm="mock")
    index = tmp_path / "index"
    rag.save(str(index))
    store = rag._retriever.vector_store
    legacy = [
        {"document_id": r.document_id, "embedding": list(r.embedding), "document_text": r.document_text, "metadata": r.metadata}
        for r in store.records()
    ]
    (index / "vectors.json").write_text(json.dumps(legacy), encoding="utf-8")
    os.remove(index / "vectors.f32")
    os.remove(index / "vectors.meta.json")
    loaded = RAG(llm="mock", embedding_llm="mock").load(str(index))
    assert loaded._retriever.vector_store.count() == 1


def test_graph_entity_vectors_are_binary(tmp_path) -> None:
    from tests.unit.test_graph_llm import _Recorder

    rag = RAG(["Ryan Petersen returned as CEO of Flexport in 2023."], chat_llm=_Recorder("answers"),
              graph_llm=_Recorder("graph"), embedding_llm="mock", rag_type="graph", llm_cache=False)
    rag.save(str(tmp_path / "index"))
    graph_dir = tmp_path / "index" / "graph"
    assert InMemoryVectorStore.has_dump(str(graph_dir), "entity_vectors")
    state = json.loads((graph_dir / "graph_state.json").read_text(encoding="utf-8"))
    assert "entity_vectors" not in state
    loaded = RAG(chat_llm=_Recorder("answers"), graph_llm=_Recorder("graph"), embedding_llm="mock",
                 rag_type="graph", llm_cache=False).load(str(tmp_path / "index"))
    names = {entity.name for entity, _ in loaded._graph_indexer.entity_search("Flexport CEO", top_k=3)}
    assert "Flexport" in names
