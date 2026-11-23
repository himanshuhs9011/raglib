"""Chunking settings recorded in the index, mismatch warnings, and the token_counter deprecation."""

from __future__ import annotations

import json
import warnings

import pytest

import raglib.loaders.chunk_splitter as splitter_module
from raglib import RAG, ChunkSplitter, IndexSettingsWarning
from raglib.schemas import Document

TEXT = (
    "The refund policy gives customers fourteen days to return a product with its receipt. "
    "Store credit is offered when the receipt is missing, and opened software cannot be returned."
)


def _rag(**kwargs) -> RAG:
    return RAG(chat_llm="mock", embedding_llm="mock", token_counter="units", **kwargs)


def test_default_token_counter_warns_once_and_explicit_values_do_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(splitter_module, "_DEFAULT_COUNTER_WARNED", False)
    monkeypatch.setattr(splitter_module, "resolve_token_counter", lambda spec: (len, "tiktoken" if spec == "auto" else "units"))
    with pytest.warns(DeprecationWarning, match="will become 'auto'"):
        ChunkSplitter()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChunkSplitter()  # once per process
        monkeypatch.setattr(splitter_module, "_DEFAULT_COUNTER_WARNED", False)
        ChunkSplitter(token_counter="units")  # explicit: no warning
        RAG(chat_llm="mock", embedding_llm="mock", token_counter="units")


def test_default_counter_without_tiktoken_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(splitter_module, "_DEFAULT_COUNTER_WARNED", False)
    monkeypatch.setattr(splitter_module, "resolve_token_counter", lambda spec: (len, "units"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChunkSplitter()  # "auto" would also be "units": nothing changes, nothing to warn about


def test_settings_are_recorded_in_the_index_and_on_chunks(tmp_path) -> None:
    rag = _rag(chunk_size=60, chunk_overlap=10, chunking_strategy="embedding", semantic_breakpoint="gradient")
    rag.add(Document(id="policy", content=TEXT))
    settings = rag.chunking_settings()
    assert settings["chunk_size"] == 60 and settings["token_counter"] == "units"
    assert settings["semantic_breakpoint"] == "gradient" and settings["buffer_size"] == 2
    assert all(d.metadata["chunking_signature"] == rag.chunking_signature() for d in rag.retriever.documents)
    rag.save(str(tmp_path))
    index = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))["index"]
    assert index["chunking"] == json.loads(json.dumps(settings)) and index["chunking_signature"] == rag.chunking_signature()
    fixed = _rag(chunk_size=60).chunking_settings()
    assert "semantic_breakpoint" not in fixed  # semantic options only matter for the embedding strategy


def test_opening_an_index_with_other_settings_warns_with_the_changes(tmp_path) -> None:
    built = _rag(chunk_size=60, chunk_overlap=10)
    built.add(Document(id="policy", content=TEXT))
    built.save(str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _rag(chunk_size=60, chunk_overlap=10).load(str(tmp_path))  # same settings: silent
    with pytest.warns(IndexSettingsWarning) as caught:
        _rag(chunk_size=500, chunk_overlap=100).load(str(tmp_path))
    message = str(caught[0].message)
    assert "chunk_size: index=60 now=500" in message and "chunk_overlap: index=10 now=100" in message
    assert "save() to a new directory" in message and "opened" in message


def test_adding_to_an_index_with_other_settings_warns(tmp_path) -> None:
    built = _rag(chunk_size=60, chunk_overlap=10)
    built.add(Document(id="policy", content=TEXT))
    built.save(str(tmp_path))
    rag = RAG(chat_llm="mock", embedding_llm="mock", token_counter="auto", chunk_size=60, chunk_overlap=10)
    if rag.chunking_settings()["token_counter"] == "units":
        pytest.skip("tiktoken not installed: 'auto' equals 'units', so the settings match")
    with pytest.warns(IndexSettingsWarning, match="token_counter: index='units' now='tiktoken'"):
        rag.load(str(tmp_path))
    rag._settings_warned = False  # the next add() reports again for this test
    with pytest.warns(IndexSettingsWarning, match="added to"):
        rag.add(Document(id="shipping", content="Orders ship within two business days and arrive within a week."))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rag.add(Document(id="returns", content="Returns are free for all members of the loyalty programme."))  # once


def test_old_indexes_without_recorded_settings_load_silently(tmp_path) -> None:
    built = _rag()
    built.add(Document(id="policy", content=TEXT))
    built.save(str(tmp_path))
    payload = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    payload["index"].pop("chunking"), payload["index"].pop("chunking_signature")
    for chunk in payload["chunks"]:
        chunk["metadata"].pop("chunking_signature")
    (tmp_path / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _rag(chunk_size=500).load(str(tmp_path))
