"""Out-of-the-box fixes: the production preset fits chunk_size to the embedding model, existing indexes are
not re-chunked, the 2.2 sentence rule for "fixed", web citations under access control, re-tagging a
v2.2 index, and no unit longer than the model limit (offline, no network)."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import List, Optional

import pytest

from raglib import RAG, EmbeddingLimitError, EmbeddingLimitWarning, UserContext, access_tags
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.guards import CitationAccessGuard, GuardContext
from raglib.loaders.chunk_splitter import SPLITTER_VERSION, ChunkSplitter
from raglib.providers.base_provider import BaseSearchProvider
from raglib.schemas import Document

LONG = " ".join(f"word{i}." if i % 9 == 8 else f"word{i}" for i in range(1500))


class Tokenized(BaseEmbedding):
    """A fake model: ``per_word`` tokens per word (or ~4 characters per token), at most ``limit`` tokens."""

    def __init__(self, limit: int, name: str = "all-MiniLM-L6-v2", per_word: Optional[float] = 1.3) -> None:
        self.limit, self._model_name, self.per_word = limit, name, per_word
        self.seen: List[str] = []

    @property
    def max_input_tokens(self) -> Optional[int]:
        return self.limit

    def count_input_tokens(self, text: str) -> Optional[int]:
        if self.per_word is None:
            return math.ceil(len(text) / 4)
        return math.ceil(self.per_word * len(text.split()))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.seen.extend(texts)
        return [[float(len(t) % 7) + 1.0, float(len(t.split()) % 5), 1.0] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        return 3


# ------------------------------------------------------------------ #5 production preset + small embedding model


def test_production_preset_fits_chunk_size_to_a_256_token_model() -> None:
    model = Tokenized(limit=256)
    with warnings.catch_warnings():
        warnings.simplefilter("error", EmbeddingLimitWarning)  # no warning, no error: it just fits
        rag = RAG(preset="production", chat_llm="mock", embedding_llm=model, reranker="none")
    assert rag._splitter._chunk_size < 500 and rag._splitter._overlap == rag._splitter._chunk_size // 5
    rag.add(Document(id="doc", content=LONG))
    assert rag.retriever.documents
    assert all(model.count_input_tokens(text) <= 256 for text in model.seen)


def test_production_preset_raises_for_an_explicit_chunk_size_that_cannot_fit() -> None:
    with pytest.raises(EmbeddingLimitError, match="chunk_size<="):
        RAG(preset="production", chat_llm="mock", embedding_llm=Tokenized(limit=256), reranker="none", chunk_size=500)


# ------------------------------------------------------------------ existing indexes are not re-chunked


def _legacy_index(tmp_path: Path, text: str) -> Path:
    """An index as raglib 2.2 saved it: no chunking signatures, no permission map."""

    unknown = Tokenized(limit=None)  # 2.2 enforced no limit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = RAG(chat_llm="mock", embedding_llm=unknown, token_counter="units", chunk_size=400, chunk_overlap=50)
    old.add(Document(id="doc", content=text))
    folder = tmp_path / "index"
    old.save(str(folder))
    chunks_file = folder / "chunks.json"
    payload = json.loads(chunks_file.read_text(encoding="utf-8"))
    for chunk in payload["chunks"]:
        chunk["metadata"].pop("chunking_signature", None)
    payload.pop("acl", None)
    payload.pop("chunking_signature", None)
    if isinstance(payload.get("settings"), dict):
        payload["settings"].pop("chunking", None)
    chunks_file.write_text(json.dumps(payload), encoding="utf-8")
    return folder


def test_an_index_built_before_2_3_is_not_rechunked_by_default(tmp_path: Path) -> None:
    folder = _legacy_index(tmp_path, LONG)
    model = Tokenized(limit=256)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rag = RAG(chat_llm="mock", embedding_llm=model, token_counter="units", chunk_size=400, chunk_overlap=50)
        rag.load(str(folder), allow_model_change=True)
        before = sorted(doc.id for doc in rag.retriever.documents)
        model.seen.clear()
        added = rag.add(Document(id="doc", content=LONG))
    assert added == 0 and sorted(doc.id for doc in rag.retriever.documents) == before  # same chunks
    assert not model.seen  # nothing re-embedded
    assert any("kept whole" in str(w.message) for w in caught if issubclass(w.category, EmbeddingLimitWarning))


def test_new_indexes_and_explicit_split_do_split(tmp_path: Path) -> None:
    model = Tokenized(limit=256)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = RAG(chat_llm="mock", embedding_llm=model, token_counter="units", chunk_size=400, chunk_overlap=50)
    fresh.add(Document(id="doc", content=LONG))
    assert fresh.ingest_usage()["oversized_chunks_split"] > 0
    assert all(model.count_input_tokens(text) <= 256 for text in model.seen)
    folder = _legacy_index(tmp_path, LONG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opted = RAG(chat_llm="mock", embedding_llm=Tokenized(limit=256), token_counter="units", chunk_size=400,
                    chunk_overlap=50, oversized_chunks="split")
        opted.load(str(folder), allow_model_change=True)
        assert opted.add(Document(id="doc", content=LONG)) > 0  # opted in: re-chunked to fit


# ------------------------------------------------------------------ the 2.2 sentence rule for "fixed"


def test_fixed_keeps_the_2_2_sentence_rule_and_semantic_strategies_use_the_new_one() -> None:
    text = "पहला वाक्य ॥ दूसरा वाक्य। तीसरा"
    fixed = ChunkSplitter(chunk_size=50, overlap=0, strategy="fixed", token_counter="units")
    semantic = ChunkSplitter(chunk_size=50, overlap=0, strategy="embedding", token_counter="units",
                             on_embedding_failure="fallback")
    assert [s.strip() for s in fixed._split_sentence_segments(text)] == ["पहला वाक्य ॥ दूसरा वाक्य।", "तीसरा"]
    assert [s.strip() for s in semantic._split_sentence_segments(text)] == ["पहला वाक्य ॥", "दूसरा वाक्य।", "तीसरा"]
    assert fixed.settings()["splitter_version"] == SPLITTER_VERSION == 2


# ------------------------------------------------------------------ #7 web citations under access control


class Provider(BaseSearchProvider):
    def __init__(self, internal: bool = False) -> None:
        self.internal = internal

    @property
    def name(self) -> str:
        return "intranet" if self.internal else "publicweb"

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        return [Document(id="w1", content="Paris weather today: sunny, 21 degrees.", source="https://example.com/w")]


@pytest.mark.parametrize("internal", [False, True])
def test_web_results_are_public_and_internal_ones_restricted(internal: bool) -> None:
    rag = RAG(rag_type="web", chat_llm="mock", embedding_llm="mock", web_search_provider="local", access_control=True,
              token_counter="units")
    rag._web_retriever.provider = Provider(internal=internal)
    result = rag.query("What is the Paris weather today?", user=UserContext(user_id="u", max_access_level="executive"))
    assert result.sources or result.metadata.get("blocked_by")
    if internal:
        assert result.metadata.get("blocked_by") == "citation_access"  # untagged internal result: restricted
    else:
        assert "blocked_by" not in result.metadata and result.sources[0].metadata["source_type"] == "web"
        assert not rag._private_document(result.sources[0])
    tagged = Document(id="w", content="x", metadata={"access_level": "public", "source_type": "web"})
    guard = CitationAccessGuard()
    guard.bind(rag)
    context = GuardContext(stage="output", documents=[tagged], permission_filter=rag.permission_filter(None))
    assert guard.check("Sunny [1].", context).action == "allow"


# ------------------------------------------------------------------ #13 re-tagging a v2.2 index


def test_readding_with_access_tags_applies_them_to_a_2_2_index(tmp_path: Path) -> None:
    text = "The office opens at eight in the morning on weekdays."
    folder = _legacy_index(tmp_path, text)
    rag = RAG(chat_llm="mock", embedding_llm=Tokenized(limit=None), token_counter="units", chunk_size=400,
              chunk_overlap=50, access_control=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rag.load(str(folder), allow_model_change=True)
    assert rag.query("When does the office open?").sources == []  # untagged: invisible (deny by default)
    rag.add(Document(id="doc", content=text), access=access_tags("public"))
    assert rag.query("When does the office open?").sources  # the tags were applied


# ------------------------------------------------------------------ #14 one unit longer than the model limit


def test_a_single_long_unit_is_cut_to_the_model_limit() -> None:
    blob = "data:image/png;base64," + "QUJD" * 300  # 1,222 characters, no whitespace: one "unit"
    model = Tokenized(limit=50, name="char-model", per_word=None)  # ~4 characters per token
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rag = RAG(chat_llm="mock", embedding_llm=model, token_counter="units", chunk_size=40, chunk_overlap=0)
        rag.add(Document(id="doc", content=f"Intro text. {blob} Outro text."))
    assert model.seen and all(model.count_input_tokens(text) <= 50 for text in model.seen)
    assert "".join(doc.content for doc in rag.retriever.documents).count("QUJD") == 300  # nothing lost
