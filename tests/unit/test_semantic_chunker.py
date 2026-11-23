"""Embedding chunker correctness (Phase A): failures, token counting, formatting, blocks, abbreviations, versions.

Fake embeddings only, no network.
"""

from __future__ import annotations

from typing import List

import pytest

from raglib import RAG, LatencyStats
from raglib.core.text import count_units, resolve_token_counter
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.loaders.chunk_splitter import ChunkSplitter, EmbeddingChunkingError
from raglib.schemas import Document

CATS = [
    "Cats sleep for most of the day in warm places.",
    "A cat grooms its fur with a rough tongue.",
    "Kittens learn to hunt by playing with their siblings.",
    "Most cats dislike water and avoid baths.",
]
ROCKETS = [
    "Rockets burn fuel to produce thrust for launch.",
    "A rocket engine needs an oxidizer to burn in space.",
    "The launch tower holds the rocket before liftoff.",
    "Rocket stages separate after their fuel is spent.",
]


class TopicEmbedding(BaseEmbedding):
    """2-d vectors counting cat and rocket words; optionally fails its first ``fail_first`` calls."""

    def __init__(self, fail_first: int = 0) -> None:
        self.calls = 0
        self.fail_first = fail_first

    def _vector(self, text: str) -> List[float]:
        lower = text.lower()
        return [
            float(sum(lower.count(word) for word in ("cat", "kitten", "fur", "pet"))) + 0.01,
            float(sum(lower.count(word) for word in ("rocket", "launch", "fuel", "thrust"))) + 0.01,
        ]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("embedding API unavailable")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)

    @property
    def dimension(self) -> int:
        return 2


def _splitter(**kwargs) -> ChunkSplitter:
    options = dict(chunk_size=200, overlap=0, min_tokens=1, strategy="embedding", embedding=TopicEmbedding(), semantic_min_tokens=3)
    options.update(kwargs)
    return ChunkSplitter(**options)


# ------------------------------------------------------------------- 1. no silent fallback


def test_embedding_failure_raises_when_asked() -> None:
    splitter = _splitter(embedding=TopicEmbedding(fail_first=1), on_embedding_failure="raise")
    with pytest.raises(EmbeddingChunkingError, match="doc"):
        splitter.split([Document(id="doc", content=" ".join(CATS + ROCKETS))])
    with pytest.raises(ValueError, match="embedding model"):
        ChunkSplitter(strategy="embedding", on_embedding_failure="raise")
    with pytest.raises(ValueError):
        ChunkSplitter(strategy="embedding", embedding=TopicEmbedding(), on_embedding_failure="ignore")


def test_embedding_failure_fallback_is_labelled_and_reported() -> None:
    reported = []
    splitter = _splitter(embedding=TopicEmbedding(fail_first=1), on_fallback=lambda doc, err: reported.append((doc.id, err)))
    chunks = splitter.split([Document(id="doc", content=" ".join(CATS + ROCKETS))])
    assert chunks and {chunk.metadata["chunking_strategy"] for chunk in chunks} == {"recursive_fallback"}
    assert reported and reported[0][0] == "doc" and isinstance(reported[0][1], ConnectionError)


def test_rag_records_fallbacks_in_ingest_usage_and_on_event() -> None:
    events = []
    stats = LatencyStats()
    # max_retries=0: CachedEmbedding would otherwise retry the transient error (and succeed).
    rag = RAG(chat_llm="mock", embedding_llm=CachedEmbedding(TopicEmbedding(fail_first=1), max_retries=0),
              chunking_strategy="embedding", on_event=[events.append, stats])
    rag.add(Document(id="doc", content=" ".join(CATS + ROCKETS)))
    usage = rag.ingest_usage()
    assert usage["chunking_fallbacks"] == 1 and usage["chunking_fallback_documents"] == ["doc"]
    warning = [event for event in events if event.stage == "chunking_fallback"]
    assert warning and warning[0].meta["document_id"] == "doc" and "ConnectionError" in warning[0].error
    assert "chunking_fallback" not in stats.stages()  # a notice, not a latency sample
    assert {d.metadata["chunking_strategy"] for d in rag.retriever.documents} == {"recursive_fallback"}

    strict = RAG(chat_llm="mock", embedding_llm=CachedEmbedding(TopicEmbedding(fail_first=1), max_retries=0),
                 chunking_strategy="embedding", on_embedding_failure="raise")
    with pytest.raises(EmbeddingChunkingError):
        strict.add(Document(id="doc", content=" ".join(CATS + ROCKETS)))


def test_production_settings_use_raise_and_real_tokens() -> None:
    from raglib.presets import resolve_preset

    values = resolve_preset("production")
    assert values["on_embedding_failure"] == "raise" and values["token_counter"] == "auto"


# ------------------------------------------------------------------- 2. token counting


def test_token_counter_options() -> None:
    assert resolve_token_counter("units")[0] is count_units
    counter, name = resolve_token_counter("auto")
    assert name in {"tiktoken", "units"}
    if name == "tiktoken":
        assert counter("Hello world, tokens!") == 5  # cl100k_base: not the 3 words "units" would count
    with pytest.raises(ValueError):
        resolve_token_counter("bpe")
    chars = _splitter(chunk_size=80, token_counter=len)  # a callable: sizes in characters
    chunks = chars.split([Document(id="doc", content=" ".join(CATS + ROCKETS))])
    assert all(len(chunk.content) <= 80 for chunk in chunks) and chars.token_counter_name == "len"


def test_chunk_size_and_cost_share_the_counter() -> None:
    def counter(text: str) -> int:
        return len(text) // 4

    rag = RAG(chat_llm="mock", embedding_llm=TopicEmbedding(), chunking_strategy="embedding", token_counter=counter,
              chunk_size=60, semantic_min_tokens=3)
    rag.add(Document(id="doc", content=" ".join(CATS + ROCKETS)))
    assert rag._splitter.token_counter is counter and rag._embedding.token_counter is counter
    assert all(counter(d.content) <= 60 for d in rag.retriever.documents)
    assert rag._splitter._semantic_min_tokens == 3  # semantic_min_tokens is exposed on RAG(...)
    words = RAG(chat_llm="mock", embedding_llm=TopicEmbedding())  # "units": cost keeps the token estimator
    assert words._embedding.token_counter is None


# -------------------------------------------------- 3. + 4. formatting, offsets, tables and lists

DOC = """Our pricing changed this year. Every plan is now billed monthly.

| Plan | Price | Seats |
|------|-------|-------|
| Basic | $10 | 1 |
| Team | $50 | 10 |

The steps to upgrade are:
- Open the billing page
- Pick a new plan
- Confirm the payment

Rockets burn fuel to produce thrust for launch. The launch tower holds the rocket before liftoff."""


def test_chunks_are_exact_slices_with_offsets_and_formatting() -> None:
    document = Document(id="doc", content=DOC)
    chunks = _splitter(chunk_size=60, semantic_min_tokens=1).split([document])
    assert len(chunks) > 1
    for chunk in chunks:
        start, end = chunk.metadata["start_char"], chunk.metadata["end_char"]
        assert DOC[start:end] == chunk.content
    table = next(chunk for chunk in chunks if "| Plan |" in chunk.content)
    assert "| Plan | Price | Seats |\n|------|-------|-------|\n| Basic | $10 | 1 |\n| Team | $50 | 10 |" in table.content
    steps = next(chunk for chunk in chunks if "- Open the billing page" in chunk.content)
    assert "- Open the billing page\n- Pick a new plan\n- Confirm the payment" in steps.content


def test_tables_and_lists_are_single_units_unless_too_large() -> None:
    splitter = _splitter()
    units = [DOC[a:b] for a, b in splitter._semantic_units(DOC)]
    assert any(unit.startswith("| Plan |") and unit.endswith("| Team | $50 | 10 |") for unit in units)
    assert "- Open the billing page\n- Pick a new plan\n- Confirm the payment" in units
    big_table = "\n".join(["| Name | Value |", "|---|---|"] + [f"| row{i} | value number {i} |" for i in range(40)])
    small = _splitter(chunk_size=30)
    fitted = [big_table[a:b] for a, b in small._fit_units(big_table, small._semantic_units(big_table))]
    assert len(fitted) > 1 and all(count_units(piece) <= 30 for piece in fitted)
    assert all(line.startswith("|") and line.endswith("|") for piece in fitted for line in piece.splitlines())


def test_loose_lists_with_blank_lines_and_numbered_items_stay_together() -> None:
    text = "Intro sentence here.\n\n1. First step\n\n2. Second step\n   continued detail\n3) Third step\n\nAfter the list."
    units = [text[a:b] for a, b in _splitter()._semantic_units(text)]
    assert "1. First step\n\n2. Second step\n   continued detail\n3) Third step" in units
    assert units[0] == "Intro sentence here." and units[-1] == "After the list."


# ------------------------------------------------------------------- 5. abbreviations + scripts


def test_abbreviations_do_not_end_sentences() -> None:
    text = "Dr. Sharma joined Acme Inc. in 2020. He works with Mr. Rao, e.g. on audits. No. 5 is next."
    units = [text[a:b] for a, b in _splitter()._semantic_units(text)]
    assert units == ["Dr. Sharma joined Acme Inc. in 2020.", "He works with Mr. Rao, e.g. on audits.", "No. 5 is next."]
    custom = _splitter(abbreviations=["mr."])
    assert [text[a:b] for a, b in custom._semantic_units(text)][0] == "Dr."  # Dr. no longer protected


def test_hindi_danda_ends_sentences() -> None:
    text = "राम घर गया। सीता बाजार गई। बच्चे खेल रहे हैं॥ अंत"
    units = [text[a:b] for a, b in _splitter()._semantic_units(text)]
    assert units == ["राम घर गया।", "सीता बाजार गई।", "बच्चे खेल रहे हैं॥", "अंत"]


# ------------------------------------------------------------------- 6. document_version on chunks


def test_document_version_is_stamped_on_chunks_and_sections() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", parent_retrieval="heading", chunk_size=20, chunk_overlap=0)
    rag.add(Document(id="handbook", content="# Refunds\nRefunds are paid within fourteen days of the return.\n\n# Shipping\nOrders ship within two days."))
    version = rag.document_version("handbook")
    assert version and all(d.metadata["document_version"] == version for d in rag.retriever.documents)
    assert all(s.metadata["document_version"] == version for s in rag._sections.values())
    stored = rag.retriever.vector_store.query(rag._embedding.embed_query("refunds"), top_k=5)
    assert all(hit.metadata["document_version"] == version for hit in stored)  # filterable in the vector store
    result = rag.query("When are refunds paid?", filters={"document_version": version})
    assert result.sources
    rag.add(Document(id="handbook", content="# Refunds\nRefunds are paid within thirty days of the return."))
    assert rag.document_version("handbook") != version
    assert all(d.metadata["document_version"] == rag.document_version("handbook") for d in rag.retriever.documents)


# ------------------------------------------------------------------- topic changes still found


@pytest.mark.parametrize("breakpoint,threshold", [("percentile", 95.0), ("fixed", 0.2)])
def test_topic_change_found_on_offsets(breakpoint: str, threshold: float) -> None:
    text = " ".join(CATS) + "\n" + " ".join(ROCKETS)
    chunks = _splitter(semantic_breakpoint=breakpoint, semantic_threshold=threshold).split([Document(id="d", content=text)])
    assert [("rocket" in c.content.lower(), "cat" in c.content.lower()) for c in chunks] == [(False, True), (True, False)]
