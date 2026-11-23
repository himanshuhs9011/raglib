"""Embedding chunker options (Phase B): buffer size, threshold types, number_of_chunks, custom splitter, overlap."""

from __future__ import annotations

import re
from typing import List

import pytest

from raglib import RAG
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.loaders.chunk_splitter import SEMANTIC_THRESHOLD_DEFAULTS, ChunkSplitter, _gradient
from raglib.schemas import Document

TOPICS = {
    "cats": ["cat", "kitten", "fur", "purr"],
    "rockets": ["rocket", "launch", "fuel", "thrust"],
    "cooking": ["oven", "flour", "dough", "bake"],
}


def _sentences(topic: str, count: int = 5) -> List[str]:
    words = TOPICS[topic]
    return [f"The {words[i % 4]} and the {words[(i + 1) % 4]} were noted on day {i}." for i in range(count)]


# Three topics, two known changes: after sentence 5 and after sentence 10.
DOC = " ".join(_sentences("cats") + _sentences("rockets") + _sentences("cooking"))


class TopicEmbedding(BaseEmbedding):
    """One dimension per topic: the count of its words (no network)."""

    def __init__(self) -> None:
        self.texts: List[str] = []

    def _vector(self, text: str) -> List[float]:
        lower = text.lower()
        return [float(sum(lower.count(word) for word in words)) + 0.01 for words in TOPICS.values()]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.texts.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)

    @property
    def dimension(self) -> int:
        return len(TOPICS)


def _split(**kwargs) -> List[Document]:
    options = dict(chunk_size=300, overlap=0, min_tokens=1, strategy="embedding", embedding=TopicEmbedding(), semantic_min_tokens=5)
    options.update(kwargs)
    return ChunkSplitter(**options).split([Document(id="doc", content=DOC)])


def _topics(chunks: List[Document]) -> List[str]:
    return ["+".join(name for name, words in TOPICS.items() if any(w in c.content.lower() for w in words)) for c in chunks]


@pytest.mark.parametrize(
    "breakpoint,threshold",
    [
        ("percentile", 80.0),
        ("fixed", 0.5),
        ("standard_deviation", 1.0),
        ("interquartile", 1.5),
        ("gradient", 80.0),
    ],
)
def test_every_threshold_type_finds_the_known_topic_changes(breakpoint: str, threshold: float) -> None:
    chunks = _split(semantic_breakpoint=breakpoint, semantic_threshold=threshold)
    assert _topics(chunks) == ["cats", "rockets", "cooking"], breakpoint


def test_threshold_defaults_and_validation() -> None:
    assert SEMANTIC_THRESHOLD_DEFAULTS == {
        "percentile": 95.0, "fixed": 0.75, "standard_deviation": 3.0, "interquartile": 1.5, "gradient": 95.0,
    }
    splitter = ChunkSplitter(strategy="embedding", embedding=TopicEmbedding(), semantic_breakpoint="interquartile")
    assert splitter._semantic_threshold == 1.5
    for breakpoint, bad in [("gradient", 120), ("standard_deviation", -1), ("fixed", 2)]:
        with pytest.raises(ValueError):
            ChunkSplitter(strategy="embedding", embedding=TopicEmbedding(), semantic_breakpoint=breakpoint, semantic_threshold=bad)
    assert _gradient([0.0, 1.0, 4.0]) == [1.0, 2.0, 3.0]  # same as numpy.gradient


def test_buffer_size_sets_the_window_on_each_side_of_a_gap() -> None:
    for width in (1, 2, 3):
        embedding = TopicEmbedding()
        _split(embedding=embedding, buffer_size=width, semantic_breakpoint="fixed", semantic_threshold=0.5)
        sentence_counts = {len(re.findall(r"day \d+\.", text)) for text in embedding.texts}
        assert max(sentence_counts) == width  # windows hold at most buffer_size sentences
    assert _topics(_split(buffer_size=1, semantic_breakpoint="fixed", semantic_threshold=0.5)) == ["cats", "rockets", "cooking"]
    with pytest.raises(ValueError):
        ChunkSplitter(strategy="embedding", embedding=TopicEmbedding(), buffer_size=0)


def test_number_of_chunks_targets_an_approximate_count() -> None:
    assert len(_split(number_of_chunks=3)) == 3 and _topics(_split(number_of_chunks=3)) == ["cats", "rockets", "cooking"]
    assert len(_split(number_of_chunks=1)) == 1
    assert len(_split(number_of_chunks=6)) == 6
    capped = _split(number_of_chunks=1, chunk_size=40)  # size limits still apply: the count is approximate
    assert len(capped) > 1
    with pytest.raises(ValueError):
        ChunkSplitter(strategy="embedding", embedding=TopicEmbedding(), number_of_chunks=0)


def test_custom_sentence_splitter_regex_and_callable() -> None:
    text = "alpha one; alpha two; beta three; beta four"
    regex = ChunkSplitter(strategy="embedding", embedding=TopicEmbedding(), sentence_splitter=r";\s*")
    assert [text[a:b] for a, b in regex._semantic_units(text)] == ["alpha one", "alpha two", "beta three", "beta four"]
    callable_splitter = ChunkSplitter(strategy="embedding", embedding=TopicEmbedding(), sentence_splitter=lambda t: t.split("; "))
    assert [text[a:b] for a, b in callable_splitter._semantic_units(text)] == ["alpha one", "alpha two", "beta three", "beta four"]
    offsets = ChunkSplitter(strategy="embedding", embedding=TopicEmbedding(), sentence_splitter=lambda t: [(0, 9), (11, 20)])
    assert [text[a:b] for a, b in offsets._semantic_units(text)] == ["alpha one", "alpha two"]
    broken = ChunkSplitter(strategy="embedding", embedding=TopicEmbedding(), sentence_splitter=lambda t: ["not in the text"])
    with pytest.raises(ValueError, match="sentence_splitter"):
        broken._semantic_units(text)
    table = "Intro text here.\n| a | b |\n| c | d |"  # blocks still win over the custom splitter
    assert "| a | b |\n| c | d |" in [table[a:b] for a, b in regex._semantic_units(table)]


def test_sentence_overlap_repeats_the_tail_of_the_previous_chunk() -> None:
    plain = _split(number_of_chunks=3)
    overlapped = _split(number_of_chunks=3, semantic_overlap_sentences=1)
    assert len(overlapped) == 3
    for previous, chunk in zip(overlapped, overlapped[1:]):
        last_sentence = re.findall(r"[^.]+\.", plain[overlapped.index(previous)].content)[-1].strip()
        assert chunk.content.startswith(last_sentence)
        assert DOC[chunk.metadata["start_char"] : chunk.metadata["end_char"]] == chunk.content
    tight = _split(number_of_chunks=3, semantic_overlap_sentences=5, chunk_size=60)  # never beyond chunk_size
    assert all(chunk.metadata["chunk_tokens"] <= 60 for chunk in tight)


def test_rag_exposes_the_options() -> None:
    rag = RAG(chat_llm="mock", embedding_llm=TopicEmbedding(), chunking_strategy="embedding", semantic_buffer_size=1,
              semantic_number_of_chunks=3, semantic_overlap_sentences=1, sentence_splitter=r"(?<=\.)\s+",
              semantic_breakpoint="gradient", semantic_min_tokens=5)
    rag.add(Document(id="doc", content=DOC))
    splitter = rag._splitter
    assert (splitter._buffer_size, splitter._number_of_chunks, splitter._semantic_overlap) == (1, 3, 1)
    assert rag.stats()["chunks"] == 3
