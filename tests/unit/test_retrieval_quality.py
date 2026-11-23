"""Phase 5: embedding-based semantic chunking, parent-section retrieval, model routing by tier."""

from __future__ import annotations

from typing import List

import pytest

from raglib import RAG, HeuristicTierRouter, LatencyStats, TierDecision, access_tags
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.llm.mock_client import MockLLMClient
from raglib.loaders.chunk_splitter import ChunkSplitter
from raglib.loaders.sections import build_sections, split_sections
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
    """Fake embedding: a 2-d vector counting cat words and rocket words (no network)."""

    def __init__(self) -> None:
        self.calls = 0

    def _vector(self, text: str) -> List[float]:
        lower = text.lower()
        return [
            float(sum(lower.count(word) for word in ("cat", "kitten", "fur"))) + 0.01,
            float(sum(lower.count(word) for word in ("rocket", "launch", "fuel", "thrust"))) + 0.01,
        ]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)

    @property
    def dimension(self) -> int:
        return 2


# --------------------------------------------------------------------- semantic chunking


@pytest.mark.parametrize("breakpoint,threshold", [("percentile", 95.0), ("fixed", 0.2)])
def test_embedding_chunking_breaks_where_the_topic_changes(breakpoint: str, threshold: float) -> None:
    splitter = ChunkSplitter(
        chunk_size=200, overlap=0, min_tokens=1, strategy="embedding", embedding=TopicEmbedding(),
        semantic_breakpoint=breakpoint, semantic_threshold=threshold, semantic_min_tokens=5,
    )
    chunks = splitter.split([Document(id="doc", content=" ".join(CATS + ROCKETS))])
    assert len(chunks) == 2
    assert all(word not in chunks[0].content.lower() for word in ("rocket", "launch"))
    assert "rocket" in chunks[1].content.lower() and "cat" not in chunks[1].content.lower()
    assert chunks[0].metadata["chunking_strategy"] == "embedding"


def test_embedding_chunking_enforces_max_and_min_chunk_tokens() -> None:
    splitter = ChunkSplitter(chunk_size=25, overlap=0, min_tokens=1, strategy="embedding", embedding=TopicEmbedding())
    chunks = splitter.split([Document(id="doc", content=" ".join(CATS + ROCKETS))])
    assert len(chunks) > 2 and all(chunk.metadata["chunk_tokens"] <= 25 for chunk in chunks)


def test_embedding_chunking_reuses_the_embedding_cache_and_falls_back_without_a_model() -> None:
    inner = TopicEmbedding()
    splitter = ChunkSplitter(chunk_size=200, min_tokens=1, overlap=0, strategy="embedding", embedding=CachedEmbedding(inner))
    document = Document(id="doc", content=" ".join(CATS + ROCKETS))
    splitter.split([document])
    splitter.split([document])
    assert inner.calls == 1  # unchanged text: sentence windows come from the cache
    plain = ChunkSplitter(chunk_size=200, min_tokens=1, overlap=0, strategy="embedding")
    assert plain.split([document])  # no embedding: recursive fallback
    with pytest.raises(ValueError):
        ChunkSplitter(strategy="embedding", semantic_breakpoint="median")
    with pytest.raises(ValueError):
        ChunkSplitter(strategy="embedding", semantic_breakpoint="fixed", semantic_threshold=3)


def test_rag_uses_its_embedding_model_for_semantic_chunking() -> None:
    rag = RAG(chat_llm="mock", embedding_llm=TopicEmbedding(), chunking_strategy="embedding", chunk_size=200,
              semantic_threshold=95)
    rag.add(" ".join(CATS * 3 + ROCKETS * 3))
    assert rag.stats()["chunks"] >= 2


# ------------------------------------------------------------------------ parent sections

HANDBOOK = (
    "# Refunds\n"
    "Refunds are paid within fourteen days. The receipt must be included with every returned item. "
    "Store credit is offered when the receipt is missing. Opened software cannot be refunded.\n\n"
    "# Shipping\n"
    "Orders ship within two business days. Express shipping costs extra and arrives the next day. "
    "International orders may take up to three weeks and pay customs on delivery.\n"
)


def test_sections_split_at_headings_and_fall_back_to_fixed_windows() -> None:
    sections = split_sections(HANDBOOK, "heading", 2000)
    assert [title for title, _ in sections] == ["Refunds", "Shipping"]
    long = split_sections("word " * 1000, "heading", 300)  # no headings: fixed windows
    assert len(long) >= 3 and all(len(text.split()) <= 300 for _, text in long)
    doc = Document(id="handbook", content=HANDBOOK, metadata={"file": "handbook.md"})
    built = build_sections(doc, "heading", 2000)
    assert built[0].id.startswith("handbook#s1_") and built[0].metadata["parent_id"] == "handbook"
    with pytest.raises(ValueError):
        split_sections(HANDBOOK, "chapter", 100)


def test_parent_retrieval_returns_whole_deduplicated_sections() -> None:
    rag = RAG(offline=True, parent_retrieval="heading", chunk_size=20, chunk_overlap=0, top_k=2)
    rag.add(Document(id="handbook", content=HANDBOOK))
    assert rag.stats()["chunks"] > 2 and rag.stats()["parent_sections"] == 2
    result = rag.query("When is the refund paid and what about the receipt?")
    sections = [doc for doc in result.sources if doc.metadata.get("parent_expanded")]
    assert sections, "children were not replaced by their section"
    refund = next(doc for doc in sections if "Refunds" in doc.content)
    assert "Opened software cannot be refunded" in refund.content  # the whole section, not one child
    assert len({doc.id for doc in result.sources}) == len(result.sources)  # de-duplicated
    assert refund.metadata["child_ids"]


def test_parent_sections_follow_updates_deletes_and_save_load(tmp_path) -> None:
    rag = RAG(offline=True, parent_retrieval="fixed", parent_chunk_size=30, chunk_size=15, chunk_overlap=0)
    rag.add(Document(id="handbook", content=HANDBOOK))
    before = set(rag._sections)
    rag.save(str(tmp_path))
    loaded = RAG(offline=True, parent_retrieval="fixed", parent_chunk_size=30, chunk_size=15, chunk_overlap=0).load(str(tmp_path))
    assert set(loaded._sections) == before
    rag.add(Document(id="handbook", content=HANDBOOK.replace("fourteen", "thirty")))
    assert set(rag._sections) != before and all(s.startswith("handbook#") for s in rag._sections)
    rag.delete("handbook")
    assert rag._sections == {} and rag.stats()["chunks"] == 0


def test_parent_sections_respect_access_control() -> None:
    rag = RAG(offline=True, parent_retrieval="heading", chunk_size=20, chunk_overlap=0, access_control=True)
    rag.add(Document(id="public", content=HANDBOOK), access=access_tags("public"))
    rag.add(Document(id="secret", content="# Bonus\nThe executive bonus pool is ZX-991 for this year only."),
            access=access_tags("executive"))
    result = rag.query("What is the executive bonus pool?")
    assert all("ZX-991" not in doc.content for doc in result.sources) and "ZX-991" not in result.answer


# ------------------------------------------------------------------------------- tiers


class NamedLLM(MockLLMClient):
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def complete(self, prompt: str, system: str = "") -> str:
        self.calls += 1
        return super().complete(prompt, system)


def test_heuristic_router_decisions() -> None:
    router = HeuristicTierRouter()
    context = {"answer_cached": False, "cheap_available": True}
    assert router("When do offices close?", context).tier == "cheap"
    assert router("Why did revenue grow, and how does that compare with last year?", context).tier == "full"
    assert router("Compare the refund and shipping policies", context).tier == "full"
    assert router("When do offices close?", {**context, "answer_cached": True}).tier == "faq"
    assert router("When do offices close?", {**context, "cheap_available": False}).tier == "full"


def test_easy_questions_go_to_the_cheap_model_and_hard_ones_to_the_big_model() -> None:
    big, cheap = NamedLLM("big"), NamedLLM("cheap")
    stats = LatencyStats()
    rag = RAG(HANDBOOK, chat_llm=big, cheap_llm=cheap, embedding_llm="mock", reranker="llm", on_event=stats)
    easy = rag.query("When do orders ship?")
    assert easy.metadata["tier"] == "cheap" and cheap.calls >= 1 and big.calls == 0
    hard = rag.query("Why are international orders slower, and how does that compare with express shipping?")
    assert hard.metadata["tier"] == "full" and big.calls >= 1
    again = rag.query("When do orders ship?")
    assert again.metadata["tier"] == "faq" and again.metadata["answer_cache_hit"]  # answer cache is on by default
    assert [s.tags.get("tier") for s in stats.samples("total")] == ["cheap", "full", "faq"]
    assert rag.stats()["tiers"] == ["faq", "cheap", "full"]


def test_custom_router_is_pluggable_and_validated() -> None:
    seen = []

    def always_full(question, context):
        seen.append(context)
        return TierDecision("full", "custom rule")

    rag = RAG(HANDBOOK, offline=True, cheap_llm=NamedLLM("cheap"), tier_router=always_full, answer_cache=False)
    result = rag.query("When do orders ship?")
    assert result.metadata["tier"] == "full" and result.metadata["tier_reason"] == "custom rule"
    assert seen[0]["cheap_available"] is True and seen[0]["answer_cached"] is False
    broken = RAG(HANDBOOK, offline=True, tier_router=lambda q, c: "premium")
    with pytest.raises(ValueError, match="tier router"):
        broken.query("When do orders ship?")
