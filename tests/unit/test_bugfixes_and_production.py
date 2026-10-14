"""Regression and production-readiness tests for raglib components and orchestrators."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

import raglib
from raglib import (
    RAG,
    AdvancedRAG,
    AgenticRAG,
    ChunkSplitter,
    ContextReducer,
    CorrectiveRAG,
    DecisionEngine,
    Deduplicator,
    Document,
    Evaluator,
    GenerationResult,
    Generator,
    HybridRAG,
    HybridRetriever,
    MemoryModule,
    MemoryRAG,
    MockLLMClient,
    MultiHopRAG,
    MultiHopRetriever,
    MultiQueryRAG,
    MultiQueryRetriever,
    NaiveRAG,
    Planner,
    QueryExpander,
    QueryRewriter,
    Refiner,
    ReflectionModule,
    Reranker,
    Retriever,
    RouterRetriever,
    RoutingRAG,
    SelfRAG,
    ToolAugmentedRAG,
    WebRAG,
    WebRetriever,
)
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.mock_embedding import MockEmbedding
from raglib.llm.base_client import BaseLLMClient
from raglib.loaders.document_loader import DocumentLoader
from raglib.loaders.web_loader import WebLoader
from raglib.vectorstores.in_memory_store import InMemoryVectorStore


class _CountingEmbedding(BaseEmbedding):
    """Embedding stub that tracks how many document batches were embedded."""

    def __init__(self) -> None:
        self.embedded_texts: List[str] = []
        self._inner = MockEmbedding()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embedded_texts.extend(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(text)

    @property
    def dimension(self) -> int:
        return self._inner.dimension


class _CountingLLM(BaseLLMClient):
    """LLM stub that counts calls and returns configurable responses."""

    def __init__(self, fixed_response: str = "vector") -> None:
        self.calls: int = 0
        self.fixed_response = fixed_response

    def complete(self, prompt: str, system: str = "") -> str:
        self.calls += 1
        return self.fixed_response

    def chat(self, messages: List[Dict[str, str]]) -> str:
        self.calls += 1
        return self.fixed_response


def test_top_level_public_exports_are_importable() -> None:
    """Verify all documented classes are exported from the root raglib package."""

    expected = [
        "RAG",
        "Document",
        "QueryResult",
        "GenerationResult",
        "MockLLMClient",
        "ChunkSplitter",
        "Retriever",
        "Generator",
        "Reranker",
        "Evaluator",
        "Refiner",
        "NaiveRAG",
        "AdvancedRAG",
        "CorrectiveRAG",
        "SelfRAG",
        "AgenticRAG",
        "HybridRAG",
        "MultiQueryRAG",
        "MultiHopRAG",
        "RoutingRAG",
        "MemoryRAG",
        "WebRAG",
        "ToolAugmentedRAG",
    ]
    for symbol in expected:
        assert hasattr(raglib, symbol), f"Missing export: {symbol}"


def test_generator_handles_custom_template_with_literal_braces() -> None:
    """Verify Generator does not raise KeyError when prompt_template contains JSON braces."""

    template = (
        'System format example: {"status": "ok"}\n'
        "Question: {query}\n"
        "Context: {context}\n"
        "Answer:"
    )
    gen = Generator(llm_client=MockLLMClient(), prompt_template=template)
    doc = Document(id="d1", content="RAG grounds answers in retrieved documents.", source="vector")
    result = gen.generate(query="What does RAG do?", documents=[doc])

    assert isinstance(result, GenerationResult)
    assert result.answer.strip() != ""
    assert len(result.sources) == 1


def test_web_loader_strips_scripts_styles_and_collapses_whitespace() -> None:
    """Verify WebLoader._html_to_text properly strips multiline script/style tags and whitespace."""

    loader = WebLoader()
    raw_html = """
    <html>
      <head>
        <style>
          body { color: black; }
        </style>
        <script type="text/javascript">
          const secret = "should_not_appear";
          console.log(secret);
        </script>
      </head>
      <body>
        <h1>Title   Here</h1>
        <p>Paragraph   one
           with   newlines.</p>
      </body>
    </html>
    """
    text = loader._html_to_text(raw_html)
    assert "should_not_appear" not in text
    assert "color: black" not in text
    assert "Title Here" in text
    assert "Paragraph one with newlines." in text
    assert "  " not in text


def test_document_loader_raises_on_missing_file_path() -> None:
    """Verify missing file paths raise FileNotFoundError instead of becoming raw text."""

    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_document_12345.pdf")

    with pytest.raises(FileNotFoundError):
        loader.load(Path("non_existent_folder/missing.txt"))


def test_mock_embedding_ranks_lexically_matching_document_first() -> None:
    """Verify MockEmbedding produces higher similarity for documents sharing query terms."""

    docs = [
        Document(id="irrelevant", content="Bananas and oranges are tropical fruits grown in orchards."),
        Document(id="relevant", content="Hallucinations in RAG pipelines are reduced by grounding context."),
    ]
    retriever = Retriever(
        documents=docs,
        embedding=MockEmbedding(),
        top_k=2,
        vector_db="memory",
    )
    results = retriever.retrieve("How are hallucinations in RAG pipelines reduced?")
    assert len(results) == 2
    assert results[0].id == "relevant"
    assert results[0].score > results[1].score


def test_retriever_incremental_add_documents_embeds_only_new_docs() -> None:
    """Verify Retriever.add_documents only embeds newly added documents."""

    counting_embedding = _CountingEmbedding()
    initial = [
        Document(id="d1", content="First document about vector databases."),
        Document(id="d2", content="Second document about query expansion."),
    ]
    retriever = Retriever(
        documents=initial,
        embedding=counting_embedding,
        top_k=3,
        vector_db="memory",
    )
    assert len(counting_embedding.embedded_texts) == 2

    retriever.add_documents([Document(id="d3", content="Third document about reranking.")])
    assert len(counting_embedding.embedded_texts) == 3

    hits = retriever.retrieve("reranking")
    assert any(doc.id == "d3" for doc in hits)


def test_retriever_reconstructs_documents_from_prepopulated_vector_store() -> None:
    """Verify Retriever can query a pre-populated vector store when initialized with empty docs."""

    store = InMemoryVectorStore()
    embedding = MockEmbedding()
    seed_retriever = Retriever(
        documents=[Document(id="persisted-1", content="Persistent knowledge about thermal detection.")],
        embedding=embedding,
        top_k=2,
        vector_db=store,
    )
    assert len(seed_retriever.retrieve("thermal detection")) == 1

    # Fresh retriever attached to the same vector store without re-passing documents:
    reattached = Retriever(
        documents=[],
        embedding=embedding,
        top_k=2,
        vector_db=store,
    )
    results = reattached.retrieve("thermal detection")
    assert len(results) == 1
    assert results[0].id == "persisted-1"
    assert "thermal detection" in results[0].content


def test_routing_rag_calls_router_llm_only_once() -> None:
    """Verify RoutingRAG does not make duplicate route() LLM calls per query."""

    counting_llm = _CountingLLM(fixed_response="vector")
    docs = [Document(id="d1", content="Vector corpus entry.")]
    vec = Retriever(documents=docs, top_k=1)
    web = WebRetriever(provider_name="duckduckgo", top_k=1)
    router = RouterRetriever(vector_retriever=vec, web_retriever=web, llm_client=counting_llm)
    rag = RoutingRAG(router_retriever=router, generator=Generator(llm_client=MockLLMClient()))

    rag.run("What is in the vector corpus?")
    assert counting_llm.calls == 1


def test_base_rag_calls_decision_engine_llm_only_once() -> None:
    """Verify BaseRAG calls DecisionEngine.retrieval_type only once per query."""

    counting_llm = _CountingLLM(fixed_response="vector")
    docs = [Document(id="d1", content="BaseRAG corpus entry.")]
    base_rag = raglib.core.base.BaseRAG(
        retriever=Retriever(documents=docs, top_k=1),
        decision_engine=DecisionEngine(llm_client=counting_llm),
        generator=Generator(llm_client=MockLLMClient()),
    )
    base_rag.run("Explain BaseRAG")
    assert counting_llm.calls == 1


def test_self_rag_word_boundary_verdict_parsing() -> None:
    """Verify SelfRAG does not misclassify 'KNOWLEDGE' as 'NO' or 'NOTE' as 'NOT'."""

    self_rag = SelfRAG(
        retriever=Retriever(documents=[Document(id="d1", content="RAG overview.")], top_k=1),
        generator=Generator(llm_client=MockLLMClient()),
        llm_client=_CountingLLM(fixed_response="YES, KNOWLEDGE RETRIEVAL IS REQUIRED"),
    )
    assert self_rag._llm_decide_retrieval("What is RAG?") == "yes"

    self_rag.llm_client = _CountingLLM(fixed_response="RELEVANT (NOTE: HIGH QUALITY)")
    assert self_rag._llm_is_relevant("What is RAG?", [Document(id="d1", content="RAG")]) == "relevant"


@pytest.mark.parametrize(
    "rag_type",
    [
        "naive",
        "advanced",
        "corrective",
        "self",
        "agentic",
        "hybrid",
        "multi_query",
        "multi_hop",
        "routing",
        "memory",
        "web",
        "tool",
    ],
)
def test_all_12_rag_types_run_end_to_end_via_facade(rag_type: str) -> None:
    """Verify all 12 RAG strategies run cleanly through the RAG facade in offline mode."""

    corpus = (
        "Retrieval-Augmented Generation (RAG) grounds language model outputs in retrieved documents. "
        "Reranking and evaluation improve retrieval precision and reduce hallucinations."
    )
    rag = RAG(
        source=corpus,
        rag_type=rag_type,
        web_search_provider="local",
        vector_db="memory",
        top_k=3,
    )
    result = rag.query("How does RAG reduce hallucinations?")
    assert isinstance(result, GenerationResult)
    assert result.answer.strip() != ""
