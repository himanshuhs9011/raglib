"""Unit and side-by-side comparison tests for the standalone GraphContextWindow."""

from __future__ import annotations

from raglib.components.memory import MemoryModule
from raglib.graph_context import GraphContextWindow
from raglib.llm.base_client import BaseLLMClient
from raglib.llm.mock_client import MockLLMClient
from raglib.loaders.chunk_splitter import ChunkSplitter
from raglib.schemas import Document


class _StructuredGraphLLM(BaseLLMClient):
    """Test LLM returning structured CHUNK + SUMMARY + KEYWORDS lines."""

    def __init__(self) -> None:
        self.calls: int = 0

    def complete(self, prompt: str, system: str = "") -> str:
        self.calls += 1
        return (
            "CHUNK: 1-2 | SUMMARY: CorrectiveRAG scores retrieved chunks with an Evaluator threshold | KEYWORDS: correctiverag, evaluator, threshold\n"
            "CHUNK: 3-4 | SUMMARY: When evaluation fails Refiner strips noise and triggers web fallback | KEYWORDS: refiner, fallback, web"
        )

    def chat(self, messages):
        return self.complete(prompt=str(messages))


def test_hierarchical_document_graph_and_connected_summaries():
    """DocumentNode -> SectionNode -> ChunkNode hierarchy should wire PART_OF and NEXT/PREV_CHUNK summaries."""

    doc_text = (
        "# Section 1: Ingestion Pipeline\n"
        "DocumentLoader reads PDF, Word, and PowerPoint files into raw Documents. "
        "It computes deterministic SHA-256 identifiers for every source file.\n"
        "ChunkSplitter divides large documents into smaller semantic segments. "
        "It supports fixed, recursive, and LLM-based splitting strategies.\n"
        "# Section 2: Retrieval and Ranking\n"
        "Retriever embeds chunks into dense vectors and queries the VectorStore. "
        "Reranker re-scores top hits using token overlap and metadata relevance."
    )

    graph = GraphContextWindow(
        llm_client=MockLLMClient(),
        chunking_strategy="llm",
        chunk_size=18,
        chunk_overlap=3,
    )
    graph.add_documents([Document(id="arch_doc", content=doc_text, source="architecture.md")])

    assert "arch_doc" in graph.nodes
    doc_node = graph.nodes["arch_doc"]
    assert doc_node.node_type == "document"
    assert doc_node.summary

    section_ids = [ref.node_id for ref in doc_node.get_connections("HAS_SECTION")]
    assert len(section_ids) == 2

    sec1 = graph.nodes[section_ids[0]]
    sec2 = graph.nodes[section_ids[1]]
    assert sec1.node_type == "section"
    assert sec2.node_type == "section"

    next_sec_refs = sec1.get_connections("NEXT_SECTION")
    assert len(next_sec_refs) == 1
    assert next_sec_refs[0].node_id == sec2.id
    assert next_sec_refs[0].summary == sec2.summary

    chunk_ids = graph.doc_to_chunks["arch_doc"]
    assert len(chunk_ids) >= 2

    first_chunk = graph.nodes[chunk_ids[0]]
    second_chunk = graph.nodes[chunk_ids[1]]

    next_refs = first_chunk.get_connections("NEXT_CHUNK")
    assert len(next_refs) == 1
    assert next_refs[0].node_id == second_chunk.id
    assert next_refs[0].summary == second_chunk.summary
    assert len(next_refs[0].summary) > 0

    prev_refs = second_chunk.get_connections("PREV_CHUNK")
    assert len(prev_refs) == 1
    assert prev_refs[0].node_id == first_chunk.id
    assert prev_refs[0].summary == first_chunk.summary

    parent_refs = first_chunk.get_connections("PART_OF")
    assert len(parent_refs) == 1
    assert parent_refs[0].node_id == sec1.id


def test_llm_structured_chunking_populates_node_summaries_and_hopping():
    """Structured LLM chunking should populate chunk summaries and enable lookahead hopping to NEXT_CHUNK."""

    llm = _StructuredGraphLLM()
    graph = GraphContextWindow(
        llm_client=llm,
        chunking_strategy="llm",
        chunk_size=25,
        chunk_overlap=4,
    )

    crag_doc = Document(
        id="crag_guide",
        content=(
            "# Section 1: CorrectiveRAG Workflow\n"
            "CorrectiveRAG first retrieves candidate passages from the vector index. "
            "The Evaluator module scores each candidate against a strict relevance threshold. "
            "When the threshold check fails, the Refiner strips noisy sentences from the context. "
            "Finally, CorrectiveRAG triggers a WebRetriever fallback to fetch fresh external evidence."
        ),
        source="crag_guide.md",
    )
    graph.add_documents([crag_doc])
    assert llm.calls >= 1

    chunk_ids = graph.doc_to_chunks["crag_guide"]
    assert len(chunk_ids) == 2

    chunk1 = graph.nodes[chunk_ids[0]]
    chunk2 = graph.nodes[chunk_ids[1]]
    assert "Evaluator threshold" in chunk1.summary
    assert "Refiner strips noise" in chunk2.summary

    # Query matches chunk1 ("Evaluator threshold") and asks what happens next -> should hop to chunk2!
    result = graph.build_context_window(
        query="How does CorrectiveRAG use the Evaluator threshold and what happens next when it fails?",
        top_k_seeds=1,
        max_hops=1,
    )
    retrieved_ids = [doc.id for doc in result.documents]
    assert chunk1.id in retrieved_ids
    assert chunk2.id in retrieved_ids
    assert any("NEXT_CHUNK" in step for step in result.traversal_trace)


def test_cross_document_shared_entity_navigation():
    """Chunks across different documents sharing entities should connect via SHARED_ENTITY and be traversable."""

    doc_a = Document(
        id="service_auth",
        content=(
            "# Authentication Service\n"
            "The AegisGateway validates JWT bearer tokens before forwarding user requests. "
            "Expired tokens are rejected with HTTP 401."
        ),
    )
    doc_b = Document(
        id="service_billing",
        content=(
            "# Billing Settlement\n"
            "Before charging an invoice, BillingWorker verifies the caller identity through AegisGateway. "
            "Ledger entries are committed atomically in Postgres."
        ),
    )

    graph = GraphContextWindow(chunking_strategy="recursive", chunk_size=40, chunk_overlap=5)
    graph.add_documents([doc_a, doc_b])

    chunk_a = graph.nodes[graph.doc_to_chunks["service_auth"][0]]
    chunk_b = graph.nodes[graph.doc_to_chunks["service_billing"][0]]

    shared_refs = chunk_a.get_connections("SHARED_ENTITY")
    assert any(ref.node_id == chunk_b.id for ref in shared_refs)

    # Query starting at AegisGateway JWT validation and asking about BillingWorker invoice settlement
    ctx = graph.build_context_window(
        query="How does AegisGateway token validation connect to BillingWorker invoice settlement?",
        top_k_seeds=1,
        max_hops=1,
    )
    doc_ids = [d.id for d in ctx.documents]
    assert chunk_a.id in doc_ids
    assert chunk_b.id in doc_ids


def test_side_by_side_graph_context_vs_linear_chunks_and_fifo_memory():
    """Side-by-side comparison showing GraphContextWindow beating flat chunks + FIFO memory on topic return."""

    corpus = [
        Document(
            id="doc_crag",
            content=(
                "# CorrectiveRAG Architecture\n"
                "CorrectiveRAG evaluates retrieved documents using the Evaluator threshold score. "
                "Documents scoring above 0.75 are kept directly.\n"
                "If evaluation scores fall below the threshold, Refiner compresses the text and "
                "WebRetriever executes a fallback search over external providers."
            ),
            source="crag.md",
        ),
        Document(
            id="doc_chroma",
            content=(
                "# Chroma Vector Store\n"
                "ChromaVectorStore persists dense embeddings on disk using SQLite and HNSW segments. "
                "It supports cosine similarity search."
            ),
            source="chroma.md",
        ),
    ]

    linear_splitter = ChunkSplitter(chunk_size=22, overlap=3, strategy="recursive")
    linear_chunks = linear_splitter.split(corpus)
    fifo_memory = MemoryModule(max_turns=2)

    graph = GraphContextWindow(
        llm_client=MockLLMClient(),
        chunking_strategy="recursive",
        chunk_size=22,
        chunk_overlap=3,
    )
    graph.add_documents(corpus)

    crag_chunk_1 = graph.nodes[graph.doc_to_chunks["doc_crag"][0]]
    crag_chunk_2 = graph.nodes[graph.doc_to_chunks["doc_crag"][1]]
    chroma_chunk_1 = graph.nodes[graph.doc_to_chunks["doc_chroma"][0]]

    t1_q = "What is the Evaluator threshold in CorrectiveRAG?"
    t1_a = "CorrectiveRAG uses an Evaluator threshold of 0.75 to filter retrieved documents."
    fifo_memory.add(t1_q, t1_a, [ Document(id=crag_chunk_1.id, content=crag_chunk_1.content) ])
    graph.add_chat_turn(t1_q, t1_a, cited_documents=[Document(id=crag_chunk_1.id, content=crag_chunk_1.content)])

    t2_q = "Why does CorrectiveRAG check that score?"
    t2_a = "It checks the Evaluator score to decide whether local context is reliable."
    fifo_memory.add(t2_q, t2_a, [ Document(id=crag_chunk_1.id, content=crag_chunk_1.content) ])
    graph.add_chat_turn(t2_q, t2_a, cited_documents=[Document(id=crag_chunk_1.id, content=crag_chunk_1.content)])

    t3_q = "How does ChromaVectorStore store embeddings?"
    t3_a = "ChromaVectorStore persists embeddings on disk using HNSW segments."
    fifo_memory.add(t3_q, t3_a, [ Document(id=chroma_chunk_1.id, content=chroma_chunk_1.content) ])
    graph.add_chat_turn(t3_q, t3_a, cited_documents=[Document(id=chroma_chunk_1.id, content=chroma_chunk_1.content)])

    t4_q = "What similarity metric does ChromaVectorStore use?"
    t4_a = "It uses cosine similarity over the HNSW index."
    fifo_memory.add(t4_q, t4_a, [ Document(id=chroma_chunk_1.id, content=chroma_chunk_1.content) ])
    graph.add_chat_turn(t4_q, t4_a, cited_documents=[Document(id=chroma_chunk_1.id, content=chroma_chunk_1.content)])

    followup_query = "Going back to CorrectiveRAG, what happens next after that Evaluator threshold check fails?"

    # Check FIFO Memory: Turn 1 and Turn 2 were evicted by Turn 3 and Turn 4!
    fifo_context = fifo_memory.get_context()
    assert "CorrectiveRAG" not in fifo_context
    assert "ChromaVectorStore" in fifo_context

    assert "connected_nodes" not in linear_chunks[0].metadata

    # Check GraphContextWindow:
    # 1) Recovers Turn 1 & Turn 2 via SAME_TOPIC_TURN / entity match on 'CorrectiveRAG' + 'Evaluator'
    # 2) Seeds on crag_chunk_1 AND hops to crag_chunk_2 via NEXT_CHUNK because the query asks "what happens next"!
    graph_result = graph.build_context_window(
        query=followup_query,
        top_k_seeds=2,
        max_hops=2,
    )

    chat_turn_ids = [node.id for node in graph_result.chat_nodes]
    assert "turn_1" in chat_turn_ids or "turn_2" in chat_turn_ids

    selected_doc_ids = [doc.id for doc in graph_result.documents]
    assert crag_chunk_1.id in selected_doc_ids
    assert crag_chunk_2.id in selected_doc_ids
    assert "Refiner" in graph_result.formatted_context
    assert "WebRetriever" in graph_result.formatted_context
