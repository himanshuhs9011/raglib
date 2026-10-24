"""Run raglib locally with Ollama chat, Ollama embeddings, and Chroma vector DB."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raglib import RAG


def main() -> None:
    """Execute a real-user style local RAG test over a DOCX file."""

    doc_path = sys.argv[1] if len(sys.argv) > 1 else "docs/"
    questions = [
        "What is the core concept of this paper?",
        "What problem does this paper try to solve?",
        "What methodology or approach is proposed?",
        "What are the key findings or contributions?",
    ]

    rag = RAG(
        source=doc_path,
        chat_llm="ollama",
        chat_model="gemma3:4b",
        embedding_llm="ollama",
        embedding_model="nomic-embed-text:latest",
        vector_db="chroma",
        vector_db_kwargs={
            "collection_name": "raglib_local_test",
            "persist_directory": ".raglib_chroma",
        },
        rag_type="advanced",
        top_k=5,
    )

    print(f"Index: {rag.stats()}")

    for idx, question in enumerate(questions, start=1):
        result = rag.query(question)
        print("\n" + "=" * 80)
        print(f"Q{idx}: {question}")
        print("-" * 80)
        print(result.answer)
        print("-" * 80)
        for citation in result.citations:
            print(f"  [{citation.index}] {citation.document_id}")
        print(f"  latency: {result.usage.get('latency_ms')} ms")


if __name__ == "__main__":
    main()
