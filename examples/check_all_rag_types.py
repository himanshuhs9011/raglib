"""Run pass/fail checks for all supported raglib rag_type strategies."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raglib import RAG

DEFAULT_QUESTIONS = [
    "What is the core concept of this paper?",
    "What problem does this paper try to solve?",
    "What methodology or approach is proposed?",
    "What are the key findings or contributions?",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the RAG strategy checker."""

    parser = argparse.ArgumentParser(description="Check all raglib rag_type variants")
    parser.add_argument(
        "--doc",
        required=True,
        help="Path to the source document",
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Question to run. Use multiple times for multiple questions. Defaults to built-in 4 questions.",
    )
    parser.add_argument("--chat-model", default="gemma3:4b", help="Ollama chat model name")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest", help="Ollama embedding model")
    parser.add_argument("--vector-db", default="chroma", help="Vector backend: chroma or memory")
    parser.add_argument("--persist-dir", default=".raglib_chroma", help="Chroma persistence directory")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval size")
    parser.add_argument(
        "--web-provider",
        default="duckduckgo",
        help="Web search provider for web / hybrid_web / corrective / routing (use 'none' to disable)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the offline mock LLM + embeddings (no Ollama needed; answers are templated)",
    )
    parser.add_argument(
        "--report-dir",
        default="examples/reports",
        help="Directory to save JSON/Markdown full output reports",
    )
    return parser.parse_args()


def _render_markdown_report(results: List[Dict[str, Any]], questions: List[str], metadata: Dict[str, Any]) -> str:
    """Build a markdown report containing full outputs for all rag types."""

    lines: List[str] = []
    lines.append("# raglib all rag types output report")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Questions")
    lines.append("")
    for index, question in enumerate(questions, start=1):
        lines.append(f"{index}. {question}")
    lines.append("")

    for item in results:
        lines.append(f"## RAG Type: {item['rag_type']}")
        lines.append("")
        lines.append(f"- status: {item['status']}")
        if item.get("error"):
            lines.append(f"- error: {item['error']}")
            lines.append("")
            continue

        outputs = item.get("outputs", [])
        for output in outputs:
            lines.append(f"### Q{output['question_index']}: {output['question']}")
            lines.append("")
            lines.append("Answer:")
            lines.append(output.get("answer", ""))
            lines.append("")
            lines.append(f"Source count: {output.get('source_count', 0)}")
            lines.append(f"Top source ids: {', '.join(output.get('source_ids', []))}")
            trace = output.get("reasoning_trace", [])
            lines.append(f"Reasoning trace: {' -> '.join(trace) if trace else ''}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Execute all RAG strategies and print a pass/fail report."""

    args = parse_args()
    questions = args.questions if args.questions else list(DEFAULT_QUESTIONS)

    rag_types = [
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
        "graph",
        "hybrid_web",
    ]
    web_provider = None if args.web_provider.lower() == "none" else args.web_provider

    results: List[Dict[str, Any]] = []

    for rag_type in rag_types:
        result: Dict[str, Any] = {
            "rag_type": rag_type,
            "status": "pass",
            "error": None,
            "outputs": [],
        }

        try:
            model_kwargs: Dict[str, Any] = (
                {"offline": True}
                if args.offline
                else {
                    "chat_llm": "ollama",
                    "chat_model": args.chat_model,
                    "embedding_llm": "ollama",
                    "embedding_model": args.embedding_model,
                }
            )
            needs_web = rag_type in {"web", "hybrid_web", "corrective", "routing"}
            rag = RAG(
                source=args.doc,
                **model_kwargs,
                web_search_provider=web_provider if needs_web else None,
                vector_db=args.vector_db,
                vector_db_kwargs={
                    "collection_name": f"raglib_check_{rag_type}",
                    "persist_directory": args.persist_dir,
                }
                if args.vector_db.lower() in {"chroma", "chromadb"}
                else None,
                rag_type=rag_type,
                top_k=args.top_k,
            )

            for question_index, question in enumerate(questions, start=1):
                output = rag.query(question)
                result["outputs"].append(
                    {
                        "question_index": question_index,
                        "question": question,
                        "answer": output.answer,
                        "source_count": len(output.sources),
                        "source_ids": [doc.id for doc in output.sources[:5]],
                        "citations": [citation.document_id for citation in output.citations],
                        "latency_ms": output.usage.get("latency_ms"),
                        "reasoning_trace": output.reasoning_trace,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            result["status"] = "fail"
            result["error"] = f"{type(exc).__name__}: {exc}"

        results.append(result)

    passed = sum(1 for item in results if item["status"] == "pass")
    failed = sum(1 for item in results if item["status"] == "fail")

    summary = {"total": len(results), "passed": passed, "failed": failed}
    print("SUMMARY", json.dumps(summary))

    for item in results:
        print("\n" + "=" * 100)
        print(f"RAG TYPE: {item['rag_type']} | STATUS: {item['status']}")
        if item["status"] == "fail":
            print(f"ERROR: {item['error']}")
            continue

        for output in item["outputs"]:
            print("\n" + "-" * 100)
            print(f"Q{output['question_index']}: {output['question']}")
            print("-" * 100)
            print(output["answer"])
            print("-" * 100)
            print("Source count:", output["source_count"])
            print("Top source ids:", output["source_ids"])
            print("Reasoning trace:", output["reasoning_trace"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "doc": args.doc,
        "chat_model": args.chat_model,
        "embedding_model": args.embedding_model,
        "vector_db": args.vector_db,
        "top_k": args.top_k,
        "offline": args.offline,
        "web_provider": web_provider,
    }

    json_path = report_dir / f"all_rag_types_full_output_{timestamp}.json"
    markdown_path = report_dir / f"all_rag_types_full_output_{timestamp}.md"

    json_payload = {
        "summary": summary,
        "metadata": metadata,
        "questions": questions,
        "results": results,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(
        _render_markdown_report(results=results, questions=questions, metadata=metadata),
        encoding="utf-8",
    )

    print("\nREPORT_JSON", str(json_path))
    print("REPORT_MD", str(markdown_path))


if __name__ == "__main__":
    main()
