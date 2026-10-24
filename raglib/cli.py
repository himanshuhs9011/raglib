"""Command-line interface: ``raglib ask``, ``raglib chat``, ``raglib providers``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence

USAGE_EXAMPLES = """examples:
  raglib ask docs/ "What is our refund policy?"
  raglib ask paper.pdf notes.md "Summarize the method" --llm openrouter/anthropic/claude-sonnet-4
  raglib ask notes.md "Key dates?" --llm ollama/llama3.1 --embedding ollama
  raglib chat docs/ --llm openrouter/deepseek/deepseek-chat
  raglib graph docs/ --out graph.html         # knowledge graph you can open in a browser
  raglib providers
"""


def _add_common(command: argparse.ArgumentParser) -> None:
    """Options shared by ask and chat."""

    command.add_argument("--llm", help="provider or provider/model (default: detected from API key env vars)")
    command.add_argument("--embedding", help="embedding provider or provider/model (openai, ollama, huggingface, none, ...)")
    command.add_argument("--rag-type", default="advanced", help="advanced | hybrid | corrective | self | agentic | graph | ...")
    command.add_argument("--top-k", type=int, default=5)
    command.add_argument("--web", help="enable web search with this provider (e.g. duckduckgo, tavily)")
    command.add_argument("--offline", action="store_true", help="use the offline mock models (no API keys)")
    command.add_argument("-v", "--verbose", action="store_true", help="show info logs")


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""

    parser = argparse.ArgumentParser(
        prog="raglib",
        description="Ask questions about your documents with retrieval-augmented generation.",
        epilog=USAGE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer one question: raglib ask SOURCE [SOURCE...] QUESTION")
    ask.add_argument("inputs", nargs="+", metavar="SOURCE... QUESTION", help="files/folders/URLs, then the question")
    _add_common(ask)

    chat = sub.add_parser("chat", help="interactive chat with conversation memory")
    chat.add_argument("inputs", nargs="+", metavar="SOURCE", help="files, folders or URLs to index")
    _add_common(chat)

    graph = sub.add_parser("graph", help="build a knowledge graph and write an interactive HTML view")
    graph.add_argument("inputs", nargs="+", metavar="SOURCE", help="files, folders or URLs to index")
    graph.add_argument("--out", default="raglib_graph.html", help="output HTML file")
    graph.add_argument("--max-nodes", type=int, default=400)
    _add_common(graph)

    sub.add_parser("providers", help="list supported LLM providers")
    return parser


def _make_rag(args: argparse.Namespace, sources: Sequence[str]):
    """Build a RAG instance from CLI arguments."""

    from raglib.facade import RAG

    return RAG(
        sources[0] if len(sources) == 1 else list(sources),
        llm=args.llm,
        embedding_llm=args.embedding,
        rag_type="graph" if args.command == "graph" else args.rag_type,
        top_k=args.top_k,
        web_search_provider=args.web,
        offline=True if args.offline else None,
        memory=True if args.command == "chat" else None,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""

    args = _build_parser().parse_args(argv)
    if args.command == "providers":
        from raglib.llm.providers import provider_table

        print("Supported LLM providers (name, API key env var, description):")
        print(provider_table())
        print("\nUse --llm <name> or --llm <name>/<model>, e.g. --llm openrouter/google/gemini-2.5-flash")
        return 0

    if args.command == "ask" and len(args.inputs) < 2:
        print('error: usage is  raglib ask SOURCE [SOURCE...] "QUESTION"', file=sys.stderr)
        return 2
    sources = args.inputs[:-1] if args.command == "ask" else args.inputs

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    try:
        rag = _make_rag(args, sources)
    except (ValueError, ImportError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "chat":
        rag.chat()
        return 0
    if args.command == "graph":
        path = rag.visualize_graph(args.out, max_nodes=args.max_nodes)
        stats = rag.stats()["graph"]
        print(f"Wrote {path}: {stats['entities']} entities, {stats['relations']} relations, {stats['communities']} communities")
        return 0

    result = rag.query(args.inputs[-1])
    print(result.answer)
    if result.citations:
        print("\nSources:")
        by_id = {doc.id: doc for doc in result.sources}
        for citation in result.citations:
            doc = by_id.get(citation.document_id)
            label = (doc.metadata.get("file") or doc.metadata.get("url")) if doc else None
            print(f"  [{citation.index}] {label or citation.document_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
