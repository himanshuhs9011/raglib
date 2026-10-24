"""A runnable tour of every rag_type: how to configure it and how to read every part of the result.

    python examples/rag_types_tour.py --offline            # no API key (mock LLM)
    python examples/rag_types_tour.py --llm openrouter/<model> --embedding openrouter/<embed-model>
    python examples/rag_types_tour.py --offline --web      # also run the web types (DuckDuckGo, free)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from a source checkout

from raglib import RAG  # noqa: E402
from raglib.components.tools import Tool  # noqa: E402
from raglib.schemas import GenerationResult  # noqa: E402

DOCS = """
# Company handbook

## Refunds
Customers can request a refund within 30 days of purchase. Refunds are paid to the original card.

## Support
The support team is led by Maria Chen. Maria Chen reports to the COO, David Park.

## Infrastructure
The billing service runs on Postgres. The auth service depends on the billing service.
"""


def show(result: GenerationResult) -> None:
    """Every field a GenerationResult carries."""

    print("answer      :", result.answer)
    for citation in result.citations:                      # which source backs each [n]
        print(f"citation [{citation.index}]:", citation.source, "|", citation.snippet[:60])
    for doc in result.sources:                             # the chunks given to the LLM
        print("source      :", doc.id, round(doc.score, 3), doc.metadata.get("file") or doc.source)
    print("trace       :", result.reasoning_trace)        # human-readable pipeline steps
    for event in result.trace_events:                      # timed stages (retrieve, generate, ...)
        print("stage       :", event.stage, f"{event.latency_ms:.1f} ms", event.meta)
    print("usage       :", result.usage)                  # tokens + total latency
    print("metadata    :", {k: v for k, v in result.metadata.items() if k != "graph_context_docs"})
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default=None)
    parser.add_argument("--embedding", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--web", action="store_true", help="also run web / hybrid_web (DuckDuckGo)")
    args = parser.parse_args()
    common = {"llm": args.llm, "embedding_llm": args.embedding, "offline": args.offline or None, "chunk_size": 30, "chunk_overlap": 0}

    # 1. naive: retrieve top-k, answer. 1 LLM call.
    rag = RAG(DOCS, rag_type="naive", top_k=3, **common)
    show(rag.query("How long do customers have to ask for a refund?"))

    # 2. advanced (default): hybrid BM25 + dense, optional reranker, cited answer. 1 LLM call.
    rag = RAG(DOCS, rag_type="advanced", reranker="none", mmr_lambda=None, neighbor_window=0, **common)
    show(rag.query("Who leads support?"))

    # 3. hybrid: keyword + semantic fusion; tune the blend.
    rag = RAG(DOCS, rag_type="hybrid", retrieval_mode="hybrid", fusion="relative", hybrid_alpha=0.5, **common)
    show(rag.query("Postgres"))

    # 4. corrective (CRAG): grades chunks; web search only if a provider is set and nothing is relevant.
    rag = RAG(DOCS, rag_type="corrective", **common)
    result = rag.query("Which database does billing use?")
    print("CRAG verdict:", result.metadata.get("crag_verdict"))
    show(result)

    # 5. self (Self-RAG): retrieve? relevant? supported? useful? Capped LLM calls.
    rag = RAG(DOCS, rag_type="self", max_llm_calls=8, **common)
    show(rag.query("What does the auth service depend on?"))

    # 6. multi_query: rewrites the question into variants and fuses the results. 2 LLM calls.
    rag = RAG(DOCS, rag_type="multi_query", **common)
    show(rag.query("money back?"))

    # 7. multi_hop: each hop asks what is still missing.
    rag = RAG(DOCS, rag_type="multi_hop", max_hops=3, **common)
    result = rag.query("Who does the support team lead report to?")
    print("hop queries:", result.metadata.get("hop_queries"))
    show(result)

    # 8. agentic: the LLM chooses tools (search_docs, plus web / graph / your own) step by step.
    rag = RAG(DOCS, rag_type="agentic", max_steps=4, **common)
    result = rag.query("Which service does auth depend on, and what database does that service use?")
    print("agent steps:", result.metadata.get("agent_steps"))
    show(result)

    # 9. tool: your own tools; the agent decides when to call them.
    def days_until_refund_deadline(purchase_date: str) -> str:
        return f"A purchase made on {purchase_date or 'that date'} can be refunded for 30 days after it."

    refund_tool = Tool(name="refund_deadline", description="Explain the refund deadline for a purchase date.",
                       func=days_until_refund_deadline)
    rag = RAG(DOCS, rag_type="tool", tools=[refund_tool], **common)
    show(rag.query("I bought something today. Until when can I get a refund?"))

    # 10. memory: per-session history; follow-ups are rewritten into standalone questions.
    rag = RAG(DOCS, rag_type="memory", max_history_turns=10, **common)
    rag.query("Who leads the support team?", session_id="user-1")
    show(rag.query("Who does she report to?", session_id="user-1"))
    rag.reset_memory("user-1")

    # 11. routing: picks vector / web / graph / no retrieval per question.
    rag = RAG(DOCS, rag_type="routing", **common)
    show(rag.query("hello!"))
    show(rag.query("What is the refund window?"))

    # 12. graph (GraphRAG): knowledge graph; local search for entities, global for themes.
    rag = RAG(DOCS, rag_type="graph", search_mode="auto", graph_kwargs={"max_workers": 4}, **common)
    result = rag.query("How are the auth and billing services related?")
    print("graph mode:", result.metadata.get("graph_mode"))
    show(result)
    print("graph viewer:", rag.visualize_graph("tour_graph.html"))

    if args.web:
        # 13. web: answers from live web search (DuckDuckGo is free, no key).
        rag = RAG(rag_type="web", web_search_provider="duckduckgo", **common)
        show(rag.query("Latest stable Python release"))

        # 14. hybrid_web: your documents plus the web, each given a share of the context.
        rag = RAG(DOCS, rag_type="hybrid_web", web_search_provider="duckduckgo", **common)
        show(rag.query("What is our refund window, and what is the EU legal minimum for online purchases?"))

    # Streaming and async work with every rag_type:
    rag = RAG(DOCS, **common)
    for token in rag.stream("Summarize the refund policy"):
        print(token, end="", flush=True)
    print()
    print("stats:", rag.stats())


if __name__ == "__main__":
    main()
