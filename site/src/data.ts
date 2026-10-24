export const VERSION = "2.0.4";

const REPO = "https://github.com/himanshuhs9011/raglib";
export const LINKS = {
  github: REPO,
  pypi: "https://pypi.org/project/raglib-py/",
  readme: `${REPO}#readme`,
  usage: `${REPO}/blob/main/docs/USAGE.md`,
  changelog: `${REPO}/blob/main/CHANGELOG.md`,
  issues: `${REPO}/issues`,
};

export interface RagType {
  id: string;
  name: string;
  what: string;
  when: string;
  calls: string;
  code: string;
}

export const RAG_TYPES: RagType[] = [
  { id: "naive", name: "Naive", what: "Retrieve top-k chunks, answer with citations.", when: "Baselines, simple FAQs.", calls: "1", code: `rag = RAG("docs/", llm="openai", rag_type="naive", top_k=3)
result = rag.query("How long do customers have to ask for a refund?")
print(result.answer)` },
  { id: "advanced", name: "Advanced (default)", what: "Hybrid BM25 + dense retrieval, optional rerank, grounded cited answer.", when: "Most document Q&A.", calls: "1", code: `rag = RAG(
    "docs/", llm="openai",               # rag_type="advanced" is the default
    reranker="cross_encoder",            # "none" | "cross_encoder" | "cohere" | "llm"
    mmr_lambda=0.5, neighbor_window=1,   # diversity + neighbouring chunks
)
result = rag.query("Who leads support?", filters={"file": {"$in": ["docs/team.md"]}})` },
  { id: "hybrid", name: "Hybrid", what: "Keyword and semantic search fused, so exact codes and paraphrases both match.", when: "IDs, product codes, mixed queries.", calls: "1", code: `rag = RAG(
    "docs/", llm="openai", rag_type="hybrid",
    fusion="relative",      # or "rrf"
    hybrid_alpha=0.5,       # 0 = keywords only, 1 = vectors only
)
result = rag.query("INV-2024-0042 status")   # exact codes and paraphrases both match` },
  { id: "corrective", name: "Corrective (CRAG)", what: "Grades retrieved chunks; refines, or searches the web only when nothing relevant was found.", when: "When retrieval is often off-target.", calls: "2–3", code: `rag = RAG("docs/", llm="openai", rag_type="corrective",
          web_search_provider="duckduckgo")   # optional: used only when nothing relevant is found
result = rag.query("Which database does billing use?")
print(result.metadata["crag_verdict"])        # "correct" | "ambiguous" | "incorrect"` },
  { id: "self", name: "Self-RAG", what: "Decides whether to retrieve, checks relevance, support and usefulness of its answer.", when: "High-stakes answers that must be supported.", calls: "≤ 8 (capped)", code: `rag = RAG("docs/", llm="openai", rag_type="self", max_llm_calls=8)
result = rag.query("What does the auth service depend on?")
print(result.reasoning_trace)   # decide_retrieval, is_relevant, is_sup, is_use, ...
print(result.metadata["self_rag_llm_calls"])` },
  { id: "agentic", name: "Agentic", what: "ReAct loop: the LLM picks tools (docs, web, graph, your own) until it has enough evidence.", when: "Open-ended, multi-part questions.", calls: "≤ steps + 1", code: `rag = RAG("docs/", llm="openai", rag_type="agentic", max_steps=4,
          web_search_provider="duckduckgo",   # adds a search_web tool
          enable_graph=True)                  # adds a graph_search tool
result = rag.query("Which service does auth depend on, and what database does it use?")
for step in result.metadata["agent_steps"]:
    print(step["action"], step["input"])` },
  { id: "tool", name: "Tool", what: "Agentic loop with your own tools, e.g. a calculator or an internal API.", when: "Answers that need live data or computation.", calls: "≤ steps + 1", code: `from raglib.components.tools import Tool

def convert(text: str) -> str:
    return "1 EUR = 1.10 USD"

rag = RAG("docs/", llm="openai", rag_type="tool",
          tools=[Tool(name="fx", description="Convert EUR to USD", func=convert)])
result = rag.query("What is the travel cap in US dollars?")
print(result.metadata["agent_steps"])   # shows when the LLM called "fx"` },
  { id: "multi_hop", name: "Multi-hop", what: "Iterative retrieval: each hop asks what is still missing.", when: "Questions that chain facts across documents.", calls: "1 per hop + 1", code: `rag = RAG("docs/", llm="openai", rag_type="multi_hop", max_hops=3)
result = rag.query("Who does the support team lead report to?")
print(result.metadata["hop_queries"])   # the follow-up question asked at each hop` },
  { id: "multi_query", name: "Multi-query", what: "Rewrites the question into variants and fuses their results.", when: "Vague or short questions.", calls: "2", code: `rag = RAG("docs/", llm="openai", rag_type="multi_query")
result = rag.query("money back?")       # rewritten into several variants, results fused` },
  { id: "memory", name: "Memory", what: "Per-session chat history; follow-ups are rewritten into standalone questions.", when: "Chatbots.", calls: "1 (+1 on follow-ups)", code: `rag = RAG("docs/", llm="openai", rag_type="memory", max_history_turns=10)
rag.query("Who leads the support team?", session_id="user-1")
result = rag.query("Who does she report to?", session_id="user-1")   # "she" is resolved
print(result.metadata["search_query"])  # the standalone question used for retrieval
rag.reset_memory("user-1")` },
  { id: "routing", name: "Routing", what: "Chooses vector, web, graph or no retrieval per question.", when: "Mixed workloads.", calls: "1–2", code: `rag = RAG("docs/", llm="openai", rag_type="routing",
          web_search_provider="duckduckgo", enable_graph=True)
print(rag.query("hello!").reasoning_trace)              # no retrieval
print(rag.query("refund window?").reasoning_trace)      # route:vector` },
  { id: "web", name: "Web", what: "Answers from live web search (8 providers, DuckDuckGo is free).", when: "Fresh information.", calls: "1 + search", code: `rag = RAG(llm="openai", rag_type="web",
          web_search_provider="duckduckgo")   # free; also tavily, serpapi, searxng, ...
result = rag.query("Latest stable Python release")
for doc in result.sources:
    print(doc.metadata.get("url"))` },
  { id: "hybrid_web", name: "Hybrid web", what: "Corpus and web results, each given a share of the context.", when: "Internal docs plus current facts.", calls: "1 + search", code: `rag = RAG("docs/", llm="openai", rag_type="hybrid_web",
          web_search_provider="duckduckgo")
result = rag.query("Our refund window vs the EU legal minimum?")
print([doc.source for doc in result.sources])   # mix of your files and "duckduckgo"` },
  { id: "graph", name: "GraphRAG", what: "LLM-built knowledge graph with communities; local (entity) and global (theme) search.", when: "Relationships and corpus-wide themes.", calls: "1 per query; 1 per chunk to index", code: `rag = RAG("papers/", llm="openai", rag_type="graph",
          search_mode="auto",                    # "local" | "global"
          graph_kwargs={"max_workers": 8, "resolve_with_embeddings": True})
print(rag.query("How are auth and billing related?").metadata["graph_mode"])   # local
print(rag.query("Main themes across all papers?").metadata["graph_mode"])      # global
rag.visualize_graph("graph.html")` },
];

export const FEATURES = [
  { title: "One line to start", text: 'RAG("docs/", llm="openrouter/<model>") loads, chunks, indexes and answers. ask() and a CLI for the simplest cases.' },
  { title: "Hybrid retrieval", text: "BM25 + dense with score fusion, metadata filters, MMR, rerankers (cross-encoder, Cohere, LLM), neighbour expansion." },
  { title: "14 RAG architectures", text: "From naive to Self-RAG, CRAG, agentic and GraphRAG, all on one shared, traced pipeline." },
  { title: "GraphRAG", text: "Parallel cached entity extraction, Leiden/Louvain communities, local and global search, NetworkX or Neo4j." },
  { title: "17 LLM providers", text: "OpenAI, Anthropic, Google, Groq, OpenRouter, Mistral, DeepSeek, Ollama, vLLM, LM Studio, Azure and more." },
  { title: "Free-tier friendly", text: "Shared rate limiter, Retry-After aware retries, quota errors that fail fast, model fallbacks." },
  { title: "Grounded answers", text: "Numbered sources, [n] citations, \"I don't know\" when the docs lack the answer, prompt-injection guard." },
  { title: "Safe by default", text: "SSRF guard for URLs, path sandbox, no queries in logs, delete() also purges caches." },
];

export interface TestRow {
  test: string;
  result: string;
  status: "pass" | "fixed";
}

export const LIVE_RESULTS: TestRow[] = [
  { test: "Needle in a haystack: 4 needles in 95 chunks with 16 hard negatives", result: "All 4 at rank 1 (dense and hybrid). BM25 alone missed the paraphrased question.", status: "pass" },
  { test: "Cited answers to needle questions", result: "3 / 3 exact, with citations", status: "pass" },
  { test: "GraphRAG local, 3-hop question across documents", result: "Correct; 25 entities and 24 relations extracted", status: "pass" },
  { test: "GraphRAG global, corpus-wide themes", result: "5 community reports and a cited summary", status: "pass" },
  { test: "GraphContextWindow", result: "Graph built in 21 ms without an LLM; correct answer", status: "pass" },
  { test: "naive, hybrid, routing, corrective, self, memory, multi_query, multi_hop, tool, web", result: "All correct", status: "pass" },
  { test: "agentic, hybrid_web", result: "Bugs found live, fixed in 2.0.1; live re-check pending", status: "fixed" },
];

export const WHY = [
  { title: "Instead of wiring ten pieces", text: "Loaders, chunking, embeddings, a vector store, BM25, fusion, reranking, prompts, citations and memory are already connected and tested together." },
  { title: "Switch techniques with one argument", text: "rag_type=\"corrective\" or \"graph\" changes the architecture, not your code. Compare them on your own data." },
  { title: "Production details included", text: "Retries, rate limits, caching, concurrency locks, streaming, async, tracing, evaluation and safe deletion." },
];
