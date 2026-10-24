export const VERSION = "2.0.3";

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
}

export const RAG_TYPES: RagType[] = [
  { id: "naive", name: "Naive", what: "Retrieve top-k chunks, answer with citations.", when: "Baselines, simple FAQs.", calls: "1" },
  { id: "advanced", name: "Advanced (default)", what: "Hybrid BM25 + dense retrieval, optional rerank, grounded cited answer.", when: "Most document Q&A.", calls: "1" },
  { id: "hybrid", name: "Hybrid", what: "Keyword and semantic search fused, so exact codes and paraphrases both match.", when: "IDs, product codes, mixed queries.", calls: "1" },
  { id: "corrective", name: "Corrective (CRAG)", what: "Grades retrieved chunks; refines, or searches the web only when nothing relevant was found.", when: "When retrieval is often off-target.", calls: "2–3" },
  { id: "self", name: "Self-RAG", what: "Decides whether to retrieve, checks relevance, support and usefulness of its answer.", when: "High-stakes answers that must be supported.", calls: "≤ 8 (capped)" },
  { id: "agentic", name: "Agentic", what: "ReAct loop: the LLM picks tools (docs, web, graph, your own) until it has enough evidence.", when: "Open-ended, multi-part questions.", calls: "≤ steps + 1" },
  { id: "tool", name: "Tool", what: "Agentic loop with your own tools, e.g. a calculator or an internal API.", when: "Answers that need live data or computation.", calls: "≤ steps + 1" },
  { id: "multi_hop", name: "Multi-hop", what: "Iterative retrieval: each hop asks what is still missing.", when: "Questions that chain facts across documents.", calls: "1 per hop + 1" },
  { id: "multi_query", name: "Multi-query", what: "Rewrites the question into variants and fuses their results.", when: "Vague or short questions.", calls: "2" },
  { id: "memory", name: "Memory", what: "Per-session chat history; follow-ups are rewritten into standalone questions.", when: "Chatbots.", calls: "1 (+1 on follow-ups)" },
  { id: "routing", name: "Routing", what: "Chooses vector, web, graph or no retrieval per question.", when: "Mixed workloads.", calls: "1–2" },
  { id: "web", name: "Web", what: "Answers from live web search (9 providers, DuckDuckGo is free).", when: "Fresh information.", calls: "1 + search" },
  { id: "hybrid_web", name: "Hybrid web", what: "Corpus and web results, each given a share of the context.", when: "Internal docs plus current facts.", calls: "1 + search" },
  { id: "graph", name: "GraphRAG", what: "LLM-built knowledge graph with communities; local (entity) and global (theme) search.", when: "Relationships and corpus-wide themes.", calls: "1 per query; 1 per chunk to index" },
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

export const LIMITATIONS = [
  { title: "In-memory store has a ceiling", text: "About 1 GB of RAM per 30,000 chunks (1536-dim). Above ~100k chunks use Chroma or another vector database." },
  { title: "One process by default", text: "Chat memory and the in-memory index live in one process. Several server workers need a shared vector DB and chat store; no Redis store ships yet." },
  { title: "GraphRAG costs one LLM call per chunk", text: "Indexing is parallel and cached, but large corpora are expensive. Similar names (\"Quillstone\" vs \"Quillstone Robotics\") stay separate unless resolve_with_embeddings=True." },
  { title: "Free tiers are small", text: "An OpenRouter free account allows about 50 free-model requests per day, and embeddings count." },
  { title: "Not every backend is live-tested", text: "OpenAI-compatible (OpenRouter) is tested live. Neo4j, Anthropic and Groq adapters are covered by code paths but not yet live-tested." },
  { title: "Answers are only as good as the model", text: "Citations and the injection guard reduce, not eliminate, hallucination and prompt injection. Treat answers as untrusted output." },
  { title: "Ingestion is single-threaded for PDFs", text: "Parsing many PDFs takes a while on the first run. Re-runs resume and skip unchanged files." },
  { title: "GraphContextWindow is not thread-safe", text: "Use one instance per conversation or guard it with a lock." },
];

export const WHY = [
  { title: "Instead of wiring ten pieces", text: "Loaders, chunking, embeddings, a vector store, BM25, fusion, reranking, prompts, citations and memory are already connected and tested together." },
  { title: "Switch techniques with one argument", text: "rag_type=\"corrective\" or \"graph\" changes the architecture, not your code. Compare them on your own data." },
  { title: "Production details included", text: "Retries, rate limits, caching, concurrency locks, streaming, async, tracing, evaluation and safe deletion." },
];
