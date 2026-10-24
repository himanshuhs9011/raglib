// Every RAG(...) argument, generated from inspect.signature(RAG.__init__) in raglib 2.0.4 and grouped by purpose.

export interface Param {
  name: string;
  default: string;
  text: string;
}

export interface ParamGroup {
  id: string;
  title: string;
  intro: string;
  params: Param[];
}

export const PARAM_GROUPS: ParamGroup[] = [
  {
    id: "sources",
    title: "Documents & chunking",
    intro: "What to index and how it is split.",
    params: [
      { name: "source", default: "None", text: "File, folder, URL, raw text, Document, or a list of these. Can also be added later with rag.add()." },
      { name: "chunk_size", default: "400", text: "Chunk length in words." },
      { name: "chunk_overlap", default: "50", text: "Words shared by neighbouring chunks, so sentences are not cut off at a boundary." },
      { name: "chunking_strategy", default: '"fixed"', text: '"fixed" (word windows), "recursive" (paragraph → sentence aware) or "llm" (the LLM picks semantic boundaries; costs LLM calls).' },
      { name: "output_dir", default: "None", text: "If set, every question and answer is also saved as JSON in this folder." },
    ],
  },
  {
    id: "models",
    title: "Models & API keys",
    intro: 'Chat, embedding and vision (OCR) models. "provider/model" strings work everywhere.',
    params: [
      { name: "llm", default: "None", text: 'Shorthand for chat_llm, e.g. "openai", "anthropic/claude-haiku-5-5", "openrouter/google/gemini-2.5-flash". None: first provider whose key is in the environment.' },
      { name: "chat_llm", default: "None", text: "Same as llm. Also accepts a BaseLLMClient or any LangChain chat model object." },
      { name: "chat_model", default: "None", text: "Model name if chat_llm is only a provider name." },
      { name: "chat_base_url", default: "None", text: "Custom endpoint: vLLM, LM Studio, a proxy, an Azure endpoint." },
      { name: "llm_key", default: "None", text: "One API key used for every model of that provider." },
      { name: "chat_api_key", default: "None", text: "Key for the chat model (overrides llm_key). Default: the provider's environment variable." },
      { name: "embedding_llm", default: "None", text: 'Embedding model: "openai", "google", "mistral", "ollama", "huggingface" (local), "openrouter/<model>", "none" (BM25 only). None: chat provider\'s embeddings, else local HuggingFace, else BM25.' },
      { name: "embedding_model", default: "None", text: "Embedding model name." },
      { name: "embedding_base_url", default: "None", text: "Custom embedding endpoint." },
      { name: "embedding_api_key", default: "None", text: "Key for embeddings. A chat key is only reused when the embedding provider is the same." },
      { name: "vision_llm", default: "auto", text: "Reads scanned PDF pages (OCR). Auto-enabled when the chat provider is OpenAI, Anthropic or Google." },
      { name: "vision_model", default: "None", text: "Vision model name." },
      { name: "vision_base_url", default: "None", text: "Custom vision endpoint." },
      { name: "vision_api_key", default: "None", text: "Key for the vision model." },
    ],
  },
  {
    id: "reliability",
    title: "Reliability, cost & rate limits",
    intro: "How provider calls behave under load, outages and quotas.",
    params: [
      { name: "llm_timeout", default: "60.0", text: "Seconds before a provider call is abandoned." },
      { name: "llm_max_retries", default: "2", text: "Retries for 429 / 5xx / timeouts, waiting as long as the server asks (Retry-After)." },
      { name: "llm_rpm", default: "None", text: "Maximum requests per minute, shared by chat and embedding calls to the same provider and key. OpenRouter :free models default to 20; 0 disables." },
      { name: "llm_fallbacks", default: "None", text: 'Backup models tried in order when the main one fails, e.g. ["openrouter/anthropic/claude-haiku-5-5"].' },
      { name: "llm_cache", default: "True", text: "Caches identical LLM calls (query rewriting, grading, graph extraction). Cleared when documents are deleted." },
    ],
  },
  {
    id: "pipeline",
    title: "Pipeline",
    intro: "Which architecture runs and how many chunks it uses.",
    params: [
      { name: "rag_type", default: '"advanced"', text: "One of the 14 architectures: naive, advanced, hybrid, corrective, self, agentic, tool, multi_hop, multi_query, memory, routing, web, hybrid_web, graph." },
      { name: "top_k", default: "5", text: "Chunks given to the LLM per question." },
      { name: "rewrite_query", default: "False", text: "Rewrite each question for retrieval before searching (+1 LLM call)." },
      { name: "**rag_type_kwargs", default: "", text: "Options specific to one rag_type, such as max_hops or search_mode (see below)." },
    ],
  },
  {
    id: "retrieval",
    title: "Retrieval",
    intro: "How chunks are found and ranked.",
    params: [
      { name: "retrieval_mode", default: "auto", text: '"hybrid" (BM25 + vectors; default when an embedding model is set), "dense" (vectors only) or "sparse" (BM25 keywords only).' },
      { name: "fusion", default: '"relative"', text: 'How keyword and vector scores are combined: "relative" (normalized score blend) or "rrf" (rank fusion).' },
      { name: "hybrid_alpha", default: "0.5", text: "Weight of vectors vs keywords in relative fusion: 0 = keywords only, 1 = vectors only." },
      { name: "reranker", default: '"none"', text: '"cross_encoder" (local model, [rerank] extra), "cohere", "llm" (1 extra LLM call) or "none".' },
      { name: "reranker_model", default: "None", text: 'Reranker model, e.g. "BAAI/bge-reranker-base".' },
      { name: "reranker_api_key", default: "None", text: "Key for a hosted reranker (Cohere). Default: COHERE_API_KEY." },
      { name: "fetch_k", default: "auto", text: "Candidates fetched before reranking / MMR. Auto: 4 × top_k (at least 20 with a reranker)." },
      { name: "mmr_lambda", default: "None", text: "Turns on MMR diversity: 1.0 = relevance only, lower = more varied chunks (0.5 is a good start)." },
      { name: "neighbor_window", default: "0", text: "Also include N neighbouring chunks around each hit (small-to-big context)." },
    ],
  },
  {
    id: "generation",
    title: "Context & conversation",
    intro: "What the LLM sees when it writes the answer.",
    params: [
      { name: "max_context_tokens", default: "3000", text: "Token budget for retrieved text in the prompt. Chunks are trimmed, not dropped, to fit." },
      { name: "memory", default: "auto", text: "Per-session chat history. On by default only for rag_type=\"memory\"; True enables it for any type." },
      { name: "max_history_turns", default: "10", text: "Conversation turns kept per session." },
      { name: "max_history_tokens", default: "1200", text: "Token cap for history included in the prompt." },
    ],
  },
  {
    id: "web",
    title: "Web search",
    intro: "Used by web, hybrid_web, corrective (fallback), agentic and routing.",
    params: [
      { name: "web_search_provider", default: "None", text: '"duckduckgo" (free, no key), "tavily", "serpapi", "brave", "bing", "google_cse", "exa" or "searxng". None: no web access at all.' },
      { name: "web_search_api_key", default: "None", text: "Key for the search provider. Default: its environment variable (e.g. TAVILY_API_KEY)." },
      { name: "web_search_base_url", default: "None", text: "Your SearXNG server URL." },
      { name: "web_search_cse_id", default: "None", text: "Google Custom Search engine id." },
      { name: "web_search_provider_kwargs", default: "None", text: "Extra options for the chosen search provider; unknown options raise an error." },
      { name: "validate_web_search_api_key", default: "False", text: "Check the search key at startup instead of on first use." },
    ],
  },
  {
    id: "storage",
    title: "Vector store",
    intro: "Where embeddings live.",
    params: [
      { name: "vector_db", default: '"memory"', text: '"memory" (in RAM, save with rag.save()) or "chroma" (persistent / client-server). A BaseVectorStore instance also works.' },
      { name: "vector_db_kwargs", default: "None", text: 'Store options, e.g. {"persist_directory": "./chroma", "collection_name": "docs"} or {"host": ..., "port": ...}.' },
    ],
  },
  {
    id: "graph",
    title: "GraphRAG & tools",
    intro: "Knowledge graph and agent tools.",
    params: [
      { name: "enable_graph", default: "False", text: "Build the knowledge graph for non-graph types too, so agentic and routing gain a graph_search tool / route." },
      { name: "graph_store", default: '"networkx"', text: '"networkx" (in memory, saved as JSON) or "neo4j".' },
      { name: "graph_kwargs", default: "None", text: "entity_types, max_workers, gleanings, max_levels, max_community_size, community_rebuild_threshold, resolve_with_embeddings, resolution_threshold, min_llm_report_size, top_k_entities, hops, max_neighbors, max_relations, max_reports, top_communities, max_points, global_level, store_options." },
      { name: "tools", default: "None", text: "Your own Tool(name, description, func) objects for rag_type agentic or tool." },
    ],
  },
  {
    id: "security",
    title: "Security & observability",
    intro: "Guard rails for user-supplied input, and hooks for monitoring.",
    params: [
      { name: "allowed_dirs", default: "None", text: "File and folder sources must be inside these directories (.. and symlinks resolved). Always set it when paths come from users." },
      { name: "allow_private_urls", default: "False", text: "Allow URLs pointing to private / localhost / cloud-metadata addresses. Keep False unless the source is a trusted intranet." },
      { name: "on_event", default: "None", text: "Callback receiving a TraceEvent(stage, latency_ms, meta) for every stage: send to logs, OpenTelemetry, LangSmith." },
      { name: "offline", default: "None", text: "True: mock LLM + hash embeddings, no network. For tests and demos only (see below). None: reads RAGLIB_OFFLINE." },
    ],
  },
];

export const TYPE_OPTIONS: Param[] = [
  { name: "search_mode", default: '"auto"', text: 'graph: "local" (entities), "global" (themes) or "auto".' },
  { name: "fuse_with_chunks", default: "True", text: "graph: also add normal hybrid chunk results to graph context." },
  { name: "max_hops", default: "3", text: "multi_hop: maximum retrieval rounds." },
  { name: "max_steps", default: "4", text: "agentic / tool: maximum tool calls before answering." },
  { name: "max_llm_calls", default: "8", text: "self: hard cap on LLM calls per question." },
  { name: "max_retrieve / max_revise / max_rewrite", default: "—", text: "self: retries for each Self-RAG loop." },
  { name: "refine_strips", default: "True", text: "corrective: keep only the relevant sentences of graded chunks." },
  { name: "max_strips", default: "40", text: "corrective: skip sentence refinement when the chunks contain more sentences than this (cost guard)." },
  { name: "include_web", default: "False", text: "hybrid: add web results (same as rag_type=\"hybrid_web\")." },
  { name: "tool_call_hook", default: "None", text: "tool: a fixed function(query, docs) -> docs run on every question (instead of LLM-chosen tools)." },
];

export const METHODS: Param[] = [
  { name: "query(question, session_id=None, filters=None)", default: "GenerationResult", text: "Answer one question." },
  { name: "await aquery(question, session_id=None, filters=None)", default: "GenerationResult", text: "Async version for FastAPI and other async servers." },
  { name: "stream(question, session_id=None, filters=None)", default: "Iterator[str]", text: "Yield the answer token by token." },
  { name: "add(source, batch_size=256, on_progress=None)", default: "int", text: "Index more documents. Idempotent; resumes after failures. Returns new chunks." },
  { name: "update(source)", default: "int", text: "Re-index changed files." },
  { name: "delete(source)", default: "int", text: "Remove chunks, vectors, graph entities and cached copies of a source." },
  { name: "save(path) / load(path)", default: "None", text: "Persist and restore the index without re-embedding." },
  { name: "reset_memory(session_id=None)", default: "None", text: "Clear one conversation, or all." },
  { name: "evaluate(dataset, judge_llm=None)", default: "EvalReport", text: "Retrieval and answer-quality metrics." },
  { name: "visualize_graph(path)", default: "str", text: "Interactive HTML of the knowledge graph." },
  { name: "rebuild_graph_communities()", default: "int", text: "Re-cluster the graph and refresh changed community reports." },
  { name: "stats()", default: "dict", text: "Chunks, documents, models, retrieval mode, graph size." },
  { name: "chat(session_id=\"interactive\")", default: "None", text: "Interactive terminal chat with memory." },
];

export const TOTAL_PARAMS = 56;
