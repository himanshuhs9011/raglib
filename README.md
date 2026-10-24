# raglib-py (v2.0.0) — Complete Documentation & Developer Guide

**`raglib-py`** is a modular Retrieval-Augmented Generation (RAG) library for Python. It provides a one-line `ask()` helper, a `raglib` command-line tool, a high-level `RAG` facade, **17 LLM providers** (OpenAI, Anthropic, Gemini, OpenRouter, DeepSeek, Mistral, Groq, Ollama, LM Studio, vLLM, ...), **14 RAG architectures** (including **Microsoft-style GraphRAG**), **hybrid BM25 + vector search**, **pluggable rerankers** (cross-encoder, Cohere, LLM), **grounded generation with citations**, **per-session conversation memory**, **3 chunking strategies**, a standalone **Graph-Based Context Window**, **9 web search providers**, **ChromaDB / in-memory / Neo4j storage**, and a built-in **evaluation** module.

- **PyPI Package:** [https://pypi.org/project/raglib-py/](https://pypi.org/project/raglib-py/) (`pip install raglib-py`)
- **Python Import Name:** `raglib` (`import raglib`)
- **GitHub Repository:** [https://github.com/himanshuhs9011/raglib](https://github.com/himanshuhs9011/raglib)

---

## Table of Contents

1. [Installation](#1-installation)
2. [60-Second Quickstart (Offline & Cloud)](#2-60-second-quickstart-offline--cloud)
3. [The `RAG` Facade API (Complete Reference)](#3-the-rag-facade-api-complete-reference)
4. [Supported Input Sources & Loaders](#4-supported-input-sources--loaders)
5. [Chunking Strategies (`fixed`, `recursive`, `llm`)](#5-chunking-strategies-fixed-recursive-llm)
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
    - [Seeing the Graph](#seeing-the-graph) · [Security & Privacy](#security--privacy) · [Scaling & Concurrency](#scaling--concurrency)
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
pip install "raglib-py[openai]"          # also: anthropic, groq, google, ollama
pip install "raglib-py[hf]"              # local HuggingFace / SentenceTransformers embeddings
pip install "raglib-py[rerank]"          # cross-encoder reranker (bge-reranker)
pip install "raglib-py[fast]"            # numpy vector math + tiktoken token counting
pip install "raglib-py[chroma]"          # persistent ChromaDB vector store
pip install "raglib-py[graph]"           # GraphRAG community detection (networkx)
pip install "raglib-py[neo4j]"           # Neo4j graph store
pip install "raglib-py[pdf,docx,pptx]"   # document loaders
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
    # ---- pipeline
    rag_type="advanced",                # One of the 14 RAG architectures
    top_k=5,                            # Documents passed to generation
    max_context_tokens=3000,            # Generation context budget (tiktoken when installed)
    rewrite_query=False,                # LLM query rewriting before retrieval
    # ---- chunking
    chunk_size=400, chunk_overlap=50,   # words
    chunking_strategy="fixed",          # "fixed" | "recursive" | "llm"
    # ---- retrieval
    retrieval_mode=None,                # "hybrid" (default with embeddings) | "dense" | "sparse"
    fusion="relative",                  # hybrid fusion: "relative" (score-based) | "rrf"
    hybrid_alpha=0.5,                   # weight of the dense score in relative fusion
    fetch_k=None,                       # candidates fetched before rerank/MMR
    mmr_lambda=None,                    # e.g. 0.5 to diversify results with MMR
    neighbor_window=0,                  # small-to-big: attach N neighbouring chunks to each hit
    reranker="none",                    # "none" | "cross_encoder" | "cohere" | "llm"
    reranker_model=None, reranker_api_key=None,
    vector_db=None,                     # "memory" (default) | "chroma" | BaseVectorStore
    vector_db_kwargs=None,              # e.g. {"persist_directory": "...", "collection_name": "...", "host": ...}
    # ---- memory
    memory=None,                        # True to enable conversation memory for any rag_type
    max_history_turns=10, max_history_tokens=1200,
    # ---- web search (OFF unless configured)
    web_search_provider=None,           # "duckduckgo" | "local" | "tavily" | "serpapi" | "brave" | "bing" | "google_cse" | "exa" | "searxng"
    web_search_api_key=None, web_search_base_url=None, web_search_cse_id=None,
    web_search_provider_kwargs=None, validate_web_search_api_key=False,
    # ---- GraphRAG
    enable_graph=False,                 # build the knowledge graph for non-graph rag_types (agentic/routing tools)
    graph_store=None,                   # "networkx" (default) | "neo4j" | BaseGraphStore
    graph_kwargs=None,                  # see section 8
    # ---- agents & observability
    tools=None,                         # [Tool(...)] for rag_type="agentic"
    on_event=None,                      # callback(TraceEvent) for every pipeline stage
    output_dir=None,                    # write each query result as JSON
    **rag_type_kwargs,                  # forwarded to the orchestrator (e.g. max_hops=3, search_mode="global")
)
```

### Facade Methods

| Method | Description |
| :--- | :--- |
| `rag.query(question, session_id=None, filters=None)` | Run the pipeline. Returns `GenerationResult(answer, sources, citations, reasoning_trace, trace_events, usage)`. |
| `rag.stream(question, ...)` | Generator of answer tokens; its return value is the final `GenerationResult`. |
| `await rag.aquery(question, ...)` | Async query. |
| `rag.add(source) -> int` | Idempotent upsert: unchanged documents are skipped, changed ones replace their old chunks. Returns new chunks indexed. |
| `rag.update(source)` / `rag.delete(source)` | Re-ingest changed sources / remove every chunk (and graph entity) from a source. |
| `rag.save(dir)` / `rag.load(dir)` | Persist and restore the chunk index, in-memory vectors and knowledge graph without re-embedding. |
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
| **Text / Markdown** (`.txt`, `.md`) | `"notes.md"` | `TextLoader` |
| **Web URL** (`http://`, `https://`) | `"https://example.com/docs"` | `WebLoader` (strips `<script>`/`<style>` and extracts visible text) |
| **Raw text string** | `"RAG grounds LLM outputs."` | Wrapped directly as a `Document` (content-hash id) |
| **List of mixed sources** | `["a.pdf", "https://...", "raw note"]` | Each item is routed to its matching loader |

> **Safety Note:** If a string looks like a file path (e.g., `"missing_file.pdf"`) and does not exist on disk, `DocumentLoader` raises `FileNotFoundError` immediately rather than silently indexing the filename string.

---

## 5. Chunking Strategies (`fixed`, `recursive`, `llm`)

| Strategy | Aliases | How It Works | Best For |
| :--- | :--- | :--- | :--- |
| **`"fixed"`** *(default)* | `"fixed_size"`, `"sliding"`, `"sentence"` | Packs complete sentences up to `chunk_size` with sentence-aware `chunk_overlap`. | Clean prose, fast ingestion. |
| **`"recursive"`** | `"recursive_character"` | Splits along structural separators (`"\n\n"`, `"\n"`, `". "`, ... `""`) to keep paragraphs together, then applies overlap. | Markdown, structured reports. |
| **`"llm"`** | `"semantic"`, `"llm_semantic"` | The LLM groups numbered sentences at topic shifts (`CHUNK: start-end`); falls back to `"recursive"` without an LLM. | High-value documents with shifting topics. |

Every chunk gets a stable id `{parent_id}_chunk_{n}_{content_hash}` plus `parent_id`, `prev_id` and `next_id` metadata. That's what makes re-ingestion idempotent and neighbour expansion possible.

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

Tokenization is Unicode-aware, including combining marks, so Hindi, Bengali, Tamil, Arabic, Thai, Cyrillic and accented Latin all work for BM25.

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
| **`"advanced"`** *(default)* | `AdvancedRAG` | Hybrid retrieve -> rerank -> dedupe -> cited generate | 1 | Most production Q&A |
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
| **Azure OpenAI** | `"azure/<deployment>"` | – | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | yes (deployment) |
| **Ollama** (local) | `"ollama"` | `llama3.1` | – | yes |
| **LM Studio** (local) | `"lmstudio/<model>"` | – | – | with `embedding_model` |
| **vLLM** (self-hosted) | `"vllm/<model>"` + `chat_base_url` | – | – | with `embedding_model` |
| **Any OpenAI-compatible API** | `"openai_compatible/<model>"` + `chat_base_url` | – | `OPENAI_COMPATIBLE_API_KEY` | with `embedding_model` |
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
- **`"chroma"`**: ChromaDB in cosine space. Each `RAG` gets its **own collection** unless you name one, so instances never see each other's data. Use `{"persist_directory": ..., "collection_name": ...}` for a durable index or `{"host": ..., "port": ...}` for a Chroma server.
- **Your own**: subclass `BaseVectorStore` (`upsert`, `query(filters=...)`, `delete`, `count`).

Initialization errors are raised. Pass `vector_db_kwargs={"fallback_to_memory": True}` to opt into a fallback.

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

# Streaming
for token in rag.stream("Summarize the onboarding guide"):
    print(token, end="", flush=True)

# Async (FastAPI etc.)
# result = await rag.aquery("...", session_id=request.user_id)

# Tracing: every stage is timed; plug in LangSmith / OpenTelemetry / logging
rag = RAG("docs/", chat_llm="openai", on_event=lambda e: print(e.stage, e.latency_ms, e.meta))
result = rag.query("...")
print(result.usage)            # {"prompt_tokens": ..., "completion_tokens": ..., "latency_ms": ...}

rag.chat()                     # interactive terminal session with memory
```

---

## 13. Evaluation

```python
from raglib import RAG
from raglib.eval import evaluate
from raglib.llm.langchain_adapter import LangChainAdapter

rag = RAG("docs/", chat_llm="openai", embedding_llm="openai")
dataset = [
    {"question": "What is the refund window?", "relevant_ids": ["docs/refunds.md"], "reference": "30 days"},
]
judge = LangChainAdapter.build("openai", model_name="gpt-4.1")   # use a strong judge model
report = evaluate(rag, dataset, judge_llm=judge, k=5)
print(report)   # hit_rate@5, mrr@5, recall@5, ndcg@5, faithfulness, answer_relevance, context_precision, correctness, latency_ms
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

llm = MockLLMClient()                                   # use LangChainAdapter.build("openai") in production
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

Subclass `BaseRAG` and override `retrieve(ctx)` (or `process_documents` / `generate` / `answer`) to build your own strategy. Condensation, memory, tracing, citations and streaming come for free.

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
  - `usage: Dict` — prompt/completion tokens and total latency
  - `metadata: Dict` — pipeline name, standalone search query, session id, strategy details
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
| User-supplied URLs | Private, loopback, link-local and metadata addresses are blocked (SSRF), redirects are re-checked, and responses are capped at 20 MB. | `allow_private_urls=True` only for trusted intranet sources. |
| User-supplied file paths | `allowed_dirs=[...]` sandboxes file and folder sources (resolves `..` and symlinks). | Always set it when paths come from users. |
| Prompt injection | Retrieved text is wrapped in `<source>` blocks with an instruction to treat it as data. | Treat answers as untrusted output; don't give agent tools destructive permissions. |
| Multi-user | `session_id` isolates chat history. Documents are shared within one `RAG` instance. | Use one `RAG` per tenant, or `filters={"tenant": ...}` on every query. |
| Data at rest | `output_dir`, `save()`, Chroma and graph files contain document text and Q&A. | Store them on encrypted disks; delete with `rag.delete(source)` (removes chunks, vectors and graph entities). |

If a key was ever committed to git, rotating it is the only fix: deleting the line does not remove it from history.

## Scaling & Concurrency

- **Many users at once:** one `RAG` instance is thread-safe. Queries run in parallel (read lock) while `add` / `delete` take a short write lock. Embedding calls happen outside the lock. Give every chat window its own `session_id`. Tested with 16 concurrent sessions plus a writer thread: 0 errors, 0 history leaks.
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
