# Changelog

## 2.2.0 - 2026-11-21

**Benchmark.** raglib's RAG types are now measured on MultiHop-RAG, a public third-party dataset: method, code and every answer are in `benchmarks/multihop_rag/`, and results are on the website's Benchmarks page.

**Fixes found by benchmarking on MultiHop-RAG** (see `benchmarks/multihop_rag/`).
- **Corrective RAG answered from an empty context.** Its grader asked whether each chunk *answers* the question, so for questions whose evidence spans several documents every chunk was graded irrelevant. Without web search the INCORRECT path then had nothing to use: on 30 unseen questions, 9 answers were generated with zero sources. The grader now counts a chunk about any one of the entities, sources or events the question names as partially relevant. Without web search, Corrective RAG falls back to the retrieved chunks instead of an empty context (`result.metadata["crag_fallback"]`). Empty-context answers went from 9/30 to 0/30.
- **Multi-query retrieval lost parts of compound questions.** The expander only paraphrased the whole question, and RRF fusion rewards chunks found by many queries, so for "does source A say X while source B says Y" one article took every slot. It now writes one focused query per entity, source or claim, and the fusion keeps each query's best hit before filling by RRF.
- **GraphContextWindow:** entity detection no longer joins words across line breaks (junk `SHARED_ENTITY` links), and `retrieve()` returns chunks before section summaries, with a `title` on every result.
- **GraphRAG kept holes in the graph after provider errors.** When an extraction call failed (rate limit, timeout, unparsable reply), the chunk was cached and marked indexed with no entities, so a temporary 429 permanently removed its facts from the graph. Failed chunks are now neither cached nor marked indexed. `rag.stats()["graph"]["failed_extractions"]` counts them, `rag.retry_graph_extraction()` extracts them again, and loading a saved index retries them automatically.
- **`graph_llm`:** build the GraphRAG graph with a cheaper or faster model than the answers.
- **GraphRAG answers saw half the evidence.** The entity, relation and community summaries were packed first into the 3,000-token context budget and used about 60% of it, so answers got 2.6 article chunks instead of 5. Questions that name two articles then failed with "the second source is not provided". Documents marked `metadata["supplementary"]` (GraphRAG's summaries) now have their own budget (`Generator(max_supplementary_tokens=1200)`) and can no longer push evidence out of the context.
- **Bounded extraction retries.** A chunk whose extraction keeps failing (some small models loop until `max_tokens` and return an empty reply) was retried, and paid for, on every add and every load. It is now retried at most `graph_kwargs={"max_extraction_attempts": 3}` times in total, then skipped. Its text stays searchable and `stats()["graph"]["skipped_extractions"]` counts it. `retry_graph_extraction(include_skipped=True)` tries again, for example with another model.
- **Truncated JSON is salvaged.** `parse_json` keeps every complete item of a reply cut off at `max_tokens` (`{"entities": [{...}, {...}, {"na` keeps the first two entities) instead of discarding the whole reply. It also no longer returns an inner list when the outer object is truncated.
- **Extraction replies are capped** at `graph_kwargs={"extraction_max_tokens": 4096}`, and the prompt states the entity and relation limits, so a looping model cannot run for minutes on one chunk.
- **Indexes load in seconds.** In-memory vectors (chunks and GraphRAG entities) are saved as binary float32 (`vectors.f32` + `vectors.meta.json`) instead of JSON numbers. A 609-article GraphRAG index went from 1.7 GB and an 81 s load to 402 MB and 2–5 s, with identical search results. Indexes saved by older versions still load, and the next `save()` converts them.
- **Background graph builds.** `RAG(rag_type="graph", graph_build="background")` returns from `add()` and `load()` as soon as chunks are searchable. Queries are answered immediately from hybrid search plus the part of the graph built so far, while one worker extracts in batches and builds community reports when the queue is empty. `rag.graph_status()` reports progress and `rag.wait_for_graph(timeout)` blocks until it is ready. Graph reads now use a read-write lock: queries run concurrently with each other and with community report generation, and are only paused for the short merge of each batch. Entity embedding calls no longer run while the graph is locked.
- **Per-model `json_mode`.** `models=[{"id": "...", "json_mode": False}]` on a custom provider stops raglib from sending `response_format` to models that accept JSON mode but then loop in it. JSON is parsed from the plain reply instead.

- **`SQLTool` broke after its first query on Python 3.9 and 3.10.** Clearing the read-only SQLite authorizer with `None` only works from Python 3.11. Before that, every later statement, `schema()` included, failed with "not authorized".
- **Packaging:** the source distribution no longer picks up README and LICENSE files from `site/node_modules`.
- **Citation labels no longer leak full Windows paths on Linux** (`C:\Users\...\report.pdf` is shown as `report.pdf`).

**LangChain is no longer used to call providers.**
- New native clients in `raglib.llm.clients` speak the three wire protocols directly with the standard library (`urllib`): `OpenAICompatibleClient` (OpenAI, Azure OpenAI, Groq, Ollama, OpenRouter, DeepSeek, Mistral, Together, Fireworks, xAI, Perplexity, Cerebras, LM Studio, vLLM), `AnthropicClient` and `GeminiClient`. They support SSE streaming, JSON mode (OpenAI-compatible and Gemini), provider token usage, retries and rate limiting. `build_chat_client(provider, ...)` builds one.
- Embeddings (OpenAI / Azure / compatible, Gemini, Ollama) call the REST APIs directly; `huggingface` uses `sentence-transformers` without `langchain-huggingface`. Class names are unchanged, so saved indexes still load.
- Vision OCR (OpenAI, Anthropic, Gemini) reuses the native chat clients. DuckDuckGo search uses `ddgs` directly.
- Retries use the HTTP status when there is one (408/409/425/429/5xx), not only the error text. An error event at the start of a stream (e.g. Anthropic `overloaded_error`) is retried.
- Importing raglib and building a client no longer loads LangChain: creating a `RAG` with a hosted provider went from about 4.3 s to 0.45 s.
- The `[openai]`, `[anthropic]`, `[groq]`, `[google]`, `[ollama]`, `[openrouter]` and `[all-llm]` extras are now empty (kept so existing install commands work); `[hf]` is just `sentence-transformers`. New `[langchain]` extra for wrapping your own LangChain model.
**Custom providers.**
- `register_provider("myprovider", api="openai" | "anthropic" | "gemini", base_url=..., api_key="{env:NAME}", headers=..., models=[{"id", "name", "reasoning", "image"}], embedding_model=...)` makes `RAG(llm="myprovider/model")` work for any third-party or self-hosted API. The key is optional (auth can go in headers), and `{env:NAME}` is expanded in keys and header values.
- `load_providers("providers.json")`, the `RAGLIB_PROVIDERS_FILE` environment variable, and `raglib ... --providers FILE` register providers from JSON. `raglib providers` lists them with their models.
- Model flags: `reasoning` stops sending `temperature`; `image` enables OCR of scanned pages with that model. A leading `<think>...</think>` block from reasoning models is removed from answers and streams.
- Custom providers get embeddings through their own API (OpenAI and Gemini protocols) and keep the key-isolation rule: a key is never sent to a different provider.

**NVIDIA NIM.** New built-in provider `nvidia` (alias `nim`, key prefix `nvapi-`, `NVIDIA_API_KEY`), for build.nvidia.com or a self-hosted NIM via `NVIDIA_BASE_URL`. Chat defaults to `nvidia/nemotron-3.5-lightning-30b-a3b`, embeddings to `nvidia/nemotron-3-embed-1b` (with the `input_type` passage/query fields these models require), and OCR of scanned pages to `nvidia/nemotron-3.5-super-vl-120b-a12b`. Tested live: cited answers, streaming with real usage, Corrective RAG grading in JSON mode, hybrid search, and OCR of a scanned PDF.
- Custom providers also accept `extra_body`, `embedding_input_types`, `embedding_params` and `stream_usage`. `vision_llm` works for any built-in provider whose models accept images (e.g. `vision_llm="openrouter/<model>"`).

- `LangChainAdapter` still wraps a LangChain chat model you pass in (`RAG(chat_llm=ChatOpenAI(...))`). `LangChainAdapter.build()` is deprecated and now returns the native client.

## 2.1.0 - 2026-11-06

Fixes and improvements from a full library review.

**Bug fixes**
- **Chunking ignored `chunk_size` for text without sentence punctuation.** The default `fixed` strategy put a 5,000-line file (logs, CSV-like records, one fact per line) into a single 50,000-word chunk, so the model saw only the first part after trimming. Lines are now boundaries, and anything longer than `chunk_size` is split further, for `fixed` and `recursive` alike. Chunks keep their line breaks.
- **Chinese, Japanese and Thai text was not chunked or searchable.** Sizes were counted in space-separated words, so 7,200 characters of Japanese became one chunk, and BM25 indexed whole clauses as single tokens. Each CJK/Thai/Lao/Khmer/Myanmar character now counts as one unit, `。！？` end sentences, keyword search uses character bigrams, the token estimate counts these characters (when tiktoken is missing), and truncation no longer empties such text.
- **A mistyped folder path was indexed as text.** `RAG("./my_docs")` with a missing folder created a document containing `"./my_docs"`. Path-like strings now raise `FileNotFoundError`.
- **A changed embedding model gave meaningless results silently.** Stores now refuse vectors of a different size, and `save()` records the embedding model and dimension; `load()` raises on a mismatch (`allow_model_change=True` re-embeds).
- **DNS rebinding could bypass the URL safety check.** The web loader now connects to the exact address it validated.
- **Conversation memory could raise "deque mutated during iteration"** when one session got concurrent requests; it also kept every turn's retrieved documents (unused), which grows with thousands of sessions. History is read under the lock and documents are not stored unless `store_documents=True`.
- **Streaming had no retries**: a 429 before the first token failed the request. Opening the stream is now retried like `complete`.
- **`rag.chat()` left memory switched on** for later `query()` calls.
- Folder ingestion no longer walks `.git`, `node_modules`, virtualenvs or caches, and skips binary files.
- PyMuPDF 1.24+/2.x: imports `pymupdf` (the `fitz` name is deprecated).

**New**
- `result.usage` reports totals over every LLM call of a query (`llm_calls`, `total_prompt_tokens`, `total_completion_tokens`, `total_tokens`), using the provider's real counts when LangChain returns them.
- `result.metadata["degraded"]` lists optional steps that failed and fell back (grader, router, reranker, rewrite, web search, tools, ...); `RAG(strict=True)` raises instead.
- `SQLTool`: exact counting/aggregation over CSV/TSV/XLSX/JSON tables for `rag_type="agentic"` (read-only SQLite, timeout, row cap).
- `SQLiteConversationStore` and `RAG(conversation_store=...)`: chat history that survives restarts and is shared by worker processes.
- Vector stores: `vector_db="qdrant"` (`[qdrant]` extra) and `vector_db="pgvector"` (`[pgvector]` extra).
- Loaders: HTML files, CSV/TSV, JSON/JSONL, XLSX (`[xlsx]` extra), EPUB, and source-code/config files. Web pages keep headings, paragraphs, lists and tables, prefer `<main>`/`<article>`, and drop navigation and footers; PDF responses are parsed as PDFs.
- `RAG(llm_temperature=..., llm_max_tokens=...)`.
- JSON mode for OpenAI and OpenAI-compatible providers in `complete_json` (falls back automatically when a model rejects it); the LLM cache and fallback clients keep it.
- Consecutive PDF pages and slides are linked, so `neighbor_window` can follow text across page breaks.

**Performance**
- MMR uses numpy when available.
- `delete` no longer blocks queries while the vector store deletes over the network.
- Chroma queries skip the extra `count()` round trip.

**Tooling**
- GitHub Actions CI: ruff, and pytest on Python 3.9-3.13 (plus Windows, and a pgvector service job).
- Ruff configuration in `pyproject.toml` (Python 3.9 target).

## 2.0.4 - 2026-10-24

- **`result.metadata` now carries pipeline diagnostics:** `crag_verdict`, `agent_steps`, `hop_queries`, `graph_mode`, `self_rag_llm_calls` were computed but dropped before the result was returned. Internal Document lists stay out, so metadata is JSON-serializable.
- **Examples:** `examples/rag_types_tour.py` (every rag_type, its options, and how to read each result field) and `examples/fastapi_app.py` (ask, stream, upload, delete, sessions, health; tested).

## 2.0.3 - 2026-10-24

- **Graph viewer layout:** nodes that started almost on top of each other were flung thousands of pixels away (repulsion had no minimum distance), so "Fit" zoomed out until the graph was a speck. Repulsion now has a minimum distance and node speed is capped.

## 2.0.2 - 2026-10-24

- **Runs in the browser (Pyodide) and other thread-less runtimes:** all parallel work goes through `raglib.core.concurrency.parallel_map`, which runs sequentially when threads are unavailable or cannot be started. Every `rag_type` now works offline in the browser, which powers the website playground.

## 2.0.1 - 2026-10-24

Fixes found by live testing every `rag_type` on OpenRouter free models.

- **hybrid_web:** web results never reached the answer. Weighted RRF (0.6/0.4, k=60) ranked the best web hit below corpus hit #31. Top-k slots are now split by weight (at least 1 web slot when the web returns results) and interleaved.
- **agentic / multi_hop:** evidence from separate steps or hops was merged with RRF, so repeated searches for one sub-question pushed out the evidence for another (the answer said the Bluefin code was "not mentioned" although step 1 found it). Steps and hops are now merged round-robin (`raglib.core.fusion.interleave`).
- **tool:** `rag_type="tool"` with `tools=[...]` raised an error. It now runs the agentic tool loop, where the LLM chooses which tools to call.
- **Generation prompt:** answer directly when the sources contain the answer (weaker models hedged with "not stated... however source 3 says...").

## 2.0.0 - 2026-10-24

Production-hardening release with breaking changes. See README section 16 for the migration table.

- **New: GraphRAG** (`rag_type="graph"`, `raglib.graphrag`): parallel, cached LLM entity/relation extraction; NetworkX (JSON-persisted) or Neo4j graph stores; hierarchical Leiden/Louvain communities with incremental reports; local search (no extra LLM calls) and global map-reduce search over embedding-prefiltered communities.
- **Retrieval:** BM25 + dense hybrid search by default with relative-score fusion (or RRF), metadata filters, MMR, neighbour (small-to-big) expansion, Unicode-aware tokenization (Indic, Arabic, Thai, Cyrillic, ...).
- **Reranking:** cross-encoder, Cohere, listwise-LLM and custom rerankers; the lossy lexical reranker was removed.
- **Generation:** delimited sources with prompt-injection guard, `[n]` citations parsed into `GenerationResult.citations`, "don't know" instruction, token-budgeted packing that trims instead of dropping, lost-in-the-middle ordering, streaming and async.
- **Conversations:** per-`session_id` memory with token budget and standalone-question condensation for every rag_type.
- **Ingestion:** idempotent upserts with content-hashed chunk ids, `update()`, `delete()`, `save()` / `load()` without re-embedding.
- **Strategies rewritten on one shared `BaseRAG` pipeline:** paper-faithful CRAG verdicts with batched grading and no implicit web search; Self-RAG with an LLM-call budget; real ReAct `AgenticRAG` with tools; iterative `MultiHopRAG`; parallel RRF `MultiQueryRAG`; `hybrid_web`; routing with a graph route.
- **Reliability:** LLM timeouts, jittered exponential-backoff retries, LRU response cache, cached + batched embeddings, per-stage `TraceEvent`s with an `on_event` hook, `raglib.eval` (hit rate, MRR, recall, nDCG, faithfulness, answer relevance, context precision, correctness).
- **Safety fixes:** substring keyword routing ("this" matched "hi" and skipped retrieval), silent mock LLM / embedding / vector-store fallbacks, shared Chroma collection across instances, deprecated default models.
- **LLM providers:** 17 providers from one registry: OpenRouter, DeepSeek, Mistral, Together, Fireworks, xAI, Perplexity, Cerebras, Azure OpenAI, LM Studio, vLLM and any OpenAI-compatible endpoint, alongside OpenAI, Anthropic, Gemini, Groq and Ollama. `llm="provider/model"` strings, provider auto-detection from environment keys, and longest-prefix key detection (OpenRouter `sk-or-` keys were previously treated as OpenAI). Chat keys are no longer sent to a different embedding provider.
- **Simpler usage:** `raglib.ask(source, question)` one-liner and a `raglib` CLI (`raglib ask`, `raglib chat`, `raglib providers`, also `python -m raglib`).
- **Graph visualization:** `rag.visualize_graph()`, `GraphContextWindow.to_html()` and `raglib graph` write a self-contained interactive HTML view (no CDN; content escaped against script injection).
- **Concurrency:** read-write locks in the retriever and in-memory store and locked graph-store reads. v1/early-v2 crashed in 25-42% of queries when documents were added during chats; now 0 errors under 16 concurrent sessions.
- **Scale:** float32 in-memory vectors (2.5 GB -> 1 GB at 30k chunks), argpartition top-k, streaming batched `add()` with `on_progress` and resume-after-failure, concurrent embedding batches.
- **Security:** SSRF guard for URL sources (private/metadata IPs, redirects, 20 MB cap), `allowed_dirs` path sandbox, user queries removed from INFO logs, `.env.example`, committed run reports with local paths untracked.
- **Rate limits and resilience:** a shared client-side limiter (`llm_rpm`; OpenRouter `:free` models default to 20/min, with chat and embeddings sharing one budget), retries that honour `Retry-After` / `X-RateLimit-Reset`, `QuotaExceededError` for daily or credit caps (fail fast), `llm_fallbacks=[...]` model fallback, and timeouts and retries for embedding calls.
- **GraphRAG cost:** communities with fewer than `min_llm_report_size` (3) entities get a template report instead of an LLM call.
- **GraphContextWindow:** optional `embedding=` for semantic seeding (finds paraphrased questions).
- **Ingestion and citations:** `rag.add()` accepts `Document` objects. Citations and prompt source labels use the file name and page, never the full local path (which used to be sent to the LLM provider).
- **Live-verified** on OpenRouter free chat and embedding models: needle-in-a-haystack, GraphRAG local and global, GraphContextWindow.
- **Packaging:** the core has no required dependencies; integrations are extras. Classifier set to Beta.
- **Removed:** `raglib.config`, `raglib.main`, duplicate `powerpoint_loader` / `plain_text_loader` / `webpage_loader`; `raglib.preprocessing` is a deprecated shim.

## 1.0.0 - 2026-10-14

- **Major Release (v1.0.0):**
  - Added selectable chunking strategies (`fixed`, `recursive`, and `llm` semantic chunking) in `ChunkSplitter` and `RAG(..., chunking_strategy=...)`.
  - Added standalone Graph-Based Context Window engine (`GraphContextWindow`, `GraphNode`, `ConnectedNodeRef`, `GraphContextResult`) supporting hierarchical document graphs (`Document -> Section -> Chunk`), neighbor summary signposts (`NEXT_CHUNK`, `PREV_CHUNK`, `SHARED_ENTITY`), and multi-turn conversational chat graphs (`PREV_TURN`, `SAME_TOPIC_TURN`, `CITED_CHUNK`).
  - Upgraded `Retriever` to perform incremental document indexing and reconstruct documents directly from pre-populated persistent vector stores (`ChromaVectorStore`, `InMemoryVectorStore`).
  - Replaced non-deterministic `hash()` IDs with deterministic SHA-256 document IDs across all loaders, providers, and retrievers.
  - Hardened `SelfRAG`, `RoutingRAG`, `MultiQueryRAG`, `MultiHopRAG`, `Generator`, `WebLoader`, and `DuckDuckGoProvider` for production reliability.
  - Exported all 12 RAG strategy classes, core components, and graph context classes from top-level `raglib`.
  - Updated repository URLs to `https://github.com/himanshuhs9011/raglib`.

## 0.1.10 - 2026-04-06

- Added additional interactive chat kill-switch commands: `bye` and `stop`
- Updated interactive session banner/help text to show all exit commands
- Updated README and usage docs so users can clearly see how to stop continuous chat sessions

## 0.1.9 - 2026-04-06

- Clarified chat continuity behavior in README for PyPI users (default stateless turn handling vs memory-enabled multi-turn context)
- Added two explicit usage snippets: one-time/stateless query mode and continuous/contextual memory mode
- Updated usage guide with matching chat continuity section and corrected section numbering

## 0.1.8 - 2026-04-06

- Added first-class web search integration in `RAG(...)` with new inputs:
	- `web_search_provider`
	- `web_search_api_key`
	- `web_search_base_url`
	- `web_search_cse_id`
	- `web_search_provider_kwargs`
	- `validate_web_search_api_key`
- Added authenticated web provider key validation with environment variable fallback support
- Added cloud chat/embedding key validation for explicit provider selection (`openai`, `anthropic`, `groq`, `google`)
- Updated default web provider to DuckDuckGo for out-of-the-box internet search
- Integrated `langchain_community` DuckDuckGo tools (`DuckDuckGoSearchRun` with validated output) for default web search
- Upgraded `WebRetriever` into a plug-and-play provider orchestrator with provider-name factory and automatic DuckDuckGo fallback
- Updated `WebRetriever` to fail safely by default and return empty results when web provider calls fail
- Updated `raglib/main.py` demo wiring to match latest orchestrator constructor signatures
- Added explicit web-provider count summary to README for PyPI package page clarity
- Updated README and usage docs with production "3-key" input patterns and provider/key requirements

## 0.1.7 - 2026-04-05

- Reworked CorrectiveRAG into a faithful three-path CRAG flow: CORRECT, AMBIGUOUS, and INCORRECT routing
- Added LLM-based retrieval evaluator and knowledge-refinement strip filtering pipeline
- Added CRAG web-search branch with query rewriting and context merge policy (k_in / k_ex)
- Added robust fallback behavior when optional LLM/web components are not configured

## 0.1.6 - 2026-04-05

- Updated default package dependencies so `pip install raglib-py` includes the full runtime stack used by core code paths
- Fixed common first-run missing module issue for Ollama chat/embedding by bundling `langchain-ollama` in base install
- Kept optional extras for users who prefer explicit feature grouping

## 0.1.5 - 2026-04-05

- Added explicit per-provider key options in the RAG facade: chat_api_key, embedding_api_key, vision_api_key
- Added endpoint options in the RAG facade: chat_base_url and vision_base_url
- Updated README and usage docs with clear API key ownership guidance and constructor examples

## 0.1.4 - 2026-04-05

- Expanded README into a full PyPI-first user guide with complete setup and usage coverage
- Documented all 12 built-in RAG strategies and usage patterns
- Standardized install instructions and runtime dependency hints to use raglib-py

## 0.1.3 - 2026-04-05

- Release version bump for PyPI publication
- Packaging and distribution preflight improvements

## 0.1.2 - 2026-04-04

- Initial release of raglib
- Added modular RAG components, providers, and orchestrators
- Added offline-capable MockLLMClient
- Added full production-grade test suite
