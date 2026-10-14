# Changelog

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
