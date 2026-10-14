# raglib-py

raglib-py is a production-focused Retrieval-Augmented Generation library for Python.

Important naming:

- PyPI package name: raglib-py
- Python import name: raglib

## Current Support Counts

- RAG strategies: 12
- Built-in chat providers: 5
- Built-in web providers: 9 total
- Internet web providers: 8
- Offline local web provider: 1

## Web Provider Count In The Library

- Total web providers: 9
- Free default: duckduckgo
- Auth-required: tavily, serpapi, brave, bing, google_cse, exa, searxng
- Offline/local: local

## Installation

```bash
pip install raglib-py
```

Optional extras:

```bash
pip install "raglib-py[ollama]"
pip install "raglib-py[chroma]"
pip install "raglib-py[pdf,docx,pptx]"
pip install "raglib-py[all]"
```

Quick import check:

```bash
python -c "from raglib import RAG; print('OK')"
```

## Quickstart (Offline)

```python
from raglib import RAG

rag = RAG("RAG improves grounded generation using retrieved context.")
result = rag.query("What does RAG improve?")

print(result.answer)
print([doc.id for doc in result.sources])
```

## Recommended Production Input (3 Main Keys)

Use this pattern when you want cloud chat + cloud embedding + live web search:

```python
from raglib import RAG

rag = RAG(
    source="docs/",
    rag_type="web",  # web/hybrid/routing/corrective can involve web search

    # 1) Chat API key
    chat_llm="openai",
    chat_api_key="YOUR_CHAT_API_KEY",

    # 2) Embedding API key
    embedding_llm="openai",
    embedding_api_key="YOUR_EMBEDDING_API_KEY",

    # 3) Web search API key (required for authenticated web providers)
    web_search_provider="tavily",
    web_search_api_key="YOUR_WEB_SEARCH_API_KEY",

    # Optional: verify provider credentials/connectivity during init
    validate_web_search_api_key=True,
)

print(rag.query("latest AI news and key trends").answer)
```

## RAG Constructor

```python
RAG(
    source=None,
    chat_llm=None,
    embedding_llm=None,
    vision_llm=None,
    llm_key=None,
    chat_api_key=None,
    embedding_api_key=None,
    vision_api_key=None,
    rag_type="corrective",
    top_k=5,
    chunk_size=400,
    chunk_overlap=50,
    chunking_strategy="fixed",  # "fixed" | "recursive" | "llm"
    output_dir=None,
    chat_model=None,
    chat_base_url=None,
    embedding_model=None,
    embedding_base_url=None,
    vision_model=None,
    vision_base_url=None,
    web_search_provider="duckduckgo",
    web_search_api_key=None,
    web_search_base_url=None,
    web_search_cse_id=None,
    web_search_provider_kwargs=None,
    validate_web_search_api_key=False,
    vector_db=None,
    vector_db_kwargs=None,
    **rag_type_kwargs,
)
```

## API Key Rules

raglib never provides API keys. Users must provide their own valid keys.

Chat keys:

- chat_llm=openai, anthropic, groq, google -> requires chat_api_key or llm_key (or env var)
- chat_llm=ollama -> no cloud key required

Embedding keys:

- embedding_llm=openai, google -> requires embedding_api_key (or llm_key/chat_api_key/env)
- embedding_llm=ollama, huggingface/local/free, mock -> no cloud key required

Web keys:

- web_search_provider=duckduckgo -> free, no API key required
- web_search_provider=local -> offline local search over ingested docs
- web_search_provider=tavily, serpapi, brave, bing, google_cse, exa, searxng -> requires web_search_api_key
- web_search_provider=google_cse -> also requires web_search_cse_id
- web_search_provider=searxng -> may require web_search_base_url

If web provider fails at runtime, raglib returns empty web results instead of crashing the whole RAG run.

If you do not set web_search_provider, raglib defaults to duckduckgo.

## Web Providers

Supported web_search_provider values:

1. local
2. duckduckgo
3. tavily
4. serpapi
5. brave
6. bing
7. google_cse
8. exa
9. searxng

## Built-in RAG Strategies

1. naive
2. advanced
3. corrective
4. self
5. agentic
6. hybrid
7. multi_query
8. multi_hop
9. routing
10. memory
11. web
12. tool

## Common Examples

Use a free web provider:

```python
from raglib import RAG

rag = RAG(
    source="docs/",
    rag_type="web",
    chat_llm="openai",
    chat_api_key="YOUR_CHAT_API_KEY",
    embedding_llm="openai",
    embedding_api_key="YOUR_EMBEDDING_API_KEY",
    web_search_provider="duckduckgo",
)
```

Use Google CSE:

```python
from raglib import RAG

rag = RAG(
    source="docs/",
    rag_type="hybrid",
    chat_llm="openai",
    chat_api_key="YOUR_CHAT_API_KEY",
    embedding_llm="openai",
    embedding_api_key="YOUR_EMBEDDING_API_KEY",
    web_search_provider="google_cse",
    web_search_api_key="YOUR_GOOGLE_API_KEY",
    web_search_cse_id="YOUR_CSE_ID",
)
```

Use local-only web mode (offline):

```python
from raglib import RAG

rag = RAG(
    source="docs/",
    rag_type="web",
    web_search_provider="local",
)
```

## Useful Methods

- query(question)
- add(source)
- chat()

## Chat Continuity And Context

How chat works today:

- You create one RAG instance and can ask many questions without re-running setup code.
- By default (`rag_type="corrective"`), each query is processed independently for retrieval/generation quality.
- If you need conversational memory across turns, use `rag_type="memory"`.
- Memory is in-process; if your script exits, memory resets.

One-time / stateless style (default):

```python
from raglib import RAG

rag = RAG(
    source="docs/",
    rag_type="corrective",  # default behavior
)

answer_1 = rag.query("What is RAG?")
answer_2 = rag.query("Give me key benefits in 3 points")

print(answer_1.answer)
print(answer_2.answer)
```

Continuous chat with context (memory-enabled):

```python
from raglib import RAG

rag = RAG(
    source="docs/",
    rag_type="memory",  # keeps previous turns in the same running instance
)

turn_1 = rag.query("Summarize the architecture")
turn_2 = rag.query("Now explain that in simpler terms")

print(turn_1.answer)
print(turn_2.answer)

# Optional interactive terminal chat loop (single run, continuous session)
rag.chat()
```

Interactive session kill-switch commands:

- exit
- quit
- q
- bye
- stop

## Chunking Strategies & Graph-Based Context Window (v1.0.0)

Select a chunking strategy directly in `RAG(...)` or `ChunkSplitter(...)`:

- `"fixed"` (default): Sentence-aware token window with overlap
- `"recursive"`: Hierarchical separator splitting (`\n\n`, `\n`, `. `, etc.)
- `"llm"`: LLM-guided semantic boundary detection with automatic fallback to `"recursive"`

Standalone Graph-Based Context Window (`GraphContextWindow`):

```python
from raglib import Document, GraphContextWindow

graph = GraphContextWindow(chunking_strategy="recursive", chunk_size=180, chunk_overlap=25)
graph.add_documents([Document(id="doc1", content="# Overview\nCorrectiveRAG uses an Evaluator threshold.")])

ctx = graph.build_context_window("How does CorrectiveRAG use the Evaluator threshold?")
print(ctx.formatted_context)
```

## License

MIT
