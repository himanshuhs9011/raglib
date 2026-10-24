"""Serve raglib over HTTP with FastAPI.

    pip install "raglib-py[openai,pdf]" fastapi uvicorn python-multipart
    export OPENROUTER_API_KEY=...                       # or any provider key
    export RAGLIB_LLM=openrouter/openai/gpt-4.1-mini    # optional: provider/model
    export RAGLIB_DOCS=./docs                           # folder indexed at startup (uploads go here too)
    uvicorn examples.fastapi_app:app --port 8000

    curl -X POST localhost:8000/ask -H "content-type: application/json" \
         -d '{"question": "What is the refund window?", "session_id": "user-42"}'

No key? RAGLIB_OFFLINE=1 runs everything with the offline mock model.
One RAG instance is shared by all requests: queries run in parallel, add/delete take a short write lock,
and session_id keeps every user's chat history separate.
"""

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from raglib import RAG
from raglib.components.generator import source_label
from raglib.llm.resilience import QuotaExceededError

DOCS_DIR = Path(os.getenv("RAGLIB_DOCS", "docs")).resolve()
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def build_rag() -> RAG:
    """One shared RAG for the whole app (configured from environment variables)."""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    has_docs = any(path.suffix.lower() in ALLOWED_EXTENSIONS for path in DOCS_DIR.rglob("*"))
    return RAG(
        str(DOCS_DIR) if has_docs else None,
        llm=os.getenv("RAGLIB_LLM"),                      # None: first provider key found in the environment
        embedding_llm=os.getenv("RAGLIB_EMBEDDING"),      # None: the chat provider's embeddings / local / BM25
        rag_type=os.getenv("RAGLIB_RAG_TYPE", "advanced"),
        allowed_dirs=[str(DOCS_DIR)],                     # file sources can never escape DOCS_DIR
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = build_rag()
    yield


app = FastAPI(title="raglib API", lifespan=lifespan)


# ---------------------------------------------------------------- schemas
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None, max_length=128, description="Per-user chat memory")
    filters: Optional[Dict[str, Any]] = Field(default=None, description='e.g. {"file": {"$in": ["policy.md"]}}')


class CitationOut(BaseModel):
    index: int
    source: str
    snippet: str


class SourceOut(BaseModel):
    id: str
    source: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[CitationOut]
    sources: List[SourceOut]
    usage: Dict[str, Any]
    metadata: Dict[str, Any]


def rag() -> RAG:
    return app.state.rag


def safe_filename(name: str) -> str:
    """Reject path tricks; keep a plain file name with an allowed extension."""

    clean = Path(name or "").name
    if not re.fullmatch(r"[\w .()-]{1,120}", clean) or Path(clean).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File name must be simple and end with one of {sorted(ALLOWED_EXTENSIONS)}")
    return clean


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", **rag().stats()}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        result = await rag().aquery(request.question, session_id=request.session_id, filters=request.filters)
    except QuotaExceededError as exc:
        raise HTTPException(429, "The model provider's quota is exhausted; try again later.") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return AskResponse(
        answer=result.answer,
        citations=[CitationOut(index=c.index, source=c.source, snippet=c.snippet) for c in result.citations],
        sources=[
            SourceOut(id=d.id, source=source_label(d), score=round(float(d.score), 4), snippet=d.content[:300])
            for d in result.sources
        ],
        usage=result.usage,
        metadata=result.metadata,  # pipeline, crag_verdict, agent_steps, graph_mode, ...
    )


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    tokens = rag().stream(request.question, session_id=request.session_id, filters=request.filters)
    return StreamingResponse(tokens, media_type="text/plain; charset=utf-8")


@app.post("/documents")
async def upload_document(file: UploadFile) -> Dict[str, Any]:
    name = safe_filename(file.filename or "")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 20 MB)")
    path = DOCS_DIR / name
    path.write_bytes(data)
    chunks = await run_in_threadpool(rag().add, str(path))  # embedding runs off the event loop
    return {"file": name, "chunks_added": chunks, "total_chunks": rag().stats()["chunks"]}


@app.delete("/documents/{filename}")
async def delete_document(filename: str) -> Dict[str, Any]:
    path = DOCS_DIR / safe_filename(filename)
    try:
        removed = await run_in_threadpool(rag().delete, str(path))  # chunks, vectors, graph entities, caches
    except FileNotFoundError:
        removed = 0
    if not removed and not path.exists():
        raise HTTPException(404, "Unknown document")
    path.unlink(missing_ok=True)
    return {"file": path.name, "chunks_removed": removed}


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str) -> Dict[str, str]:
    rag().reset_memory(session_id)
    return {"session_id": session_id, "status": "cleared"}
