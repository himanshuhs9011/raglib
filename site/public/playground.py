# Runs inside the browser (Pyodide). Uses the real raglib package from PyPI in offline mode:
# no network calls, no API keys - the "LLM" is raglib's deterministic offline client.
import json
import time

from raglib import RAG

_rag = None


def build(text, rag_type="advanced", retrieval_mode="hybrid", chunk_size=60, top_k=4):
    """Index ``text`` and return corpus stats as JSON."""

    global _rag
    started = time.perf_counter()
    kwargs = {}
    if rag_type == "graph":
        kwargs["graph_kwargs"] = {"max_workers": 1}  # browsers cannot start threads
    _rag = RAG(
        text,
        offline=True,
        rag_type=rag_type,
        retrieval_mode=retrieval_mode,
        chunk_size=int(chunk_size),
        chunk_overlap=0,
        top_k=int(top_k),
        **kwargs,
    )
    stats = _rag.stats()
    stats["index_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return json.dumps(stats)


def _extractive_answer(question, sources):
    """Best-matching sentence from the retrieved sources (no LLM): (sentence, 1-based source index)."""

    import re

    from raglib.core.text import token_set

    terms = token_set(question, remove_stopwords=True)
    best = ("", 0, 0.0)
    for index, doc in enumerate(sources, start=1):
        text = re.sub(r"#{1,6}\s[^\r\n]*", " ", doc.content)  # drop markdown headings (they run to the line end)
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = " ".join(sentence.split())
            words = token_set(sentence, remove_stopwords=True)
            if len(sentence) < 15 or not words:
                continue
            overlap = len(terms & words)
            # prefer more query terms; earlier (higher-ranked) sources break ties
            score = overlap + 0.2 * overlap / len(words) - 0.01 * index
            if overlap and score > best[2]:
                best = (sentence, index, score)
    return best[0], best[1]


def ask(question):
    """Answer ``question`` against the last built index and return the result as JSON."""

    if _rag is None:
        return json.dumps({"error": "Build the index first."})
    started = time.perf_counter()
    result = _rag.query(question)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    sentence, cited = _extractive_answer(question, result.sources)
    return json.dumps(
        {
            "answer": f"{sentence} [{cited}]" if sentence else "No sentence in the retrieved sources matches the question.",
            "ms": elapsed,
            "trace": list(result.reasoning_trace),
            "citations": [cited] if cited else [],
            "sources": [
                {"id": doc.id, "score": round(float(doc.score), 4), "content": doc.content}
                for doc in result.sources
            ],
        }
    )


def graph_html():
    """Interactive HTML view of the knowledge graph (rag_type='graph' only)."""

    if _rag is None or _rag.stats().get("rag_type") != "graph":
        return ""
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "graph.html")
    _rag.visualize_graph(path)
    with open(path, encoding="utf-8") as handle:
        return handle.read()
