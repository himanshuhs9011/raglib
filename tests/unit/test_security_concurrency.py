"""Security guards (SSRF, path sandbox, log hygiene) and concurrent-use safety."""

from __future__ import annotations

import logging
import threading

import pytest

from raglib import RAG
from raglib.loaders.document_loader import DocumentLoader
from raglib.loaders.web_loader import UnsafeURLError, WebLoader


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata (credential theft)
        "http://127.0.0.1:6379/",
        "http://localhost:8080/admin",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://[::1]/",
        "file:///etc/passwd",
    ],
)
def test_ssrf_guard_blocks_internal_urls(url: str) -> None:
    """User-supplied URLs can't reach internal services."""

    with pytest.raises(UnsafeURLError):
        WebLoader().load(url)


def test_allowed_dirs_sandbox(tmp_path) -> None:
    """With allowed_dirs, files outside the sandbox (incl. '..' tricks) are refused."""

    inside = tmp_path / "kb"
    inside.mkdir()
    (inside / "ok.txt").write_text("allowed content", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret", encoding="utf-8")

    loader = DocumentLoader(allowed_dirs=[str(inside)])
    assert loader.load(str(inside / "ok.txt"))
    with pytest.raises(PermissionError):
        loader.load(str(outside))
    with pytest.raises(PermissionError):
        loader.load(str(inside / ".." / "secret.txt"))
    rag = RAG(allowed_dirs=[str(inside)])
    with pytest.raises(PermissionError):
        rag.add(str(outside))


def test_user_queries_are_not_logged_at_info(caplog) -> None:
    """Questions may contain personal data: INFO+ logs must not contain them."""

    secret_question = "my email is alice@example.com, what is the refund policy?"
    rag = RAG(["Refunds are allowed within 30 days."], rag_type="corrective")
    with caplog.at_level(logging.INFO, logger="raglib"):
        rag.query(secret_question)
        RAG(["Refunds are allowed within 30 days."], rag_type="multi_hop").query(secret_question)
    assert "alice@example.com" not in caplog.text


def test_large_folder_ingestion_streams_reports_progress_and_resumes(tmp_path) -> None:
    """Batches are indexed as they go: a crash mid-way keeps earlier files; re-running resumes."""

    from raglib.embedding.mock_embedding import MockEmbedding

    for index in range(12):
        (tmp_path / f"doc{index:02d}.txt").write_text(f"Document {index} talks about topic{index}.", encoding="utf-8")

    class FlakyEmbedding(MockEmbedding):
        calls = 0
        fail = True

        def embed_documents(self, texts):
            FlakyEmbedding.calls += 1
            if self.fail and FlakyEmbedding.calls == 3:
                raise RuntimeError("API outage")
            return super().embed_documents(texts)

    embedding = FlakyEmbedding()
    rag = RAG(embedding_llm=embedding)
    progress = []
    with pytest.raises(RuntimeError, match="API outage"):
        rag.add(str(tmp_path), batch_size=4, on_progress=progress.append)
    assert len(rag.retriever.documents) == 8  # two batches survived the outage
    assert progress[-1]["chunks_indexed"] == 8

    embedding.fail = False
    added = rag.add(str(tmp_path), batch_size=4)
    assert added == 4 and len(rag.retriever.documents) == 12


def test_concurrent_sessions_while_ingesting() -> None:
    """Many chat sessions + concurrent add/delete on one instance: no crashes, no history leaks."""

    rag = RAG([f"Item{i} costs {i} dollars." for i in range(150)], rag_type="memory", chunk_size=40, chunk_overlap=5)
    errors = []
    stop = threading.Event()

    def chat(user: int) -> None:
        for turn in range(8):
            try:
                rag.query(f"user{user} asks about Item{turn}", session_id=f"s{user}")
                history = rag.pipeline.conversation_store.get(f"s{user}").get_context()
                assert all(f"user{other} " not in history for other in range(10) if other != user)
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

    def writer() -> None:
        index = 0
        while not stop.is_set():
            try:
                text = f"Extra{index} costs {index} rupees."
                rag.add(text)
                if index % 2:
                    rag.delete(text)
                index += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    threads = [threading.Thread(target=chat, args=(user,)) for user in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stop.set()
    writer_thread.join()
    assert errors == []


def test_graph_visualization_is_self_contained_and_escapes_content(tmp_path) -> None:
    """The HTML viewer works offline and document text cannot inject scripts."""

    from raglib import GraphContextWindow
    from raglib.schemas import Document

    evil = 'Alice Smith said </script><script>alert("x")</script> about Acme Corp.'
    rag = RAG([evil, "Acme Corp hired Alice Smith in Berlin."], rag_type="graph", chunk_size=40, chunk_overlap=5)
    path = rag.visualize_graph(str(tmp_path / "graph.html"))
    html = open(path, encoding="utf-8").read()
    assert html.count("<script>") == 1 and "alert(" not in html.split("const DATA")[0]
    assert '</script><script>alert' not in html
    assert "http://" not in html and "https://" not in html  # no CDN / network
    assert "acme corp" in html.lower()

    gcw = GraphContextWindow(chunking_strategy="recursive", chunk_size=20, chunk_overlap=2)
    gcw.add_documents([Document(id="d", content="# Intro\nAcme Corp builds rockets.\n# Team\nAlice Smith leads Acme Corp.")])
    assert '"type": "section"' in gcw.to_html(str(tmp_path / "ctx.html"))


def test_delete_leaves_no_copies_in_caches(tmp_path) -> None:
    """Right-to-erasure: after delete, the text is gone from indexes, caches and the saved graph state."""

    secret = "Patient Ravi Kumar was diagnosed with Diabetes at Apollo Hospital."
    rag = RAG([secret, "Acme Corp sells rockets in Berlin."], rag_type="graph", chunk_size=40, chunk_overlap=5)
    rag.query("What was Ravi Kumar diagnosed with?")
    rag.delete(secret)
    rag.save(str(tmp_path / "idx"))
    saved = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "idx").rglob("*.json"))
    assert "Ravi" not in saved and "Diabetes" not in saved
    assert not rag._llm._cache and not rag._embedding._cache
    assert rag.graph.store.get_entity("ravi kumar") is None
