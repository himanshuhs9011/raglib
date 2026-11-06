"""Regression tests for the 2.1 review fixes: chunking, CJK text, loaders, safety, usage, stores, SQL."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import zipfile

import pytest

from raglib import RAG, SQLiteConversationStore, SQLTool
from raglib.components.grader import RelevanceGrader
from raglib.components.memory import ConversationStore, MemoryModule
from raglib.components.retriever import Retriever, _mmr_numpy
from raglib.core.runtime import report_degraded, run_scope, usage_scope
from raglib.core.text import count_tokens, count_units, tokenize, truncate_to_tokens
from raglib.embedding.mock_embedding import MockEmbedding
from raglib.llm.base_client import BaseLLMClient, CachedLLMClient
from raglib.llm.langchain_adapter import LangChainAdapter
from raglib.loaders.chunk_splitter import ChunkSplitter
from raglib.loaders.document_loader import DocumentLoader
from raglib.loaders.html_text import html_to_text
from raglib.loaders.web_loader import UnsafeURLError, WebLoader
from raglib.schemas import Document
from raglib.vectorstores.base_store import VectorRecord
from raglib.vectorstores.in_memory_store import InMemoryVectorStore
from raglib.vectorstores.pgvector_store import filter_to_sql

RECORDS = "\n".join(f"Entry {i}: Alice B., age 30, engineer, based in Tokyo" for i in range(5000))
JAPANESE = "東京は日本の首都です。人口は約1400万人です。" * 300
THAI = "ประเทศไทยมีประชากรมาก " * 400


# --------------------------------------------------------------------------- chunking
@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
@pytest.mark.parametrize("text", [RECORDS, JAPANESE, THAI], ids=["records", "japanese", "thai"])
def test_no_chunk_exceeds_chunk_size(strategy: str, text: str) -> None:
    """Text without sentence punctuation or spaces is still split within the budget."""

    chunks = ChunkSplitter(chunk_size=400, overlap=50, strategy=strategy).split([Document("d", text)])
    assert len(chunks) > 10
    assert max(count_units(chunk.content) for chunk in chunks) <= 400


def test_fixed_chunks_keep_line_breaks_and_overlap() -> None:
    chunks = ChunkSplitter(chunk_size=100, overlap=20).split([Document("d", RECORDS)])
    assert "\n" in chunks[0].content
    first_tail = chunks[0].content.splitlines()[-1]
    assert first_tail in chunks[1].content  # whole-line overlap


def test_cjk_sentence_punctuation_is_a_boundary() -> None:
    splitter = ChunkSplitter(chunk_size=30, overlap=0)
    chunks = splitter.split([Document("d", JAPANESE[:200])])
    assert all(chunk.content.endswith("。") for chunk in chunks[:-1])


def test_consecutive_pdf_pages_are_linked() -> None:
    pages = [Document(f"p{i}", f"Page {i} text about topic {i}.", {"file": "a.pdf", "page": i}) for i in (1, 2, 3)]
    chunks = ChunkSplitter(chunk_size=50, overlap=0).split(pages)
    assert chunks[0].metadata["next_id"] == chunks[1].id
    assert chunks[2].metadata["prev_id"] == chunks[1].id


# --------------------------------------------------------------------------- CJK text
def test_tokenize_uses_bigrams_for_cjk() -> None:
    tokens = tokenize("東京は日本の首都です Tokyo 1400万人")
    assert "東京" in tokens and "首都" in tokens and "tokyo" in tokens and "1400" in tokens and "万人" in tokens


def test_bm25_keyword_search_finds_japanese() -> None:
    retriever = Retriever(
        documents=[Document("jp", "大阪は日本の第二の都市です。"), Document("jp2", "東京は日本の首都です。")], top_k=1
    )
    assert retriever.retrieve("首都はどこ")[0].id == "jp2"


def test_truncate_keeps_cjk_text() -> None:
    truncated = truncate_to_tokens(JAPANESE, 20)
    assert truncated and count_tokens(truncated) <= 20 and JAPANESE.startswith(truncated)


def test_token_estimate_counts_cjk_characters(monkeypatch) -> None:
    import raglib.core.text as text_module

    monkeypatch.setattr(text_module, "_tiktoken_encoder", lambda: None)
    assert text_module.count_tokens("東京都") == 3
    assert text_module.count_tokens("abcdefgh") == 2


# --------------------------------------------------------------------------- loaders
@pytest.mark.parametrize("value", ["./my_docs_folder", "data/docs", "notes.md", "C:\\nope\\file", "~/docs"])
def test_missing_paths_raise(value: str) -> None:
    with pytest.raises(FileNotFoundError):
        DocumentLoader().load(value)


def test_plain_text_is_still_raw_text() -> None:
    docs = DocumentLoader().load("Refunds are processed within 14 days.")
    assert docs[0].metadata["loader"] == "raw_text"


def test_folder_skips_vcs_dirs_and_binary_files(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("junk()", encoding="utf-8")
    (tmp_path / "blob").write_bytes(b"\x00\x01binary")
    (tmp_path / "README").write_text("Project readme.", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('hi')", encoding="utf-8")
    contents = sorted(doc.content for doc in DocumentLoader().load(str(tmp_path)))
    assert contents == ["Project readme.", "print('hi')"]


def test_structured_loaders(tmp_path) -> None:
    (tmp_path / "people.csv").write_text("name,city\nAlice,Tokyo\nBob,Paris\n", encoding="utf-8")
    (tmp_path / "rows.jsonl").write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    (tmp_path / "page.html").write_text(
        "<html><head><title>T</title></head><body><nav>MENU</nav><main><h1>Hi</h1><p>Body.</p></main>"
        "<footer>FOOT</footer></body></html>",
        encoding="utf-8",
    )
    book = tmp_path / "book.epub"
    with zipfile.ZipFile(book, "w") as archive:
        archive.writestr("META-INF/container.xml", (
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf"/></rootfiles></container>'
        ))
        archive.writestr("OEBPS/content.opf", (
            '<package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="c1" href="ch1.xhtml"/>'
            '<item id="c2" href="ch2.xhtml"/></manifest><spine><itemref idref="c2"/><itemref idref="c1"/></spine></package>'
        ))
        archive.writestr("OEBPS/ch1.xhtml", "<html><body><p>Chapter one.</p></body></html>")
        archive.writestr("OEBPS/ch2.xhtml", "<html><body><p>Chapter two.</p></body></html>")
    by_loader = {}
    for doc in DocumentLoader().load(str(tmp_path)):
        by_loader.setdefault(doc.metadata["loader"], []).append(doc)
    assert by_loader["csv"][0].content == "name: Alice; city: Tokyo\nname: Bob; city: Paris"
    assert by_loader["json"][0].content == "a: 1\na: 2"
    assert by_loader["html"][0].content == "# Hi\n\nBody." and by_loader["html"][0].metadata["title"] == "T"
    assert [doc.content for doc in by_loader["epub"]] == ["Chapter two.", "Chapter one."]  # spine order


def test_xlsx_loader(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.append(["name", "age"])
    workbook.active.append(["Zed", 40])
    workbook.save(tmp_path / "w.xlsx")
    docs = DocumentLoader().load(str(tmp_path / "w.xlsx"))
    assert docs[0].content == "name: Zed; age: 40"


def test_html_keeps_structure() -> None:
    title, text = html_to_text(
        "<title>X</title><p>One</p><ul><li>a</li><li>b</li></ul><table><tr><td>k</td><td>v</td></tr></table>"
    )
    assert title == "X"
    assert text == "One\n\n- a\n- b\n\nk | v"


# --------------------------------------------------------------------------- SSRF
def test_dns_rebinding_is_blocked_at_connect_time(monkeypatch) -> None:
    """The host resolves to a public IP for the check and to 127.0.0.1 for the connection."""

    answers = iter(["93.184.216.34", "127.0.0.1"])

    def fake_getaddrinfo(host, port, *args, **kwargs):
        address = next(answers, "127.0.0.1")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(UnsafeURLError):
        WebLoader().load("http://rebind.example/")


# --------------------------------------------------------------------------- embeddings / persistence
def test_in_memory_store_rejects_dimension_mismatch() -> None:
    store = InMemoryVectorStore()
    store.upsert([VectorRecord("a", [0.1, 0.2, 0.3], "a", {})])
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.upsert([VectorRecord("b", [0.1, 0.2], "b", {})])
    with pytest.raises(ValueError, match="different embedding model"):
        store.query([0.1, 0.2], top_k=1)


class _OtherModel(MockEmbedding):
    _model_name = "other-model"


def test_load_refuses_a_different_embedding_model(tmp_path) -> None:
    corpus = ["Photosynthesis converts light into chemical energy.", "Mitochondria produce ATP."]
    built = RAG(corpus, embedding_llm=_OtherModel())
    built.save(str(tmp_path / "index"))

    class NewModel(MockEmbedding):
        _model_name = "new-model"

    with pytest.raises(ValueError, match="different embedding model"):
        RAG(embedding_llm=NewModel()).load(str(tmp_path / "index"))
    restored = RAG(embedding_llm=NewModel()).load(str(tmp_path / "index"), allow_model_change=True)
    assert restored.query("What does photosynthesis convert?").sources


def test_save_is_atomic_and_versioned(tmp_path) -> None:
    rag = RAG(["Alpha beta gamma."])
    rag.save(str(tmp_path))
    payload = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    assert payload["version"] == 3 and "index" in payload
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".tmp")]


# --------------------------------------------------------------------------- memory
def test_memory_does_not_keep_documents_by_default() -> None:
    memory = MemoryModule()
    memory.add("q", "a", [Document("d", "x")])
    assert memory.turns[0].documents == []
    assert MemoryModule(store_documents=True).turns == []


def test_memory_context_is_safe_under_concurrent_writes() -> None:
    memory = ConversationStore(max_turns=50).get("s")
    errors = []

    def writer():
        for i in range(2000):
            memory.add(f"q{i}", "a" * 20)

    def reader():
        try:
            for _ in range(2000):
                memory.get_context()
        except RuntimeError as exc:  # "deque mutated during iteration"
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors


def test_sqlite_conversation_store_persists(tmp_path) -> None:
    path = str(tmp_path / "chats.db")
    store = SQLiteConversationStore(path, max_turns=2)
    memory = store.get("alice")
    for i in range(3):
        memory.add(f"question {i}", f"answer {i}")
    store.close()

    reopened = SQLiteConversationStore(path, max_turns=2)
    assert [turn.query for turn in reopened.get("alice").turns] == ["question 1", "question 2"]
    assert reopened.sessions() == ["alice"]
    reopened.clear("alice")
    assert reopened.get("alice").turns == []


def test_rag_uses_custom_conversation_store(tmp_path) -> None:
    store = SQLiteConversationStore(str(tmp_path / "c.db"))
    rag = RAG(["Acme sells rockets."], rag_type="memory", conversation_store=store)
    rag.query("What does Acme sell?", session_id="u1")
    assert len(store.get("u1").turns) == 1


def test_chat_restores_memory_setting(monkeypatch) -> None:
    from raglib.session import interactive

    monkeypatch.setattr(interactive.InteractiveSession, "start", lambda self: None)
    rag = RAG(["Some text."])
    rag.chat()
    assert rag.pipeline.use_memory is False


# --------------------------------------------------------------------------- usage + degradation
def test_result_reports_totals_over_every_llm_call() -> None:
    rag = RAG(["Photosynthesis converts light into chemical energy."], rag_type="corrective")
    usage = rag.query("What does photosynthesis convert?").usage
    assert usage["llm_calls"] >= 2  # grading + generation (+ strip filter)
    assert usage["total_tokens"] == usage["total_prompt_tokens"] + usage["total_completion_tokens"] > 0


class _Message:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage_metadata = usage


class _FakeChatModel:
    """Minimal LangChain-like chat model."""

    def __init__(self, reply="ok", usage=None, stream_failures=0, json_supported=True):
        self.reply, self.usage = reply, usage
        self.stream_failures = stream_failures
        self.json_supported = json_supported
        self.bound_with = None

    def invoke(self, payload):
        return _Message(self.reply, self.usage)

    def stream(self, payload):
        if self.stream_failures:
            self.stream_failures -= 1
            raise RuntimeError("Error code: 429 rate limit")
        for part in ("Hel", "lo"):
            yield _Message(part)

    def bind(self, **kwargs):
        if not self.json_supported:
            raise ValueError("Error code: 400 - response_format is not supported")
        self.bound_with = kwargs
        return self


def test_real_provider_usage_is_used_for_generation() -> None:
    model = _FakeChatModel(reply="Answer [1].", usage={"input_tokens": 1234, "output_tokens": 56})
    rag = RAG(["Doc one."], chat_llm=LangChainAdapter(model, provider_name="openai"), embedding_llm="mock")
    usage = rag.query("Question?").usage
    assert usage["prompt_tokens"] == 1234 and usage["completion_tokens"] == 56
    assert usage["generation_usage_estimated"] is False


def test_stream_retries_before_first_token(monkeypatch) -> None:
    monkeypatch.setattr("raglib.llm.resilience.time.sleep", lambda seconds: None)
    adapter = LangChainAdapter(_FakeChatModel(stream_failures=1), provider_name="openai", max_retries=2)
    assert "".join(adapter.stream("hi")) == "Hello"


def test_json_mode_is_used_and_falls_back() -> None:
    model = _FakeChatModel(reply='{"ranking": [2, 1]}')
    adapter = LangChainAdapter(model, provider_name="openai")
    assert adapter.complete_json("Return JSON") == {"ranking": [2, 1]}
    assert model.bound_with == {"response_format": {"type": "json_object"}}

    unsupported = LangChainAdapter(_FakeChatModel(reply='{"a": 1}', json_supported=False), provider_name="openrouter")
    assert unsupported.complete_json("Return JSON") == {"a": 1}
    assert unsupported._json_mode is False


def test_cached_client_caches_json() -> None:
    model = _FakeChatModel(reply='{"x": 1}')
    cached = CachedLLMClient(LangChainAdapter(model, provider_name="openai"))
    assert cached.complete_json("Return JSON") == {"x": 1}
    model.reply = '{"x": 2}'
    assert cached.complete_json("Return JSON") == {"x": 1}


def test_llm_max_tokens_reaches_the_model() -> None:
    pytest.importorskip("langchain_openai")
    rag = RAG(["Doc."], llm="openai/gpt-4.1-mini", chat_api_key="sk-test", embedding_llm="mock", llm_max_tokens=77)
    assert rag._llm_raw._model.max_tokens == 77


class _BrokenLLM(BaseLLMClient):
    """Fails every call except answer generation."""

    def complete(self, prompt: str, system: str = "") -> str:
        if "<sources>" in prompt:
            return "Answer [1]."
        raise RuntimeError("Error code: 400 bad request")

    def chat(self, messages):
        raise RuntimeError("unused")


def test_degraded_steps_are_reported() -> None:
    rag = RAG(["Photosynthesis converts light."], chat_llm=_BrokenLLM(), embedding_llm="mock", rag_type="corrective")
    result = rag.query("What does photosynthesis convert?")
    components = {item["component"] for item in result.metadata["degraded"]}
    assert "relevance_grader" in components


def test_strict_mode_raises() -> None:
    rag = RAG(["Photosynthesis converts light."], chat_llm=_BrokenLLM(), embedding_llm="mock", rag_type="corrective", strict=True)
    with pytest.raises(RuntimeError, match="bad request"):
        rag.query("What does photosynthesis convert?")


def test_runtime_state_reaches_worker_threads() -> None:
    from raglib.core.concurrency import parallel_map
    from raglib.core.runtime import record_usage

    with run_scope() as state, usage_scope() as meter:
        def work(index):
            record_usage(10, 1)
            try:
                raise ValueError(f"boom {index}")
            except ValueError as exc:
                report_degraded("worker", exc)

        parallel_map(work, range(4), max_workers=4)
    assert meter.llm_calls == 4 and len(state.degraded) == 4


def test_grader_without_scope_still_works() -> None:
    assert RelevanceGrader(llm_client=_BrokenLLM()).grade("photosynthesis", [Document("d", "photosynthesis light")])


# --------------------------------------------------------------------------- retrieval performance paths
def test_numpy_mmr_matches_python_mmr() -> None:
    pytest.importorskip("numpy")
    docs = [Document(f"d{i}", f"topic {i % 3} text number {i}") for i in range(12)]
    retriever = Retriever(documents=docs, embedding=MockEmbedding(), top_k=4, mmr_lambda=0.5)
    vectors = retriever.vector_store.get_embeddings([doc.id for doc in docs])
    query = retriever.embedding.embed_query("topic 1")
    fast = _mmr_numpy(query, [vectors[doc.id] for doc in docs], 4, 0.5)
    import raglib.components.retriever as retriever_module

    original = retriever_module._mmr_numpy
    retriever_module._mmr_numpy = lambda *args: None
    try:
        slow = retriever._mmr("topic 1", docs, 4)
    finally:
        retriever_module._mmr_numpy = original
    assert [docs[i].id for i in fast] == [doc.id for doc in slow]


def test_deleted_documents_never_come_back() -> None:
    docs = [Document(f"d{i}", f"shared words alpha {i}") for i in range(5)]
    retriever = Retriever(documents=docs, embedding=MockEmbedding(), top_k=5)
    retriever.delete_documents(["d1", "d2"])
    ids = {doc.id for doc in retriever.retrieve("shared words alpha", top_k=5)}
    assert ids == {"d0", "d3", "d4"}


# --------------------------------------------------------------------------- vector backends
def test_qdrant_store_end_to_end() -> None:
    pytest.importorskip("qdrant_client")
    docs = [
        Document("a", "Photosynthesis converts light into chemical energy.", {"lang": "en", "year": 2020}),
        Document("b", "Mitochondria are the powerhouse of the cell.", {"lang": "en", "year": 2023}),
        Document("c", "La fotosintesis convierte la luz.", {"lang": "es", "year": 2021}),
    ]
    rag = RAG(docs, embedding_llm="mock", vector_db="qdrant")
    retriever = rag.retriever
    assert {d.metadata["parent_id"] for d in retriever.retrieve("light", top_k=3, mode="dense", filters={"lang": "es"})} == {"c"}
    hits = retriever.retrieve("cell", top_k=3, mode="dense", filters={"year": {"$gte": 2021}, "$or": [{"lang": "en"}, {"lang": "es"}]})
    assert {d.metadata["parent_id"] for d in hits} == {"b", "c"}
    assert rag.delete("a") == 1 and retriever.vector_store.count() == 2
    with pytest.raises(ValueError, match="dimension mismatch"):
        retriever.vector_store.upsert([VectorRecord("z", [0.1, 0.2], "x", {})])


def test_pgvector_filter_translation() -> None:
    sql, params = filter_to_sql({"lang": "en", "year": {"$gte": 2021}, "$or": [{"tag": {"$in": ["a", "b"]}}, {"x": {"$ne": 1}}]})
    assert sql.count("%s") == len(params)
    assert "metadata @> %s::jsonb" in sql and "OR" in sql
    assert json.loads(params[0]) == {"lang": "en"}
    assert filter_to_sql(None) == ("TRUE", [])
    with pytest.raises(ValueError):
        filter_to_sql({"k": {"$regex": "x"}})


@pytest.mark.skipif(not os.getenv("RAGLIB_TEST_PG_DSN"), reason="set RAGLIB_TEST_PG_DSN to run against PostgreSQL")
def test_pgvector_store_end_to_end() -> None:
    from raglib.vectorstores.pgvector_store import PGVectorStore

    store = PGVectorStore(os.environ["RAGLIB_TEST_PG_DSN"], table_name="raglib_test_vectors")
    store.clear()
    try:
        rag = RAG(
            [Document("a", "Photosynthesis converts light.", {"lang": "en"}), Document("b", "La luz.", {"lang": "es"})],
            embedding_llm="mock",
            vector_db=store,
        )
        hits = rag.retriever.retrieve("light", top_k=2, mode="dense", filters={"lang": "es"})
        assert {d.metadata["parent_id"] for d in hits} == {"b"}
        assert store.count() == 2
    finally:
        store.clear()


# --------------------------------------------------------------------------- SQL tool
def _people_tool(tmp_path, llm=None) -> SQLTool:
    rows = ["name,city,job,age"] + [f"P{i},{'Tokyo' if i % 3 == 0 else 'Paris'},{'engineer' if i % 2 == 0 else 'chef'},{20 + i % 40}" for i in range(5000)]
    (tmp_path / "people.csv").write_text("\n".join(rows), encoding="utf-8")
    return SQLTool.from_files(str(tmp_path / "people.csv"), llm_client=llm)


def test_sql_tool_counts_exactly(tmp_path) -> None:
    tool = _people_tool(tmp_path)
    expected = sum(1 for i in range(5000) if i % 3 == 0 and i % 2 == 0)
    result = tool.run_sql("SELECT COUNT(*) FROM people WHERE city = 'Tokyo' AND job = 'engineer'")
    assert result["rows"][0][0] == expected
    assert "INTEGER" in tool.schema() and "5000 rows" in tool.schema()


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM people",
        "INSERT INTO people VALUES ('x','y','z',1)",
        "DROP TABLE people",
        "ATTACH DATABASE 'evil.db' AS evil",
        "PRAGMA writable_schema = 1",
        "SELECT 1; DROP TABLE people",
    ],
)
def test_sql_tool_is_read_only(tmp_path, sql: str) -> None:
    tool = _people_tool(tmp_path)
    with pytest.raises((sqlite3.DatabaseError, ValueError)):
        tool.run_sql(sql)
    assert tool.run_sql("SELECT COUNT(*) FROM people")["rows"][0][0] == 5000


def test_sql_tool_times_out(tmp_path) -> None:
    tool = _people_tool(tmp_path)
    tool.timeout_seconds = 0.05
    with pytest.raises(TimeoutError):
        tool.run_sql("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n")


class _SQLAgentLLM(BaseLLMClient):
    """Agent that calls the SQL tool once, then finishes; writes SQL and the final answer."""

    def complete(self, prompt: str, system: str = "") -> str:
        if "research agent" in prompt:
            if "Step 1:" in prompt:
                return json.dumps({"action": "finish", "input": ""})
            return json.dumps({"action": "query_tables", "input": "How many engineers are based in Tokyo?"})
        if "SQLite SELECT" in prompt:
            return "```sql\nSELECT COUNT(*) AS n FROM people WHERE LOWER(city) = 'tokyo' AND LOWER(job) = 'engineer'\n```"
        if "<sources>" in prompt:
            import re

            match = re.search(r"Result:\nn\n(\d+)", prompt)
            return f"{match.group(1) if match else 'unknown'} [1]"
        return "ok"

    def chat(self, messages):
        return "ok"


def test_agent_answers_aggregate_question_with_sql(tmp_path) -> None:
    tool = _people_tool(tmp_path)
    rag = RAG(["People directory."], chat_llm=_SQLAgentLLM(), embedding_llm="mock", rag_type="agentic", tools=[tool])
    expected = sum(1 for i in range(5000) if i % 3 == 0 and i % 2 == 0)
    assert rag.query("How many engineers are based in Tokyo?").answer.startswith(str(expected))
