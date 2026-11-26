"""2.4 / 3.1: file size, file type, zip-bomb and page limits; clean loader errors; question limit."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from raglib import RAG
from raglib.llm.base_client import BaseLLMClient
from raglib.loaders.document_loader import DocumentLoader
from raglib.loaders.exceptions import DocumentLoadError


def test_a_file_over_max_file_bytes_is_refused_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "big.txt"
    path.write_text("word " * 5000, encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="over max_file_bytes") as info:
        DocumentLoader(max_file_bytes=1000).load(str(path))
    assert str(tmp_path) not in str(info.value)  # names the file, not its full path
    assert DocumentLoader(max_file_bytes=None).load(str(path))  # None turns the limit off


def test_a_file_that_is_not_what_its_extension_says_is_refused(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "report.pdf"
    fake_pdf.write_text("<html>not a pdf</html>", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="not a PDF"):
        DocumentLoader().load(str(fake_pdf))
    fake_docx = tmp_path / "memo.docx"
    fake_docx.write_bytes(b"MZ\x90\x00 executable, not a document")
    with pytest.raises(DocumentLoadError, match="not a DOCX"):
        DocumentLoader().load(str(fake_docx))


def test_a_zip_bomb_is_refused_before_unpacking(tmp_path: Path) -> None:
    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"\0" * (5 * 1024 * 1024))  # 5 MB that packs to a few KB
    assert bomb.stat().st_size < 50_000
    with pytest.raises(DocumentLoadError, match="zip bomb"):
        DocumentLoader(max_file_bytes=200_000).load(str(bomb))  # may unpack to 2 MB


def test_a_corrupt_zip_document_raises_a_clean_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.docx"
    broken.write_bytes(b"PK\x03\x04" + b"\x00garbage" * 50)
    with pytest.raises(DocumentLoadError):
        DocumentLoader().load(str(broken))


def test_a_pdf_over_max_pages_is_refused(tmp_path: Path) -> None:
    fitz = pytest.importorskip("pymupdf")
    path = tmp_path / "three.pdf"
    doc = fitz.open()
    for number in range(3):
        doc.new_page().insert_text((72, 72), f"Page {number + 1} text.")
    doc.save(str(path))
    with pytest.raises(DocumentLoadError, match="3 pages, over max_pages=2"):
        DocumentLoader(max_pages=2).load(str(path))
    assert len(DocumentLoader(max_pages=3).load(str(path))) == 3


def test_rag_passes_the_limits_and_normal_files_still_load(tmp_path: Path) -> None:
    small = tmp_path / "notes.txt"
    small.write_text("The office closes at 5pm on Fridays.", encoding="utf-8")
    rag = RAG(chat_llm="mock", embedding_llm="mock", token_counter="units", max_file_bytes=100)
    assert rag.add(str(small)) > 0
    big = tmp_path / "big.txt"
    big.write_text("x " * 200, encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        rag.add(str(big))
    with pytest.raises(ValueError):
        RAG(chat_llm="mock", embedding_llm="mock", max_pages=0)


class _Counting(BaseLLMClient):
    calls = 0

    @property
    def provider(self) -> str:
        return "test"

    def complete(self, prompt: str, system: str = "") -> str:
        _Counting.calls += 1
        return "Fridays at 5pm [1]."

    def chat(self, messages) -> str:
        return self.complete("")


def test_plain_rag_refuses_an_absurdly_long_question_without_a_model_call() -> None:
    rag = RAG(["The office closes at 5pm on Fridays."], chat_llm=_Counting(), embedding_llm="mock", token_counter="units")
    _Counting.calls = 0
    result = rag.query("When does the office close? " + "padding " * 20000)
    assert result.metadata.get("blocked_by") == "max_question_tokens" and _Counting.calls == 0
    normal = rag.query("When does the office close on Fridays?")
    assert "blocked_by" not in normal.metadata and "guards" not in normal.metadata  # quiet for normal questions
    assert "blocked_by" not in RAG(["x"], chat_llm=_Counting(), embedding_llm="mock", secure_defaults={"question_limit": False}).query(
        "q " + "padding " * 20000).metadata


def test_raw_text_is_not_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    secret = "Salary of Jane Doe is 98k"
    with caplog.at_level("INFO", logger="raglib.loaders.document_loader"):
        DocumentLoader().load(secret)
    info = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert info and all(secret not in message for message in info)
    assert any("25-character source" in message for message in info)
