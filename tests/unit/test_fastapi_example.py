"""The FastAPI example in examples/fastapi_app.py works end to end (offline)."""

import importlib.util
import os
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "fastapi_app.py")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGLIB_DOCS", str(tmp_path))
    monkeypatch.setenv("RAGLIB_OFFLINE", "1")
    spec = importlib.util.spec_from_file_location("fastapi_app_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fastapi_app_example"] = module
    spec.loader.exec_module(module)
    with TestClient(module.app) as test_client:
        yield test_client


def test_upload_ask_stream_delete(client):
    assert client.get("/health").json()["chunks"] == 0

    text = b"Refunds are accepted within 30 days of purchase. Refunds go back to the original card."
    upload = client.post("/documents", files={"file": ("refunds.md", text, "text/markdown")})
    assert upload.status_code == 200 and upload.json()["chunks_added"] >= 1

    answer = client.post("/ask", json={"question": "What is the refund window?", "session_id": "u1"})
    body = answer.json()
    assert answer.status_code == 200
    assert body["sources"] and body["sources"][0]["source"] == "refunds.md"  # file name, not a local path
    assert body["metadata"]["pipeline"] == "advanced"

    streamed = client.post("/ask/stream", json={"question": "Summarize the refund policy"})
    assert streamed.status_code == 200 and streamed.text

    assert client.delete("/sessions/u1").json()["status"] == "cleared"
    deleted = client.delete("/documents/refunds.md")
    assert deleted.status_code == 200 and deleted.json()["chunks_removed"] >= 1
    assert client.get("/health").json()["chunks"] == 0


def test_rejects_unsafe_uploads(client):
    traversal = client.post("/documents", files={"file": ("../evil.md", b"some text here", "text/markdown")})
    assert traversal.status_code == 200 and traversal.json()["file"] == "evil.md"  # stays inside RAGLIB_DOCS
    assert client.post("/documents", files={"file": ("run.exe", b"x", "application/octet-stream")}).status_code == 400
    assert client.delete("/documents/missing.md").status_code == 404
    assert client.post("/ask", json={"question": ""}).status_code == 422
