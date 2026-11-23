"""preset="production", RAG.production(...) and the ``raglib init`` scaffold (offline, no network)."""

from __future__ import annotations

import inspect
import logging

import pytest

from raglib import RAG
from raglib.cli import main
from raglib.presets import PRESETS, describe_preset, resolve_preset
from raglib.scaffold import render_env, render_pipeline

TEXT = "Retrieval grounds every answer in passages that were retrieved from the corpus for the question."


def test_production_preset_sets_the_documented_values() -> None:
    rag = RAG(TEXT, preset="production", embedding_llm="mock", llm="mock", reranker="llm")
    assert rag._splitter._chunk_size == 500 and rag._splitter._overlap == 100
    assert rag._top_k == 3 and rag._reranker.top_k == 3
    assert rag.pipeline.candidate_k() == 10  # retrieve 10, rerank to 3
    assert rag.stats()["preset"] == "production"


def test_explicit_arguments_override_the_preset() -> None:
    rag = RAG(TEXT, preset="production", offline=True, chunk_size=400, top_k=5, reranker="none")
    assert rag._splitter._chunk_size == 400 and rag._splitter._overlap == 100
    assert rag._top_k == 5 and rag._reranker.backend == "none"
    assert rag.pipeline.candidate_k() is None  # rerank_candidates is ignored without a reranker


def test_production_shortcut_matches_the_preset() -> None:
    rag = RAG.production(TEXT, offline=True)
    assert rag.stats()["preset"] == "production" and rag._top_k == 3
    assert RAG.production(TEXT, offline=True, top_k=4)._top_k == 4


def test_offline_preset_never_loads_a_cross_encoder() -> None:
    assert resolve_preset("production", offline=True)["reranker"] == "llm"
    assert resolve_preset("production")["reranker"] == "cross_encoder"
    assert RAG(TEXT, preset="production", offline=True)._reranker.backend == "llm"


def test_missing_cross_encoder_dependency_warns(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "sentence_transformers" else real(name, *a))
    with caplog.at_level(logging.WARNING, logger="raglib.presets"):
        rag = RAG(TEXT, preset="production", llm="mock", embedding_llm="mock", offline=False)
    assert rag._reranker.backend == "cross_encoder"
    assert "raglib-py[rerank]" in caplog.text
    assert rag.query("What grounds every answer?").answer  # degrades to the hybrid order


def test_without_a_preset_nothing_changes() -> None:
    rag = RAG(TEXT, offline=True)
    assert rag._splitter._chunk_size == 400 and rag._top_k == 5 and rag._reranker.backend == "none"
    assert rag.stats()["preset"] is None
    assert "preset" in inspect.signature(RAG.__init__).parameters  # signature is kept for IDEs/docs


def test_unknown_preset_and_bad_rerank_candidates_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        RAG(offline=True, preset="turbo")
    with pytest.raises(ValueError, match="rerank_candidates"):
        RAG(offline=True, rerank_candidates=0)


def test_describe_preset_lists_every_setting() -> None:
    text = describe_preset("production")
    assert all(key in text for key in PRESETS["production"])


def test_rendered_pipeline_is_valid_python_and_links_the_docs() -> None:
    source = render_pipeline(preset="production", llm="openrouter/google/gemini-2.5-flash", docs="kb/")
    compile(source, "rag_pipeline.py", "exec")
    assert 'source="kb/"' in source and 'llm="openrouter/google/gemini-2.5-flash"' in source
    assert 'chat_api_key=os.getenv("OPENROUTER_API_KEY")' in source
    assert 'embedding_api_key=os.getenv("OPENAI_API_KEY")' in source  # OpenRouter has no embeddings API
    assert "To customize more" in source and "github.com/himanshuhs9011/raglib" in source
    env = render_env("openrouter")
    assert "OPENROUTER_API_KEY=" in env and "OPENAI_API_KEY=" in env
    assert "needs no API key" in render_env("ollama")
    assert render_env("openai").count("OPENAI_API_KEY=") == 1  # one key for chat and embeddings


def test_rendered_pipeline_writes_out_every_rag_option_with_production_values() -> None:
    import re

    from raglib.scaffold import TEMPLATE_SKIPPED_OPTIONS

    source = render_pipeline(preset="production", llm="openai")
    written = set(re.findall(r"(?m)^    ([a-z_]+)=", source))
    options = set(inspect.signature(RAG.__init__).parameters) - TEMPLATE_SKIPPED_OPTIONS - {"llm"}
    assert options - written == set(), "add new RAG options to the raglib init template"
    assert "preset=" not in source.split("rag = RAG(")[1]  # values are spelled out, not hidden in a preset
    for key, value in PRESETS["production"].items():
        assert re.search(rf"(?m)^    {key}={value!r}," .replace("'", '"'), source), key
    assert all(len(line) - len(line.lstrip()) == 4 for line in re.findall(r"(?m)^    [a-z_]+=.*$", source))


def test_generated_pipeline_runs(tmp_path) -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.txt").write_text(
        "The Orion launch window opens in March and closes in April for the whole launch team.", encoding="utf-8"
    )
    source = render_pipeline(llm="mock").replace('reranker="cross_encoder"', 'reranker="llm"')  # no model download
    (tmp_path / "rag_pipeline.py").write_text(source, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    run = subprocess.run(
        [sys.executable, "rag_pipeline.py", "When does the Orion launch window open?", "--latency"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert "March" in run.stdout and "p95 ms" in run.stdout


def test_raglib_init_writes_files_and_refuses_to_overwrite(tmp_path, capsys: pytest.CaptureFixture) -> None:
    target = tmp_path / "app" / "pipeline.py"
    assert main(["init", str(target), "--llm", "anthropic", "--docs", "data/"]) == 0
    out = capsys.readouterr().out
    assert target.exists() and (target.parent / ".env.example").exists()
    assert "ANTHROPIC_API_KEY=" in (target.parent / ".env.example").read_text(encoding="utf-8")
    assert "To customize more, read the docs" in out
    assert main(["init", str(target)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert main(["init", str(target), "--force"]) == 0


def test_production_flag_shortcut(tmp_path, capsys: pytest.CaptureFixture) -> None:
    target = tmp_path / "my_rag.py"
    assert main(["--production", str(target), "--llm", "groq", "--docs", "kb/"]) == 0
    text = target.read_text(encoding="utf-8")
    assert 'llm="groq"' in text and "chunk_size=500," in text and 'source="kb/"' in text
    assert "lists every RAG option" in capsys.readouterr().out
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    assert 'raglib-py = "raglib.cli:main"' in pyproject.read_text(encoding="utf-8")  # the raglib-py command


def test_raglib_init_into_a_folder(tmp_path) -> None:
    assert main(["init", str(tmp_path)]) == 0
    assert (tmp_path / "rag_pipeline.py").exists()


def test_raglib_presets_command(capsys: pytest.CaptureFixture) -> None:
    assert main(["presets"]) == 0
    assert "chunk_size=500" in capsys.readouterr().out


def test_cli_ask_accepts_a_preset(tmp_path, capsys: pytest.CaptureFixture) -> None:
    doc = tmp_path / "notes.txt"
    doc.write_text("The Orion launch window opens in March and closes in April for the whole team.", encoding="utf-8")
    assert main(["ask", str(doc), "When does the Orion launch window open?", "--offline", "--preset", "production"]) == 0
    assert "March" in capsys.readouterr().out
