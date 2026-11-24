"""Security fix 2.2.1: the routing pipeline applies filters on every route (no network)."""

from __future__ import annotations

from typing import List

import pytest

from raglib import RAG
from raglib.schemas import Document

PUBLIC = Document(id="pub", content="The Orion launch window opens in March for the whole team.", metadata={"tenant": "acme"})
PRIVATE = Document(id="priv", content="The Orion launch budget is 42 million, board only.", metadata={"tenant": "globex"})


def _rag(route: str) -> RAG:
    rag = RAG([PUBLIC, PRIVATE], rag_type="routing", chat_llm="mock", embedding_llm="mock", web_search_provider="local")
    rag.pipeline.router_retriever.route = lambda query: route  # force the route
    return rag


@pytest.mark.parametrize("route", ["vector", "hybrid", "web"])
def test_every_route_applies_the_callers_filters(route: str) -> None:
    rag = _rag(route)
    result = rag.query("What about the Orion launch window and budget?", filters={"tenant": "acme"})
    texts: List[str] = [doc.content for doc in result.sources]
    assert all("42 million" not in text for text in texts), route
    assert "42 million" not in result.answer


def test_without_filters_the_web_route_is_unchanged() -> None:
    result = _rag("web").query("What about the Orion launch window and budget?")
    assert any("42 million" in doc.content for doc in result.sources)


@pytest.mark.parametrize("route", ["vector", "hybrid", "web"])
def test_every_route_applies_access_control(route: str) -> None:
    from raglib import UserContext, access_tags

    rag = RAG(rag_type="routing", chat_llm="mock", embedding_llm="mock", web_search_provider="local",
              access_control=True, token_counter="units")
    rag.add("The Orion launch window opens in March for the whole team.", access=access_tags("public"))
    rag.add("The Orion launch budget is 42 million, board only.", access=access_tags("executive"))
    rag.add("Untagged note: the Orion launch password is hunter2.")  # untagged: never visible
    rag.pipeline.router_retriever.route = lambda query: route
    result = rag.query("What about the Orion launch window, budget and password?", user=UserContext(user_id="visitor"))
    texts = " ".join(doc.content for doc in result.sources) + result.answer
    assert "42 million" not in texts and "hunter2" not in texts, route
