"""Access control, leak and filter-semantics tests against every vector store raglib ships
(in-memory, Chroma, Qdrant in-process; live pgvector and Qdrant, plain and strict mode).

``test_store_filters_match_raglib_semantics`` is the shared filter test: the same filters must
select the same documents in every store, list-valued fields included.

The store is queried directly as well as through raglib, because the retriever re-checks dense
hits against the filter: an end-to-end test alone would pass even if a store returned too much.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

import pytest

from raglib import RAG, UserContext, access_tags
from raglib.access import merge_filters
from raglib.schemas import Document
from raglib.vectorstores.base_store import matches_filter

pytestmark = pytest.mark.integration

DOCS = {
    "public": ("Orion handbook: the Northfield base opens at eight.", access_tags("public")),
    "internal": ("Orion staff rota: Northfield shifts start at six.", access_tags("internal")),
    "finance": ("Orion budget at Northfield: 42 million, finance only.",
                access_tags("confidential", department="finance", allowed_roles=["cfo"])),
    "hr": ("Orion salaries at Northfield: reviewed by HR in May.", access_tags("confidential", department=["hr", "legal"])),
    "untagged": ("Orion untagged note about Northfield.", None),
}
ANONYMOUS = None
STAFF = UserContext(user_id="staff", max_access_level="internal")
CFO = UserContext(user_id="cfo", roles=["cfo"], departments=["finance"], max_access_level="confidential")
FINANCE_CLERK = UserContext(user_id="clerk", departments=["finance"], max_access_level="confidential")
LAWYER = UserContext(user_id="lawyer", departments=["legal"], max_access_level="executive")
EXPECTED = {
    "anonymous": (ANONYMOUS, {"public"}),
    "staff": (STAFF, {"public", "internal"}),
    "cfo": (CFO, {"public", "internal", "finance"}),
    "finance_clerk": (FINANCE_CLERK, {"public", "internal"}),  # right department, missing the cfo role
    "lawyer": (LAWYER, {"public", "internal", "hr"}),  # one of the two departments is enough
}
QUESTION = "What happens at the Orion Northfield base?"


def _rag(make_store: Callable[[int], Any], **kwargs: Any) -> RAG:
    probe = RAG(chat_llm="mock", embedding_llm="mock", token_counter="units")
    store = make_store(probe._embedding.dimension)
    rag = RAG(chat_llm="mock", embedding_llm="mock", vector_db=store, access_control=True, token_counter="units",
              llm_cache=False, **kwargs)
    for doc_id, (text, tags) in DOCS.items():
        rag.add(Document(id=doc_id, content=text, metadata={"tenant": "acme" if doc_id != "hr" else "globex"}), access=tags)
    rag.store_under_test = store  # type: ignore[attr-defined]
    return rag


def _store_hits(rag: RAG, filters: Optional[Dict[str, Any]]) -> Set[str]:
    """Parent document ids the *store itself* returns for ``filters`` (no raglib re-check)."""

    store = rag.store_under_test  # type: ignore[attr-defined]
    vector = rag._embedding.embed_query(QUESTION)
    return {str(hit.metadata.get("parent_id")) for hit in store.query(vector, top_k=50, filters=filters)}


def _retrieved(rag: RAG, user: Optional[UserContext]) -> Set[str]:
    docs = rag.retriever.retrieve(QUESTION, top_k=50, filters=rag.permission_filter(user), mode="dense")
    return {str(doc.metadata.get("parent_id")) for doc in docs}


# ------------------------------------------------------------------------------ access control


@pytest.mark.parametrize("who", sorted(EXPECTED))
def test_the_store_applies_the_permission_filter(make_store: Callable[[int], Any], who: str) -> None:
    rag = _rag(make_store)
    user, visible = EXPECTED[who]
    assert _store_hits(rag, rag.permission_filter(user)) == visible
    assert _retrieved(rag, user) == visible


def test_query_answers_only_from_documents_the_user_may_read(make_store: Callable[[int], Any]) -> None:
    rag = _rag(make_store)
    result = rag.query(QUESTION, user=FINANCE_CLERK)
    assert result.sources
    assert {doc.metadata.get("parent_id") for doc in result.sources} <= {"public", "internal"}
    assert all("42 million" not in doc.content for doc in result.sources)


def test_caller_filters_narrow_the_permission_filter(make_store: Callable[[int], Any]) -> None:
    rag = _rag(make_store)
    merged = merge_filters(rag.permission_filter(LAWYER), {"tenant": "acme"})
    assert _store_hits(rag, merged) == {"public", "internal"}  # "hr" is globex
    merged = merge_filters(rag.permission_filter(ANONYMOUS), {"tenant": "globex"})
    assert _store_hits(rag, merged) == set()


# ------------------------------------------------------------------------------ leaks


def test_retagging_a_document_updates_the_store(make_store: Callable[[int], Any]) -> None:
    rag = _rag(make_store)
    text, _ = DOCS["public"]
    rag.add(Document(id="public", content=text, metadata={"tenant": "acme"}),
            access=access_tags("confidential", department="finance", allowed_roles=["cfo"]))
    assert "public" not in _store_hits(rag, rag.permission_filter(ANONYMOUS))
    assert "public" in _store_hits(rag, rag.permission_filter(CFO))


def test_a_deleted_document_is_gone_from_the_store(make_store: Callable[[int], Any]) -> None:
    rag = _rag(make_store)
    assert rag.delete("finance") > 0
    assert "finance" not in _store_hits(rag, rag.permission_filter(CFO))
    assert "finance" not in _store_hits(rag, None)


def test_untagged_documents_match_no_permission_filter(make_store: Callable[[int], Any]) -> None:
    rag = _rag(make_store)
    everyone = UserContext(user_id="root", roles=["cfo"], departments=["finance", "hr", "legal"], max_access_level="executive")
    hits = _store_hits(rag, rag.permission_filter(everyone))
    assert hits == {"public", "internal", "finance", "hr"}
    assert "untagged" in _store_hits(rag, None)  # it is stored, only never permitted


# ------------------------------------------------------------------------------ filter semantics

FILTERS: List[Dict[str, Any]] = [
    {"$or": []},
    {"$and": []},
    {"$and": [{"$or": []}, {"tenant": "acme"}]},
    {"$or": [{"$or": []}, {"tenant": "globex"}]},
    {"tenant": {"$in": []}},
    {"tenant": {"$nin": []}},
    {"tenant": {"$ne": "acme"}},
    {"$or": [{"access_level": "internal"}, {"tenant": "globex"}]},
    {"department": "legal"},  # list-valued metadata: "contains"
    {"department": {"$in": ["finance", "nobody"]}},
    {"department": {"$in": ["legal"]}},
    {"department": {"$nin": ["legal"]}},
    {"$and": [{"department": "hr"}, {"tenant": "globex"}]},
    {"department": {"$ne": "hr"}},
]


@pytest.mark.parametrize("filters", FILTERS, ids=[str(f) for f in FILTERS])
def test_store_filters_match_raglib_semantics(make_store: Callable[[int], Any], filters: Dict[str, Any]) -> None:
    rag = _rag(make_store)
    expected = {
        str(doc.metadata.get("parent_id"))
        for doc in rag.retriever.documents
        if matches_filter({**doc.metadata, "source": doc.source}, filters, array_contains=True)
    }
    assert _store_hits(rag, filters) == expected


def test_empty_or_matches_nothing_and_empty_and_everything(make_store: Callable[[int], Any]) -> None:
    rag = _rag(make_store)
    assert _store_hits(rag, {"$or": []}) == set()
    assert _store_hits(rag, {"$and": []}) == set(DOCS)
