"""GraphRAG summaries inherit the strictest access tags of their source chunks (offline, no network)."""

from __future__ import annotations

from typing import List

from raglib import RAG, AccessControl, UserContext, access_tags
from raglib.access import NO_DEPARTMENT, NO_ROLE
from raglib.guards import CitationAccessGuard, GuardContext
from raglib.schemas import Document

PUBLIC = [
    "The Orion Program trains new pilots at the Northfield Base every spring.",
    "The Orion Program publishes its flight schedule for the Northfield Base online.",
    "Visitors can tour the Northfield Base while the Orion Program runs open days.",
]
CONFIDENTIAL = "The Orion Program budget for the Northfield Base is 42 million, approved by the Finance Board."
PUBLIC_USER = UserContext(user_id="visitor")
CLEARED_USER = UserContext(user_id="cfo", roles=["executive"], departments=["finance"], max_access_level="confidential")


def _graph_rag() -> RAG:
    rag = RAG(chat_llm="mock", embedding_llm="mock", rag_type="graph", access_control=True, token_counter="units",
              llm_cache=False, graph_kwargs={"min_llm_report_size": 99})
    for text in PUBLIC:
        rag.add(text, access=access_tags("public"))
    rag.add(CONFIDENTIAL, access=access_tags("confidential", department="finance", allowed_roles=["executive"]))
    rag.rebuild_graph_communities()
    return rag


def _summaries(rag: RAG, user: UserContext, mode: str = "local") -> List[Document]:
    docs = rag._graph_retriever.retrieve("What does the Orion Program do at the Northfield Base?", mode=mode,
                                         filters=rag.permission_filter(user))
    return [doc for doc in docs if doc.metadata.get("graph") != "chunk"]


def test_strictest_tags_highest_level_and_intersection() -> None:
    access = AccessControl()
    assert access.strictest([{"access_level": "public"}, {"access_level": "confidential"}]) == {"access_level": "confidential"}
    tags = access.strictest([
        {"access_level": "internal", "department": ["finance", "legal"], "allowed_roles": ["cfo", "auditor"]},
        {"access_level": "public", "department": "finance"},
        {"access_level": "public"},  # unrestricted sources do not widen or narrow the intersection
    ])
    assert tags == {"access_level": "internal", "department": ["finance"], "allowed_roles": ["auditor", "cfo"]}
    disjoint = access.strictest([{"access_level": "public", "department": "hr"}, {"access_level": "public", "department": "it"}])
    assert disjoint["department"] == [NO_DEPARTMENT]
    assert access.strictest([{"access_level": "public", "allowed_roles": "a"}, {"access_level": "public", "allowed_roles": "b"}])["allowed_roles"] == [NO_ROLE]
    assert access.strictest([{"access_level": "public"}, {}]) == {}  # an untagged source: deny by default
    assert access.strictest([]) == {}


def test_community_reports_record_their_source_chunks_at_build_time() -> None:
    rag = _graph_rag()
    store = rag._graph_indexer.store
    mixed = [c for c in store.communities() if len(c.source_chunk_ids) == 4]
    assert mixed, "the Orion Program community draws on all four chunks"
    confidential_id = next(doc.id for doc in rag._documents if "42 million" in doc.content)
    assert confidential_id in mixed[0].source_chunk_ids


def test_summary_from_three_public_and_one_confidential_chunk_is_invisible_to_public_users() -> None:
    rag = _graph_rag()
    public = _summaries(rag, PUBLIC_USER)
    assert not [doc for doc in public if doc.metadata["graph"] == "community"]  # the mixed report is hidden
    assert all("42 million" not in doc.content for doc in public)
    assert all(doc.metadata.get("access_level") == "public" for doc in public)  # built from visible chunks only
    cleared = _summaries(rag, CLEARED_USER)
    report = next(doc for doc in cleared if doc.metadata["graph"] == "community")
    assert report.metadata["access_level"] == "confidential"
    assert report.metadata["department"] == ["finance"] and report.metadata["allowed_roles"] == ["executive"]
    assert len(report.metadata["source_chunk_ids"]) == 4
    points = [doc for doc in _summaries(rag, CLEARED_USER, mode="global") if doc.metadata["graph"] == "global_point"]
    assert points and all(doc.metadata["access_level"] == "confidential" for doc in points)
    assert not _summaries(rag, PUBLIC_USER, mode="global") or all(
        doc.metadata["access_level"] == "public" for doc in _summaries(rag, PUBLIC_USER, mode="global")
    )


def test_citation_access_guard_checks_summaries_by_their_sources_and_tags() -> None:
    rag = _graph_rag()
    report = next(doc for doc in _summaries(rag, CLEARED_USER) if doc.metadata["graph"] == "community")
    guard = next(g for g in rag._guards.output if g.name == "citation_access")
    slipped = GuardContext(stage="output", documents=[report], permission_filter=rag.permission_filter(PUBLIC_USER))
    assert guard.check("The budget is 42 million [1].", slipped).action == "block"
    allowed = GuardContext(stage="output", documents=[report], permission_filter=rag.permission_filter(CLEARED_USER))
    assert guard.check("The budget is 42 million [1].", allowed).action == "allow"
    untagged = Document(id="graph-x", content="summary", metadata={"graph": "community", "supplementary": True})
    strict = CitationAccessGuard(summary_rule="strict_tags")
    strict.bind(rag)
    assert strict.check("The budget is 42 million [1].", slipped).action == "block"
    assert strict.check("The budget is 42 million [1].", allowed).action == "allow"
    lone = CitationAccessGuard()
    lone.bind(rag)
    denied = GuardContext(stage="output", documents=[untagged], permission_filter=rag.permission_filter(CLEARED_USER))
    assert lone.check("Fact [1].", denied).action == "block"  # no inherited tags: denied, not skipped


def test_cleared_users_get_graph_answers_and_public_users_never_see_the_budget() -> None:
    rag = _graph_rag()
    cleared = rag.query("What is the Orion Program budget at the Northfield Base?", user=CLEARED_USER)
    assert "blocked_by" not in cleared.metadata
    public = rag.query("What is the Orion Program budget at the Northfield Base?", user=PUBLIC_USER)
    assert "42 million" not in public.answer and all("42 million" not in d.content for d in public.sources)
