"""Multi-query RAG orchestration with query expansion."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import Dict, Optional

from raglib.components.deduplicator import Deduplicator
from raglib.components.generator import Generator
from raglib.components.multi_query_retriever import MultiQueryRetriever
from raglib.components.query_expander import QueryExpander
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)


class MultiQueryRAG(BaseRAG):
    """Runs expanded-query retrieval and merged-context generation."""

    def __init__(
        self,
        query_expander: Optional[QueryExpander] = None,
        retriever: Optional[Retriever] = None,
        deduplicator: Optional[Deduplicator] = None,
        generator: Optional[Generator] = None,
        multi_query_retriever: Optional[MultiQueryRetriever] = None,
        **kwargs,
    ):
        """Initialize MultiQueryRAG with expansion and retrieval components."""

        super().__init__(
            query_expander=query_expander,
            retriever=retriever,
            deduplicator=deduplicator,
            generator=generator,
            multi_query_retriever=multi_query_retriever,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Expand query, retrieve per variant, merge with frequency boost, and generate."""

        if self.generator is None or (
            self.multi_query_retriever is None and (self.query_expander is None or self.retriever is None)
        ):
            raise ValueError("MultiQueryRAG requires query_expander, retriever, and generator")

        active_query = self.pre_retrieve(query)

        if self.multi_query_retriever is not None:
            documents = self.multi_query_retriever.retrieve(active_query)
            trace = ["multi_query_retriever"]
        else:
            assert self.query_expander is not None and self.retriever is not None
            variants = self.query_expander.expand(active_query)
            trace = [f"variants:{len(variants)}"]

            merged: Dict[str, Document] = {}
            counts: Dict[str, int] = {}
            for variant in variants:
                for doc in self.retriever.retrieve(variant):
                    digest = hashlib.sha256(doc.content.lower().strip().encode("utf-8")).hexdigest()[:16]
                    key = doc.id or f"doc-{digest}"
                    if key in merged:
                        previous = merged[key]
                        merged[key] = replace(previous, score=max(previous.score, doc.score))
                        counts[key] += 1
                    else:
                        merged[key] = doc
                        counts[key] = 1

            blended = [
                replace(doc, score=doc.score * (1.0 + 0.1 * (counts[key] - 1)))
                for key, doc in merged.items()
            ]
            documents = sorted(blended, key=lambda d: d.score, reverse=True)
            if self.deduplicator is not None:
                documents = self.deduplicator.deduplicate(documents)

        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(query=active_query, documents=documents, reasoning_trace=trace)
        return self.post_generate(result)
