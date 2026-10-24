"""One-line helpers for the simplest possible RAG usage."""

from __future__ import annotations

import logging
from typing import Any, List, Sequence, Union

from raglib.facade import RAG
from raglib.loaders.document_loader import SourceInput
from raglib.schemas import GenerationResult

logger = logging.getLogger(__name__)


def ask(
    source: SourceInput,
    question: Union[str, Sequence[str]],
    llm: Any = None,
    **kwargs: Any,
) -> Union[GenerationResult, List[GenerationResult]]:
    """Index ``source`` and answer one question (or a list of questions) in a single call.

    >>> from raglib import ask
    >>> print(ask("docs/", "What is the refund policy?", llm="openrouter/openai/gpt-4.1-mini").answer)

    With ``llm`` omitted, the provider is picked from your API key environment variables
    (``OPENAI_API_KEY``, ``OPENROUTER_API_KEY``, ...). For repeated questions over the same
    documents, create a ``RAG`` once instead, so the index is built only once.
    """

    rag = RAG(source, llm=llm, **kwargs)
    if isinstance(question, str):
        return rag.query(question)
    return [rag.query(item) for item in question]
