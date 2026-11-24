"""Base interfaces and shared errors for web search providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Represents an operational provider error."""


class ProviderNotConfiguredError(ProviderError, ValueError):
    """Raised when a provider is missing required credentials."""


class BaseSearchProvider(ABC):
    """Abstract contract for web search providers.

    Metadata filters (and access control) restrict *your corpus*; results from the public
    internet are not corpus documents, so external providers ignore them. A provider that
    searches the local corpus must set ``searches_corpus = True`` and, to serve filtered
    queries, ``supports_filters = True`` with ``search(query, num_results, filters=None)``;
    otherwise a filtered query raises ``FilterNotSupportedError``.
    """

    searches_corpus: bool = False
    supports_filters: bool = False
    # An internal search (intranet, a private wiki). Its results are not public: at retrieval they
    # must pass the query's filters, including the access-control permission filter, judged by the
    # tags the provider puts in each result's metadata (access_level, department, allowed_roles).
    # A result without tags is hidden from users without access (deny by default). Public web
    # results are tagged with the lowest access level instead.
    internal: bool = False

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Search the web for a query and return documents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider identifier."""


# Set by ProviderChain on each result: the index of the provider in the chain that returned it.
CHAIN_INDEX_KEY = "_raglib_chain_index"


class ProviderChain(BaseSearchProvider):
    """Tries providers in order and falls back when one fails.

    Each result records which provider returned it (``CHAIN_INDEX_KEY``), so access control
    judges it by that provider: a public provider's results stay public, an internal one's stay
    restricted.
    """

    CHAIN_NAME: str = "provider_chain"

    def __init__(self, providers: List[BaseSearchProvider]):
        """Initialize a fallback chain of providers."""

        self.providers = providers

    @property
    def searches_corpus(self) -> bool:  # type: ignore[override]
        """True when any provider in the chain searches the local corpus."""

        return any(getattr(provider, "searches_corpus", False) for provider in self.providers)

    @property
    def internal(self) -> bool:  # type: ignore[override]
        """True when any provider in the chain is internal (each result is judged by its own provider)."""

        return any(getattr(provider, "internal", False) for provider in self.providers)

    @property
    def name(self) -> str:
        """Return the chain identifier."""

        return self.CHAIN_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute provider search with ordered fallback on failure."""

        errors: List[str] = []
        for index, provider in enumerate(self.providers):
            try:
                logger.info("ProviderChain trying provider=%s", provider.name)
                docs = [
                    replace(doc, metadata={**doc.metadata, CHAIN_INDEX_KEY: index})
                    for doc in provider.search(query=query, num_results=num_results)
                ]
                if docs:
                    logger.info(
                        "ProviderChain succeeded with provider=%s docs=%d",
                        provider.name,
                        len(docs),
                    )
                    return docs
                logger.warning("Provider %s returned no results", provider.name)
            except Exception as exc:  # noqa: BLE001
                err = f"{provider.name}: {exc}"
                errors.append(err)
                logger.warning("Provider fallback triggered: %s", err)

        error_message = "All providers failed in ProviderChain. Errors: " + " | ".join(errors)
        raise ProviderError(error_message)
