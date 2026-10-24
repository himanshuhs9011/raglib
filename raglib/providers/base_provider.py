"""Base interfaces and shared errors for web search providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Represents an operational provider error."""


class ProviderNotConfiguredError(ProviderError, ValueError):
    """Raised when a provider is missing required credentials."""


class BaseSearchProvider(ABC):
    """Abstract contract for web search providers."""

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Search the web for a query and return documents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider identifier."""


class ProviderChain(BaseSearchProvider):
    """Tries providers in order and falls back when one fails."""

    CHAIN_NAME: str = "provider_chain"

    def __init__(self, providers: List[BaseSearchProvider]):
        """Initialize a fallback chain of providers."""

        self.providers = providers

    @property
    def name(self) -> str:
        """Return the chain identifier."""

        return self.CHAIN_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute provider search with ordered fallback on failure."""

        errors: List[str] = []
        for provider in self.providers:
            try:
                logger.info("ProviderChain trying provider=%s", provider.name)
                docs = provider.search(query=query, num_results=num_results)
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
