"""Abstract embedding interface for raglib retrievers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseEmbedding(ABC):
    """Abstract interface all embedding implementations must satisfy."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document strings."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of vectors produced by this embedding model."""

    @property
    def model_name(self) -> Optional[str]:
        """Return the model identifier when known (used to detect a changed model on load)."""

        return getattr(self, "_model_name", None)

    def fingerprint(self) -> Dict[str, Any]:
        """Identify the model that produced the vectors (class + model name)."""

        return {"embedding_class": type(self).__name__, "embedding_model": self.model_name}
