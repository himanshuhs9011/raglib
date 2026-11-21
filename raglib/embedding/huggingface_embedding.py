"""Free local embeddings powered by sentence-transformers."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.constants import PROVIDER_DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class HuggingFaceEmbedding(BaseEmbedding):
    """Run a sentence-transformers model locally (CPU or GPU)."""

    def __init__(self, model_name: Optional[str] = None, normalize: bool = False, batch_size: int = 32):
        """Load the model. ``normalize=True`` returns unit-length vectors."""

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Free local embeddings require sentence-transformers. Install with: pip install raglib-py[hf]"
            ) from exc

        resolved_model = model_name or PROVIDER_DEFAULT_EMBEDDING_MODEL["huggingface"]
        self._model = SentenceTransformer(resolved_model)
        self.normalize = normalize
        self.batch_size = batch_size
        getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        self._dimension = int(getter() or 0) if callable(getter) else 0
        self._model_name = resolved_model
        logger.info("Using HuggingFace embedding model='%s'", resolved_model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents."""

        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result = [[float(value) for value in row] for row in vectors.tolist()]
        if result:
            self._dimension = len(result[0])
        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""

        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""

        return self._dimension
