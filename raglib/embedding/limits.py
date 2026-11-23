"""Maximum input length of embedding models, so chunks are never silently truncated.

Most local models (sentence-transformers, Ollama) silently cut text longer than their limit - the
end of a chunk is then never embedded and can never be found. OpenAI returns an error instead,
which fails ingestion. raglib checks chunks against the limit at ingest and splits (or raises).

Limits are in the model's own tokens. ``known_limit`` matches by model-name substring; pass
``RAG(embedding_max_tokens=...)`` for models not listed (or to be more conservative).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

# Most specific patterns first. Sources: model cards / provider docs (2026).
KNOWN_EMBEDDING_LIMITS: Sequence[Tuple[str, int]] = (
    ("bge-m3", 8192),
    ("bge-", 512),  # bge-small/base/large(-en)(-v1.5)
    ("all-minilm", 256),  # all-MiniLM-L6/L12 (sentence-transformers and Ollama)
    ("all-mpnet", 384),
    ("multilingual-e5", 512),
    ("e5-", 512),
    ("gte-", 512),
    ("mxbai-embed-large", 512),
    ("snowflake-arctic-embed", 512),
    ("nomic-embed-text", 2048),  # Ollama's default context window (the model supports 8192 with num_ctx raised)
    ("jina-embeddings-v", 8192),
    ("text-embedding-3", 8191),
    ("text-embedding-ada-002", 8191),
    ("gemini-embedding", 2048),
    ("text-embedding-004", 2048),
    ("text-embedding-005", 2048),
    ("embedding-001", 2048),
    ("mistral-embed", 8192),
)

# Estimated counts (no tokenizer for the model) are checked against this share of the limit.
ESTIMATE_MARGIN = 0.9


class EmbeddingLimitError(ValueError):
    """A chunk is longer than the embedding model can read (``oversized_chunks="raise"``)."""


class EmbeddingLimitWarning(UserWarning):
    """``chunk_size`` does not fit the embedding model, or chunks had to be split further."""


def known_limit(model_name: Optional[str]) -> Optional[int]:
    """The documented input limit of ``model_name`` (``None`` when unknown)."""

    name = (model_name or "").lower()
    for pattern, limit in KNOWN_EMBEDDING_LIMITS:
        if pattern in name:
            return limit
    return None


# Representative English prose (with numbers and names) to compare two tokenizers' counts.
TOKENIZER_SAMPLE = (
    "Quarterly revenue grew 34% to $2.1M, driven by enterprise contracts signed in Berlin, Austin and "
    "Bengaluru. Operating costs fell 23% after the infrastructure migration, while the support team "
    "resolved 1,284 tickets with a median first-response time of 3.5 hours. The board approved the "
    "2026 hiring plan: twelve engineers, four account managers and two compliance specialists, "
    "subject to the usual background checks and the renewal of the office lease on Hamilton Street."
)


_UNKNOWN_WARNED: set = set()


def warn_unknown_limit(model: str) -> bool:
    """Warn once per process per model that its input limit is unknown (never raises).

    Returns True when the warning was emitted now.
    """

    import logging
    import warnings

    if model in _UNKNOWN_WARNED:
        return False
    _UNKNOWN_WARNED.add(model)
    message = (
        f"The input limit of embedding model '{model}' is unknown, so raglib cannot check that chunks fit: "
        "a provider that truncates long input would do so silently. Set RAG(embedding_max_tokens=<limit>) "
        "(e.g. 512 for bge / e5 models, 8192 for OpenAI text-embedding-3-*) to have oversized chunks split "
        "or rejected (oversized_chunks=)."
    )
    logging.getLogger(__name__).warning(message)
    warnings.warn(message, EmbeddingLimitWarning, stacklevel=4)
    return True
