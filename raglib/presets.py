"""Named configuration presets for ``RAG`` (``RAG(..., preset="production")``).

A preset only fills in arguments you did not pass yourself: ``RAG(preset="production", top_k=5)``
uses every production value except ``top_k``. Print ``describe_preset("production")`` to see one.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import logging
import os
from typing import Any, Callable, Dict

from raglib.constants import OFFLINE_ENV_VAR

logger = logging.getLogger(__name__)

DOCS_URL = "https://github.com/himanshuhs9011/raglib#readme"
PRESETS_DOCS_URL = "https://github.com/himanshuhs9011/raglib#presets--production-setup"

PRESETS: Dict[str, Dict[str, Any]] = {
    # Production defaults from the "Production RAG" checklist: 500-token chunks with 20% overlap,
    # hybrid retrieval of 10 candidates, a local cross-encoder keeps the best 3.
    "production": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "reranker": "cross_encoder",
        "rerank_candidates": 10,
        "top_k": 3,
        # Repeated questions reuse their query embedding for a day (the corpus vectors live in the store).
        "embedding_cache_ttl": 86400.0,
        # Chunk sizes in real model tokens (tiktoken when installed), and no silent chunking fallback.
        "token_counter": "auto",
        "on_embedding_failure": "raise",
        # Tier 0 guardrails (free rules): PII masking, question length, injection keywords; a
        # "couldn't find this" exit without an LLM call; leak + citation-access checks on answers.
        "input_guards": "default",
        "retrieval_guards": "default",
        "output_guards": "default",
        "tool_policy": "default",
        # Caps what each answer can cost: the model stops generating here (MaxOutputTokens only
        # trims what is shown). 1024 tokens is ~750 words, plenty for a cited answer.
        "llm_max_tokens": 4096,  # room for a reasoning model to think and still answer
    },
}

PRESET_DESCRIPTIONS: Dict[str, str] = {
    "production": (
        "500-token chunks (100 overlap), hybrid search for 10 candidates, cross-encoder rerank to the best 3, "
        "embeddings cached for 24 hours, tier 0 guardrails on every stage, answers capped at 1024 tokens"
    ),
}


def resolve_preset(name: str, offline: bool = False) -> Dict[str, Any]:
    """Return the settings of preset ``name`` (``ValueError`` for an unknown name).

    In offline mode (mock models, no network) the cross-encoder, which would download a model,
    is replaced by the mock LLM reranker so the pipeline keeps its rerank step.
    """

    key = (name or "").strip().lower()
    if key not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Available: {sorted(PRESETS)}")
    values = dict(PRESETS[key])
    if offline and values.get("reranker") == "cross_encoder":
        values["reranker"] = "llm"
    return values


def describe_preset(name: str) -> str:
    """One readable line per setting of a preset."""

    values = resolve_preset(name)
    lines = [f"preset={name!r}: {PRESET_DESCRIPTIONS.get(name, '')}"]
    lines += [f"  {key}={value!r}" for key, value in values.items()]
    return "\n".join(lines)


def apply_presets(init: Callable[..., None]) -> Callable[..., None]:
    """Decorate ``RAG.__init__`` so ``preset=`` fills in only the arguments the caller left out."""

    signature = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        preset = kwargs.get("preset")
        if preset:
            bound = signature.bind_partial(self, *args, **kwargs)
            offline = kwargs.get("offline")
            if offline is None:
                offline = os.getenv(OFFLINE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}
            filled = []
            for key, value in resolve_preset(preset, offline=bool(offline)).items():
                if key not in bound.arguments:
                    kwargs[key] = value
                    filled.append(key)
            self._preset_filled = frozenset(filled)  # e.g. a preset chunk_size may be fitted to the model
            if kwargs.get("reranker") == "cross_encoder" and importlib.util.find_spec("sentence_transformers") is None:
                logger.warning(
                    "preset=%r uses reranker='cross_encoder', but sentence-transformers is not installed; "
                    "answers use the hybrid-search order until you `pip install raglib-py[rerank]` "
                    "(or pass reranker='cohere' / 'llm' / 'none')",
                    preset,
                )
        init(self, *args, **kwargs)

    return wrapper

