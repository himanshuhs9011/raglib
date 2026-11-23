"""Model choice for model-based guards: local model -> configured API -> skipped (with a tip).

Order, per guard (resolved once, when ``RAG`` binds the guard):

1. **local** - only when you configured one: ``local_model=`` on the guard, or
   ``RAG(guard_models={"injection": ..., "toxicity": ..., "nli": ...})``. A local model is either a
   callable (your own scorer, e.g. an ONNX session) or a Hugging Face model id. A model id is loaded
   with ``local_files_only=True``: raglib never downloads a model unless you pass
   ``allow_download=True``. Missing ``transformers`` or a model that is not in the local cache falls
   through to the next step with a one-time tip saying how to enable it.
2. **api** - your already-configured models: the provider's moderation endpoint where one exists
   (OpenAI moderation for toxicity - free), otherwise the cheapest configured chat model
   (``cheap_llm`` / ``graph_llm`` / the main model; with ``prices=`` the lowest input price wins).
   ``api=False`` on a guard turns this step off.
3. **skipped** - the guard allows everything and records ``mode="skipped"``; a one-time
   ``GuardSkippedWarning`` says what was skipped and which local model or setting enables it.

The chosen mode is on every decision (``result.metadata["guards"][i]["mode"]``) and on the guard's
trace stage meta.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# task -> (recommended local model, transformers pipeline task)
RECOMMENDED_LOCAL_MODELS: Dict[str, Tuple[str, str]] = {
    "injection": ("protectai/deberta-v3-base-prompt-injection-v2", "text-classification"),
    "toxicity": ("unitary/toxic-bert", "text-classification"),
    "nli": ("cross-encoder/nli-deberta-v3-small", "text-classification"),
}
OPENAI_MODERATION_URL = "https://api.openai.com/v1/moderations"
OPENAI_MODERATION_MODEL = "omni-moderation-latest"


class LocalModelUnavailable(Exception):
    """A configured local model cannot be used (not installed / not downloaded); ``tip`` says why."""

    def __init__(self, tip: str):
        super().__init__(tip)
        self.tip = tip


def load_local_model(spec: Any, task: str, allow_download: bool = False) -> Callable[..., Any]:
    """A callable for ``spec``: returned as is when callable, else a local transformers pipeline.

    Never downloads unless ``allow_download=True`` (``local_files_only`` otherwise).
    """

    if callable(spec):
        return spec
    if not isinstance(spec, str) or not spec.strip():
        raise TypeError(f"local_model must be a callable or a Hugging Face model id, got {type(spec).__name__}")
    try:
        import transformers  # type: ignore[import-not-found]
    except ImportError:
        raise LocalModelUnavailable(
            f"local model '{spec}' needs transformers: pip install raglib-py[guards]"
        ) from None
    pipeline_task = RECOMMENDED_LOCAL_MODELS.get(task, ("", "text-classification"))[1]
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(spec, local_files_only=not allow_download)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(spec, local_files_only=not allow_download)
    except OSError:
        raise LocalModelUnavailable(
            f"local model '{spec}' is not downloaded. Download it once with `huggingface-cli download {spec}` "
            "(raglib never downloads models on its own), or pass allow_download=True"
        ) from None
    return transformers.pipeline(pipeline_task, model=model, tokenizer=tokenizer, top_k=None)


def label_score(output: Any, positive: Sequence[str]) -> float:
    """Probability of the positive label(s) from a scorer's output.

    Accepts a float, ``{"label": ..., "score": ...}``, or a (nested) list of those - the shapes
    transformers pipelines return (``top_k=None`` gives every label).
    """

    if isinstance(output, (int, float)):
        return float(output)
    while isinstance(output, list) and len(output) == 1 and isinstance(output[0], list):
        output = output[0]
    items = output if isinstance(output, list) else [output]
    wanted = {label.lower() for label in positive}
    best, seen = 0.0, False
    for item in items:
        if not isinstance(item, dict):
            continue
        label, score = str(item.get("label", "")).lower(), float(item.get("score", 0.0))
        if label in wanted:
            best, seen = max(best, score), True
    if not seen and len(items) == 1 and isinstance(items[0], dict):
        # A single "SAFE" / "neutral" label: its score is the confidence in the negative class.
        return 1.0 - float(items[0].get("score", 0.0))
    return best


def cheapest_llm(rag: Any) -> Tuple[Optional[Any], Optional[str]]:
    """The cheapest configured chat model: lowest input price with ``prices=``, else cheap_llm >
    graph_llm > the main model. Returns ``(client, "provider/model" or None)``."""

    from raglib.llm.base_client import unwrap_llm

    candidates: List[Any] = [client for client in (
        getattr(rag, "_cheap_llm", None), getattr(rag, "_graph_llm", None), getattr(rag, "_llm", None),
    ) if client is not None]
    if not candidates:
        return None, None

    def name(client: Any) -> Optional[str]:
        raw = unwrap_llm(client)
        provider, model = getattr(raw, "provider", None), getattr(raw, "model_name", None)
        return f"{provider}/{model}" if provider and model else None

    prices = getattr(rag, "_prices", None)
    if prices is not None:
        priced = [(prices.llm_price(name(c) or ""), c) for c in candidates]
        priced = [(p.input_per_1m, c) for p, c in priced if p is not None]
        if priced:
            client = min(priced, key=lambda item: item[0])[1]
            return client, name(client)
    return candidates[0], name(candidates[0])


class ModerationClient:
    """OpenAI's moderation endpoint (free). ``transport(url, payload, headers, timeout)`` is replaceable in tests."""

    def __init__(self, api_key: str, base_url: str = OPENAI_MODERATION_URL, model: str = OPENAI_MODERATION_MODEL,
                 timeout: float = 10.0, transport: Optional[Callable[..., Any]] = None):
        self.api_key, self.url, self.model, self.timeout = api_key, base_url, model, timeout
        if transport is None:
            from raglib.llm.http import post_json

            transport = post_json
        self.transport = transport

    def score(self, text: str) -> float:
        """Highest category score (0..1) for ``text``."""

        body = self.transport(
            self.url, {"model": self.model, "input": text},
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, timeout=self.timeout,
        )
        result = (body or {}).get("results", [{}])[0]
        scores = result.get("category_scores") or {}
        top = max((float(v) for v in scores.values()), default=0.0)
        return max(top, 1.0 if result.get("flagged") else 0.0) if not scores else top


def moderation_client(rag: Any) -> Optional[ModerationClient]:
    """OpenAI moderation when the chat provider is OpenAI with a key (no other provider has one)."""

    from raglib.llm.base_client import unwrap_llm

    raw = unwrap_llm(getattr(rag, "_llm_raw", None))
    if getattr(rag, "_chat_provider", None) != "openai" or raw is None:
        return None
    key = getattr(raw, "api_key", None) or getattr(raw, "_api_key", None)
    return ModerationClient(api_key=key) if key else None


def ask_json(client: Any, prompt: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """``complete_json`` with a dict fallback (an unparseable reply counts as the default)."""

    data = client.complete_json(prompt, default=default)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return default
    return data if isinstance(data, dict) else default
