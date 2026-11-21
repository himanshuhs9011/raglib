"""Anthropic vision (OCR) client over the native Messages client."""

from __future__ import annotations

import base64
import logging
from typing import Optional

from raglib.llm.constants import PROVIDER_DEFAULT_VISION_MODEL
from raglib.vision.base_vision import OCR_PROMPT, BaseVisionClient, image_mime_type

logger = logging.getLogger(__name__)


class AnthropicVisionClient(BaseVisionClient):
    """Use Anthropic vision models to extract text from page images."""

    def __init__(self, api_key: Optional[str], model_name: Optional[str] = None, base_url: Optional[str] = None):
        """Build the native chat client used for OCR."""

        from raglib.llm.clients import AnthropicClient

        resolved_model = model_name or PROVIDER_DEFAULT_VISION_MODEL["anthropic"]
        self._client = AnthropicClient(
            model=resolved_model,
            api_key=api_key,
            base_url=base_url,
            provider_name="anthropic",
            temperature=0.0,
            max_tokens=4096,
        )
        self._model_name = resolved_model
        logger.info("Using Anthropic vision model='%s'", resolved_model)

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract plain text from an image by prompting a vision-capable model."""

        logger.debug("AnthropicVisionClient extracting text from image_path=%s", image_path)
        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return self._client.complete_with_image(OCR_PROMPT, encoded, image_mime_type(image_path)).strip()
