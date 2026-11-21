"""OCR through any native chat client whose model accepts images (custom providers included)."""

from __future__ import annotations

import base64
import logging

from raglib.vision.base_vision import OCR_PROMPT, BaseVisionClient, image_mime_type

logger = logging.getLogger(__name__)


class ChatVisionClient(BaseVisionClient):
    """Extract text from page images with an ``HTTPChatClient`` (OpenAI, Anthropic or Gemini protocol)."""

    def __init__(self, client) -> None:
        """Wrap a client from ``raglib.llm.build_chat_client``."""

        self._client = client
        self._model_name = getattr(client, "model_name", None)

    def extract_text_from_image(self, image_path: str) -> str:
        """Send the image with an OCR instruction and return the text."""

        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return self._client.complete_with_image(OCR_PROMPT, encoded, image_mime_type(image_path)).strip()
