"""Abstract interface for vision-capable LLM clients."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

OCR_PROMPT = "Extract all readable text from this image. Return only the extracted text with no additional commentary."

_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}


def image_mime_type(path: str) -> str:
    """MIME type from the file extension (PNG when unknown, which is what the PDF loader renders)."""

    return _MIME_TYPES.get(os.path.splitext(path)[1].lower(), "image/png")


class BaseVisionClient(ABC):
    """Send an image to a vision-capable model and return extracted text."""

    @abstractmethod
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract readable text from an image file path."""
