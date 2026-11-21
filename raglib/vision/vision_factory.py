"""Factory for building vision clients by provider name."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.vision.anthropic_vision import AnthropicVisionClient
from raglib.vision.base_vision import BaseVisionClient
from raglib.vision.google_vision import GoogleVisionClient
from raglib.vision.mock_vision import MockVisionClient
from raglib.vision.openai_vision import OpenAIVisionClient

logger = logging.getLogger(__name__)


class VisionFactory:
    """Build a vision implementation from provider name and credentials."""

    @staticmethod
    def build(
        provider: Optional[str],
        api_key: Optional[str],
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> BaseVisionClient:
        """Return the right vision client for the configured provider."""

        provider_name = (provider or "mock").strip().lower()

        if provider_name == "mock":
            logger.info("VisionFactory selected provider='mock'")
            return MockVisionClient()
        if provider_name == "openai":
            logger.info("VisionFactory selected provider='openai'")
            return OpenAIVisionClient(api_key=api_key, model_name=model_name, base_url=base_url)
        if provider_name == "anthropic":
            logger.info("VisionFactory selected provider='anthropic'")
            return AnthropicVisionClient(api_key=api_key, model_name=model_name, base_url=base_url)
        if provider_name == "google":
            logger.info("VisionFactory selected provider='google'")
            return GoogleVisionClient(api_key=api_key, model_name=model_name, base_url=base_url)

        raise ValueError(f"Unknown vision provider '{provider_name}'.")
