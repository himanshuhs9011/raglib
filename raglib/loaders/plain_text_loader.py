"""Loader for plain text and markdown sources."""

from __future__ import annotations

from raglib.loaders.text_loader import TextLoader


class PlainTextLoader(TextLoader):
    """Backwards-compatible alias for TextLoader."""


__all__ = ["PlainTextLoader"]
