"""Loader for webpage sources addressed by URL."""

from __future__ import annotations

from raglib.loaders.web_loader import WebLoader, _VisibleTextParser


class WebPageLoader(WebLoader):
    """Backwards-compatible alias for WebLoader."""


__all__ = ["WebPageLoader", "_VisibleTextParser"]
