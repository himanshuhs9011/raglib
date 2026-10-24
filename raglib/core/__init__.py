"""Core exports for raglib orchestration primitives.

``BaseRAG`` is imported lazily so low-level helpers (``raglib.core.text`` etc.)
can be used by components without creating an import cycle.
"""

from typing import Any

__all__ = ["BaseRAG", "PipelineContext"]


def __getattr__(name: str) -> Any:
    """Lazily resolve orchestration classes on first access."""

    if name in __all__:
        from raglib.core import base

        return getattr(base, name)
    raise AttributeError(f"module 'raglib.core' has no attribute {name!r}")
