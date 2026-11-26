"""Custom exceptions used by raglib document loaders."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DocumentLoadError(RuntimeError):
    """A file could not be loaded: over a limit, not what its extension says, or unparsable.

    The message names the file (not its full path) and the reason; the parser's own error, if
    any, is chained as ``__cause__``.
    """


class InstructionError(RuntimeError):
    """Represents a user-actionable loader error with remediation guidance."""
