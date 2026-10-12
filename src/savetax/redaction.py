"""Token redaction for safe logging."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"savetax_[A-Za-z0-9_-]+")


def redact(text: str) -> str:
    """Return text with token-shaped substrings replaced by <redacted>."""
    return _TOKEN_PATTERN.sub("<redacted>", text)
