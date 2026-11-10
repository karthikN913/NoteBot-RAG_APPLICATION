"""Secure logging configuration for NoteBot.
Ensures API keys and credentials are never emitted to logs or standard output.
"""

import logging
import re
import sys
from typing import Optional

KEY_PATTERNS = [
    re.compile(r'(AIza[0-9A-Za-z-_]{35})'),
    re.compile(r'(sk-[a-zA-Z0-9_-]{20,})'),
    re.compile(r'(gsk_[a-zA-Z0-9_-]{20,})'),
]


class RedactingFormatter(logging.Formatter):
    """Filters out sensitive API keys from log messages."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for pattern in KEY_PATTERNS:
            msg = pattern.sub(r'***REDACTED_API_KEY***', msg)
        return msg


def setup_logger(name: str = "notebot", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a thread-safe, secure logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = RedactingFormatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

