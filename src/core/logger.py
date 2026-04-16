"""Thin wrapper over structlog for uniform, JSON-friendly logs."""
from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:  # fallback so the package imports even without structlog
    _HAS_STRUCTLOG = False


def configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    )
    if _HAS_STRUCTLOG:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(numeric),
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.dev.ConsoleRenderer(),
            ],
        )


def get_logger(name: str) -> Any:
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)
