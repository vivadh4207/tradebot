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


class _StdlibKwargAdapter(logging.LoggerAdapter):
    """Accepts arbitrary kwargs (structlog style) and folds them into the
    log message so calls like `log.info("event", reason=x, symbol=y)` work
    whether or not structlog is installed.
    """

    def process(self, msg, kwargs):
        stdlib = {}
        for k in ("exc_info", "stack_info", "stacklevel", "extra"):
            if k in kwargs:
                stdlib[k] = kwargs.pop(k)
        if kwargs:
            payload = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
            msg = f"{msg} {payload}"
        return msg, stdlib


def get_logger(name: str) -> Any:
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return _StdlibKwargAdapter(logging.getLogger(name), {})
