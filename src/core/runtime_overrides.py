"""Runtime knob overrides — Discord buttons write here, the bot reads.

File-backed JSON at `data/runtime_overrides.json`. Filter chain and order
pricing check this first, fall through to settings.yaml. Means operators
can bump max_0dte_per_day, tighten entry_spread_pct, etc. without a
bot restart.

Write path: `set_override('max_0dte_per_day', 20)`.
Read path: `get_override('max_0dte_per_day', default=5)`.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import RLock
from typing import Any, Optional


_log = logging.getLogger(__name__)
_lock = RLock()


def _store() -> Path:
    try:
        from .data_paths import data_path
        return Path(data_path("runtime_overrides.json"))
    except Exception:
        return Path(os.getenv("TRADEBOT_DATA_DIR", "data")) / "runtime_overrides.json"


def _load() -> dict:
    p = _store()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text() or "{}")
    except Exception as e:                                  # noqa: BLE001
        _log.info("runtime_overrides_load_err err=%s", e)
        return {}


def _save(d: dict) -> None:
    p = _store()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True))
    tmp.replace(p)


def get_override(key: str, default: Any = None) -> Any:
    """Return override for `key`, or `default` if none. Thread-safe."""
    with _lock:
        d = _load()
        return d.get(key, default)


def set_override(key: str, value: Any) -> Any:
    """Set override and persist. Pass value=None to clear."""
    with _lock:
        d = _load()
        if value is None:
            d.pop(key, None)
        else:
            d[key] = value
        _save(d)
        return value


def all_overrides() -> dict:
    with _lock:
        return _load()
