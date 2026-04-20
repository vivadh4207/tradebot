"""Centralized path resolver for everything the bot reads and writes.

Single env var `TRADEBOT_DATA_ROOT` relocates the entire on-disk
footprint (SQLite journal, LSTM checkpoints, LLM models, data cache,
log files) to a different directory. Designed for the Jetson plug-and-
play bring-up: set it to the SD-card mount (`/media/orin/tradebot-data`)
and all persistent state lives on the SD card, not the eMMC.

When `TRADEBOT_DATA_ROOT` is unset, every path resolves under the repo
root — identical to the pre-existing behavior, so no config migration
is required.

Usage:
    from src.core.data_paths import data_path
    db = data_path("logs/tradebot.sqlite")        # → $ROOT/logs/... if unset
                                                   # → $TRADEBOT_DATA_ROOT/logs/... if set

This module has no external deps and never crashes — if the resolved
path's parent doesn't exist it creates it on first write (caller uses
the Path).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union


# Repo root = parent of `src/core` = parents[2] of this file.
# Stays correct even if the workspace is moved to the SD card.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Absolute path to the tradebot repo root (where settings.yaml lives)."""
    return _REPO_ROOT


def data_root() -> Path:
    """Resolve the data root from `TRADEBOT_DATA_ROOT` if set, otherwise
    fall back to the repo root. Always returns an absolute Path.

    The SD-card setup script sets `TRADEBOT_DATA_ROOT=/media/$USER/tradebot-data`
    in `.env` so every subsequent `data_path(...)` call lands on the SD.
    """
    env = os.environ.get("TRADEBOT_DATA_ROOT", "").strip()
    if not env:
        return _REPO_ROOT
    p = Path(env).expanduser()
    # Normalize (but don't demand it exist — the caller may be about to
    # create it). Resolve relative paths against the repo.
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def data_path(rel: Union[str, Path]) -> Path:
    """Resolve `rel` under the data root.

    Accepts either a relative path ("logs/tradebot.sqlite") or an
    absolute path (returned unchanged). Returns an absolute Path.
    Parent directories are NOT auto-created here — callers that open
    for write should `p.parent.mkdir(parents=True, exist_ok=True)`.
    """
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p
    return (data_root() / p).resolve()


def is_sd_card_root() -> bool:
    """Heuristic: does the data root look like an SD-card mount?

    Used by `doctor.sh` and the startup log to distinguish "bot is on
    the SD" from "bot is on eMMC". Matches common Jetson mount points.
    """
    r = str(data_root()).lower()
    return any(marker in r for marker in (
        "/media/", "/mnt/", "tradebot-data",
    )) and data_root() != _REPO_ROOT


def describe() -> dict:
    """Ops-friendly summary for logs and the dashboard."""
    return {
        "repo_root": str(repo_root()),
        "data_root": str(data_root()),
        "on_sd": is_sd_card_root(),
        "env_override": bool(os.environ.get("TRADEBOT_DATA_ROOT", "").strip()),
    }
