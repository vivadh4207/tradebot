"""Tests for src.core.data_paths — the SD-card data-root indirection."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core import data_paths
from src.core.data_paths import data_path, data_root, is_sd_card_root, repo_root


def test_repo_root_is_absolute():
    assert repo_root().is_absolute()


def test_data_root_defaults_to_repo_root_when_env_unset(monkeypatch):
    monkeypatch.delenv("TRADEBOT_DATA_ROOT", raising=False)
    assert data_root() == repo_root()


def test_data_root_honors_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADEBOT_DATA_ROOT", str(tmp_path))
    assert data_root() == tmp_path.resolve()


def test_data_root_expands_user_home(monkeypatch):
    """~/… should expand to $HOME."""
    monkeypatch.setenv("TRADEBOT_DATA_ROOT", "~/some_dir")
    expected = Path(os.path.expanduser("~/some_dir"))
    assert data_root() == expected


def test_data_root_relative_path_resolves_under_repo(monkeypatch):
    """A relative value is resolved against the repo root, not CWD."""
    monkeypatch.setenv("TRADEBOT_DATA_ROOT", "some_relative_dir")
    resolved = data_root()
    assert resolved == (repo_root() / "some_relative_dir").resolve()


def test_data_path_with_relative_input_joins_to_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADEBOT_DATA_ROOT", str(tmp_path))
    got = data_path("logs/tradebot.sqlite")
    assert got == (tmp_path / "logs" / "tradebot.sqlite").resolve()


def test_data_path_passes_absolute_paths_through(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADEBOT_DATA_ROOT", str(tmp_path))
    abs_in = "/tmp/some/explicit.sqlite"
    got = data_path(abs_in)
    assert got == Path(abs_in)


def test_is_sd_card_root_false_on_default(monkeypatch):
    monkeypatch.delenv("TRADEBOT_DATA_ROOT", raising=False)
    assert is_sd_card_root() is False


def test_is_sd_card_root_true_for_media_mount(monkeypatch):
    monkeypatch.setenv("TRADEBOT_DATA_ROOT", "/media/orin/tradebot-data")
    assert is_sd_card_root() is True


def test_is_sd_card_root_true_for_mnt_path(monkeypatch):
    monkeypatch.setenv("TRADEBOT_DATA_ROOT", "/mnt/sdcard/tradebot-data")
    assert is_sd_card_root() is True


def test_describe_returns_structured_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADEBOT_DATA_ROOT", str(tmp_path))
    d = data_paths.describe()
    assert d["env_override"] is True
    assert d["data_root"] == str(tmp_path)
    assert d["repo_root"] == str(repo_root())
