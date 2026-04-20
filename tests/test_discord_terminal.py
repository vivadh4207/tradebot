"""Tests for the Discord-terminal bot.

Focus: the security-critical pieces — parsing, auth, rate limit,
command whitelist, destructive-action gating. The discord.py event
loop itself is NOT exercised in tests (it would require network + the
lib).
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import time
from pathlib import Path
from typing import Tuple

import pytest


@pytest.fixture
def dt_module(tmp_path, monkeypatch):
    """Load scripts.discord_terminal with safe env overrides so it
    doesn't try to connect to Discord or write to real logs."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-token-for-tests")
    monkeypatch.setenv("DISCORD_TERMINAL_CHANNEL_IDS", "111,222")
    monkeypatch.setenv("DISCORD_TERMINAL_AUTHORIZED_USERS", "42,99")
    monkeypatch.setenv("DISCORD_TERMINAL_AUDIT_LOG",
                        str(tmp_path / "audit.jsonl"))

    # Import from scripts/ — not a package, so we use importlib.util
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "discord_terminal_test_module",
        str(Path(__file__).resolve().parents[1] / "scripts" / "discord_terminal.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- parsing ----------


def test_parse_id_set_ignores_blanks_and_bad_values(dt_module):
    s = dt_module._parse_id_set("1, 2,, 3,abc,4")
    assert s == {1, 2, 3, 4}


def test_parse_id_set_empty_string_returns_empty_set(dt_module):
    assert dt_module._parse_id_set("") == set()
    assert dt_module._parse_id_set(None) == set()


def test_multi_channel_env_is_parsed(dt_module):
    """Both plural DISCORD_TERMINAL_CHANNEL_IDS and the legacy singular
    var should feed into the same set."""
    assert 111 in dt_module.CHANNEL_IDS
    assert 222 in dt_module.CHANNEL_IDS


def test_authorized_users_set_parsed(dt_module):
    assert dt_module.AUTH_USERS == {42, 99}


# ---------- whitelist ----------


def test_command_map_contains_expected_keys(dt_module):
    expected = {"status", "start", "stop", "restart", "logs",
                "positions", "doctor", "walkforward", "risk-switch",
                "audit", "reset-paper", "wipe"}
    assert expected.issubset(set(dt_module.COMMAND_MAP.keys()))


def test_destructive_commands_flagged_correctly(dt_module):
    """reset-paper and wipe MUST be marked destructive; the rest MUST NOT."""
    for name, (_, destroy) in dt_module.COMMAND_MAP.items():
        if name in ("reset-paper", "wipe"):
            assert destroy is True, f"{name} should require DESTROY"
        else:
            assert destroy is False, f"{name} should not require DESTROY"


# ---------- rate limiter ----------


def test_rate_limiter_allows_under_cap(dt_module):
    rl = dt_module.RateLimiter(max_count=3, window_sec=60.0)
    assert rl.allow(1) is True
    assert rl.allow(1) is True
    assert rl.allow(1) is True


def test_rate_limiter_blocks_over_cap(dt_module):
    rl = dt_module.RateLimiter(max_count=3, window_sec=60.0)
    for _ in range(3):
        rl.allow(1)
    assert rl.allow(1) is False    # 4th in window blocked


def test_rate_limiter_is_per_user(dt_module):
    rl = dt_module.RateLimiter(max_count=2, window_sec=60.0)
    rl.allow(1); rl.allow(1)
    # user 2 starts fresh
    assert rl.allow(2) is True


def test_rate_limiter_releases_after_window(dt_module, monkeypatch):
    """After the window slides past an event, the slot is reusable."""
    rl = dt_module.RateLimiter(max_count=1, window_sec=0.1)
    assert rl.allow(1) is True
    assert rl.allow(1) is False
    time.sleep(0.15)
    assert rl.allow(1) is True


# ---------- CommandRunner auth + dispatch ----------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_unauthorized_user_rejected(dt_module, monkeypatch):
    runner = dt_module.CommandRunner()
    # user_id=7 not in AUTH_USERS={42, 99}
    msg = _run(runner.run(7, "attacker", "status"))
    assert "Not authorized" in msg
    # Ensure the audit log captured the rejection
    assert dt_module.AUDIT_LOG_PATH.exists()
    lines = dt_module.AUDIT_LOG_PATH.read_text().splitlines()
    assert any('"kind":"rejected_auth"' in ln for ln in lines)


def test_unknown_command_rejected(dt_module):
    runner = dt_module.CommandRunner()
    msg = _run(runner.run(42, "me", "rm_rf_slash"))
    assert "Unknown command" in msg
    # `!help` hint included
    assert "!help" in msg


def test_destructive_command_requires_confirm(dt_module):
    """First call to reset-paper must NOT execute — must arm a pending
    DESTROY and wait for confirmation."""
    runner = dt_module.CommandRunner()
    msg = _run(runner.run(42, "me", "reset-paper"))
    assert "DESTRUCTIVE" in msg
    assert runner.pending_destructive.get(42) is not None


def test_destructive_confirm_fires_within_window(dt_module, monkeypatch):
    """When the user sends the DESTROY confirm within the window, the
    command actually runs. We mock _sh to avoid touching real state."""
    runner = dt_module.CommandRunner()
    _run(runner.run(42, "me", "reset-paper"))          # arm
    recorded = {}

    async def fake_sh(self, argv, *, cwd, timeout):
        recorded["argv"] = argv
        return (0, "ok", "")

    monkeypatch.setattr(dt_module.CommandRunner, "_sh", fake_sh)
    msg = _run(runner.run(42, "me", "DESTROY"))
    assert "reset-paper" in msg or "exit 0" in msg
    # The --yes flag MUST be forwarded so the underlying script skips
    # its own interactive prompt.
    assert "--yes" in " ".join(recorded["argv"])
    # Pending cleared after confirm
    assert 42 not in runner.pending_destructive


def test_expired_destroy_confirm_is_rejected(dt_module, monkeypatch):
    """If the user types DESTROY after the 30s window, nothing runs."""
    runner = dt_module.CommandRunner()
    _run(runner.run(42, "me", "reset-paper"))
    # Manually expire the pending entry
    cmd, _exp = runner.pending_destructive[42]
    runner.pending_destructive[42] = (cmd, time.time() - 1.0)
    msg = _run(runner.run(42, "me", "DESTROY"))
    assert "expired" in msg.lower()


def test_rate_limit_applies_to_authorized_users(dt_module):
    """Auth alone is not enough — authorized users can still flood."""
    runner = dt_module.CommandRunner()
    # force a small limit
    runner.rate_limiter = dt_module.RateLimiter(max_count=2, window_sec=60.0)
    msg1 = _run(runner.run(42, "me", "status"))
    msg2 = _run(runner.run(42, "me", "status"))
    msg3 = _run(runner.run(42, "me", "status"))
    # Third call should return the rate-limit message
    assert "Rate limit" in msg3


# ---------- truncation ----------


def test_truncate_leaves_short_text_untouched(dt_module):
    assert dt_module._truncate("hello") == "hello"


def test_truncate_chops_long_text_and_marks_it(dt_module):
    long_text = "A" * 10_000
    t = dt_module._truncate(long_text, limit=500)
    assert len(t) < 10_000
    assert "truncated" in t
