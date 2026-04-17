"""Test harness — isolates every test from the production journal, the
real broker_state.json, and the live Cockroach connection.

WHY: tests that construct `TradeBot(s)` end up loading the real
`config/settings.yaml` and reading env vars (COCKROACH_DSN, ALPACA keys,
DISCORD webhook) from the real .env. Without the defensive
overrides below, a test run would:
  - Write test positions to the live `logs/broker_state.json`
  - INSERT test fills into the live Cockroach `tradebot.fills` table
  - Ping Discord with `startup: tradebot started` on every run
  - Call out to Alpaca (network) during unit tests

The `_sandbox_production_state` autouse fixture forces every test to:
  1. Set TRADEBOT_NO_NETWORK=1 (skips dividend_yield yfinance pulls,
     news fetches, etc.)
  2. Clear the webhook env vars so build_notifier returns NullNotifier
  3. Point `logs/broker_state.json`, `logs/heartbeat.txt`, and the
     calibration JSONLs at per-test tmp paths
  4. Force `storage.backend=sqlite` in loaded settings (no Cockroach)

Tests that NEED the real settings can opt out with
`@pytest.mark.uses_real_config` — currently unused, here for future
integration tests.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _sandbox_production_state(tmp_path, monkeypatch):
    """Autouse: every test gets a private sandbox for all side-effectful
    paths + disables outbound network."""
    # 1. Block any yfinance / Alpaca SDK / external HTTP that tests trigger.
    monkeypatch.setenv("TRADEBOT_NO_NETWORK", "1")

    # 2. Strip webhook URLs so build_notifier() returns NullNotifier.
    #    Prevents tests from posting "startup: tradebot started" to the
    #    user's Discord every time pytest runs.
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

    # 3. Disable the Cockroach backend for tests. Any TradeBot constructed
    #    during the test will fall back to sqlite (in tmp_path).
    monkeypatch.delenv("COCKROACH_DSN", raising=False)
    monkeypatch.delenv("COCKROACH_HOST", raising=False)
    monkeypatch.setenv("TRADEBOT_STORAGE_BACKEND", "sqlite")

    # 4. Per-test sandbox dir for all on-disk state. Tests can read
    #    `tradebot_sandbox_logs` if they want to inspect what the bot
    #    wrote. Uses a uniquely-named subdir so tests that ALSO create
    #    `tmp_path/logs` themselves don't collide.
    sandbox_logs = tmp_path / ".tradebot-sandbox"
    sandbox_logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRADEBOT_SANDBOX_LOGS", str(sandbox_logs))

    # 5. Redirect calibration JSONLs to sandbox so our slippage tests
    #    never write into the live calibration history.
    monkeypatch.setenv("TRADEBOT_SLIPPAGE_LOG",
                        str(sandbox_logs / "slippage_calibration.jsonl"))

    yield


# Expose ROOT to legacy tests that imported it from conftest
__all__ = ["ROOT"]
