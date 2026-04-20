"""Test harness — isolates every test from the production journal,
the real broker_state.json, and outbound network.

WHY: tests that construct `TradeBot(s)` end up loading the real
`config/settings.yaml` and reading env vars (ALPACA keys, DISCORD
webhook) from the real .env. Without the defensive overrides below,
a test run would:
  - Write test positions to the live `logs/broker_state.json`
  - INSERT test fills into the live SQLite journal
  - Ping Discord with `startup: tradebot started` on every run
  - Call out to Alpaca (network) during unit tests

The `_sandbox_production_state` autouse fixture forces every test to:
  1. Set TRADEBOT_NO_NETWORK=1 (skips yfinance pulls, news fetches, etc.)
  2. Clear the webhook env vars so build_notifier returns NullNotifier
  3. Point broker_state.json, heartbeat.txt, and calibration JSONLs
     at per-test tmp paths
  4. Point the SQLite journal at a per-test tmp file
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

    # 2. Strip EVERY DISCORD_WEBHOOK_URL* + SLACK_WEBHOOK_URL so
    #    build_notifier() returns NullNotifier by default. Prevents
    #    tests from posting to real webhooks AND prevents the dynamic
    #    DISCORD_WEBHOOK_URL_<NAME> scan in base.py from accidentally
    #    building a MultiChannelNotifier from a leaked env var.
    for key in list(os.environ.keys()):
        if key.startswith("DISCORD_WEBHOOK_URL") or key == "SLACK_WEBHOOK_URL":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("DISCORD_EXTRA_CHANNEL_ROUTES", raising=False)

    # 3. Per-test sandbox dir for all on-disk state. Tests can read
    #    `tradebot_sandbox_logs` if they want to inspect what the bot
    #    wrote. Uses a uniquely-named subdir so tests that ALSO create
    #    `tmp_path/logs` themselves don't collide.
    sandbox_logs = tmp_path / ".tradebot-sandbox"
    sandbox_logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRADEBOT_SANDBOX_LOGS", str(sandbox_logs))

    # 4. Redirect calibration JSONLs to sandbox so our slippage tests
    #    never write into the live calibration history.
    monkeypatch.setenv("TRADEBOT_SLIPPAGE_LOG",
                        str(sandbox_logs / "slippage_calibration.jsonl"))

    yield


# Expose ROOT to legacy tests that imported it from conftest
__all__ = ["ROOT"]
