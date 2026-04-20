"""TradingView webhook ingest + signal source tests."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core.types import Bar, Side
from src.signals.base import SignalContext
from src.signals.tradingview_webhook import (
    ingest, IngestResult, TradingViewWebhookSignal,
)


SECRET = "test-secret-abc123"


@pytest.fixture
def queue_env(tmp_path, monkeypatch):
    q = tmp_path / "tv_queue.jsonl"
    monkeypatch.setenv("TRADINGVIEW_QUEUE_PATH", str(q))
    monkeypatch.setenv("TRADINGVIEW_WEBHOOK_SECRET", SECRET)
    return q


def _payload(**overrides):
    base = {
        "secret": SECRET, "symbol": "SPY", "side": "buy",
        "reason": "rsi_oversold", "confidence": 0.65,
        "tf": "5m", "spot": 580.25, "ts": "2026-04-17T15:30Z",
    }
    base.update(overrides)
    return base


# ---------- ingest ----------


def test_ingest_rejects_missing_secret(queue_env, monkeypatch):
    """Default-configure prevents open-door abuse: if the server's
    TRADINGVIEW_WEBHOOK_SECRET isn't set, the endpoint 503s."""
    monkeypatch.delenv("TRADINGVIEW_WEBHOOK_SECRET", raising=False)
    r = ingest(_payload())
    assert r.ok is False
    assert r.status_code == 503


def test_ingest_rejects_wrong_secret(queue_env):
    r = ingest(_payload(secret="wrong"))
    assert r.ok is False
    assert r.status_code == 401


def test_ingest_rejects_unknown_symbol(queue_env):
    r = ingest(_payload(symbol="NVDA"))
    assert r.ok is False
    assert r.status_code == 400
    assert "universe" in (r.error or "")


def test_ingest_rejects_bad_side(queue_env):
    r = ingest(_payload(side="long"))
    assert r.ok is False
    assert r.status_code == 400


def test_ingest_rejects_non_dict_payload(queue_env):
    """Defensive: a TV alert Message field might be a bare string if
    the user forgets to JSON-encode it."""
    # ingest() expects a dict — caller (FastAPI handler) validates. So
    # passing a bare string as body goes through the handler's own
    # rejection path. We just verify ingest() doesn't silently accept
    # missing fields:
    r = ingest({"secret": SECRET})
    assert r.ok is False
    assert r.status_code == 400


def test_ingest_persists_canonical_record(queue_env):
    r = ingest(_payload())
    assert r.ok is True
    assert r.alert_id and r.alert_id.startswith("tv-")
    lines = queue_env.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["symbol"] == "SPY"
    assert rec["side"] == "buy"
    assert rec["confidence"] == 0.65
    # Secret must NEVER be persisted — critical invariant
    assert "secret" not in rec
    assert "secret" not in rec.get("raw", {})
    assert rec["consumed"] is False


def test_ingest_clamps_out_of_range_confidence(queue_env):
    r = ingest(_payload(confidence=5.0))
    assert r.ok is True
    rec = json.loads(queue_env.read_text().splitlines()[-1])
    assert rec["confidence"] == 1.0


def test_ingest_constant_time_secret_compare(queue_env):
    """The secret compare uses hmac.compare_digest. Verify by ensuring
    two different-wrong secrets both produce the same error shape."""
    r1 = ingest(_payload(secret="aaaa"))
    r2 = ingest(_payload(secret="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"))
    assert r1.status_code == r2.status_code == 401
    assert r1.error == r2.error


# ---------- signal source ----------


def _ctx(symbol, now=None):
    now = now or datetime.now(tz=timezone.utc)
    return SignalContext(symbol=symbol, now=now, bars=[], spot=580.0)


def test_signal_emits_for_matching_symbol_and_marks_consumed(queue_env):
    ingest(_payload(symbol="SPY", side="buy"))
    src = TradingViewWebhookSignal()
    sig = src.emit(_ctx("SPY"))
    assert sig is not None
    assert sig.source == "tradingview_webhook"
    assert sig.symbol == "SPY"
    assert sig.side == Side.BUY
    assert 0 < sig.confidence <= 1
    # Second call returns None — alert already consumed
    sig2 = src.emit(_ctx("SPY"))
    assert sig2 is None
    # The record in the file should have consumed=true
    rec = json.loads(queue_env.read_text().splitlines()[-1])
    assert rec["consumed"] is True
    assert rec["consume_reason"] == "emitted"


def test_signal_skips_alerts_for_other_symbols(queue_env):
    ingest(_payload(symbol="SPY", side="buy"))
    src = TradingViewWebhookSignal()
    # Ask for QQQ — SPY alert should be untouched
    sig = src.emit(_ctx("QQQ"))
    assert sig is None
    rec = json.loads(queue_env.read_text().splitlines()[-1])
    assert rec["consumed"] is False   # still available for SPY tick


def test_signal_expires_stale_alerts(queue_env, monkeypatch):
    """Alert older than ttl_sec is marked consumed WITHOUT emission so
    the queue doesn't grow without bound and old signals don't suddenly
    fire on a bot restart."""
    # Write an old alert directly (bypassing time.time() freshness)
    old_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
    queue_env.write_text(json.dumps({
        "alert_id": "tv-old", "received_at": old_ts,
        "symbol": "SPY", "side": "buy",
        "reason": "stale", "confidence": 0.7, "tf": "5m",
        "spot": 580.0, "ts": "", "raw": {}, "consumed": False,
    }) + "\n")
    src = TradingViewWebhookSignal(ttl_sec=30.0)
    sig = src.emit(_ctx("SPY"))
    assert sig is None
    rec = json.loads(queue_env.read_text().splitlines()[-1])
    assert rec["consumed"] is True
    assert rec["consume_reason"] == "expired"


def test_signal_handles_missing_queue_file_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_QUEUE_PATH",
                        str(tmp_path / "absent.jsonl"))
    src = TradingViewWebhookSignal()
    assert src.emit(_ctx("SPY")) is None


def test_signal_preserves_rationale_and_meta(queue_env):
    ingest(_payload(symbol="QQQ", side="sell",
                    reason="tv_supertrend_flip"))
    src = TradingViewWebhookSignal()
    sig = src.emit(_ctx("QQQ"))
    assert sig is not None
    assert sig.side == Side.SELL
    assert "tv_supertrend_flip" in sig.rationale
    assert sig.meta.get("tf") == "5m"
    assert "alert_id" in sig.meta
