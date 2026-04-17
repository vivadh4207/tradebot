"""Regression tests for the VWAP-alignment and momentum-confirmation
exec-chain filters that locked in operator feedback: 'don't enter
anything without checking patterns and momentum'."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest


def _ctx(*, direction: str, spot: float, vwap: float,
          bars=None, enabled_keys: dict | None = None):
    from src.core.types import Signal, Side, Bar
    from src.risk.execution_chain import ExecutionContext
    sig = Signal(
        source="ensemble", symbol="NVDA", side=Side.BUY,
        confidence=0.9, rationale="t",
        meta={"direction": direction},
    )
    return ExecutionContext(
        signal=sig,
        now=datetime.now(tz=timezone.utc),
        account_equity=10_000.0, day_pnl=0.0, open_positions_count=0,
        current_bar_volume=1.0, avg_bar_volume=1.0,
        opening_range_high=spot * 1.01, opening_range_low=spot * 0.99,
        spot=spot, vwap=vwap, vix=15.0,
        recent_bars=bars or [],
    )


def _chain(settings_dict):
    from src.risk.execution_chain import ExecutionChain
    from src.core.clock import MarketClock
    clock = MarketClock(
        market_open="09:30", market_close="16:00",
        no_new_entries_after="15:30", eod_force_close="15:45",
    )
    return ExecutionChain(settings_dict, clock)


def _mkbars(prices):
    """Build bars with the given close prices. Open = previous close."""
    from src.core.types import Bar
    bars = []
    now = datetime.now(tz=timezone.utc)
    prev = prices[0]
    for i, p in enumerate(prices):
        bars.append(Bar(
            symbol="NVDA", ts=now + timedelta(minutes=i),
            open=float(prev), high=float(max(prev, p) * 1.001),
            low=float(min(prev, p) * 0.999), close=float(p),
            volume=1000.0,
        ))
        prev = p
    return bars


# ---------- VWAP alignment ----------
def test_vwap_bullish_below_vwap_blocks():
    ec = _chain({"execution": {"vwap_alignment_enabled": True}})
    ctx = _ctx(direction="bullish", spot=100.0, vwap=101.0)  # spot below
    r = ec.f16_vwap_alignment(ctx)
    assert r.passed is False
    assert "vwap_align_wrong_side" in r.reason


def test_vwap_bullish_above_vwap_passes():
    ec = _chain({"execution": {"vwap_alignment_enabled": True}})
    ctx = _ctx(direction="bullish", spot=102.0, vwap=101.0)
    r = ec.f16_vwap_alignment(ctx)
    assert r.passed is True


def test_vwap_bearish_above_vwap_blocks():
    ec = _chain({"execution": {"vwap_alignment_enabled": True}})
    ctx = _ctx(direction="bearish", spot=102.0, vwap=101.0)
    r = ec.f16_vwap_alignment(ctx)
    assert r.passed is False


def test_vwap_bearish_below_vwap_passes():
    ec = _chain({"execution": {"vwap_alignment_enabled": True}})
    ctx = _ctx(direction="bearish", spot=100.0, vwap=101.0)
    r = ec.f16_vwap_alignment(ctx)
    assert r.passed is True


def test_vwap_disabled_always_passes():
    ec = _chain({"execution": {"vwap_alignment_enabled": False}})
    ctx = _ctx(direction="bullish", spot=50.0, vwap=100.0)   # deeply wrong
    r = ec.f16_vwap_alignment(ctx)
    assert r.passed is True


# ---------- Momentum confirmation ----------
def test_momentum_bullish_with_rally_passes():
    ec = _chain({"execution": {"momentum_confirmation_enabled": True,
                                  "momentum_confirmation_min_move": 0.002}})
    bars = _mkbars([100, 100.1, 100.3, 100.5, 100.8, 101.0])
    ctx = _ctx(direction="bullish", spot=101.0, vwap=100.0, bars=bars)
    r = ec.f17_momentum_confirmation(ctx)
    assert r.passed is True


def test_momentum_bullish_with_no_move_blocks():
    ec = _chain({"execution": {"momentum_confirmation_enabled": True,
                                  "momentum_confirmation_min_move": 0.002}})
    # Flat bars — no move
    bars = _mkbars([100.0] * 6)
    ctx = _ctx(direction="bullish", spot=100.0, vwap=100.0, bars=bars)
    r = ec.f17_momentum_confirmation(ctx)
    assert r.passed is False
    assert "momentum_weak" in r.reason


def test_momentum_bearish_with_decline_passes():
    ec = _chain({"execution": {"momentum_confirmation_enabled": True,
                                  "momentum_confirmation_min_move": 0.002}})
    bars = _mkbars([100, 99.8, 99.5, 99.3, 99.1, 98.9])
    ctx = _ctx(direction="bearish", spot=98.9, vwap=100.0, bars=bars)
    r = ec.f17_momentum_confirmation(ctx)
    assert r.passed is True


def test_momentum_bearish_with_rally_blocks():
    ec = _chain({"execution": {"momentum_confirmation_enabled": True,
                                  "momentum_confirmation_min_move": 0.002}})
    bars = _mkbars([100, 100.2, 100.4, 100.6, 100.8, 101.0])
    ctx = _ctx(direction="bearish", spot=101.0, vwap=100.0, bars=bars)
    r = ec.f17_momentum_confirmation(ctx)
    assert r.passed is False


def test_momentum_insufficient_bars_skips():
    """With < 5 bars, the filter doesn't have enough data → skip advisory."""
    ec = _chain({"execution": {"momentum_confirmation_enabled": True}})
    bars = _mkbars([100, 100.5])
    ctx = _ctx(direction="bullish", spot=100.5, vwap=100.0, bars=bars)
    r = ec.f17_momentum_confirmation(ctx)
    assert r.passed is True   # advisory pass — not enough data to block
    assert r.advisory is True


def test_momentum_disabled_always_passes():
    ec = _chain({"execution": {"momentum_confirmation_enabled": False}})
    ctx = _ctx(direction="bullish", spot=100.0, vwap=100.0,
                bars=_mkbars([100.0] * 10))
    r = ec.f17_momentum_confirmation(ctx)
    assert r.passed is True
