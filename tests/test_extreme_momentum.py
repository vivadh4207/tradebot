"""Tests for the extreme-momentum shock scanner."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from src.core.types import Bar, OptionRight, Side
from src.signals.base import SignalContext
from src.signals.extreme_momentum import (
    ExtremeMomentumConfig, ExtremeMomentumSignal, compute_shock,
)


def _bar(idx, close, vol, *, o=None, h=None, l=None):
    o = o if o is not None else close
    return Bar(
        symbol="SPY",
        ts=datetime(2026, 4, 17, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=idx),
        open=o,
        high=h if h is not None else max(o, close) * 1.001,
        low=l if l is not None else min(o, close) * 0.999,
        close=close,
        volume=vol,
    )


def _flat(n, price=580.0, vol=1000.0):
    return [_bar(i, price, vol) for i in range(n)]


# ---------- compute_shock pure function ----------


def test_no_shock_on_flat_series():
    assert compute_shock(_flat(30)) is None


def test_no_shock_when_too_few_bars():
    cfg = ExtremeMomentumConfig(lookback_bars=5, baseline_bars=20)
    # Need >= 26 bars
    assert compute_shock(_flat(20), cfg) is None


def test_detects_bullish_shock():
    """Flat → 4% rip over 5 bars + volume surge."""
    bars = _flat(25, price=580.0, vol=1000)
    # Last 5 bars climb from 580 to 603 (3.97%) with 4× volume
    for i, px in enumerate([587.0, 591.0, 595.0, 599.0, 603.0]):
        bars.append(_bar(25 + i, px, 4000))
    shock = compute_shock(bars,
                          ExtremeMomentumConfig(lookback_bars=5,
                                                 baseline_bars=20,
                                                 min_move_pct=0.03,
                                                 min_volume_multiple=3.0))
    assert shock is not None
    assert shock["direction"] == "bullish"
    assert shock["move_pct"] > 0.03
    assert shock["volume_multiple"] >= 3.0


def test_detects_bearish_shock():
    bars = _flat(25, price=580.0, vol=1000)
    for i, px in enumerate([574.0, 570.0, 566.0, 562.0, 558.0]):
        bars.append(_bar(25 + i, px, 5000))
    shock = compute_shock(bars,
                          ExtremeMomentumConfig(min_move_pct=0.03,
                                                 min_volume_multiple=3.0))
    assert shock is not None
    assert shock["direction"] == "bearish"
    assert shock["move_pct"] < -0.03


def test_no_shock_when_move_large_but_volume_low():
    """A 4% move WITHOUT the volume surge isn't a shock — might be
    overnight drift or illiquid afternoon. Reject."""
    bars = _flat(25, price=580.0, vol=1000)
    for i, px in enumerate([585.0, 590.0, 595.0, 600.0, 604.0]):
        bars.append(_bar(25 + i, px, 1200))   # 1.2× not > 3×
    shock = compute_shock(bars,
                          ExtremeMomentumConfig(min_volume_multiple=3.0))
    assert shock is None


def test_no_shock_when_volume_surge_but_move_small():
    """Volume surge without directional conviction = whipsaw, not shock."""
    bars = _flat(25, price=580.0, vol=1000)
    # Tiny drift (0.5%) with big volume — shouldn't fire
    for i, px in enumerate([580.5, 581.0, 581.5, 582.0, 582.5]):
        bars.append(_bar(25 + i, px, 10000))
    shock = compute_shock(bars,
                          ExtremeMomentumConfig(min_move_pct=0.03))
    assert shock is None


def test_move_percent_calculated_from_pre_window_close():
    """The move % should compare the LAST close to the close BEFORE
    the lookback window started, not to the first bar of the window."""
    bars = _flat(25, price=580.0, vol=1000)
    # Bar 25 (the bar-before-window) has close 580
    bars.append(_bar(25, 590.0, 5000))
    for i, px in enumerate([595.0, 600.0, 603.0, 605.0]):
        bars.append(_bar(26 + i, px, 5000))
    shock = compute_shock(bars,
                          ExtremeMomentumConfig(lookback_bars=5,
                                                 baseline_bars=20,
                                                 min_move_pct=0.03,
                                                 min_volume_multiple=3.0))
    assert shock is not None
    # start_close is bars[-6].close == 580; end = 605 → ~4.3%
    assert shock["start_close"] == pytest.approx(580.0)
    assert shock["close"] == pytest.approx(605.0)
    assert abs(shock["move_pct"] - 0.0431) < 0.01


# ---------- ExtremeMomentumSignal wrapper ----------


def _ctx_from(bars):
    return SignalContext(
        symbol="SPY",
        now=datetime.now(tz=timezone.utc),
        bars=bars,
        spot=bars[-1].close,
    )


def test_signal_emits_call_on_bullish_shock():
    bars = _flat(25, vol=1000)
    for i, px in enumerate([587.0, 591.0, 595.0, 599.0, 603.0]):
        bars.append(_bar(25 + i, px, 4000))
    src = ExtremeMomentumSignal()
    sig = src.emit(_ctx_from(bars))
    assert sig is not None
    assert sig.option_right == OptionRight.CALL
    assert sig.side == Side.BUY
    assert "shock_move" in sig.rationale
    assert sig.confidence == 0.85


def test_signal_emits_put_on_bearish_shock():
    bars = _flat(25, vol=1000)
    for i, px in enumerate([574.0, 570.0, 566.0, 562.0, 558.0]):
        bars.append(_bar(25 + i, px, 5000))
    src = ExtremeMomentumSignal()
    sig = src.emit(_ctx_from(bars))
    assert sig is not None
    assert sig.option_right == OptionRight.PUT


def test_signal_returns_none_on_flat_data():
    assert ExtremeMomentumSignal().emit(_ctx_from(_flat(30))) is None
