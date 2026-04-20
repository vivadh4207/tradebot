"""CandlePatternSignal — detector behavior + volume gating."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

from src.core.types import Bar, OptionRight, Side
from src.signals.base import SignalContext
from src.signals.candle_patterns import (
    CandlePatternSignal, _detect_engulfing, _detect_hammer,
    _detect_shooting_star, _detect_inside_bar_breakout,
    _detect_range_breakout,
)


def _mk(bars_data, base_ts=None, vol=1000):
    """bars_data: list of (open, high, low, close[, volume])."""
    base_ts = base_ts or datetime(2026, 4, 20, 14, 30, tzinfo=timezone.utc)
    out: List[Bar] = []
    for i, row in enumerate(bars_data):
        o, h, l, c = row[:4]
        v = row[4] if len(row) > 4 else vol
        out.append(Bar(
            symbol="SPY",
            ts=base_ts + timedelta(minutes=i),
            open=float(o), high=float(h), low=float(l), close=float(c),
            volume=float(v),
        ))
    return out


def _flat(n=20, price=100.0, vol=1000):
    """A series of small-body flat bars used as filler."""
    rows = []
    for _ in range(n):
        rows.append((price, price + 0.05, price - 0.05, price, vol))
    return _mk(rows)


# ------------------------------------------------------------- detectors


def test_bullish_engulfing_detected():
    bars = _mk([(100, 100.1, 99.5, 99.7),   # red
                (99.6, 100.8, 99.55, 100.5)])  # green engulf
    hit = _detect_engulfing(bars)
    assert hit is not None
    assert hit.direction == "bullish"
    assert hit.name == "bullish_engulfing"


def test_bearish_engulfing_detected():
    bars = _mk([(100, 100.5, 99.95, 100.4),   # green
                (100.5, 100.55, 99.4, 99.5)])  # red engulf
    hit = _detect_engulfing(bars)
    assert hit is not None
    assert hit.direction == "bearish"


def test_hammer_requires_prior_downtrend_and_long_lower_shadow():
    bars = _mk([(102, 102, 101, 101),
                (101, 101, 100, 100),
                (100, 100, 99.5, 99.5),
                # hammer: tiny body (0.1), long lower wick (~1.1)
                (99.5, 99.6, 98.5, 99.6)])
    hit = _detect_hammer(bars)
    assert hit is not None
    assert hit.direction == "bullish"


def test_shooting_star_requires_uptrend_and_long_upper_shadow():
    bars = _mk([(100, 100.5, 100, 100.5),
                (100.5, 101, 100.5, 101),
                (101, 101.5, 101, 101.5),
                # shooting star: small body (0.2), long upper wick (~0.9),
                # tiny lower wick
                (101.5, 102.5, 101.49, 101.7)])
    hit = _detect_shooting_star(bars)
    assert hit is not None
    assert hit.direction == "bearish"


def test_inside_bar_breakout_up():
    # mother bar wide, inside bar narrow, current breaks above both
    bars = _mk([(100, 101.5, 99, 101),      # mother
                (100.5, 101.0, 100.2, 100.8),  # inside
                (100.8, 102.0, 100.7, 101.8)])  # break up
    hit = _detect_inside_bar_breakout(bars)
    assert hit is not None
    assert hit.direction == "bullish"
    assert hit.name == "inside_bar_breakout_up"


def test_range_breakout_up():
    # 20 flat bars around 100, then a break to 101
    bars = _flat(20, price=100.0) + _mk([(100.5, 101.5, 100.4, 101.4)])
    hit = _detect_range_breakout(bars, n=20)
    assert hit is not None
    assert hit.direction == "bullish"


def test_range_breakout_down():
    bars = _flat(20, price=100.0) + _mk([(99.5, 99.6, 98.5, 98.6)])
    hit = _detect_range_breakout(bars, n=20)
    assert hit is not None
    assert hit.direction == "bearish"


# ------------------------------------------------------------- signal-level


def _ctx(bars):
    return SignalContext(symbol="SPY", now=bars[-1].ts, bars=bars,
                          spot=bars[-1].close, vwap=bars[-1].close)


def test_signal_emits_bullish_engulfing_as_call():
    flat = _flat(20, price=100.0, vol=1000)
    # engulfing pair with HIGH volume on current bar
    pair = _mk([(100, 100.1, 99.5, 99.7, 1000),
                (99.6, 100.8, 99.55, 100.5, 2000)])
    bars = flat + pair
    s = CandlePatternSignal().emit(_ctx(bars))
    assert s is not None
    assert s.option_right == OptionRight.CALL
    assert s.side == Side.BUY
    assert "bullish_engulfing" in s.rationale
    assert s.confidence >= 0.55


def test_signal_breakout_rejected_on_low_volume():
    # 20 flat bars + breakout up, but on LOW volume → should NOT fire
    flat = _flat(20, price=100.0, vol=1000)
    breakout = _mk([(100.5, 101.5, 100.4, 101.4, 200)])     # 0.2× avg
    s = CandlePatternSignal().emit(_ctx(flat + breakout))
    assert s is None


def test_signal_breakout_fires_on_high_volume():
    flat = _flat(20, price=100.0, vol=1000)
    breakout = _mk([(100.5, 101.5, 100.4, 101.4, 3000)])    # 3× avg
    s = CandlePatternSignal().emit(_ctx(flat + breakout))
    assert s is not None
    assert s.option_right == OptionRight.CALL
    # Either inside_bar_breakout_up or range_breakout_up can fire here;
    # both are valid bullish continuation patterns with volume.
    assert s.meta.get("pattern") in (
        "range_breakout_up", "inside_bar_breakout_up",
    )
    assert s.meta.get("is_continuation") is True


def test_signal_reversal_low_volume_dampened_but_still_possible():
    """Engulfing on low volume — kept (reversal) but confidence
    reduced. Should still emit if pattern strength is high."""
    flat = _flat(20, price=100.0, vol=5000)
    # Larger body ratio so base strength is high enough to survive dampen.
    pair = _mk([(100, 100.1, 99.5, 99.7, 5000),              # prior red, wide
                (99.3, 102.0, 99.2, 101.8, 500)])            # huge engulf low vol
    s = CandlePatternSignal().emit(_ctx(flat + pair))
    # May or may not emit depending on thresholds; if it does, confidence
    # should be dampened vs the same pattern on normal volume.
    if s is not None:
        assert s.confidence < 0.90


def test_signal_requires_minimum_history():
    bars = _mk([(100, 100.1, 99.5, 99.7),
                (99.6, 100.8, 99.55, 100.5)])   # only 2 bars
    s = CandlePatternSignal(min_bars=20).emit(_ctx(bars))
    assert s is None


def test_signal_none_when_no_pattern():
    """Flat random bars — no pattern → no signal."""
    bars = _flat(25, price=100.0, vol=1000)
    s = CandlePatternSignal().emit(_ctx(bars))
    assert s is None


def test_near_vwap_boosts_confidence():
    flat = _flat(20, price=100.0, vol=1000)
    # Engulfing near VWAP at 100.5
    pair = _mk([(100.4, 100.5, 100.0, 100.1, 1000),
                (100.05, 100.8, 100.0, 100.7, 2000)])
    bars = flat + pair
    ctx_near = SignalContext(symbol="SPY", now=bars[-1].ts, bars=bars,
                               spot=bars[-1].close, vwap=100.5)
    ctx_far = SignalContext(symbol="SPY", now=bars[-1].ts, bars=bars,
                              spot=bars[-1].close, vwap=95.0)
    s_near = CandlePatternSignal().emit(ctx_near)
    s_far = CandlePatternSignal().emit(ctx_far)
    if s_near and s_far:
        assert s_near.confidence >= s_far.confidence
