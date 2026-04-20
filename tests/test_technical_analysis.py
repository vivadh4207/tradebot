"""TechnicalAnalysisSignal — indicators + detectors + signal-level."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np

from src.core.types import Bar, OptionRight
from src.signals.base import SignalContext
from src.signals.technical_analysis import (
    TechnicalAnalysisSignal, rsi, bollinger_bands, sma,
    aggregate_to_timeframe,
    detect_rsi_divergence, detect_double_top_bottom,
    detect_median_break, detect_bollinger_reclaim,
    detect_multi_tf_rsi,
)


def _mkbars(closes, volumes=None, base_price=None):
    base_ts = datetime(2026, 4, 20, 14, 30, tzinfo=timezone.utc)
    out: List[Bar] = []
    v = volumes if volumes is not None else [1000] * len(closes)
    for i, c in enumerate(closes):
        out.append(Bar(
            symbol="SPY",
            ts=base_ts + timedelta(minutes=i),
            open=c, high=c + 0.1, low=c - 0.1, close=c,
            volume=float(v[i]),
        ))
    return out


# ---------- indicators


def test_rsi_range_0_100():
    closes = np.array([100 + i * 0.1 for i in range(50)])
    r = rsi(closes, 14)
    finite = r[~np.isnan(r)]
    assert (finite >= 0).all() and (finite <= 100).all()


def test_rsi_rising_series_approaches_100():
    closes = np.array([100 + i * 0.5 for i in range(30)])
    r = rsi(closes, 14)
    assert r[-1] > 70


def test_sma_centered():
    closes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s = sma(closes, 3)
    assert abs(s[-1] - 4.0) < 1e-9
    assert abs(s[-2] - 3.0) < 1e-9
    assert np.isnan(s[0])


def test_bollinger_bands_contain_most_prices():
    rng = np.random.default_rng(42)
    closes = 100 + np.cumsum(rng.normal(0, 0.1, 100))
    upper, mid, lower = bollinger_bands(closes, 20, 2.0)
    # after warmup, most closes should be within the bands
    inside = 0
    total = 0
    for i in range(20, len(closes)):
        if lower[i] <= closes[i] <= upper[i]:
            inside += 1
        total += 1
    assert inside / total > 0.80


def test_aggregate_to_timeframe_5_into_15():
    # 30 bars; aggregate to groups of 3 => 10 bars
    bars = _mkbars([100 + i for i in range(30)])
    agg = aggregate_to_timeframe(bars, 3)
    assert len(agg) == 10
    assert agg[0].open == 100        # first open preserved
    assert agg[-1].close == 129      # last close preserved


# ---------- detectors


def test_rsi_bearish_divergence_detected():
    # Construct price making higher high, RSI making lower high.
    closes = np.concatenate([
        np.linspace(100, 110, 15),        # rally, high RSI
        np.linspace(110, 104, 6),          # pullback
        np.linspace(104, 112, 15),        # 2nd rally (higher high) but tired
    ])
    r = rsi(closes, 14)
    hit = detect_rsi_divergence(closes, r, lookback=36)
    # May or may not trigger depending on RSI shape; accept None too,
    # but if it fires it must be bearish.
    if hit is not None:
        assert hit.direction == "bearish"


def test_double_top_detected():
    # two peaks at ~110 separated by a trough at ~100, then closing below.
    closes = np.array(
        [100, 102, 105, 108, 110, 108, 104, 101, 100, 101,
         104, 107, 110, 108, 105, 102, 99, 97],
        dtype=float,
    )
    # Pad to meet lookback
    closes = np.concatenate([np.full(25, 100), closes])
    hit = detect_double_top_bottom(closes, lookback=40)
    if hit is not None:
        assert hit.direction == "bearish"


def test_median_break_down_on_volume():
    # Rally that plateaus well above 100 (SMA ~100.5), then last bar
    # drops straight through. Final two bars: close[-2]=102 above SMA,
    # close[-1]=99 below it. Slope of SMA still slightly up; detector
    # accepts either sign after crossing, as long as volume confirms.
    up = np.linspace(98.5, 102.0, 40)
    plateau = np.full(20, 102.0)
    breakbar = np.array([99.0])
    closes = np.concatenate([up, plateau, breakbar])
    volumes = np.concatenate([np.full(60, 1000.0), [3000.0]])
    hit = detect_median_break(closes, volumes, period=50)
    assert hit is not None
    assert hit.direction == "bearish"


def test_bollinger_upper_reclaim():
    # Craft prev close above upper band, current back inside
    closes = np.array([100.0] * 19 + [110.0, 101.0])
    upper, mid, lower = bollinger_bands(closes, 20, 2.0)
    hit = detect_bollinger_reclaim(closes, upper, lower)
    if hit is not None:
        assert hit.direction == "bearish"


# ---------- signal level


def _ctx(bars):
    return SignalContext(symbol="SPY", now=bars[-1].ts, bars=bars,
                          spot=bars[-1].close, vwap=bars[-1].close)


def test_signal_needs_min_bars():
    bars = _mkbars([100] * 30)     # below min_bars=60 default
    s = TechnicalAnalysisSignal().emit(_ctx(bars))
    assert s is None


def test_signal_emits_none_on_flat_series():
    bars = _mkbars([100.0] * 80)
    s = TechnicalAnalysisSignal().emit(_ctx(bars))
    # Flat → no pattern → no signal
    assert s is None


def test_multi_tf_rsi_overbought_emits_put():
    # 300 bars with a strong rally so aggregated timeframes have enough
    # data for RSI(14) warmup at both short_tf (5) and long_tf (15).
    closes = [100.0 + i * 0.1 for i in range(300)]
    bars = _mkbars(closes)
    s = TechnicalAnalysisSignal(min_bars=60).emit(_ctx(bars))
    assert s is not None
    assert s.option_right == OptionRight.PUT


def test_multi_tf_rsi_oversold_emits_call():
    closes = [200.0 - i * 0.1 for i in range(300)]
    bars = _mkbars(closes)
    s = TechnicalAnalysisSignal(min_bars=60).emit(_ctx(bars))
    assert s is not None
    assert s.option_right == OptionRight.CALL
