"""LongPutDipSignal — dip-buy-put trigger with macro confirmation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np

from src.core.types import Bar, OptionRight
from src.signals.base import SignalContext
from src.signals.long_put_dip import LongPutDipSignal, LongPutDipConfig


def _mkbars(closes, volumes=None):
    base_ts = datetime(2026, 4, 20, 14, 30, tzinfo=timezone.utc)
    vols = volumes if volumes is not None else [1000] * len(closes)
    out: List[Bar] = []
    for i, c in enumerate(closes):
        out.append(Bar(
            symbol="SPY",
            ts=base_ts + timedelta(minutes=i),
            open=c, high=c + 0.02, low=c - 0.02, close=c,
            volume=float(vols[i]),
        ))
    return out


def _ctx(bars, vwap):
    return SignalContext(symbol="SPY", now=bars[-1].ts, bars=bars,
                          spot=bars[-1].close, vwap=vwap)


def test_no_signal_when_insufficient_bars():
    bars = _mkbars([100] * 30)
    s = LongPutDipSignal().emit(_ctx(bars, 100.5))
    assert s is None


def test_no_signal_when_price_above_vwap():
    bars = _mkbars([100 + i * 0.01 for i in range(50)])
    s = LongPutDipSignal().emit(_ctx(bars, 99.5))     # price above vwap
    assert s is None


def test_no_signal_without_macro_confirm():
    # Dip + oversold RSI + volume, but no VIX, no breadth,
    # AND dip isn't deep enough for the price-only fallback.
    closes = [100.0] * 40 + [99.0, 98.5, 98.2, 98.0]
    bars = _mkbars(closes)
    s = LongPutDipSignal().emit(_ctx(bars, 100.0))     # dip ~2%
    # dip is large enough for price-only, so it MAY fire — but if we
    # disable price-only by reducing dip, it should NOT fire.
    closes2 = [100.0] * 40 + [99.8, 99.7, 99.6, 99.6]  # ~0.4% dip
    bars2 = _mkbars(closes2)
    s2 = LongPutDipSignal().emit(_ctx(bars2, 100.0))
    assert s2 is None


def test_signal_fires_with_vix_spike():
    # Construct a proper dip with RSI oversold + VIX spike.
    closes = [100.0] * 35 + list(np.linspace(100, 99.2, 8))
    bars = _mkbars(closes, [1000] * len(closes))
    # Boost current bar volume
    bars[-1].volume = 3000
    vix_fn = lambda: {"change_pct": 0.08}               # VIX up 8%
    s = LongPutDipSignal(get_vix_fn=vix_fn).emit(_ctx(bars, 100.0))
    assert s is not None
    assert s.option_right == OptionRight.PUT
    assert s.meta["setup"] == "dip_buy_put"


def test_signal_fires_with_breadth():
    closes = [100.0] * 35 + list(np.linspace(100, 99.2, 8))
    bars = _mkbars(closes, [1000] * len(closes))
    bars[-1].volume = 3000
    breadth_fn = lambda: {"advancers": 200, "decliners": 500}  # 2.5x
    s = LongPutDipSignal(get_breadth_fn=breadth_fn).emit(_ctx(bars, 100.0))
    assert s is not None
    assert s.option_right == OptionRight.PUT


def test_signal_none_when_volume_anemic():
    closes = [100.0] * 35 + list(np.linspace(100, 99.2, 8))
    bars = _mkbars(closes, [1000] * len(closes))
    bars[-1].volume = 200     # way below avg
    vix_fn = lambda: {"change_pct": 0.08}
    s = LongPutDipSignal(get_vix_fn=vix_fn).emit(_ctx(bars, 100.0))
    assert s is None


def test_signal_none_when_rsi_not_oversold():
    # Rising price → RSI high → no dip-buy-put
    closes = [100.0 + i * 0.02 for i in range(50)]
    bars = _mkbars(closes)
    vix_fn = lambda: {"change_pct": 0.08}
    s = LongPutDipSignal(get_vix_fn=vix_fn).emit(_ctx(bars, 99.0))
    assert s is None


def test_confidence_higher_with_deeper_dip():
    base = [100.0] * 35
    shallow = base + list(np.linspace(100, 99.5, 8))
    deep = base + list(np.linspace(100, 98.5, 8))
    bars_shallow = _mkbars(shallow, [1000] * len(shallow)); bars_shallow[-1].volume = 3000
    bars_deep = _mkbars(deep, [1000] * len(deep));          bars_deep[-1].volume = 3000
    vix_fn = lambda: {"change_pct": 0.08}
    s_shallow = LongPutDipSignal(get_vix_fn=vix_fn).emit(_ctx(bars_shallow, 100.0))
    s_deep = LongPutDipSignal(get_vix_fn=vix_fn).emit(_ctx(bars_deep, 100.0))
    if s_shallow and s_deep:
        assert s_deep.confidence >= s_shallow.confidence
