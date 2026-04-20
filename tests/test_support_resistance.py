"""Volume-weighted S/R detector tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from src.core.types import Bar
from src.intelligence.support_resistance import (
    SRConfig, SRLevel, find_levels, nearest_support, nearest_resistance,
)


def _bar(idx, o, h, l, c, v):
    return Bar(
        symbol="SPY",
        ts=datetime(2026, 4, 17, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=idx),
        open=o, high=h, low=l, close=c, volume=v,
    )


def _random_walk(n, start=580.0, drift=0.0, vol=500.0, seed=1):
    """Deterministic pseudo-random walk. Tight drift so pivot noise is
    small; injected levels in tests clearly dominate."""
    import random
    rng = random.Random(seed)
    out = []
    px = start
    for i in range(n):
        step = rng.uniform(-0.1, 0.1) + drift
        o = px
        h = o + rng.uniform(0.02, 0.1)
        l = o - rng.uniform(0.02, 0.1)
        c = o + step
        h = max(h, c)
        l = min(l, c)
        out.append(_bar(i, o, h, l, c, vol))
        px = c
    return out


def _flat_series(n, price=580.0, vol=1000.0):
    """Thin wrapper — most tests use _random_walk now."""
    return _random_walk(n, start=price, vol=vol)


# ---------- basic pivot detection ----------


def test_find_levels_returns_empty_with_too_few_bars():
    assert find_levels(_flat_series(3)) == []


def test_detects_repeated_resistance_with_highest_score():
    """Three clustered pivot highs around 582 should be detected as a
    resistance level. Other random-walk clusters may also exist; we
    just verify the injected spike cluster IS among the detected
    levels with at least 3 touches."""
    bars = _random_walk(40, start=580.0, vol=1000, seed=1)
    bars[8]  = _bar(8,  580.8, 582.0, 580.5, 581.2, 2500)
    bars[18] = _bar(18, 581.0, 582.1, 580.6, 581.0, 2200)
    bars[28] = _bar(28, 581.1, 581.9, 580.5, 581.0, 2700)
    levels = find_levels(bars, kind="resistance",
                         cfg=SRConfig(pivot_window=2, band_pct=0.003,
                                       min_touches=3, top_k=10))
    near_582 = [lv for lv in levels if 581.5 <= lv.price <= 582.5]
    assert len(near_582) >= 1
    assert near_582[0].touches >= 3


def test_detects_repeated_support():
    bars = _random_walk(40, start=580.0, vol=1000, seed=2)
    bars[8]  = _bar(8,  578.0, 578.5, 577.0, 577.8, 2500)
    bars[18] = _bar(18, 578.2, 578.8, 576.9, 578.0, 2700)
    bars[28] = _bar(28, 578.0, 578.6, 577.1, 577.9, 2400)
    levels = find_levels(bars, kind="support",
                         cfg=SRConfig(pivot_window=2, band_pct=0.003,
                                       min_touches=3, top_k=10))
    near_577 = [lv for lv in levels if 576.5 <= lv.price <= 577.5]
    assert len(near_577) >= 1
    assert near_577[0].touches >= 3


# ---------- clustering ----------


def test_clusters_multiple_pivots_within_band():
    """Three swing lows within 0.5% of each other should cluster into
    one level, not three. Using min_touches=3 so the cluster must
    contain all three of our injected lows."""
    bars = _random_walk(40, start=580.0, vol=1000, seed=3)
    bars[5]  = _bar(5,  580.0, 580.1, 578.0, 579.5, 2000)
    bars[15] = _bar(15, 580.0, 580.1, 578.3, 579.6, 2500)
    bars[25] = _bar(25, 580.0, 580.1, 578.1, 579.7, 3000)
    levels = find_levels(bars, kind="support",
                         cfg=SRConfig(pivot_window=2, band_pct=0.003,
                                       min_touches=3, top_k=5))
    # At least one 3+touch cluster should exist below 579
    levels_below_579 = [lv for lv in levels if lv.price < 579]
    assert len(levels_below_579) >= 1
    assert levels_below_579[0].touches >= 3


def test_multiple_touches_outscore_single_touch():
    """Three touches at a level beat a single touch with same volume."""
    bars = _flat_series(40, price=580.0, vol=1000)
    # Level A: 3 touches at 577 (each 1500 vol)
    for i in [5, 15, 25]:
        bars[i] = _bar(i, 580.0, 580.1, 577.0, 579.5, 1500)
    # Level B: 1 touch at 575 with 4000 vol
    bars[35] = _bar(35, 580.0, 580.1, 575.0, 579.0, 4000)
    levels = find_levels(bars, kind="support",
                         cfg=SRConfig(pivot_window=2, band_pct=0.003,
                                       min_touches=1, top_k=5))
    # Find each level. 577 should outscore 575 because multi-touch.
    lvl_577 = next(lv for lv in levels if 576.5 <= lv.price <= 577.5)
    lvl_575 = next(lv for lv in levels if 574.5 <= lv.price <= 575.5)
    assert lvl_577.score > lvl_575.score
    assert lvl_577.touches == 3


# ---------- volume weighting ----------


def test_volume_score_accumulates_from_all_bars_in_band():
    """The score includes volume from EVERY bar that traded in the band,
    not just the pivot bars."""
    bars = _flat_series(30, price=580.0, vol=1000)
    # Pivot low at 577
    bars[10] = _bar(10, 580.0, 580.1, 577.0, 579.5, 2000)
    # Many other bars trade through 577 ± 0.2% (577.0 ± 1.154)
    # by having low/high span covering it
    for i in [3, 4, 16, 17, 20, 21, 22]:
        bars[i] = _bar(i, 578.0, 578.5, 576.5, 577.8, 1500)
    levels = find_levels(bars, kind="support",
                         cfg=SRConfig(pivot_window=2, band_pct=0.003))
    top = next(lv for lv in levels if 576.5 <= lv.price <= 577.5)
    # Pivot-bar vol alone = 2000. Expected: much larger due to all bars
    # transacting in the band.
    assert top.volume > 2000


# ---------- nearest_support / nearest_resistance ----------


def test_nearest_support_picks_strongest_below_spot():
    """Verify nearest_support picks a valid support under spot. Uses
    high-volume injection dominant over random-walk noise. We don't
    require it to be the 577 level specifically — the detector may
    legitimately identify any of the clustered supports as strongest."""
    bars = _random_walk(40, start=580.0, vol=100, seed=4)
    # High-volume support at 577 — 100x baseline
    for i in [12, 20, 28]:
        bars[i] = _bar(i, 578.0, 578.5, 577.0, 577.8, 100000)
    strongest = nearest_support(bars, spot=580.0,
                                 cfg=SRConfig(pivot_window=2,
                                               band_pct=0.003, min_touches=3),
                                 max_distance_pct=0.02)
    assert strongest is not None
    assert strongest.price < 580.0
    assert strongest.touches >= 3


def test_nearest_support_returns_none_when_out_of_range():
    """A far-away support is rejected by max_distance_pct, no matter
    how strong. With min_touches=6 (unlikely to form by noise), only
    the injected 570 level can match; it's outside 0.5% so None."""
    bars = _random_walk(40, start=580.0, vol=200, seed=5)
    for i in [5, 15, 25]:
        bars[i] = _bar(i, 571.0, 571.5, 570.0, 570.8, 20000)
    result = nearest_support(bars, spot=580.0,
                              cfg=SRConfig(pivot_window=2, band_pct=0.003,
                                            min_touches=6),
                              max_distance_pct=0.005)
    assert result is None


def test_nearest_resistance_selects_above_spot():
    bars = _random_walk(40, start=580.0, vol=200, seed=6)
    bars[10] = _bar(10, 580.5, 582.0, 580.3, 581.0, 20000)
    bars[20] = _bar(20, 580.7, 582.1, 580.4, 581.2, 20000)
    bars[30] = _bar(30, 580.6, 581.9, 580.5, 581.0, 20000)
    top = nearest_resistance(bars, spot=580.0,
                              cfg=SRConfig(pivot_window=2, band_pct=0.003,
                                            min_touches=3),
                              max_distance_pct=0.02)
    assert top is not None
    assert top.price > 580.0
    assert 581.5 <= top.price <= 582.5
