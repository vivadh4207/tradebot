"""Realized-volatility-scaled sizing.

A $100 risk budget on NVDA (realized vol 40%) is NOT equivalent to $100
on KO (vol 15%). This module computes per-symbol realized volatility
from a recent bar history and returns a multiplier that normalizes
exposure to a target vol per trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.types import Bar


@dataclass
class VolScaling:
    realized_vol: float        # annualized
    multiplier: float          # qty scale factor (target_vol / realized_vol, clipped)
    note: str = ""


def realized_vol_annualized(bars: Sequence[Bar],
                             lookback: int = 60,
                             periods_per_year: int = 98_280) -> float:
    """Sample-stdev of log returns, annualized. Default periods_per_year is
    for 1-minute bars during 6.5h × 252 days = 98,280 bars/yr.
    """
    n = min(len(bars), lookback)
    if n < 2:
        return 0.0
    closes = np.array([b.close for b in bars[-n:]], dtype=np.float64)
    closes = closes[closes > 0]
    if closes.size < 2:
        return 0.0
    rets = np.diff(np.log(closes))
    if rets.size < 2:
        return 0.0
    sd = float(np.std(rets, ddof=1))
    return sd * np.sqrt(periods_per_year)


def vol_scale(bars: Sequence[Bar],
               target_annual_vol: float = 0.20,
               lookback: int = 60,
               min_mult: float = 0.25,
               max_mult: float = 2.0,
               periods_per_year: int = 98_280) -> VolScaling:
    """Return a size multiplier that targets `target_annual_vol` per position.

    High-vol names get scaled DOWN (mult < 1); low-vol names scaled UP
    (mult > 1). Clipped to [min_mult, max_mult] to avoid extreme positions.
    """
    rv = realized_vol_annualized(bars, lookback=lookback,
                                   periods_per_year=periods_per_year)
    if rv <= 0:
        return VolScaling(realized_vol=0.0, multiplier=1.0,
                           note="insufficient_data")
    raw = target_annual_vol / rv
    clipped = float(np.clip(raw, min_mult, max_mult))
    return VolScaling(
        realized_vol=rv,
        multiplier=clipped,
        note=f"rv={rv:.3f} target={target_annual_vol:.3f} raw={raw:.3f}",
    )
