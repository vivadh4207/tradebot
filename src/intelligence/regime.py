"""RegimeClassifier — six-state label describing current market regime.

Dimensions combined:
  - time-of-day (overrides everything else during open/close windows)
  - volatility (VIX): low (<15), normal (15-25), high (>25)
  - trend vs range: lag-1 autocorrelation of recent bar returns

Regime labels (one of six):
  OPENING         — first hour of the session. ORB / gap logic dominates.
  CLOSING         — last 30 min. No new entries except flat-close.
  TREND_LOWVOL    — trending + quiet. Momentum + LSTM favored.
  TREND_HIGHVOL   — trending + VIX elevated. Momentum OK, smaller size.
  RANGE_LOWVOL    — choppy + quiet. VWAP-reversion + ORB favored.
  RANGE_HIGHVOL   — choppy + VIX elevated. Premium harvest (VRP/Wheel) favored.

Deliberately coarse. We want stable classifications over 10-30 minute
windows, not per-bar flickering.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Sequence

import numpy as np

from ..core.clock import ET


class Regime(str, Enum):
    OPENING = "opening"
    CLOSING = "closing"
    TREND_LOWVOL = "trend_lowvol"
    TREND_HIGHVOL = "trend_highvol"
    RANGE_LOWVOL = "range_lowvol"
    RANGE_HIGHVOL = "range_highvol"


@dataclass
class RegimeSnapshot:
    regime: Regime
    vix: float
    trend_score: float            # lag-1 autocorr of returns, [-1, 1]
    minute_of_day: int            # minutes since 09:30 ET
    rationale: str


class RegimeClassifier:
    """Pure-function classifier. Safe to call every bar; stable labels."""

    def __init__(self,
                 opening_end_min: int = 60,         # first 60 min = OPENING
                 closing_start_min: int = 330,      # 09:30 + 330 = 15:00 = CLOSING
                 vix_high_cutoff: float = 25.0,
                 trend_threshold: float = 0.15):
        self.opening_end_min = opening_end_min
        self.closing_start_min = closing_start_min
        self.vix_high_cutoff = vix_high_cutoff
        self.trend_threshold = trend_threshold

    def classify(self, *, vix: float, now: datetime,
                 recent_closes: Sequence[float]) -> RegimeSnapshot:
        now = now if now.tzinfo else now.replace(tzinfo=ET)
        now_et = now.astimezone(ET)
        mod = (now_et.hour - 9) * 60 + (now_et.minute - 30)
        mod = max(0, min(389, mod))

        # Time-of-day regimes override volatility/trend regimes.
        if mod < self.opening_end_min:
            return RegimeSnapshot(
                regime=Regime.OPENING, vix=vix, trend_score=0.0,
                minute_of_day=mod, rationale=f"mod<{self.opening_end_min}",
            )
        if mod >= self.closing_start_min:
            return RegimeSnapshot(
                regime=Regime.CLOSING, vix=vix, trend_score=0.0,
                minute_of_day=mod, rationale=f"mod>={self.closing_start_min}",
            )

        trend = self._trend_score(recent_closes)
        high_vol = vix > self.vix_high_cutoff
        is_trending = abs(trend) > self.trend_threshold

        if is_trending and high_vol:
            r = Regime.TREND_HIGHVOL
        elif is_trending:
            r = Regime.TREND_LOWVOL
        elif high_vol:
            r = Regime.RANGE_HIGHVOL
        else:
            r = Regime.RANGE_LOWVOL

        return RegimeSnapshot(
            regime=r, vix=vix, trend_score=float(trend),
            minute_of_day=mod,
            rationale=f"trend={trend:+.3f} vix={vix:.1f}",
        )

    @staticmethod
    def _trend_score(closes: Sequence[float]) -> float:
        """Lag-1 autocorrelation of log-returns. Positive → trending."""
        arr = np.asarray(closes, dtype=np.float64)
        if arr.size < 30:
            return 0.0
        rets = np.diff(np.log(arr))
        # strip zeros / NaN
        rets = rets[np.isfinite(rets)]
        if rets.size < 20:
            return 0.0
        if rets.std() < 1e-8:
            return 0.0
        a, b = rets[:-1], rets[1:]
        cov = np.mean((a - a.mean()) * (b - b.mean()))
        denom = a.std() * b.std()
        if denom < 1e-12:
            return 0.0
        return float(np.clip(cov / denom, -1.0, 1.0))
