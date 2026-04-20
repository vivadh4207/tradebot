"""Momentum Boost — raises profit target when a clear momentum surge is
present at the target-hit moment.

Conditions (all must be true):
- Current bar volume >= 2.0x 20-bar avg
- At least 4 of the last 5 bars are green (positive close>open) for a long
  or red for a short
- Target can be raised up to cfg.hard_cap_pct (default 150%)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..core.types import Bar, Position


@dataclass
class BoostConfig:
    volume_mult_required: float = 2.0
    consecutive_bars: int = 4
    lookback_bars: int = 5
    initial_target_pct: float = 0.35
    boosted_target_pct: float = 0.60
    hard_cap_pct: float = 1.50


class MomentumBoost:
    def __init__(self, cfg: BoostConfig = BoostConfig()):
        self.cfg = cfg

    def evaluate(self, pos: Position, bars: List[Bar]) -> float:
        """Returns the (possibly-boosted) profit target pct to use NOW."""
        if len(bars) < max(20, self.cfg.lookback_bars):
            return self.cfg.initial_target_pct
        current = bars[-1]
        avg_vol = sum(b.volume for b in bars[-20:]) / 20.0
        if avg_vol <= 0:
            return self.cfg.initial_target_pct
        vol_mult = current.volume / avg_vol
        if vol_mult < self.cfg.volume_mult_required:
            return self.cfg.initial_target_pct

        last = bars[-self.cfg.lookback_bars:]
        if pos.is_long:
            greens = sum(1 for b in last if b.close > b.open)
            if greens < self.cfg.consecutive_bars:
                return self.cfg.initial_target_pct
        else:
            reds = sum(1 for b in last if b.close < b.open)
            if reds < self.cfg.consecutive_bars:
                return self.cfg.initial_target_pct

        boosted = min(self.cfg.boosted_target_pct, self.cfg.hard_cap_pct)
        return boosted
