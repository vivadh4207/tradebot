"""Market breadth: advance/decline + NH/NL ratios.

For retail we can proxy via sector ETFs or top-N names advancing vs. declining.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class MarketBreadth:
    advancers: int
    decliners: int
    new_highs: int = 0
    new_lows: int = 0

    @property
    def ad_ratio(self) -> float:
        total = self.advancers + self.decliners
        return self.advancers / total if total > 0 else 0.5

    def divergence_score(self) -> float:
        """Returns -1..+1. Negative = weak breadth / divergence, positive = broad."""
        return 2 * self.ad_ratio - 1

    @classmethod
    def from_daily_changes(cls, daily_changes: Dict[str, float]) -> "MarketBreadth":
        adv = sum(1 for v in daily_changes.values() if v > 0)
        dec = sum(1 for v in daily_changes.values() if v < 0)
        return cls(advancers=adv, decliners=dec)
