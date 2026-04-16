"""QuoteValidator — from playbook Section 7.2.

Reject crossed/stale/too-wide quotes before acting on them. This is the
single highest-ROI 'boring' component in any serious bot.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Deque

import numpy as np

from ..core.types import Quote, OptionContract


class QuoteValidator:
    def __init__(self, max_spread_pct: float = 0.10,
                 max_spread_multiplier: float = 2.5,
                 min_size: int = 1, history_len: int = 100):
        self.max_spread_pct = max_spread_pct
        self.max_spread_multiplier = max_spread_multiplier
        self.min_size = min_size
        self._spread_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=history_len)
        )

    def is_valid(self, q: Quote) -> bool:
        if q.bid <= 0 or q.ask <= 0:
            return False
        if q.bid >= q.ask:
            return False
        if q.bid_size < self.min_size or q.ask_size < self.min_size:
            return False

        mid = q.mid
        spread = q.ask - q.bid
        if mid <= 0:
            return False
        if spread / mid > self.max_spread_pct:
            return False

        hist = self._spread_history[q.symbol]
        if len(hist) > 10:
            avg = float(np.mean(hist))
            if spread > avg * self.max_spread_multiplier:
                return False
        hist.append(spread)
        return True

    def option_valid(self, c: OptionContract, max_spread_pct_override: float | None = None) -> bool:
        max_sp = max_spread_pct_override or self.max_spread_pct
        if c.bid <= 0 or c.ask <= 0 or c.bid >= c.ask:
            return False
        if c.mid <= 0:
            return False
        if c.spread_pct > max_sp:
            return False
        return True
