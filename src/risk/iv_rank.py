"""IV rank: (current - 52w_low) / (52w_high - 52w_low)."""
from __future__ import annotations

from typing import Sequence


def iv_rank(current_iv: float, iv_52w_low: float, iv_52w_high: float) -> float:
    rng = max(iv_52w_high - iv_52w_low, 1e-9)
    rank = (current_iv - iv_52w_low) / rng
    return max(0.0, min(1.0, rank))
