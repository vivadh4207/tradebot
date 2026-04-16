"""Compute auto profit target and stop-loss PRICES (per position, at entry)."""
from __future__ import annotations

from typing import Tuple

from ..core.types import Position


def compute_auto_stops(pos: Position, is_short_dte: bool,
                       pt_short_pct: float = 0.35, pt_multi_pct: float = 0.50,
                       sl_short_pct: float = 0.20, sl_multi_pct: float = 0.30
                       ) -> Tuple[float, float]:
    """Returns (profit_target_price, stop_loss_price). Direction-aware."""
    pt_pct = pt_short_pct if is_short_dte else pt_multi_pct
    sl_pct = sl_short_pct if is_short_dte else sl_multi_pct
    entry = pos.avg_price
    if pos.is_long:
        pt = entry * (1 + pt_pct)
        sl = entry * (1 - sl_pct)
    else:
        pt = entry * (1 - pt_pct)
        sl = entry * (1 + sl_pct)
    return pt, sl
