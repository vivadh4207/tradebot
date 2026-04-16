"""VIX state + regime classifier."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VixState:
    current: float
    pct_52w: float = 0.5    # 0..1 rank within 52w
    regime: str = "normal"


def vix_regime(v: float,
               halt_above: float = 40.0,
               no_short_premium_above: float = 30.0,
               no_0dte_long_below: float = 12.0) -> str:
    if v > halt_above:
        return "halt"
    if v > no_short_premium_above:
        return "high"
    if v < no_0dte_long_below:
        return "ultra_low"
    return "normal"
