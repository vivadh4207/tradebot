"""Drawdown-based leverage reduction.

Tiered guard that scales all position sizes (and in extreme cases halts
entries) based on month-to-date and peak-to-trough drawdown. Keeps you
from doubling down while a losing strategy is actively bleeding.

Default tiers (percent drawdown from equity peak):
  0% ... 5%  →  multiplier 1.00 (no scale-down)
  5% ... 8%  →  multiplier 0.75
  8% ... 12% →  multiplier 0.50
  12%+       →  multiplier 0.00 (halt new entries for the day)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DrawdownState:
    current_drawdown_pct: float         # positive number = drawdown magnitude
    peak_equity: float
    size_multiplier: float              # to apply to sizer output
    halted: bool
    reason: str


class DrawdownGuard:
    """Stateless: pass current equity + equity history; returns the state.

    Stateful alternative: caller stores `peak_equity` between calls.
    """

    DEFAULT_TIERS: List[Tuple[float, float]] = [
        # (drawdown_threshold, size_multiplier)
        (0.05, 1.00),
        (0.08, 0.75),
        (0.12, 0.50),
        (float("inf"), 0.00),    # halt
    ]

    def __init__(self, tiers: Optional[List[Tuple[float, float]]] = None):
        self.tiers = sorted(tiers or self.DEFAULT_TIERS, key=lambda x: x[0])

    def evaluate(self, current_equity: float,
                 peak_equity: float) -> DrawdownState:
        peak = max(peak_equity, current_equity, 1e-9)
        dd_pct = max(0.0, (peak - current_equity) / peak)
        mult = 1.0
        halted = False
        reason = f"dd={dd_pct:.3%}"
        for threshold, multiplier in self.tiers:
            if dd_pct <= threshold:
                mult = multiplier
                break
        if mult <= 0.0:
            halted = True
            reason = f"halt:dd={dd_pct:.3%}"
        elif mult < 1.0:
            reason = f"scale={mult:.2f}:dd={dd_pct:.3%}"
        return DrawdownState(
            current_drawdown_pct=dd_pct, peak_equity=peak,
            size_multiplier=mult, halted=halted, reason=reason,
        )

    @staticmethod
    def peak_from_series(equity_curve) -> float:
        if not equity_curve:
            return 0.0
        return max(float(x) for x in equity_curve)
