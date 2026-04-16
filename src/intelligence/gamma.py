"""Gamma exposure (GEX) regime classifier.

Approximation for retail: treat aggregate dealer gamma by summing open-interest
weighted gammas across the near chain and classify positive vs. negative GEX.

- Positive GEX: dealers long gamma → they sell rallies / buy dips → compressive
- Negative GEX: dealers short gamma → they chase direction → expansive (volatile)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from datetime import date

from ..core.types import OptionContract
from ..math_tools.pricer import bs_greeks


@dataclass
class GammaRegime:
    gex: float                # dealer gamma exposure (model-dependent units)
    label: str                # 'positive' | 'negative' | 'neutral'
    flip_distance: float = 0.0  # |spot - gamma-flip| / spot

    def against_regime(self, direction: str) -> bool:
        """'against' means we're fighting the dealer flow."""
        if self.label == "positive" and direction in {"bullish", "bearish"}:
            # compressive — trend trades are fighting it
            return True
        return False


def compute_gex(contracts: List[OptionContract], spot: float,
                r: float = 0.045, q: float = 0.015,
                today=None) -> GammaRegime:
    today = today or date.today()
    total_gamma = 0.0
    for c in contracts:
        T = max((c.expiry - today).days, 0) / 365.0 or 1e-4
        sigma = c.iv if c.iv > 0 else 0.25
        g = bs_greeks(spot, c.strike, T, r, sigma, q, c.right.value)
        # Dealer assumed short calls, long puts (standard simplification).
        sign = -1 if c.right.value == "call" else +1
        total_gamma += sign * g["gamma"] * c.open_interest * c.multiplier
    label = "positive" if total_gamma > 1e3 else ("negative" if total_gamma < -1e3 else "neutral")
    return GammaRegime(gex=total_gamma, label=label)
