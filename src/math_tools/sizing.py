"""Position sizing: Kelly-lite, VIX-regime, Hybrid.

Drawn from playbook Section 5. We default to QUARTER Kelly, hard-capped at
5% of equity per trade, because win-rate estimates are noisy in retail-scale
options trading.
"""
from __future__ import annotations

from typing import Optional
import math


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                   fraction: float = 0.25, hard_cap: float = 0.05) -> float:
    """f* = W - (1-W)/B where B = avg_win/avg_loss.

    Returns the FRACTION of equity to risk. Fractional Kelly applied to damp
    noise; hard-capped for survival.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    if not (0.0 <= win_rate <= 1.0):
        return 0.0
    b = avg_win / avg_loss
    f = win_rate - (1.0 - win_rate) / b
    return max(0.0, min(f * fraction, hard_cap))


def vix_regime_multiplier(vix_today: float, vix_52w_low: float, vix_52w_high: float,
                          low: float = 0.5, high: float = 1.5) -> float:
    """Linear scaling: low@VIX_min to high@VIX_max. Clipped to [low, high]."""
    rng = max(vix_52w_high - vix_52w_low, 1e-9)
    rank = (vix_today - vix_52w_low) / rng
    rank = max(0.0, min(1.0, rank))
    return low + (high - low) * rank


def hybrid_sizing(equity: float, max_loss_per_contract: float,
                  win_rate_est: float, avg_win: float, avg_loss: float,
                  vix_today: float, vix_52w_low: float, vix_52w_high: float,
                  vrp_zscore: float = 0.0,
                  kelly_fraction_cap: float = 0.25,
                  kelly_hard_cap: float = 0.05,
                  max_contracts: int = 10) -> int:
    """Kelly × VIX-regime × VRP multiplier, return integer contract count.

    Research: Arxiv 2508.16598 (2025) shows Hybrid Kelly+VIX outperforms
    either alone in put-writing on index options.
    """
    if max_loss_per_contract <= 0:
        return 0
    kelly_f = kelly_fraction(win_rate_est, avg_win, avg_loss,
                              fraction=kelly_fraction_cap,
                              hard_cap=kelly_hard_cap)
    vix_mult = vix_regime_multiplier(vix_today, vix_52w_low, vix_52w_high)
    vrp_mult = max(0.0, min(1.5, 1.0 + vrp_zscore * 0.3))
    capital_at_risk = equity * kelly_f * vix_mult * vrp_mult
    n = int(math.floor(capital_at_risk / max_loss_per_contract))
    return max(0, min(n, max_contracts))
