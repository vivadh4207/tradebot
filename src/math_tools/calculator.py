"""Trade-math calculator: risk/reward, EV, POP, breakevens.

This is the 'calculator' the bot references before entering a trade. Every
candidate trade must satisfy: R:R >= 2, EV > 0 at estimated POP. Anyone
claiming '90-95% guaranteed profit' is selling fantasy — the math below is
the actual gate.
"""
from __future__ import annotations

from typing import Optional
import math
from scipy.stats import norm


def risk_reward_ratio(entry: float, target: float, stop: float) -> float:
    """R:R = reward / risk. Returns inf if stop == entry."""
    reward = abs(target - entry)
    risk = abs(entry - stop)
    if risk <= 0:
        return float("inf")
    return reward / risk


def expected_value(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """EV per unit risk. Pass avg_win and avg_loss as absolute numbers."""
    return win_rate * avg_win - (1.0 - win_rate) * avg_loss


def probability_of_profit(S: float, K: float, T: float, r: float, sigma: float,
                          q: float = 0.0, option_type: str = "call",
                          long_short: str = "long") -> float:
    """Under Black-Scholes, P(S_T > K) for a long call etc.

    Useful as a *prior* for POP; reality includes vol-smile, gaps, early assignment,
    transaction costs, etc. Treat as an estimate, not a promise.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if option_type == "call":
        pop = float(norm.cdf(d2))
    else:
        pop = float(norm.cdf(-d2))
    # For a SHORT option, POP is the complement (we win if it expires OTM)
    if long_short == "short":
        pop = 1.0 - pop
    return pop


def breakeven_call(strike: float, premium: float) -> float:
    return strike + premium


def breakeven_put(strike: float, premium: float) -> float:
    return strike - premium


def contract_edge(bid: float, ask: float, fair_value: float) -> float:
    """How much the market price deviates from our fair-value estimate.

    Positive = market is rich (sellable); negative = cheap (buyable).
    """
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    if mid <= 0:
        return 0.0
    return mid - fair_value
