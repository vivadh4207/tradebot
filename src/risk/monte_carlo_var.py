"""Monte Carlo Value-at-Risk and Conditional VaR (Expected Shortfall).

Simulates 10k GBM paths for each position's underlying over the VaR
horizon (default 1 day). For option positions, re-prices at terminal
spots using BS; for equities uses the spot itself.

VaR_α   = -quantile(terminal_pnl, 1-α)        e.g. α=0.95 → 5% tail
CVaR_α  = -mean(terminal_pnl | terminal_pnl ≤ -VaR_α)

CVaR is the number your risk committee actually cares about — it measures
what you lose in the tail, not just where the tail starts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from ..core.types import Position
from ..math_tools.pricer import bs_price


@dataclass
class VaRReport:
    n_paths: int
    horizon_days: float
    var_95: float             # positive dollar loss at 95% confidence
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_pnl: float
    pnl_stdev: float
    best_case: float
    worst_case: float
    per_position: Dict[str, Dict[str, float]]


def _gbm_terminal(S0: float, mu: float, sigma: float, T: float,
                   n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """One-step GBM terminal spots."""
    z = rng.standard_normal(n_paths)
    drift = (mu - 0.5 * sigma * sigma) * T
    diff = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diff)


def monte_carlo_var(
    positions: List[Position],
    spots: Dict[str, float],
    vols: Dict[str, float],
    *,
    horizon_days: float = 1.0,
    n_paths: int = 10_000,
    r: float = 0.045,
    q_by_symbol: Optional[Dict[str, float]] = None,
    today: Optional[date] = None,
    seed: Optional[int] = 42,
) -> VaRReport:
    """Compute 1-day 95/99 VaR + CVaR for a book of positions.

    `spots`: symbol → current price (underlying for options, spot for equities)
    `vols`:  symbol → annualized volatility
    `q_by_symbol`: optional per-symbol dividend yield (defaults 0.0)
    """
    today = today or date.today()
    rng = np.random.default_rng(seed)
    T = horizon_days / 252.0
    q_by_symbol = q_by_symbol or {}

    # Bucket positions by underlying
    by_underlying: Dict[str, List[Position]] = {}
    for pos in positions:
        u = pos.underlying or pos.symbol
        by_underlying.setdefault(u, []).append(pos)

    total_pnl_paths = np.zeros(n_paths, dtype=np.float64)
    per_pos: Dict[str, Dict[str, float]] = {}

    for underlying, group in by_underlying.items():
        S0 = float(spots.get(underlying, 0.0))
        sigma = float(vols.get(underlying, 0.20))
        if S0 <= 0 or sigma <= 0:
            continue
        # Simulate terminal spots for this underlying ONCE, reuse across positions
        ST = _gbm_terminal(S0, mu=r, sigma=sigma, T=T, n_paths=n_paths, rng=rng)

        for pos in group:
            q = float(q_by_symbol.get(underlying, 0.0))
            if pos.is_option and pos.strike is not None and pos.expiry is not None:
                dte = max((pos.expiry - today).days, 0)
                T_remaining = max(dte / 365.0 - T, 1e-4)
                # Re-price at ST
                right = pos.right.value if pos.right else "call"
                new_prices = np.array([
                    bs_price(float(s), pos.strike, T_remaining, r, sigma, q, right)
                    for s in ST
                ], dtype=np.float64)
                pnl = (new_prices - pos.avg_price) * pos.qty * pos.multiplier
            else:
                # Equity: P&L = qty * (ST - avg_price)
                pnl = (ST - pos.avg_price) * pos.qty
            total_pnl_paths += pnl
            per_pos[pos.symbol] = {
                "mean_pnl": float(pnl.mean()),
                "std_pnl": float(pnl.std(ddof=1)),
                "worst": float(pnl.min()),
                "best": float(pnl.max()),
            }

    if total_pnl_paths.size == 0:
        return VaRReport(0, horizon_days, 0, 0, 0, 0, 0, 0, 0, 0, {})

    losses = -total_pnl_paths   # positive = loss
    # VaR at α = loss at (1-α) tail
    var_95 = float(np.quantile(losses, 0.95))
    var_99 = float(np.quantile(losses, 0.99))
    # CVaR = mean loss conditional on exceeding VaR
    tail_95 = losses[losses >= var_95]
    tail_99 = losses[losses >= var_99]
    cvar_95 = float(tail_95.mean()) if tail_95.size else var_95
    cvar_99 = float(tail_99.mean()) if tail_99.size else var_99

    return VaRReport(
        n_paths=n_paths, horizon_days=horizon_days,
        var_95=var_95, var_99=var_99, cvar_95=cvar_95, cvar_99=cvar_99,
        expected_pnl=float(total_pnl_paths.mean()),
        pnl_stdev=float(total_pnl_paths.std(ddof=1)),
        best_case=float(total_pnl_paths.max()),
        worst_case=float(total_pnl_paths.min()),
        per_position=per_pos,
    )
