"""Structured P&L attribution (delta / gamma / vega / theta / residual).

Between two time points t0 and t1, decompose the P&L of an option
position into first- and second-order Greek contributions:

    ΔPnL ≈ Δ·ΔS + 0.5·Γ·ΔS² + Vega·Δσ + Θ·Δt + residual

The residual captures higher-order terms, vol-of-vol, skew effects, and
any mispricing vs. BS. On a well-calibrated desk, residual should be <
20% of |ΔPnL|. Larger residuals signal model error (you're trading
something BS doesn't capture — skew, early exercise, etc.).

Useful as:
  - daily attribution report per position
  - diagnostic when a trade P&L doesn't match expectations
  - input to risk decomposition at the portfolio level
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, List, Optional

from ..core.types import Position
from ..math_tools.pricer import bs_greeks, bs_price


@dataclass
class PnLAttributionReport:
    symbol: str
    total_pnl: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    residual_pnl: float
    residual_pct_of_total: float
    notes: str = ""

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def attribute_pnl(
    pos: Position,
    *,
    S_t0: float, S_t1: float,
    sigma_t0: float, sigma_t1: float,
    T_t0: float, T_t1: float,
    r: float = 0.045, q: float = 0.015,
) -> PnLAttributionReport:
    """Attribute one position's P&L between t0 and t1.

    `T_t0` and `T_t1` are years-to-expiry at each timestamp; for a day
    between two points, T_t1 ≈ T_t0 - 1/365.

    For equity positions this reduces to delta·ΔS (trivially); we still
    return the report structure so the caller can aggregate uniformly.
    """
    if not pos.is_option or pos.strike is None or pos.expiry is None:
        # Equity position — only has delta
        pnl = float(pos.qty) * (S_t1 - S_t0) * pos.multiplier
        return PnLAttributionReport(
            symbol=pos.symbol, total_pnl=pnl,
            delta_pnl=pnl, gamma_pnl=0.0, vega_pnl=0.0,
            theta_pnl=0.0, residual_pnl=0.0,
            residual_pct_of_total=0.0, notes="equity",
        )

    right = pos.right.value if pos.right else "call"
    sign_mult = pos.qty * pos.multiplier  # signed by position direction

    # Actual P&L from BS re-price
    px_t0 = bs_price(S_t0, pos.strike, T_t0, r, sigma_t0, q, right)
    px_t1 = bs_price(S_t1, pos.strike, T_t1, r, sigma_t1, q, right)
    total_pnl = (px_t1 - px_t0) * sign_mult

    # Greeks at t0 (the "starting book" view — standard attribution convention)
    g = bs_greeks(S_t0, pos.strike, T_t0, r, sigma_t0, q, right)

    dS = S_t1 - S_t0
    dSigma = sigma_t1 - sigma_t0
    dT_days = (T_t0 - T_t1) * 365.0  # positive = calendar days elapsed

    # Contributions per Greek (remember: theta in bs_greeks is daily; vega per 1% vol)
    delta_pnl = g["delta"] * dS * sign_mult
    gamma_pnl = 0.5 * g["gamma"] * (dS * dS) * sign_mult
    vega_pnl = g["vega"] * (dSigma * 100.0) * sign_mult      # vega is per 1% vol
    theta_pnl = g["theta"] * dT_days * sign_mult
    explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    residual = total_pnl - explained

    return PnLAttributionReport(
        symbol=pos.symbol, total_pnl=float(total_pnl),
        delta_pnl=float(delta_pnl), gamma_pnl=float(gamma_pnl),
        vega_pnl=float(vega_pnl), theta_pnl=float(theta_pnl),
        residual_pnl=float(residual),
        residual_pct_of_total=(
            abs(residual) / abs(total_pnl) if abs(total_pnl) > 1e-9 else 0.0
        ),
        notes=f"dS={dS:.2f} dSigma={dSigma:+.4f} dT_d={dT_days:.3f}",
    )


def attribute_book(
    positions: List[Position],
    snapshots: Dict[str, Dict[str, float]],
    *,
    r: float = 0.045, q_by_symbol: Optional[Dict[str, float]] = None,
) -> List[PnLAttributionReport]:
    """Attribute P&L for every position. `snapshots[symbol]` must have:
      S_t0, S_t1, sigma_t0, sigma_t1, T_t0, T_t1
    """
    out = []
    q_by_symbol = q_by_symbol or {}
    for pos in positions:
        snap = snapshots.get(pos.symbol)
        if snap is None:
            continue
        q = q_by_symbol.get(pos.underlying or pos.symbol, 0.0)
        out.append(attribute_pnl(pos, r=r, q=q, **snap))
    return out
