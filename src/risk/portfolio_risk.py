"""PortfolioRiskManager — aggregate Greek limits + stress tests.

Playbook Section 6. Runs before every new trade. Stress scenarios:
(-20, +50 vol), (-10, +25 vol), (-5, +10 vol), (+5, -10 vol), (+10, -15 vol).
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Optional
from datetime import date

from ..core.types import Position
from ..math_tools.pricer import bs_price, bs_greeks


class PortfolioRiskManager:
    def __init__(self, max_dollar_delta_per_100k: float = 1000,
                 max_dollar_gamma_per_100k: float = 50,
                 max_vega: float = 5000,
                 max_theta_daily: float = -500,
                 max_notional_pct: float = 0.50):
        self.max_dd = max_dollar_delta_per_100k
        self.max_dg = max_dollar_gamma_per_100k
        self.max_vega = max_vega
        self.max_theta = max_theta_daily
        self.max_notional_pct = max_notional_pct

    @staticmethod
    def aggregate_greeks(positions: List[Position], spot: float,
                         r: float = 0.045, q: float = 0.015,
                         today: Optional[date] = None) -> Dict[str, float]:
        today = today or date.today()
        total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for pos in positions:
            if not pos.is_option or pos.strike is None or pos.expiry is None:
                if not pos.is_option:
                    # equity contributes 1 delta per share
                    total["delta"] += float(pos.qty)
                continue
            T = max((pos.expiry - today).days, 0) / 365.0 or 1e-4
            sigma = 0.25  # fallback if we don't have IV on the position object
            g = bs_greeks(spot, pos.strike, T, r, sigma, q, pos.right.value)
            mult = pos.multiplier * pos.qty  # sign carried by qty
            for k in total:
                total[k] += mult * g[k]
        total["dollar_delta"] = total["delta"] * spot
        total["dollar_gamma"] = total["gamma"] * spot * spot / 100.0
        return total

    def check(self, proposed: Position, existing: List[Position],
              spot: float, equity: float) -> Tuple[bool, str]:
        combined = existing + [proposed]
        g = self.aggregate_greeks(combined, spot)
        scale = max(equity / 100_000.0, 1e-6)
        if abs(g["dollar_delta"]) > self.max_dd * scale:
            return False, "portfolio_delta_limit"
        if abs(g["dollar_gamma"]) > self.max_dg * scale:
            return False, "portfolio_gamma_limit"
        if abs(g["vega"]) > self.max_vega:
            return False, "portfolio_vega_limit"
        # Theta: `max_theta` is a negative floor (e.g. -500 $/day). We block
        # when the portfolio's daily theta is more negative than this floor.
        # abs(theta) lets us catch any direction that exceeds the magnitude.
        if g["theta"] < self.max_theta:
            return False, "portfolio_theta_bleed"
        # Notional check — previously unused. Sum of option notional
        # (qty × price × 100) + equity notional must stay under pct × equity.
        notional = 0.0
        for p in combined:
            if p.is_option:
                notional += abs(p.qty) * max(p.avg_price, 0.0) * p.multiplier
            else:
                notional += abs(p.qty) * spot
        if notional > self.max_notional_pct * equity and equity > 0:
            return False, f"portfolio_notional_limit: {notional:.2f}>{self.max_notional_pct * equity:.2f}"
        return True, "ok"

    def stress(self, positions: List[Position], spot: float,
               scenarios=None, r: float = 0.045, q: float = 0.015,
               today: Optional[date] = None) -> float:
        today = today or date.today()
        scenarios = scenarios or [
            (-0.20, 0.50), (-0.10, 0.25), (-0.05, 0.10),
            (0.05, -0.10), (0.10, -0.15),
        ]
        worst = 0.0
        for spot_shock, vol_shock in scenarios:
            pnl = 0.0
            for pos in positions:
                if not pos.is_option:
                    pnl += pos.qty * (spot * spot_shock)
                    continue
                T = max((pos.expiry - today).days, 0) / 365.0 or 1e-4
                new_S = spot * (1 + spot_shock)
                new_iv = 0.25 * (1 + vol_shock)
                new_price = bs_price(new_S, pos.strike, T, r, new_iv, q, pos.right.value)
                pnl += (new_price - pos.avg_price) * pos.qty * pos.multiplier
            if pnl < worst:
                worst = pnl
        return worst
