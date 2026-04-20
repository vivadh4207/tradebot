"""Stochastic cost model — realistic slippage for paper / backtest fills.

Replaces the fixed-bps slippage with a model that combines:
  1. Bid-ask half-spread (always pay at least this when crossing)
  2. Size impact: slippage grows with order_qty / displayed_size
  3. Volatility impact: slippage widens under stress (scales with VIX)
  4. Quote-spread-aware multiplier: wide quotes → wider realized spread

Philosophy: slippage on a backtest should NEVER be more optimistic than
your live fills will show. If anything, err pessimistic here.

References:
  - Almgren-Chriss (2000) "Optimal execution" — square-root market impact
  - Kissell-Glantz (2003) "Optimal trading strategies"
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from ..core.types import Order, Quote, Side


@dataclass
class FillCost:
    executed_price: float
    slippage_bps: float                 # vs. mid, signed positive-bad
    components: dict                    # audit trail


@dataclass
class MarketContext:
    """What the slippage model needs to know about current conditions."""
    bid: float
    ask: float
    bid_size: float = 1000.0            # displayed size
    ask_size: float = 1000.0
    vix: float = 15.0                   # current VIX level
    recent_spread_pct: float = 0.0002   # rolling-avg spread as % of mid

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return max(self.bid, self.ask)


class StochasticCostModel:
    """Callable that takes (order, market_ctx) → FillCost.

    Tunable constants are defaults; override in __init__ to calibrate from
    your paper-trade realized slippage.
    """

    def __init__(self,
                 base_half_spread_mult: float = 1.0,
                 size_impact_coef: float = 0.25,   # √ market-impact coefficient
                 vix_impact_coef: float = 0.015,   # bps per VIX point above 15
                 stress_spread_mult: float = 2.0,  # how wide spreads go in VIX > 30
                 random_noise_bps: float = 0.5,    # ± noise to simulate queue
                 min_slippage_bps: float = 0.5,    # floor; always crossing ≥ 0.5bp
                 seed: Optional[int] = None):
        self.base_half_spread_mult = float(base_half_spread_mult)
        self.size_impact_coef = float(size_impact_coef)
        self.vix_impact_coef = float(vix_impact_coef)
        self.stress_spread_mult = float(stress_spread_mult)
        self.random_noise_bps = float(random_noise_bps)
        self.min_slippage_bps = float(min_slippage_bps)
        self._rng = random.Random(seed)

    def fill(self, order: Order, ctx: MarketContext) -> FillCost:
        """Compute a realistic fill price + slippage breakdown."""
        if order.limit_price is None or order.limit_price <= 0:
            raise ValueError("Order must have a positive limit_price")
        mid = ctx.mid if ctx.mid > 0 else float(order.limit_price)
        spread = max(ctx.ask - ctx.bid, mid * 0.0001) if ctx.ask > ctx.bid else mid * 0.0005

        # 1. Half-spread cost (we ALWAYS cross by at least this much when taking liquidity)
        half_spread_bps = (spread / 2.0) / mid * 10_000.0 * self.base_half_spread_mult

        # 2. Size impact: √(qty / displayed_size) × coef, in bps
        displayed = max(ctx.ask_size if order.side == Side.BUY else ctx.bid_size, 1.0)
        size_ratio = order.qty / displayed
        size_impact_bps = self.size_impact_coef * math.sqrt(max(size_ratio, 0.0)) * 10.0

        # 3. Volatility stress: extra slippage proportional to (VIX - 15)+
        vix_excess = max(ctx.vix - 15.0, 0.0)
        vix_impact_bps = vix_excess * self.vix_impact_coef * 10_000.0 / 100.0
        # Also widen if the recent spread has been unusually wide
        if ctx.recent_spread_pct > 0:
            stress_mult = min(self.stress_spread_mult,
                               1.0 + (ctx.recent_spread_pct / 0.0002 - 1.0) * 0.5)
            size_impact_bps *= stress_mult

        # 4. Random queue noise
        noise_bps = self._rng.uniform(-self.random_noise_bps, self.random_noise_bps)

        total_bps = half_spread_bps + size_impact_bps + vix_impact_bps + noise_bps
        total_bps = max(total_bps, self.min_slippage_bps)

        # Translate bps to absolute price; sign depends on side
        slip = mid * total_bps / 10_000.0
        executed = mid + slip if order.side == Side.BUY else mid - slip

        return FillCost(
            executed_price=float(executed),
            slippage_bps=float(total_bps),
            components={
                "mid": round(mid, 6),
                "half_spread_bps": round(half_spread_bps, 3),
                "size_impact_bps": round(size_impact_bps, 3),
                "vix_impact_bps": round(vix_impact_bps, 3),
                "noise_bps": round(noise_bps, 3),
            },
        )


class LinearCostModel:
    """Legacy fixed-bps model; kept for backward compatibility."""

    def __init__(self, slippage_bps: float = 2.0):
        self.slippage_bps = float(slippage_bps)

    def fill(self, order: Order, ctx: MarketContext) -> FillCost:
        mid = ctx.mid if ctx.mid > 0 else float(order.limit_price or 0)
        slip = mid * self.slippage_bps / 10_000.0
        executed = (order.limit_price or mid) + (slip if order.side == Side.BUY else -slip)
        return FillCost(
            executed_price=float(executed),
            slippage_bps=self.slippage_bps,
            components={"mid": mid, "fixed_bps": self.slippage_bps},
        )
