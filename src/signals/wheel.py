"""The Wheel — sell cash-secured puts; if assigned, sell covered calls.

Pattern inspired by thetagang (brndnmtthws/thetagang). Highly systematic
premium-harvesting strategy. Requires sufficient buying power to be
assigned the underlying.
"""
from __future__ import annotations

from typing import Optional

from ..core.types import Signal, Side, OptionRight
from .base import SignalSource, SignalContext


class WheelSignal(SignalSource):
    name = "wheel"

    def __init__(self, target_delta: float = -0.30, target_dte: int = 7,
                 premium_min_pct: float = 0.004):
        self.target_delta = target_delta
        self.target_dte = target_dte
        self.premium_min_pct = premium_min_pct

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        if not ctx.chain:
            return None
        puts = [c for c in ctx.chain if c.right == OptionRight.PUT]
        if not puts:
            return None
        # approximate target by strike distance (~30-delta OTM)
        target_strike = ctx.spot * 0.97
        pick = min(puts, key=lambda c: abs(c.strike - target_strike))
        if pick.mid <= 0 or ctx.spot <= 0:
            return None
        prem_pct = pick.mid / ctx.spot
        if prem_pct < self.premium_min_pct:
            return None
        return Signal(source=self.name, symbol=ctx.symbol,
                      side=Side.SELL,
                      option_right=OptionRight.PUT,
                      strike=pick.strike, expiry=pick.expiry,
                      confidence=0.7,
                      rationale=f"wheel_csp premium_pct={prem_pct:.4f}",
                      meta={"direction": "premium_harvest",
                            "premium_action": "sell",
                            "entry_tag": "wheel_csp"})
