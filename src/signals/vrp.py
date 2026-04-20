"""Variance Risk Premium (VRP) signal — playbook Section 3.

Sell short-dated OTM puts when VRP z-score > 0.5. Skip when VRP is thin or
negative.
"""
from __future__ import annotations

from typing import Optional

from ..core.types import Signal, Side, OptionRight
from ..data.options_chain import SyntheticOptionsChain
from .base import SignalSource, SignalContext


class VRPSignal(SignalSource):
    name = "vrp"

    def __init__(self, z_threshold: float = 0.5, otm_pct: float = 0.05,
                 target_dte: int = 1):
        self.z_threshold = z_threshold
        self.otm_pct = otm_pct
        self.target_dte = target_dte

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        vrp = ctx.atm_iv_30d - ctx.rv_20d
        # approximate z score with a prior; real system uses rolling series
        # (provided by intelligence layer). Here the prior is 0 mean / 0.05 std.
        z = vrp / 0.05
        if z < self.z_threshold:
            return None
        # Find a suitable OTM put from the chain if available
        right = OptionRight.PUT
        strike = None
        expiry = None
        if ctx.chain:
            target_strike = ctx.spot * (1 - self.otm_pct)
            candidates = [c for c in ctx.chain if c.right == right]
            if candidates:
                best = min(candidates, key=lambda c: abs(c.strike - target_strike))
                strike = best.strike
                expiry = best.expiry
        return Signal(source=self.name, symbol=ctx.symbol,
                      side=Side.SELL,                     # sell-to-open short put
                      option_right=right, strike=strike, expiry=expiry,
                      confidence=min(1.0, 0.55 + z * 0.1),
                      rationale=f"vrp_z={z:.2f}",
                      meta={"direction": "premium_harvest",
                            "premium_action": "sell"})
