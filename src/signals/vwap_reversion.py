"""VWAP mean-reversion signal: >1% from VWAP → fade."""
from __future__ import annotations

from typing import Optional

from ..core.types import Signal, Side, OptionRight
from .base import SignalSource, SignalContext


class VwapReversionSignal(SignalSource):
    name = "vwap_reversion"

    def __init__(self, trigger_pct: float = 0.01):
        self.trigger_pct = trigger_pct

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        if ctx.vwap <= 0 or ctx.spot <= 0:
            return None
        diff = (ctx.spot - ctx.vwap) / ctx.vwap
        if diff > self.trigger_pct:
            return Signal(source=self.name, symbol=ctx.symbol,
                          side=Side.BUY, option_right=OptionRight.PUT,
                          confidence=min(1.0, 0.55 + abs(diff) * 10),
                          rationale=f"extended_above_vwap:{diff:.4f}",
                          meta={"direction": "bearish",
                                "entry_tag": "vwap_reversion"})
        if diff < -self.trigger_pct:
            return Signal(source=self.name, symbol=ctx.symbol,
                          side=Side.BUY, option_right=OptionRight.CALL,
                          confidence=min(1.0, 0.55 + abs(diff) * 10),
                          rationale=f"extended_below_vwap:{diff:.4f}",
                          meta={"direction": "bullish",
                                "entry_tag": "vwap_reversion"})
        return None
