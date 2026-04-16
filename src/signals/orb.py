"""Opening Range Breakout (ORB) scalp signal."""
from __future__ import annotations

from typing import Optional

from ..core.types import Signal, Side, OptionRight
from .base import SignalSource, SignalContext


class OpeningRangeBreakout(SignalSource):
    name = "orb"

    def __init__(self, range_minutes: int = 30):
        self.range_minutes = range_minutes

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        if ctx.opening_range_high <= 0 or ctx.opening_range_low <= 0 or ctx.spot <= 0:
            return None
        if ctx.spot > ctx.opening_range_high:
            return Signal(source=self.name, symbol=ctx.symbol,
                          side=Side.BUY, option_right=OptionRight.CALL,
                          confidence=0.7,
                          rationale=f"close>{ctx.opening_range_high:.2f}",
                          meta={"direction": "bullish",
                                "entry_tag": "scalp"})
        if ctx.spot < ctx.opening_range_low:
            return Signal(source=self.name, symbol=ctx.symbol,
                          side=Side.BUY, option_right=OptionRight.PUT,
                          confidence=0.7,
                          rationale=f"close<{ctx.opening_range_low:.2f}",
                          meta={"direction": "bearish",
                                "entry_tag": "scalp"})
        return None
