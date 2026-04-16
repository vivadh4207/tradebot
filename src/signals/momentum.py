"""5-bar slope momentum signal. Call if slope > +0.01%, put if < -0.01%."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import Signal, Side, OptionRight
from .base import SignalSource, SignalContext


class MomentumSignal(SignalSource):
    name = "momentum"

    def __init__(self, bars: int = 5, slope_long: float = 1e-4,
                 slope_short: float = -1e-4):
        self.bars_n = bars
        self.slope_long = slope_long
        self.slope_short = slope_short

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        if len(ctx.bars) < self.bars_n:
            return None
        closes = np.array([b.close for b in ctx.bars[-self.bars_n:]], dtype=float)
        x = np.arange(len(closes), dtype=float)
        # slope per bar relative to price
        coef = np.polyfit(x, closes, 1)[0]
        rel = coef / closes.mean() if closes.mean() > 0 else 0.0
        if rel > self.slope_long:
            return Signal(source=self.name, symbol=ctx.symbol,
                          side=Side.BUY, option_right=OptionRight.CALL,
                          confidence=min(1.0, 0.6 + abs(rel) * 100),
                          rationale=f"slope {rel:.6f}",
                          meta={"direction": "bullish"})
        if rel < self.slope_short:
            return Signal(source=self.name, symbol=ctx.symbol,
                          side=Side.BUY, option_right=OptionRight.PUT,
                          confidence=min(1.0, 0.6 + abs(rel) * 100),
                          rationale=f"slope {rel:.6f}",
                          meta={"direction": "bearish"})
        return None
