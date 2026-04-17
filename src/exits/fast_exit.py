"""Fast exit evaluator — runs every 5 seconds, independent of main loop.

- 0DTE/1DTE: sells at +35% profit or -20% loss
- 2+ DTE: sells at +50% profit or -30% loss

Applies BEFORE the main 6-layer engine. Designed to lock gains and stop
bleeding with minimum latency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.types import Position, ExitDecision


@dataclass
class FastExitConfig:
    pt_short_pct: float = 0.35
    pt_multi_pct: float = 0.50
    sl_short_pct: float = 0.20
    sl_multi_pct: float = 0.30
    # Force-close same-day expiry positions after this many minutes of
    # being held, regardless of P&L. 0DTE options decay fast — holding
    # past the scalp window just donates premium to theta.
    zero_dte_max_hold_minutes: float = 30.0


class FastExitEvaluator:
    def __init__(self, cfg: FastExitConfig = FastExitConfig()):
        self.cfg = cfg

    def evaluate(self, pos: Position, current_price: float) -> Optional[ExitDecision]:
        dte = pos.dte()
        pnl = pos.unrealized_pnl_pct(current_price)
        short_dte = dte <= 1
        # 0DTE scalp-window timeout — fires BEFORE PT/SL so it reliably
        # caps the hold time on same-day contracts. Only applies to DTE==0
        # (not 1DTE), since 1DTE has overnight theta already baked into
        # the entry price.
        if dte == 0 and self.cfg.zero_dte_max_hold_minutes > 0:
            import time as _time
            hold_min = max(0.0, (_time.time() - float(pos.entry_ts)) / 60.0)
            if hold_min > self.cfg.zero_dte_max_hold_minutes:
                return ExitDecision(
                    True,
                    f"fast_0dte_scalp_timeout:{hold_min:.0f}min>"
                    f"{self.cfg.zero_dte_max_hold_minutes:.0f}min",
                    layer=0,
                )
        if short_dte:
            pt, sl = self.cfg.pt_short_pct, self.cfg.sl_short_pct
        else:
            pt, sl = self.cfg.pt_multi_pct, self.cfg.sl_multi_pct
        if pnl >= pt:
            return ExitDecision(True, f"fast_pt_hit:{pnl:.2%}", layer=0)
        if pnl <= -sl:
            return ExitDecision(True, f"fast_sl_hit:{pnl:.2%}", layer=0)
        return None
