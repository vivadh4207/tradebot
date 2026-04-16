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


class FastExitEvaluator:
    def __init__(self, cfg: FastExitConfig = FastExitConfig()):
        self.cfg = cfg

    def evaluate(self, pos: Position, current_price: float) -> Optional[ExitDecision]:
        dte = pos.dte()
        pnl = pos.unrealized_pnl_pct(current_price)
        short_dte = dte <= 1
        if short_dte:
            pt, sl = self.cfg.pt_short_pct, self.cfg.sl_short_pct
        else:
            pt, sl = self.cfg.pt_multi_pct, self.cfg.sl_multi_pct
        if pnl >= pt:
            return ExitDecision(True, f"fast_pt_hit:{pnl:.2%}", layer=0)
        if pnl <= -sl:
            return ExitDecision(True, f"fast_sl_hit:{pnl:.2%}", layer=0)
        return None
