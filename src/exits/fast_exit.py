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
    # being held. 0DTE options decay fast — holding past the scalp
    # window just donates premium to theta.
    zero_dte_max_hold_minutes: float = 30.0
    # Dynamic scalp-window — tightens when the trade is moving against
    # you (underlying going the wrong way + theta bleeding you → cut fast)
    # and loosens when it's working (let the winner run).
    #
    # The effective timeout is:
    #   base + (extension if pnl >= +favorable_threshold
    #           else -reduction if pnl <= -unfavorable_threshold
    #           else 0)
    #
    # P&L is used as the direction-alignment signal because for long
    # directional options (long call = bullish bet, long put = bearish
    # bet), unrealized P&L is equivalent to "is the underlying moving
    # the way I bet?" The option's price captures both direction AND
    # theta, which is what we actually care about.
    zero_dte_favorable_pnl_threshold: float = 0.10      # +10% unrealized → trade is working
    zero_dte_unfavorable_pnl_threshold: float = 0.10    # -10% unrealized → trade is failing
    zero_dte_favorable_extension_minutes: float = 15.0  # add time when winning
    zero_dte_unfavorable_reduction_minutes: float = 20.0  # subtract time when losing


class FastExitEvaluator:
    def __init__(self, cfg: FastExitConfig = FastExitConfig()):
        self.cfg = cfg

    def _effective_0dte_timeout(self, pnl_pct: float) -> float:
        """Compute the current max-hold window based on unrealized P&L.
        Winning trades get extra time; losing trades get cut faster."""
        base = self.cfg.zero_dte_max_hold_minutes
        if base <= 0:
            return 0.0
        if pnl_pct >= self.cfg.zero_dte_favorable_pnl_threshold:
            return base + self.cfg.zero_dte_favorable_extension_minutes
        if pnl_pct <= -self.cfg.zero_dte_unfavorable_pnl_threshold:
            return max(1.0, base - self.cfg.zero_dte_unfavorable_reduction_minutes)
        return base

    def evaluate(self, pos: Position, current_price: float) -> Optional[ExitDecision]:
        dte = pos.dte()
        pnl = pos.unrealized_pnl_pct(current_price)
        short_dte = dte <= 1
        # 0DTE scalp-window timeout — fires BEFORE PT/SL so it reliably
        # caps the hold time on same-day contracts. Timeout is DYNAMIC:
        # contracts faster when the trade is moving against you, expands
        # when it's working. Applies only to DTE==0 (not 1DTE, which has
        # overnight theta already priced in).
        if dte == 0 and self.cfg.zero_dte_max_hold_minutes > 0:
            import time as _time
            hold_min = max(0.0, (_time.time() - float(pos.entry_ts)) / 60.0)
            effective_max = self._effective_0dte_timeout(pnl)
            if hold_min > effective_max:
                return ExitDecision(
                    True,
                    f"fast_0dte_scalp_timeout:{hold_min:.0f}min>"
                    f"{effective_max:.0f}min@pnl={pnl:+.1%}",
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
