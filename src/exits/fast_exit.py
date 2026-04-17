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
    # Scale-out + trailing stop — lets winners run.
    # At first PT hit, close `scale_out_fraction` of the position. The
    # remainder then rides a trailing stop at peak * (1 - trailing_stop_pct).
    # This is what momentum-options prop desks do: lock half the gain,
    # let the rest chase higher highs, exit on first meaningful retrace.
    scale_out_fraction: float = 0.50
    trailing_stop_pct: float = 0.10
    trailing_stop_enabled: bool = True
    hard_cap_pct: float = 1.50


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

        # Update peak price for trailing-stop math. Mutating pos here
        # is intentional — main.py persists the state via the broker
        # snapshot, and the next fast_loop tick reads the updated peak.
        if pos.is_long:
            if pos.peak_price is None or current_price > pos.peak_price:
                pos.peak_price = current_price

        # Hard cap: close fully at runaway wins.
        if pnl >= self.cfg.hard_cap_pct:
            return ExitDecision(True, f"fast_hard_cap:{pnl:.2%}", layer=0)

        # Stop loss always fires first.
        if pnl <= -sl:
            return ExitDecision(True, f"fast_sl_hit:{pnl:.2%}", layer=0)

        # Scale-out at first PT hit: close half, flag pos.scaled_out,
        # let the remainder trail the high. Requires at least 2 contracts
        # to split; with 1 contract we fall back to full close.
        can_scale = (
            self.cfg.scale_out_fraction > 0
            and not pos.scaled_out
            and abs(pos.qty) >= 2
        )
        if pnl >= pt:
            if can_scale:
                half = max(1, int(round(abs(pos.qty)
                                          * self.cfg.scale_out_fraction)))
                return ExitDecision(
                    True,
                    f"fast_scale_out_at_pt:{pnl:.2%}_closing_{half}of{abs(pos.qty)}",
                    layer=0,
                    close_qty=half,
                )
            # One contract, can't split → legacy full close at PT.
            return ExitDecision(True, f"fast_pt_hit:{pnl:.2%}", layer=0)

        # Trailing stop for the runner. Only active after scale-out.
        # Close the remainder once price retraces from the peak beyond
        # `trailing_stop_pct`. Captures the bulk of a big move, exits on
        # the first real retrace.
        if (self.cfg.trailing_stop_enabled
                and pos.scaled_out
                and pos.peak_price and pos.peak_price > 0):
            if pos.is_long:
                retrace = (pos.peak_price - current_price) / pos.peak_price
            else:
                retrace = (current_price - pos.peak_price) / pos.peak_price
            if retrace >= self.cfg.trailing_stop_pct:
                peak_pnl = pos.unrealized_pnl_pct(pos.peak_price)
                return ExitDecision(
                    True,
                    f"fast_trailing_stop:peak_pnl={peak_pnl:+.2%}_now="
                    f"{pnl:+.2%}_retrace={retrace:.2%}",
                    layer=0,
                )
        return None
