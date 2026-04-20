"""Wheel exit engine — closes short-option (premium-harvest) positions.

Premium-selling exit math is INVERTED from long-option exit math:
  - Entry:  SELL option at $X  → broker position qty is NEGATIVE
  - Goal:   option price DECAYS toward 0 → we buy back cheap → we keep premium
  - Profit target: buy back at 50% of entry premium (keep half the credit)
  - Stop loss:     buy back if premium doubles (we lose 1× premium)
  - Time stop:     buy back at ≤21 DTE regardless of P&L (gamma risk explodes)

`Position.unrealized_pnl_pct` already flips sign for short positions, so
`pnl` is +0.50 when we've captured half the premium. Exit logic below
uses that convention.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from ..core.types import Position, ExitDecision


_log = logging.getLogger(__name__)


@dataclass
class WheelExitConfig:
    profit_target_pct: float = 0.50   # +50% pnl = captured half the premium
    stop_loss_pct: float = 1.00       # -100% pnl = we lose 1× premium (option doubled)
    dte_roll_threshold: int = 21      # close under this DTE regardless of P&L


class WheelExitEvaluator:
    """Runs in fast_loop just like FastExitEvaluator. Only evaluates
    SHORT option positions (qty < 0 and is_option). Long/stock positions
    are skipped — caller's other exit engine handles those."""

    def __init__(self, cfg: WheelExitConfig = WheelExitConfig()):
        self.cfg = cfg

    def evaluate(self, pos: Position, current_price: float) -> Optional[ExitDecision]:
        if not pos.is_option or pos.qty >= 0:
            return None   # not a short-option (premium-sold) position
        if current_price <= 0 or pos.avg_price <= 0:
            return None

        # `unrealized_pnl_pct` returns positive when the short is
        # winning (option decayed below entry). We interpret:
        #   pnl >= +0.50  → option's mid is below 50% of entry → PT hit
        #   pnl <= -1.00  → option's mid is 2× entry → stopped out
        pnl = pos.unrealized_pnl_pct(current_price)

        if pnl >= self.cfg.profit_target_pct:
            return ExitDecision(
                True,
                f"wheel_pt_50pct_capture: pnl={pnl:+.2%}",
                layer=0,
            )

        if pnl <= -self.cfg.stop_loss_pct:
            return ExitDecision(
                True,
                f"wheel_sl_option_doubled: pnl={pnl:+.2%}",
                layer=0,
            )

        # DTE-based time stop. Past 21 DTE a short put's gamma explodes
        # (small underlying moves = large option-price moves). Standard
        # wheel practice: close at 21 DTE or roll out/down.
        dte = pos.dte()
        if dte <= self.cfg.dte_roll_threshold:
            return ExitDecision(
                True,
                f"wheel_dte_roll: {dte}d left (<= {self.cfg.dte_roll_threshold})",
                layer=0,
            )

        return None


def build_wheel_close_order(pos: Position, limit_price: float):
    """Build the buy-to-close order that cancels a short put.

    Short put position has qty < 0; we BUY `abs(qty)` to flatten.
    Limit should be slightly above the current bid to ensure fill —
    paying a small premium to close is much cheaper than letting the
    position ride past the 21-DTE gamma wall.
    """
    from ..core.types import Order, Side
    return Order(
        symbol=pos.symbol, side=Side.BUY, qty=abs(pos.qty),
        is_option=True, limit_price=limit_price,
        tif="DAY",
        tag="wheel_close",
    )
