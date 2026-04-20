"""Credit-spread exit engine.

A credit spread is TWO legs (short + long, same right, different strikes)
that must be closed together. This evaluator:

  1. Groups open option positions by their entry_tag — legs entered
     as a combo share a tag like "weekly_pcs:SPY:<id>" or "0dte_pcs:QQQ:<id>".
  2. Computes the NET P&L of each group at current market (mid prices).
     P&L is expressed as a fraction of the credit originally captured.
  3. Returns a CreditSpreadExit decision when:
       - net pnl >= profit_target_pct × credit  (take profit)
       - net loss >= stop_loss_pct × credit     (stop loss)
       - DTE <= dte_close_threshold             (time-based close)
       - For 0DTE: it is past force_close_et    (final sweep)

Exit is executed as a SINGLE ComboOrder (buy-back-short + sell-back-long)
so the legs unwind atomically.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timezone
from typing import Dict, List, Optional, Tuple

from ..core.types import (
    Position, ComboOrder, OptionLeg, OptionContract, OptionRight, Side,
)

_log = logging.getLogger(__name__)


@dataclass
class CreditSpreadExitConfig:
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 1.50
    dte_close_threshold: int = 21
    # 0DTE final sweep (only used by ZeroDTECreditSpreadRunner's exit loop)
    zero_dte_force_close_et: dtime = dtime(15, 45)


@dataclass
class CreditSpreadExitDecision:
    should_close: bool
    reason: str
    spread_tag: str
    legs: List[Position] = field(default_factory=list)
    net_pnl_pct_of_credit: float = 0.0


def _read_tag(p: Position) -> str:
    """Pull the tag from a Position in whichever shape it has.

    Production code stores tags as `entry_tags={"tag": "<name>"}`.
    Legacy snapshot rows sometimes expose a flat `entry_tag` attribute
    (plain string). Support both.
    """
    tags = getattr(p, "entry_tags", None)
    if isinstance(tags, dict):
        v = tags.get("tag")
        if isinstance(v, str):
            return v
    # fallback: some call sites pass a plain string entry_tag attr
    flat = getattr(p, "entry_tag", "")
    return flat if isinstance(flat, str) else ""


def group_spread_positions(positions: List[Position]) -> Dict[str, List[Position]]:
    """Bucket open positions by their credit-spread entry tag.

    Returns {tag: [leg1, leg2, ...]}. Only includes tags that look
    like a credit spread (prefix in the known set). Non-spread
    positions are ignored.
    """
    groups: Dict[str, List[Position]] = {}
    for p in positions:
        tag = _read_tag(p)
        if not (tag.startswith("weekly_pcs:") or tag.startswith("0dte_pcs:")):
            continue
        groups.setdefault(tag, []).append(p)
    return groups


def _current_mid(p: Position, mark_prices: Dict[str, float]) -> Optional[float]:
    """Best-effort current mid for a position's contract. Falls back to
    avg_price if nothing better is known (which means pnl = 0 for that
    leg — we skip eval in that case)."""
    m = mark_prices.get(p.symbol)
    if m and m > 0:
        return float(m)
    return None


def evaluate_spread(
    legs: List[Position],
    mark_prices: Dict[str, float],
    cfg: CreditSpreadExitConfig,
    now_et: Optional[datetime] = None,
) -> Optional[CreditSpreadExitDecision]:
    """Decide whether to close a credit spread given its two legs + live marks."""
    if len(legs) < 2:
        return None   # half a spread — wait for the other leg to show up

    short_leg = next((p for p in legs if p.qty < 0), None)
    long_leg  = next((p for p in legs if p.qty > 0), None)
    if short_leg is None or long_leg is None:
        return None
    short_mid = _current_mid(short_leg, mark_prices)
    long_mid  = _current_mid(long_leg,  mark_prices)
    if short_mid is None or long_mid is None:
        return None

    # Net credit captured at entry = short_leg.avg - long_leg.avg (positive)
    # Net cost to close now = short_mid - long_mid
    entry_credit = short_leg.avg_price - long_leg.avg_price
    close_cost   = short_mid - long_mid
    if entry_credit <= 0:
        return None  # malformed; avoid divide-by-zero

    # How much of the original credit is still "ours"?
    # pnl_pct > 0 means the spread narrowed in our favor (profit)
    # pnl_pct = (entry_credit - close_cost) / entry_credit
    pnl_pct = (entry_credit - close_cost) / entry_credit

    tag_key = _tag_key(legs[0])
    # --- exit conditions, in priority order ---

    # 0DTE force-close (both legs have the same expiry)
    if now_et is not None and short_leg.expiry == now_et.date():
        if now_et.time() >= cfg.zero_dte_force_close_et:
            return CreditSpreadExitDecision(
                True,
                f"0dte_force_close: past {cfg.zero_dte_force_close_et}",
                spread_tag=tag_key,
                legs=legs, net_pnl_pct_of_credit=pnl_pct,
            )

    # Profit target
    if pnl_pct >= cfg.profit_target_pct:
        return CreditSpreadExitDecision(
            True,
            f"pt_hit: captured {pnl_pct:.1%} of credit (target {cfg.profit_target_pct:.0%})",
            spread_tag=tag_key,
            legs=legs, net_pnl_pct_of_credit=pnl_pct,
        )
    # Stop loss (pnl_pct negative, magnitude > threshold)
    if pnl_pct <= -cfg.stop_loss_pct:
        return CreditSpreadExitDecision(
            True,
            f"sl_hit: loss {pnl_pct:.1%} (stop {-cfg.stop_loss_pct:.0%})",
            spread_tag=tag_key,
            legs=legs, net_pnl_pct_of_credit=pnl_pct,
        )
    # DTE-based time stop (for weekly spreads; 0DTE never hits this)
    dte = short_leg.dte()
    if dte <= cfg.dte_close_threshold and not tag_key.startswith("0dte_pcs:"):
        return CreditSpreadExitDecision(
            True,
            f"dte_close: {dte}d left (<= {cfg.dte_close_threshold})",
            spread_tag=tag_key,
            legs=legs, net_pnl_pct_of_credit=pnl_pct,
        )
    return None


def build_close_combo(
    decision: CreditSpreadExitDecision,
    contracts_by_symbol: Dict[str, OptionContract],
) -> Optional[ComboOrder]:
    """Construct the debit-pay-to-close combo for an exit decision.

    Buys back the short leg + sells back the long leg, atomically.
    Returns None if we can't find both contracts in the chain.
    """
    short_leg = next((p for p in decision.legs if p.qty < 0), None)
    long_leg  = next((p for p in decision.legs if p.qty > 0), None)
    if short_leg is None or long_leg is None:
        return None
    short_c = contracts_by_symbol.get(short_leg.symbol)
    long_c  = contracts_by_symbol.get(long_leg.symbol)
    if short_c is None or long_c is None:
        return None

    # Net debit to close = short_mid - long_mid (positive, because we're
    # buying back a more-expensive option we originally sold).
    close_debit = max(0.05, short_c.mid - long_c.mid)
    return ComboOrder(
        legs=[
            OptionLeg(contract=short_c, side=Side.BUY,  ratio=1),  # close short
            OptionLeg(contract=long_c,  side=Side.SELL, ratio=1),  # close long
        ],
        qty=abs(short_leg.qty),
        net_limit=round(close_debit, 2),    # positive = net debit to pay
        tag=f"close:{decision.spread_tag}",
        tif="DAY",
    )


def _tag_key(p: Position) -> str:
    return _read_tag(p)
