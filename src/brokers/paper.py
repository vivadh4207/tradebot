"""PaperBroker — simulated fills with configurable slippage.

Deterministic, fast, and safe. Used for backtests AND for the first 30-90
days of paper trading against live data before any real broker is wired.

Accepts an optional `TradeJournal` so every fill and every realized
round-trip is persisted (SQLite by default, CockroachDB when wired).
"""
from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime, timezone

from ..core.types import Order, Fill, Position, Side
from .base import BrokerAdapter, AccountState


class PaperBroker(BrokerAdapter):
    def __init__(self, starting_equity: float = 10_000.0,
                 slippage_bps: float = 2.0,
                 commission_per_option: float = 0.0,
                 commission_per_share: float = 0.0,
                 journal=None):
        self._equity = starting_equity
        self._cash = starting_equity
        self._day_pnl = 0.0
        self._total_pnl = 0.0
        self._slippage_bps = slippage_bps
        self._comm_option = commission_per_option
        self._comm_share = commission_per_share
        self._positions: Dict[str, Position] = {}
        self._journal = journal                    # optional TradeJournal

    # --- adapter interface ---
    def account(self) -> AccountState:
        return AccountState(equity=self._equity, cash=self._cash,
                            buying_power=max(self._cash, 0.0),
                            day_pnl=self._day_pnl, total_pnl=self._total_pnl)

    def positions(self) -> List[Position]:
        return list(self._positions.values())

    def submit(self, order: Order) -> Optional[Fill]:
        if order.limit_price is None:
            return None
        slip = order.limit_price * self._slippage_bps / 10_000.0
        fill_price = order.limit_price + (slip if order.side == Side.BUY else -slip)
        fee = (self._comm_option if order.is_option else self._comm_share) * order.qty
        fill = Fill(order=order, price=fill_price, qty=order.qty, fee=fee)
        self._apply(fill)
        self._log_fill(fill)
        return fill

    def cancel_all(self) -> None:
        return

    def flatten_all(self) -> None:
        for sym, pos in list(self._positions.items()):
            side = Side.SELL if pos.qty > 0 else Side.BUY
            closing = Order(
                symbol=sym, side=side, qty=abs(pos.qty),
                is_option=pos.is_option,
                limit_price=pos.avg_price, tif="IOC", tag="eod_force_close",
            )
            self.submit(closing)

    # --- internal ---
    def _apply(self, fill: Fill) -> None:
        o = fill.order
        sym = o.symbol
        signed_qty = fill.qty if o.side == Side.BUY else -fill.qty
        mul = 100 if o.is_option else 1
        cost = fill.price * fill.qty * mul
        self._cash -= cost if o.side == Side.BUY else -cost
        self._cash -= fill.fee

        pos = self._positions.get(sym)
        if pos is None and signed_qty != 0:
            self._positions[sym] = Position(
                symbol=sym, qty=signed_qty, avg_price=fill.price,
                is_option=o.is_option, multiplier=mul,
                entry_tags={"tag": o.tag} if o.tag else {},
            )
            return

        if pos is None:
            return

        new_qty = pos.qty + signed_qty
        # closing or reducing
        if (pos.qty > 0 and signed_qty < 0) or (pos.qty < 0 and signed_qty > 0):
            closed_qty = min(abs(signed_qty), abs(pos.qty))
            sign = 1 if pos.qty > 0 else -1
            realized = sign * (fill.price - pos.avg_price) * closed_qty * mul
            self._total_pnl += realized
            self._day_pnl += realized
            self._equity = self._cash + sum(
                p.qty * p.avg_price * p.multiplier for p in self._positions.values()
            )
            self._log_trade(pos, fill, closed_qty, realized)
        if new_qty == 0:
            self._positions.pop(sym, None)
        else:
            if (pos.qty > 0 and signed_qty > 0) or (pos.qty < 0 and signed_qty < 0):
                total_cost = pos.avg_price * pos.qty * mul + fill.price * signed_qty * mul
                pos.avg_price = total_cost / (new_qty * mul) if new_qty * mul != 0 else fill.price
            pos.qty = new_qty

    def _log_fill(self, fill: Fill) -> None:
        if self._journal is None:
            return
        try:
            self._journal.record_fill(fill)
        except Exception:
            pass   # never let journaling kill the trade loop

    def _log_trade(self, pos: Position, exit_fill: Fill, closed_qty: int, realized: float) -> None:
        if self._journal is None:
            return
        from ..storage.journal import ClosedTrade
        side = "long" if pos.qty > 0 else "short"
        pnl_pct = 0.0
        if pos.avg_price > 0:
            sign = 1 if side == "long" else -1
            pnl_pct = sign * (exit_fill.price - pos.avg_price) / pos.avg_price
        try:
            self._journal.record_trade(ClosedTrade(
                symbol=pos.symbol,
                opened_at=datetime.fromtimestamp(pos.entry_ts, tz=timezone.utc),
                closed_at=datetime.fromtimestamp(exit_fill.ts, tz=timezone.utc),
                side=side, qty=int(closed_qty),
                entry_price=float(pos.avg_price),
                exit_price=float(exit_fill.price),
                pnl=float(realized) - float(exit_fill.fee),
                pnl_pct=float(pnl_pct),
                entry_tag=pos.entry_tags.get("tag") if pos.entry_tags else None,
                exit_reason=exit_fill.order.tag or None,
                is_option=pos.is_option,
            ))
        except Exception:
            pass

    def mark_to_market(self, prices: Dict[str, float]) -> None:
        unreal = 0.0
        for sym, pos in self._positions.items():
            p = prices.get(sym, pos.avg_price)
            unreal += pos.qty * (p - pos.avg_price) * pos.multiplier
        self._equity = self._cash + unreal
        if self._journal is not None:
            try:
                self._journal.record_equity(
                    datetime.now(tz=timezone.utc),
                    self._equity, self._cash, self._day_pnl,
                )
            except Exception:
                pass

    def reset_day(self) -> None:
        self._day_pnl = 0.0
