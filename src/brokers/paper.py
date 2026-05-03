"""PaperBroker — simulated fills with configurable slippage.

Deterministic, fast, and safe. Used for backtests AND for the first 30-90
days of paper trading against live data before any real broker is wired.

Accepts an optional `TradeJournal` so every fill and every realized
round-trip is persisted (SQLite by default, CockroachDB when wired).
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..core.types import Order, Fill, Position, Side, OptionContract
from .base import BrokerAdapter, AccountState

# Lazy cache of the calibration logger so we don't instantiate one per fill.
_SLIPPAGE_LOGGER = None


def _get_slippage_logger(path: str):
    global _SLIPPAGE_LOGGER
    if _SLIPPAGE_LOGGER is None or getattr(_SLIPPAGE_LOGGER, "path", None) != path:
        from ..analytics.slippage_calibration import SlippageLogger
        _SLIPPAGE_LOGGER = SlippageLogger(path)
    return _SLIPPAGE_LOGGER


class PaperBroker(BrokerAdapter):
    def __init__(self, starting_equity: float = 10_000.0,
                 slippage_bps: float = 2.0,
                 commission_per_option: float = 0.0,
                 commission_per_share: float = 0.0,
                 journal=None,
                 snapshot_path: Optional[str] = None,
                 slippage_model=None):
        self._equity = starting_equity
        self._cash = starting_equity
        self._day_pnl = 0.0
        self._total_pnl = 0.0
        self._slippage_bps = slippage_bps
        self._comm_option = commission_per_option
        self._comm_share = commission_per_share
        self._positions: Dict[str, Position] = {}
        self._journal = journal                    # optional TradeJournal
        self._snapshot_path: Optional[str] = snapshot_path
        # Optional stochastic cost model. When None, falls back to fixed-bps
        # slippage (legacy behavior) for backward compatibility.
        self._slippage_model = slippage_model
        self._last_market_ctx: Dict[str, Any] = {}   # symbol → latest MarketContext
        self._lock = threading.RLock()

    # --- adapter interface ---
    def account(self) -> AccountState:
        with self._lock:
            return AccountState(equity=self._equity, cash=self._cash,
                                buying_power=max(self._cash, 0.0),
                                day_pnl=self._day_pnl, total_pnl=self._total_pnl)

    def positions(self) -> List[Position]:
        with self._lock:
            return list(self._positions.values())

    def submit(self, order: Order, *,
                contract: Optional[OptionContract] = None,
                auto_profit_target: Optional[float] = None,
                auto_stop_loss: Optional[float] = None) -> Optional[Fill]:
        """Submit a paper order. For options, pass the full `contract`
        so the created Position has underlying/strike/expiry/right set
        (needed by the exit engine's DTE logic and by EOD option pricing).
        `auto_profit_target` / `auto_stop_loss` are tagged onto the Position
        at entry to honor the CLAUDE.md rule that every entry has both set.
        """
        if order.limit_price is None:
            return None
        cost_for_log = None
        ctx_for_log = None
        if self._slippage_model is not None:
            ctx = self._last_market_ctx.get(order.symbol)
            if ctx is None:
                # synthesize a minimal context from the limit_price
                from .slippage_model import MarketContext
                px = float(order.limit_price)
                ctx = MarketContext(
                    bid=px * 0.9998, ask=px * 1.0002,
                    bid_size=1000, ask_size=1000,
                    vix=15.0, recent_spread_pct=0.0004,
                )
            cost = self._slippage_model.fill(order, ctx)
            fill_price = cost.executed_price
            cost_for_log = cost
            ctx_for_log = ctx
        else:
            slip = order.limit_price * self._slippage_bps / 10_000.0
            fill_price = order.limit_price + (slip if order.side == Side.BUY else -slip)
        fee = (self._comm_option if order.is_option else self._comm_share) * order.qty
        fill = Fill(order=order, price=fill_price, qty=order.qty, fee=fee)
        with self._lock:
            self._apply(fill, contract=contract,
                         auto_profit_target=auto_profit_target,
                         auto_stop_loss=auto_stop_loss)
        # Journal writes happen OUTSIDE the lock to avoid blocking the fast-exit
        # loop on slow DB I/O. The fill object is immutable at this point so
        # there's no data race even across threads.
        self._log_fill(fill)
        self._log_slippage_calibration(fill, cost_for_log, ctx_for_log)
        self._snapshot_if_configured()
        return fill

    def cancel_all(self) -> None:
        return

    def flatten_all(self, mark_prices: Optional[Dict[str, float]] = None,
                     *, tag: str = "eod_force_close") -> None:
        """Close every open position. Uses `mark_prices[symbol]` if provided,
        else falls back to avg_price (zero-slippage close).

        The `tag` parameter lets callers distinguish WHY they flattened:
          - 'eod_force_close' (default) — end-of-session flatten
          - 'shutdown_flatten' — process restart/SIGTERM
          - 'halt_flatten' — emergency kill switch
          - 'manual_flatten' — operator explicit action
        Without this, every caller produced misleading 'eod_force_close'
        journal entries (happens to look like EOD but is actually a
        restart, confusing post-hoc analysis).

        Thread-safe: snapshots positions atomically, then submits sequentially.
        """
        mark_prices = mark_prices or {}
        with self._lock:
            snap = list(self._positions.items())
        for sym, pos in snap:
            side = Side.SELL if pos.qty > 0 else Side.BUY
            px = float(mark_prices.get(sym, pos.avg_price))
            closing = Order(
                symbol=sym, side=side, qty=abs(pos.qty),
                is_option=pos.is_option,
                limit_price=px, tif="IOC", tag=tag,
            )
            self.submit(closing)

    # --- internal ---
    def _apply(self, fill: Fill, *,
                contract: Optional[OptionContract] = None,
                auto_profit_target: Optional[float] = None,
                auto_stop_loss: Optional[float] = None) -> None:
        o = fill.order
        sym = o.symbol
        signed_qty = fill.qty if o.side == Side.BUY else -fill.qty
        mul = 100 if o.is_option else 1
        cost = fill.price * fill.qty * mul
        self._cash -= cost if o.side == Side.BUY else -cost
        self._cash -= fill.fee

        pos = self._positions.get(sym)
        if pos is None and signed_qty != 0:
            # Populate the full option metadata from the contract so the
            # exit engine's dte() / Greek paths work, and so EOD flatten
            # can price the option via chain lookup on (underlying, strike,
            # expiry, right) instead of trying `latest_price(OCC)` which
            # has no bar data.
            p = Position(
                symbol=sym, qty=signed_qty, avg_price=fill.price,
                is_option=o.is_option, multiplier=mul,
                entry_tags={"tag": o.tag} if o.tag else {},
            )
            if contract is not None:
                p.underlying = contract.underlying
                p.strike = contract.strike
                p.expiry = contract.expiry
                p.right = contract.right
            if auto_profit_target is not None:
                p.auto_profit_target = float(auto_profit_target)
            if auto_stop_loss is not None:
                p.auto_stop_loss = float(auto_stop_loss)
            self._positions[sym] = p
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
            self._log_trade(pos, fill, closed_qty, realized)
        if new_qty == 0:
            self._positions.pop(sym, None)
        else:
            if (pos.qty > 0 and signed_qty > 0) or (pos.qty < 0 and signed_qty < 0):
                total_cost = pos.avg_price * pos.qty * mul + fill.price * signed_qty * mul
                pos.avg_price = total_cost / (new_qty * mul) if new_qty * mul != 0 else fill.price
            pos.qty = new_qty
        # Recompute equity at AVG price (book value) after qty adjustment.
        # This is a conservative snapshot — the next mark_to_market() call
        # with real market prices overwrites it with the mark-based equity.
        # Moved after the pop so we don't double-count the closed position.
        self._equity = self._cash + sum(
            p.qty * p.avg_price * p.multiplier for p in self._positions.values()
        )

    def _log_fill(self, fill: Fill) -> None:
        if self._journal is None:
            return
        try:
            self._journal.record_fill(fill)
        except Exception as e:                         # noqa: BLE001
            # Never let journaling kill the trade loop — but DO alert.
            # A silent journal failure means every subsequent trade
            # runs without an audit trail. The user must know.
            try:
                from ..notify.issue_reporter import report_issue
                report_issue(
                    scope="journal.record_fill",
                    message=f"journal write failed for fill {fill.order.symbol}: {type(e).__name__}: {e}",
                    exc=e,
                )
            except Exception:
                pass

    def _log_slippage_calibration(self, fill: Fill, cost, ctx) -> None:
        """Append a calibration row for this fill so the auto-tuner / weekly
        report can reason about predicted-vs-observed slippage."""
        if cost is None or ctx is None:
            return
        try:
            from ..analytics.slippage_calibration import SlippageLogger
            # One global logger at the default path; swap via env if needed.
            import os as _os
            path = _os.getenv("TRADEBOT_SLIPPAGE_LOG",
                                "logs/slippage_calibration.jsonl")
            lg = _get_slippage_logger(path)
            lg.record(
                symbol=fill.order.symbol,
                side=fill.order.side.value,
                qty=int(fill.order.qty),
                is_option=bool(fill.order.is_option),
                limit_price=float(fill.order.limit_price or 0.0),
                executed_price=float(fill.price),
                predicted_bps=float(cost.slippage_bps),
                components=dict(cost.components or {}),
                mid=float(ctx.mid),
                vix=float(ctx.vix),
                tag=str(fill.order.tag or ""),
            )
        except Exception as e:                         # noqa: BLE001
            # Calibration is advisory; degraded slippage tuning won't
            # stop trading. Throttle aggressively (1 hr) — if the path
            # is bad we don't want one alert per fill.
            try:
                from ..notify.issue_reporter import report_issue
                report_issue(
                    scope="slippage_calibration",
                    message=f"slippage logger failed: {type(e).__name__}: {e}",
                    exc=e,
                    throttle_sec=3600.0,
                )
            except Exception:
                pass

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
        with self._lock:
            unreal = 0.0
            for sym, pos in self._positions.items():
                p = prices.get(sym, pos.avg_price)
                unreal += pos.qty * (p - pos.avg_price) * pos.multiplier
            self._equity = self._cash + unreal
            eq, cash, day = self._equity, self._cash, self._day_pnl
        if self._journal is not None:
            try:
                self._journal.record_equity(
                    datetime.now(tz=timezone.utc), eq, cash, day,
                )
            except Exception:
                pass
        self._snapshot_if_configured()

    def reset_day(self) -> None:
        with self._lock:
            self._day_pnl = 0.0
        self._snapshot_if_configured()

    def update_market_context(self, symbol: str, ctx) -> None:
        """Ingest a `MarketContext` for the symbol so the next fill uses it."""
        with self._lock:
            self._last_market_ctx[symbol] = ctx

    # --- snapshot hook ---
    def _snapshot_if_configured(self) -> None:
        if not self._snapshot_path:
            return
        try:
            from ..storage.position_snapshot import save_snapshot
            save_snapshot(self._snapshot_path, self)
        except Exception:
            pass   # best-effort — never let snapshot errors kill trading

    def restore_from_snapshot(self, path: str) -> int:
        """Load a broker state snapshot from disk into this instance.

        Returns the number of positions restored (0 if file missing).
        Thread-safe.
        """
        from ..storage.position_snapshot import load_snapshot, restore_into_paper_broker
        snap = load_snapshot(path)
        if snap is None:
            return 0
        return restore_into_paper_broker(self, snap)
