"""MirrorAlpacaBroker — PaperBroker that ALSO fires every order to Alpaca paper.

Purpose: operator visibility. You want to see the same trades in Alpaca's
paper-trading UI that you see in your local dashboard. This wrapper
forwards each submit to Alpaca as fire-and-forget; our PaperBroker
remains the source of truth for positions / P&L / exits / journal.

Honest caveats (documented so the next reader doesn't assume the two
match):
  - Alpaca's paper simulator uses its own fill prices + timing. Our
    PaperBroker uses the synthetic options chain + stochastic cost
    model. Expect small divergence per trade that compounds over time.
  - Alpaca is asynchronous: submit returns before the fill. We return
    the Fill from our PaperBroker (instant, in-process). The Alpaca
    side may still be queued.
  - Options orders require the user's Alpaca paper account to have
    L1+ options permission. If not, Alpaca rejects with 403 and the
    mirror silently logs the failure.

Design:
  - Subclasses PaperBroker so inherited methods (positions, account,
    mark_to_market, etc.) stay authoritative.
  - `submit()` is the only override: does the paper submit first,
    THEN queues the order for a background thread to dispatch to
    Alpaca. Fire-and-forget so Alpaca latency never blocks our loop.
  - A bounded queue (default 256) drops oldest on overflow rather
    than blocking the caller.
  - Every mirror dispatch logs its outcome at INFO (success) or
    WARNING (failure). First success triggers a one-time "mirror_ok"
    banner so the operator knows the connection works.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from ..core.types import Order, Fill, OptionContract
from .paper import PaperBroker

_log = logging.getLogger(__name__)


class MirrorAlpacaBroker(PaperBroker):
    """PaperBroker + best-effort mirror of every submit to Alpaca paper."""

    def __init__(self, *args, alpaca_broker=None,
                  mirror_queue_size: int = 256,
                  **kwargs):
        super().__init__(*args, **kwargs)
        self._alpaca = alpaca_broker
        self._mirror_stop = threading.Event()
        self._mirror_queue: "queue.Queue[Order]" = queue.Queue(maxsize=mirror_queue_size)
        self._mirror_first_ok_logged = False
        if self._alpaca is not None:
            t = threading.Thread(
                target=self._mirror_worker,
                name="alpaca-mirror",
                daemon=True,
            )
            t.start()
            self._mirror_thread = t
        else:
            self._mirror_thread = None

    def submit(self, order: Order, *,
                contract: Optional[OptionContract] = None,
                auto_profit_target: Optional[float] = None,
                auto_stop_loss: Optional[float] = None) -> Optional[Fill]:
        # Primary: our local paper broker (source of truth for journal,
        # positions, P&L, exits). Must not be bypassed.
        fill = super().submit(
            order,
            contract=contract,
            auto_profit_target=auto_profit_target,
            auto_stop_loss=auto_stop_loss,
        )
        # Mirror: only if paper actually filled (no point mirroring a
        # rejected-order path) AND the mirror is configured.
        if fill is not None and self._alpaca is not None:
            try:
                self._mirror_queue.put_nowait(order)
            except queue.Full:
                # Drop oldest, add newest — preserves recency during
                # sustained Alpaca outages.
                try:
                    self._mirror_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._mirror_queue.put_nowait(order)
                except queue.Full:
                    _log.warning(
                        "alpaca_mirror_queue_full dropping symbol=%s qty=%d",
                        order.symbol, order.qty,
                    )
        return fill

    def close(self) -> None:
        """Shut down the mirror worker cleanly. Called on bot shutdown."""
        self._mirror_stop.set()
        if self._mirror_thread is not None:
            self._mirror_thread.join(timeout=2.0)

    def reconcile_with_alpaca(self) -> dict:
        """Find Alpaca paper positions that don't exist in our local book
        and send close orders for them. Runs at startup so zombie orders
        (left behind when earlier close-mirrors were rejected due to
        the IOC-tif bug etc.) get cleaned up automatically instead of
        requiring manual clicks in the Alpaca UI.

        Returns a summary dict: {reconciled: N, errors: M, local_only: X,
        alpaca_only_symbols: [...]}.
        """
        from ..core.types import Order, Side
        result = {"reconciled": 0, "errors": 0,
                   "local_only": 0, "alpaca_only_symbols": []}
        if self._alpaca is None:
            return result
        try:
            alpaca_positions = self._alpaca.positions()
        except Exception as e:                          # noqa: BLE001
            _log.warning("alpaca_reconcile_fetch_failed err=%s", e)
            return result
        with self._lock:
            local_symbols = set(self._positions.keys())
        for ap in alpaca_positions:
            sym = ap.symbol
            if sym in local_symbols:
                continue            # both books have it; no action
            result["alpaca_only_symbols"].append(sym)
            qty = abs(int(ap.qty))
            if qty <= 0:
                continue

            # Alpaca holds a position we don't. Close it at a
            # QUOTE-DRIVEN limit — using $0.01 doesn't fill (nobody
            # crosses that spread) OR if it did, it'd fill at terrible
            # prices that destroy the account. Same rule your broker
            # applies: a close limit must be realistic relative to the
            # current bid/ask.
            #
            # Need to fetch the contract's live quote. We use the
            # Alpaca TradingClient which can't pull option quotes
            # directly, so we try our data providers via the optional
            # `self._close_quote_fn` injected by main.py. If no quote
            # is available, skip — don't dump positions blindly.
            close_limit = self._close_limit_for(ap, sym)
            if close_limit is None:
                result["errors"] += 1
                _log.warning(
                    "alpaca_reconcile_skip_no_quote symbol=%s qty=%d "
                    "(no live quote + refusing to submit blind close)",
                    sym, qty,
                )
                continue
            # Determine side: short (qty<0) → BUY to close; long (qty>0)
            # → SELL to close. AlpacaBroker.positions() now returns
            # qty with correct sign (short positions negated via the
            # side field).
            side = Side.SELL if ap.qty > 0 else Side.BUY
            close = Order(
                symbol=sym, side=side, qty=qty,
                is_option=bool(ap.is_option),
                limit_price=close_limit,
                tif="DAY",      # options don't support IOC
                tag="reconcile_zombie_close",
            )
            try:
                self._alpaca.submit(close)
                result["reconciled"] += 1
                _log.info(
                    "alpaca_reconcile_zombie_closed symbol=%s qty=%d "
                    "side=%s limit=$%.2f",
                    sym, qty, side.value, close_limit,
                )
            except Exception as e:                      # noqa: BLE001
                result["errors"] += 1
                _log.warning(
                    "alpaca_reconcile_close_failed symbol=%s err=%s",
                    sym, e,
                )
        return result

    def _close_limit_for(self, ap_position, sym: str) -> Optional[float]:
        """Pick a realistic close limit from the live options quote.
        Returns None if no quote is available (caller skips the close).

        Strategy:
          - BUY-to-close a short:  bid + 80% of spread  (pay close to ask)
          - SELL-to-close a long:  ask - 80% of spread  (accept near bid)
        Both cross the spread aggressively to ensure fill but never go
        to $0.01 on a contract with real market value.
        """
        quote_fn = getattr(self, "_close_quote_fn", None)
        if quote_fn is None:
            return None
        try:
            quote = quote_fn(sym)    # expect (bid, ask) or None
        except Exception as e:       # noqa: BLE001
            _log.warning("close_quote_fn_failed symbol=%s err=%s", sym, e)
            return None
        if not quote:
            return None
        bid, ask = quote
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            return None
        spread = ask - bid
        # Direction: if Alpaca has qty>0 we need to SELL → limit near bid
        # (accept market). If qty<0 short → BUY → limit near ask.
        if ap_position.qty > 0:
            return round(max(0.01, ask - 0.80 * spread), 2)   # ~bid+20% of spread, SELL
        else:
            return round(bid + 0.80 * spread, 2)              # ~ask-20% of spread, BUY

    # --- internal ---
    def _mirror_worker(self) -> None:
        while not self._mirror_stop.is_set():
            try:
                order = self._mirror_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._alpaca.submit(order)
                if not self._mirror_first_ok_logged:
                    _log.info(
                        "alpaca_mirror_ok first_post symbol=%s qty=%d",
                        order.symbol, order.qty,
                    )
                    self._mirror_first_ok_logged = True
                else:
                    _log.info(
                        "alpaca_mirror_submitted symbol=%s qty=%d side=%s",
                        order.symbol, order.qty, order.side.value,
                    )
            except Exception as e:                          # noqa: BLE001
                # Alpaca failures never propagate. Common causes:
                # options permission missing (403), symbol not found
                # (OCC format mismatch), network timeout, rate limit.
                _log.warning(
                    "alpaca_mirror_failed symbol=%s err=%s",
                    order.symbol, e,
                )
