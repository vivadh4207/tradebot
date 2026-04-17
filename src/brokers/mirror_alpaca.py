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
