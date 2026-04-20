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

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from ..core.types import Order, Fill, OptionContract, Side
from .paper import PaperBroker

_log = logging.getLogger(__name__)


def _order_to_dict(order: Order) -> dict:
    """Serialize an Order for on-disk retry queue."""
    return {
        "symbol": order.symbol,
        "side": order.side.value if hasattr(order.side, "value") else str(order.side),
        "qty": int(order.qty),
        "is_option": bool(getattr(order, "is_option", False)),
        "limit_price": getattr(order, "limit_price", None),
        "tif": getattr(order, "tif", "DAY"),
        "tag": getattr(order, "tag", "mirror_retry"),
    }


def _order_from_dict(d: dict) -> Order:
    """Deserialize. Keep in sync with _order_to_dict."""
    side = Side.BUY if str(d.get("side", "BUY")).upper() == "BUY" else Side.SELL
    return Order(
        symbol=d["symbol"], side=side, qty=int(d["qty"]),
        is_option=bool(d.get("is_option", False)),
        limit_price=d.get("limit_price"),
        tif=d.get("tif", "DAY"),
        tag=d.get("tag", "mirror_retry"),
    )


class MirrorAlpacaBroker(PaperBroker):
    """PaperBroker + best-effort mirror of every submit to Alpaca paper."""

    def __init__(self, *args, alpaca_broker=None,
                  mirror_queue_size: int = 256,
                  retry_queue_path: str = "logs/alpaca_mirror_retry.jsonl",
                  max_retries_per_order: int = 6,
                  backlog_alert_threshold: int = 5,
                  **kwargs):
        super().__init__(*args, **kwargs)
        self._alpaca = alpaca_broker
        self._mirror_stop = threading.Event()
        self._mirror_queue: "queue.Queue[Order]" = queue.Queue(maxsize=mirror_queue_size)
        self._mirror_first_ok_logged = False
        # Persistent retry queue. Orders that fail to submit are
        # appended here and replayed by the worker until success or
        # max_retries. Prevents trades from being silently dropped
        # during Alpaca outages / DNS hiccups.
        try:
            from ..core.data_paths import data_path
            self._retry_path = Path(data_path(retry_queue_path))
        except Exception:
            self._retry_path = Path(retry_queue_path)
        self._retry_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_retries = int(max_retries_per_order)
        self._backlog_threshold = int(backlog_alert_threshold)
        self._backlog_alerted = False
        self._retry_lock = threading.Lock()
        if self._alpaca is not None:
            # Drain persisted queue on startup — trades that didn't
            # make it during the last run get replayed now.
            self._load_persisted_retries()
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
        # Operator override: specific symbols (zombie positions kept
        # intentionally, e.g. a cheap long put left to expire) can be
        # excluded from reconcile. Keeps the noise down without hiding
        # the mirror from other positions. Comma-separated, env-only so
        # it's trivially togglable without a deploy.
        import os as _os
        skip_syms = {
            s.strip() for s in
            (_os.getenv("ALPACA_RECONCILE_SKIP_SYMBOLS", "") or "").split(",")
            if s.strip()
        }
        for ap in alpaca_positions:
            sym = ap.symbol
            if sym in local_symbols:
                continue            # both books have it; no action
            if sym in skip_syms:
                _log.info("alpaca_reconcile_skip_operator_override symbol=%s",
                          sym)
                continue
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
        """Worker loop. Drains in-memory queue + periodically replays
        any persisted retries. Each order carries a retry_count; after
        max_retries it's dropped from the queue and an alert fires."""
        last_retry_sweep = 0.0
        while not self._mirror_stop.is_set():
            now = time.time()
            # Every ~20s, replay anything in the on-disk retry file
            # (handles the case where new orders stopped coming in but
            # old failures still need retrying).
            if now - last_retry_sweep > 20.0:
                self._replay_persisted_retries()
                last_retry_sweep = now
            try:
                item = self._mirror_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # item is Order on first try, or (Order, retry_count) when
            # replayed from the retry file.
            if isinstance(item, tuple):
                order, retry_count = item
            else:
                order, retry_count = item, 0
            try:
                self._alpaca.submit(order)
                # Success — if this was a retry, note the recovery.
                if retry_count > 0:
                    _log.info(
                        "alpaca_mirror_healed symbol=%s qty=%d after_retries=%d",
                        order.symbol, order.qty, retry_count,
                    )
                    # Mirror is back; clear the backlog-alert flag so a
                    # future outage can alert again.
                    self._backlog_alerted = False
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
                next_retry = retry_count + 1
                if next_retry > self._max_retries:
                    # Permanent failure. Alert + don't re-queue.
                    _log.warning(
                        "alpaca_mirror_dropped symbol=%s qty=%d "
                        "after %d retries err=%s",
                        order.symbol, order.qty, retry_count, e,
                    )
                    try:
                        from ..notify.issue_reporter import report_issue
                        report_issue(
                            scope="alpaca.mirror_dropped",
                            message=(
                                f"Mirror PERMANENTLY dropped "
                                f"{order.symbol} after {retry_count} retries: "
                                f"{type(e).__name__}: {e}"
                            ),
                            exc=e,
                            extra={"qty": order.qty, "side": str(order.side),
                                    "is_option": order.is_option,
                                    "retries": retry_count},
                        )
                    except Exception:
                        pass
                    continue

                # Transient failure — persist + schedule retry.
                _log.warning(
                    "alpaca_mirror_failed symbol=%s try=%d/%d err=%s",
                    order.symbol, next_retry, self._max_retries, e,
                )
                self._persist_retry(order, next_retry)
                # Emit ONE backlog alert when we cross the threshold,
                # not one per failure. Reset when healed above.
                backlog = self._persisted_count()
                if (backlog >= self._backlog_threshold
                        and not self._backlog_alerted):
                    try:
                        from ..notify.issue_reporter import report_issue
                        report_issue(
                            scope="alpaca.mirror_backlog",
                            message=(
                                f"Alpaca mirror backlog: {backlog} orders "
                                f"queued for retry. Last error: "
                                f"{type(e).__name__}: {e}"
                            ),
                            exc=e,
                            extra={"backlog": backlog},
                        )
                    except Exception:
                        pass
                    self._backlog_alerted = True

    # --- persistent retry queue helpers ---
    def _persist_retry(self, order: Order, retry_count: int) -> None:
        """Append a failed order to the retry file for replay later."""
        row = _order_to_dict(order)
        row["retry_count"] = int(retry_count)
        row["ts"] = time.time()
        try:
            with self._retry_lock, self._retry_path.open("a",
                                                           encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        except Exception as e:                              # noqa: BLE001
            _log.warning("alpaca_mirror_persist_failed err=%s", e)

    def _persisted_count(self) -> int:
        try:
            if not self._retry_path.exists():
                return 0
            with self._retry_path.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def _load_persisted_retries(self) -> None:
        """Drain the on-disk retry queue into the in-memory queue on
        startup. Safe to call after __init__ — no submission happens
        here; the worker thread picks them up."""
        if not self._retry_path.exists():
            return
        rows = []
        try:
            with self._retry_lock, self._retry_path.open("r",
                                                           encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
            # Truncate — worker will re-persist any that fail again.
            self._retry_path.write_text("", encoding="utf-8")
        except Exception as e:                              # noqa: BLE001
            _log.warning("alpaca_mirror_load_failed err=%s", e)
            return
        n = 0
        for r in rows:
            try:
                order = _order_from_dict(r)
                retry_count = int(r.get("retry_count", 0))
                try:
                    self._mirror_queue.put_nowait((order, retry_count))
                    n += 1
                except queue.Full:
                    # Couldn't queue — re-persist so we try later.
                    self._persist_retry(order, retry_count)
            except Exception:
                continue
        if n > 0:
            _log.info("alpaca_mirror_replay_loaded n=%d", n)

    def _replay_persisted_retries(self) -> None:
        """Periodic sweep: re-load the disk queue into memory so the
        worker can try again. Same contract as _load_persisted_retries
        but runs during the worker loop, not just startup."""
        if self._persisted_count() == 0:
            return
        self._load_persisted_retries()
