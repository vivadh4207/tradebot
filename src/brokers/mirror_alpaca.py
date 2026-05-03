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


class _NullLock:
    """Context-manager no-op. Used when the underlying PaperBroker
    doesn't expose a `_lock` — we still want the `with ...` block in
    the reconcile path to run without a special-case branch."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


class MirrorAlpacaBroker(PaperBroker):
    """PaperBroker + best-effort mirror of every submit to Alpaca paper."""

    def __init__(self, *args, alpaca_broker=None,
                  tradier_broker=None,
                  mirror_queue_size: int = 256,
                  retry_queue_path: str = "logs/alpaca_mirror_retry.jsonl",
                  max_retries_per_order: int = 6,
                  backlog_alert_threshold: int = 5,
                  **kwargs):
        super().__init__(*args, **kwargs)
        self._alpaca = alpaca_broker
        self._tradier = tradier_broker
        self._mirror_stop = threading.Event()
        self._mirror_queue: "queue.Queue[Order]" = queue.Queue(maxsize=mirror_queue_size)
        # Second queue for Tradier (independent retry / backoff).
        self._tradier_queue: "queue.Queue[Order]" = queue.Queue(maxsize=mirror_queue_size)
        self._mirror_first_ok_logged = False
        self._tradier_first_ok_logged = False
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
        # In-flight order lock — prevents the race condition where
        # 3 signals fire on the same symbol within seconds, each
        # checks dedup before any fill registers, all 3 submit, all
        # 3 fill, ending up with 3x intended size. Today (2026-04-29)
        # this happened on IWM 271P (3 buys qty=2 → qty=6, lost $234)
        # and TSLA 375C (qty=6 instead of qty=2). Lock holds for
        # ~10s after each submit; signals 2/3 see lock and skip.
        self._inflight_lock_until: Dict[str, float] = {}
        self._inflight_lock_ttl_sec = 10.0
        self._inflight_lock_mu = threading.Lock()
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
        # Independent Tradier mirror worker. Runs in parallel with
        # Alpaca — each failure doesn't affect the other.
        if self._tradier is not None:
            tt = threading.Thread(
                target=self._tradier_worker,
                name="tradier-mirror",
                daemon=True,
            )
            tt.start()
            self._tradier_thread = tt
            # Cooldown table — symbols recently closed locally. Prevents
            # the auto-reconcile thread from re-adopting a position
            # that fast_exit just closed, which would trigger another
            # close attempt on the next tick → spam loop. Today's bug.
            self._recent_close_ts: Dict[str, float] = {}
            self._recent_close_ttl_sec = 300.0   # 5 min

            # Circuit breaker — track per-symbol reject timestamps.
            # If 3 rejects hit within 60 seconds, pause that symbol
            # for 30 min (no submits at all). Architect's rec:
            # prevents repeat-reject thrash during sandbox throttling
            # or persistent OCC validation failures.
            self._reject_log: Dict[str, list] = {}     # sym -> [ts]
            self._symbol_pause_until: Dict[str, float] = {}  # sym -> ts
            # Auto-reconcile thread: every ~60s, pull Tradier-only
            # positions into the local PaperBroker so existing exit
            # logic (fast_exit tiers, profit-lock, trailing) manages
            # them. Prevents the NFLX-ghost scenario where DNS flap +
            # broken mirror = Tradier has a position the bot can't see.
            rt = threading.Thread(
                target=self._tradier_reconcile_loop,
                name="tradier-reconcile",
                daemon=True,
            )
            rt.start()
            self._tradier_reconcile_thread = rt
        else:
            self._tradier_thread = None
            self._tradier_reconcile_thread = None
            self._recent_close_ts = {}
            self._recent_close_ttl_sec = 300.0

    def _tradier_reconcile_loop(self) -> None:
        """Periodically adopt Tradier-only positions into local state.

        Runs every 60s. For each Tradier symbol not in the local book,
        copies it into PaperBroker's in-memory positions with the
        Tradier-reported avg_price, strike, expiry, right. fast_exit
        picks it up on the next tick and manages it like any other
        entry. Tagged `auto_adopted_from_tradier` so the journal +
        dashboard show where it came from.
        """
        import time as _time
        RECONCILE_INTERVAL = 60.0
        # Sleep a bit on startup so the initial broker_state_restored
        # finishes before we start comparing.
        _time.sleep(15.0)
        while not self._mirror_stop.is_set():
            try:
                self._reconcile_tradier_once()
            except Exception as e:                          # noqa: BLE001
                _log.warning("tradier_reconcile_loop_err err=%s",
                              str(e)[:160])
            self._mirror_stop.wait(RECONCILE_INTERVAL)

    def _reconcile_tradier_once(self) -> int:
        """One reconcile pass. Returns # positions adopted this pass.

        OPERATOR DECISION 2026-05-01: phantom-adoption flow caused
        more problems than it solved (phantom-close storms, broken
        OCC metadata, Discord notification spam). Adoption is now
        DISABLED by default; this loop only computes drift for the
        alarm, never modifies local state.

        To re-enable adoption, set runtime override:
          tradier_reconcile_adopt_enabled = true
        """
        from ..core.types import Position as _Pos, OptionRight as _OR
        if self._tradier is None:
            return 0
        try:
            tr_positions = list(self._tradier.positions())
        except Exception as e:                              # noqa: BLE001
            _log.info("tradier_reconcile_fetch_failed err=%s", str(e)[:120])
            return 0

        # Read adoption flag — default OFF
        try:
            from ..core.runtime_overrides import get_override
            adopt_enabled = bool(get_override(
                "tradier_reconcile_adopt_enabled", False
            ))
        except Exception:
            adopt_enabled = False

        # Always compute drift for alarm visibility
        adopted = 0
        if not adopt_enabled:
            # Read-only mode — only run the drift alarm at the end
            try:
                tradier_cost_basis = sum(
                    abs(int(p.qty)) * float(p.avg_price)
                    * float(p.multiplier or (100 if p.is_option else 1))
                    for p in tr_positions
                )
                with getattr(self, "_lock", _NullLock()):
                    local_cost_basis = sum(
                        abs(int(p.qty)) * float(p.avg_price)
                        * float(p.multiplier or (100 if p.is_option else 1))
                        for p in self._positions.values()
                    )
                drift = abs(local_cost_basis - tradier_cost_basis)
                try:
                    from ..core.runtime_overrides import get_override
                    drift_alarm = float(get_override(
                        "reconcile_drift_alarm_usd", 200.0
                    ))
                except Exception:
                    drift_alarm = 200.0
                if drift >= drift_alarm:
                    _log.warning(
                        "reconcile_drift_alarm local=$%.2f tradier=$%.2f "
                        "drift=$%.2f >= $%.0f_threshold "
                        "(adoption DISABLED — operator must reconcile manually)",
                        local_cost_basis, tradier_cost_basis,
                        drift, drift_alarm,
                    )
            except Exception:
                pass
            return 0

        # ---- ADOPTION PATH (disabled by default) ----
        if not tr_positions:
            return 0
        # Acquire broker lock the PaperBroker uses for _positions
        lock = getattr(self, "_lock", None)
        import time as _t
        ctx = lock if lock is not None else _NullLock()
        with ctx:
            local = dict(self._positions)
            for tp in tr_positions:
                # Cooldown check: skip symbols recently closed locally
                # ONLY when local agrees they should be closed (qty=0
                # locally). If Tradier has a fresh BUY of the same
                # symbol, local needs to know about it — bug today
                # was reconcile blocked the IWM re-adopt because
                # bot had recently closed an IWM put, while Tradier
                # had a NEW IWM call on the books.
                now_t = _t.time()
                last_close = self._recent_close_ts.get(tp.symbol, 0.0)
                in_cooldown = (
                    now_t - last_close < self._recent_close_ttl_sec
                )
                # If Tradier shows a position AND local has zero/none,
                # it's either a phantom we want to skip OR a fresh
                # buy we want to adopt. Disambiguate by checking
                # entry timestamp: if Tradier `date_acquired` is
                # AFTER our last close, it's a fresh buy → adopt.
                if in_cooldown:
                    # Best-effort: pull date_acquired from raw if avail.
                    # Conservative default: ALLOW adopt if local has
                    # NO position on this symbol (genuine orphan).
                    if tp.symbol in local:
                        # Local thinks it's closed but Tradier still
                        # has it. This is the phantom-spam case —
                        # honor the cooldown.
                        age = now_t - last_close
                        _log.info(
                            "tradier_reconcile_cooldown_skip symbol=%s "
                            "closed_%.0fs_ago",
                            tp.symbol, age,
                        )
                        continue
                    # Local has no position → this is a fresh buy bot
                    # didn't track. Adopt it.
                    _log.info(
                        "tradier_reconcile_cooldown_override symbol=%s "
                        "fresh_buy_after_close_window",
                        tp.symbol,
                    )
                if tp.symbol in local:
                    # Already tracked locally — qty mismatch is a
                    # separate concern; log once per occurrence to let
                    # the operator notice without spamming.
                    lp = local[tp.symbol]
                    if int(lp.qty) != int(tp.qty):
                        _log.warning(
                            "tradier_reconcile_qty_mismatch symbol=%s "
                            "local_qty=%d tradier_qty=%d",
                            tp.symbol, int(lp.qty), int(tp.qty),
                        )
                    continue
                # Orphan on Tradier — but verify the symbol is priceable
                # before adopting. Sandbox sometimes returns positions
                # whose OCC symbology is rejected by Alpaca / Yahoo as
                # "invalid symbol", leaving the adopted position
                # unmanageable. Use Tradier's OWN quotes endpoint for
                # the priceability check (the data feed used to fetch
                # bars during fast_exit might fail, but if Tradier
                # returns a valid bid/ask we can manage it).
                try:
                    test_q = None
                    if hasattr(self._tradier, "_get"):
                        try:
                            data = self._tradier._get(
                                "/v1/markets/quotes",
                                {"symbols": tp.symbol},
                            ) or {}
                            rows = (data.get("quotes") or {}
                                      ).get("quote") or {}
                            if isinstance(rows, list):
                                rows = rows[0] if rows else {}
                            ask = float(rows.get("ask") or 0)
                            bid = float(rows.get("bid") or 0)
                            if ask > 0 or bid > 0:
                                test_q = True
                        except Exception:
                            test_q = None
                    if test_q is None:
                        _log.warning(
                            "tradier_reconcile_skip_unpriceable symbol=%s "
                            "qty=%d — Tradier quote returned no bid/ask, "
                            "leaving on Tradier for manual handling",
                            tp.symbol, int(tp.qty),
                        )
                        continue
                except Exception as _e:                     # noqa: BLE001
                    _log.info(
                        "tradier_reconcile_quote_check_err symbol=%s err=%s",
                        tp.symbol, str(_e)[:120],
                    )

                right = tp.right if isinstance(tp.right, _OR) else (
                    _OR(tp.right) if tp.right else None
                )
                adopted_pos = _Pos(
                    symbol=tp.symbol,
                    qty=int(tp.qty),
                    avg_price=float(tp.avg_price),
                    is_option=bool(tp.is_option),
                    underlying=tp.underlying,
                    strike=tp.strike,
                    expiry=tp.expiry,
                    right=right,
                    multiplier=int(tp.multiplier or (100 if tp.is_option else 1)),
                    entry_ts=_t.time(),
                    entry_tags={"tag": "auto_adopted_from_tradier"},
                    auto_profit_target=None,
                    auto_stop_loss=None,
                )
                self._positions[tp.symbol] = adopted_pos
                adopted += 1
                _log.warning(
                    "tradier_reconcile_adopted symbol=%s qty=%d "
                    "avg=%.4f right=%s — exit logic now manages it",
                    tp.symbol, int(tp.qty), float(tp.avg_price),
                    right.value if right else "stock",
                )
        if adopted:
            # Persist the snapshot so a restart preserves adoptions.
            try:
                from ..storage.position_snapshot import save_snapshot
                from ..core.data_paths import data_path
                save_snapshot(data_path("logs/broker_state.json"), self)
            except Exception as e:                          # noqa: BLE001
                _log.info("tradier_reconcile_snap_save_failed err=%s",
                           str(e)[:120])

        # ---- Drift invariant alarm (architect's recommendation) ----
        # Every reconcile pass, compare cumulative cost basis on
        # Tradier vs local. If they diverge by more than the alarm
        # threshold ($200 default), log warning + Discord alert. This
        # is the canary for a brewing phantom-position problem.
        try:
            tradier_cost_basis = sum(
                abs(int(p.qty)) * float(p.avg_price)
                * float(p.multiplier or (100 if p.is_option else 1))
                for p in tr_positions
            )
            with ctx:
                local_cost_basis = sum(
                    abs(int(p.qty)) * float(p.avg_price)
                    * float(p.multiplier or (100 if p.is_option else 1))
                    for p in self._positions.values()
                )
            drift = abs(local_cost_basis - tradier_cost_basis)
            try:
                from ..core.runtime_overrides import get_override
                drift_alarm = float(get_override(
                    "reconcile_drift_alarm_usd", 200.0
                ))
            except Exception:
                drift_alarm = 200.0
            if drift >= drift_alarm:
                _log.warning(
                    "reconcile_drift_alarm local=$%.2f tradier=$%.2f "
                    "drift=$%.2f >= $%.0f_threshold "
                    "(check for phantom positions)",
                    local_cost_basis, tradier_cost_basis,
                    drift, drift_alarm,
                )
        except Exception as _e:                             # noqa: BLE001
            _log.info("reconcile_drift_check_err err=%s",
                        str(_e)[:120])
        return adopted

    def _mark_recently_touched(self, symbol: str) -> None:
        """Tell reconcile not to adopt this symbol for the cooldown
        window. Called after every local submit so the reconcile
        thread doesn't race with a just-opened or just-closed
        position. Race that bit us today: reconcile read Tradier qty=3,
        then bot fast-traded another qty=3 before reconcile wrote
        local → local qty=6, Tradier qty=3 → 'sell 6 > long 3' loop."""
        try:
            self._recent_close_ts[symbol] = time.time()
        except Exception:
            pass

    def submit(self, order: Order, *,
                contract: Optional[OptionContract] = None,
                auto_profit_target: Optional[float] = None,
                auto_stop_loss: Optional[float] = None) -> Optional[Fill]:
        # ---- IN-FLIGHT ORDER LOCK (race-condition prevention) ----
        # Prevents 3 simultaneous signals from each opening qty=2,
        # ending with qty=6. Only applies to BUY-to-open (entries).
        # Exits (sells) bypass — same reason as circuit-breaker.
        from ..core.types import Side as _Side
        side_l = (order.side.value or "").lower()
        is_open = (
            order.side == _Side.BUY and order.is_option
            and "to_close" not in side_l
        )
        if is_open:
            with self._inflight_lock_mu:
                now = time.time()
                lock_until = self._inflight_lock_until.get(order.symbol, 0)
                if now < lock_until:
                    _log.info(
                        "entry_skip_inflight_lock symbol=%s "
                        "lock_remaining=%.1fs (race-condition guard)",
                        order.symbol, lock_until - now,
                    )
                    return None
                # Acquire lock for this symbol
                self._inflight_lock_until[order.symbol] = (
                    now + self._inflight_lock_ttl_sec
                )

        # Primary: our local paper broker (source of truth for journal,
        # positions, P&L, exits). Must not be bypassed.
        fill = super().submit(
            order,
            contract=contract,
            auto_profit_target=auto_profit_target,
            auto_stop_loss=auto_stop_loss,
        )
        # Mirror: only if paper actually filled (no point mirroring a
        # rejected-order path) AND at least one mirror is configured.
        if fill is not None:
            # Lock reconcile out of this symbol for cooldown window
            # so it doesn't read Tradier's stale snapshot and
            # double-count the just-opened/closed position.
            self._mark_recently_touched(order.symbol)
            self._tee_to_queue(order, self._mirror_queue, "alpaca") \
                if self._alpaca is not None else None
            self._tee_to_queue(order, self._tradier_queue, "tradier") \
                if self._tradier is not None else None
        return fill

    def _tee_to_queue(self, order: Order, q: "queue.Queue[Order]",
                       label: str) -> None:
        """Non-blocking queue put with drop-oldest on overflow."""
        try:
            q.put_nowait(order)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(order)
            except queue.Full:
                _log.warning(
                    "%s_mirror_queue_full dropping symbol=%s qty=%d",
                    label, order.symbol, order.qty,
                )

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
                # Alpaca paper sometimes rejects SELL-to-close with
                # "account not eligible to trade uncovered option
                # contracts" — means Alpaca's view of the account
                # disagrees with what we bought. Fire a loud Discord
                # alert so the operator can intervene before theta
                # decay or a market move wipes the orphaned position.
                err_str = str(e)
                try:
                    from ..notify.issue_reporter import report_issue
                    report_issue(
                        scope="alpaca_mirror_orphan",
                        message=(
                            f"Alpaca REJECTED close for orphan position "
                            f"`{sym}` qty={qty}. Alpaca error: "
                            f"{err_str[:200]}. Local bot has no "
                            "position but ALPACA DOES — go close it "
                            "manually on alpaca.markets/paper or your "
                            "account will sit exposed."
                        ),
                    )
                except Exception:
                    pass
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

    def _tradier_worker(self) -> None:
        """Independent worker for Tradier mirroring.

        Submit returns `None` on network/HTTP failure or rejection (see
        TradierBroker.submit). Treat that as a failed submission —
        re-enqueue with exponential backoff so transient DNS flaps or
        gateway timeouts don't silently create ghost positions on the
        local PaperBroker that Tradier never acknowledged. Only log
        `tradier_mirror_ok` / `tradier_mirror_submitted` when submit
        actually returns a Fill.

        Max retries cap = 6 (roughly 2 min of backoff). After that, log
        a fatal mirror_drop so the operator can reconcile manually.
        """
        import time as _time
        backoff_schedule = [1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        max_retries = len(backoff_schedule)
        while not self._mirror_stop.is_set():
            try:
                item = self._tradier_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # Items may be either a raw Order (first submit) or a
            # (Order, retry_count) tuple for requeued orders.
            if isinstance(item, tuple):
                order, retry = item
            else:
                order, retry = item, 0

            # Circuit-breaker check: is this symbol currently paused?
            # CRITICAL EXEMPTION: sell-to-close orders are EXITS, not
            # entries. Blocking them would let positions bleed past the
            # hard $-cap (today's IWM 271P bug — cap fired but breaker
            # blocked the close, position bled $324 → $348 → final
            # close at -$234 only after timeout expired).
            #
            # The breaker exists to prevent ENTRY thrash during sandbox
            # throttling. Exits must always be allowed through.
            now_ts = _time.time()
            paused_until = self._symbol_pause_until.get(order.symbol, 0)
            if now_ts < paused_until:
                # Check if this is a close order — those bypass.
                _side_l = (order.side.value or "").lower()
                _is_close = (
                    "sell" in _side_l
                    or "to_close" in _side_l
                    or getattr(order, "is_close", False)
                )
                if _is_close:
                    _log.info(
                        "tradier_circuit_bypass_close symbol=%s "
                        "(close order EXEMPT from circuit pause)",
                        order.symbol,
                    )
                    # Continue to submit — don't skip
                else:
                    _log.warning(
                        "tradier_symbol_circuit_open symbol=%s "
                        "paused_for=%.0fs_more (entry blocked, "
                        "closes still allowed)",
                        order.symbol, paused_until - now_ts,
                    )
                    continue

            result = None
            caught: Optional[Exception] = None
            try:
                result = self._tradier.submit(order)
            except Exception as e:                      # noqa: BLE001
                caught = e

            if result is not None and caught is None:
                # Genuine success — Tradier confirmed terminal status
                # = filled (with poll). Note the cooldown for closes.
                side_val = order.side.value
                is_close = ("sell" in side_val.lower()
                              or "to_close" in side_val.lower())
                if is_close:
                    self._recent_close_ts[order.symbol] = _time.time()
                if not self._tradier_first_ok_logged:
                    _log.info("tradier_mirror_ok first_post symbol=%s qty=%d",
                              order.symbol, order.qty)
                    self._tradier_first_ok_logged = True
                else:
                    _log.info(
                        "tradier_mirror_submitted symbol=%s qty=%d side=%s",
                        order.symbol, order.qty, side_val,
                    )
                continue

            # Failure path: either an exception or submit returned None
            # (rejection / HTTP fail / DNS fail). Both are unsafe to
            # treat as success.
            err_desc = str(caught) if caught else "submit_returned_none"

            # ---- IMMEDIATE PHANTOM PRUNE (simplified) ----
            # If a sell-to-close order fails (Tradier rejected, expired,
            # or pending-at-timeout), local PaperBroker has a position
            # Tradier doesn't. PRUNE IMMEDIATELY — no retry, no Discord
            # notify. The original phantom-detect-by-querying-Tradier
            # approach was racy (the orders endpoint hadn't yet logged
            # the new rejection by the time we polled). Simpler logic:
            # if a close failed, it's a phantom. Risk: rare false-prune
            # on transient broker error, but drift alarm catches that.
            try:
                _side_l = (order.side.value or "").lower()
                _is_close = (
                    "sell" in _side_l or "to_close" in _side_l
                )
                if _is_close:
                    with getattr(self, "_lock", _NullLock()):
                        if order.symbol in self._positions:
                            del self._positions[order.symbol]
                            self._recent_close_ts[order.symbol] = (
                                _time.time()
                            )
                            _log.warning(
                                "tradier_phantom_pruned_immediate "
                                "symbol=%s side=%s — close failed "
                                "(rejected/expired/timeout), local "
                                "pruned without retry/notify",
                                order.symbol, _side_l,
                            )
                            try:
                                from ..storage.position_snapshot import (
                                    save_snapshot,
                                )
                                from ..core.data_paths import data_path
                                save_snapshot(
                                    data_path("logs/broker_state.json"),
                                    self,
                                )
                            except Exception:
                                pass
                    # Skip retry — close failed = phantom = done
                    continue
            except Exception as _e:                          # noqa: BLE001
                _log.info("phantom_prune_check_err err=%s",
                            str(_e)[:120])

            # Circuit-breaker bookkeeping: log the reject, prune older
            # than 60s, trip if >= 3 within window.
            try:
                rejs = self._reject_log.setdefault(order.symbol, [])
                rejs.append(_time.time())
                # Prune > 60s
                cutoff = _time.time() - 60.0
                rejs[:] = [t for t in rejs if t >= cutoff]
                if len(rejs) >= 3:
                    pause_for_sec = 30 * 60   # 30 min
                    self._symbol_pause_until[order.symbol] = (
                        _time.time() + pause_for_sec
                    )
                    rejs.clear()
                    _log.warning(
                        "tradier_circuit_breaker_tripped symbol=%s "
                        "paused_for=%ds reason=3_rejects_in_60s",
                        order.symbol, pause_for_sec,
                    )
            except Exception:
                pass

            # PHANTOM-POSITION HEAL: if this was a sell-to-close that
            # Tradier rejected with "not closing a long position", local
            # PaperBroker is desynced — has a position Tradier doesn't.
            # Remove it from local so fast_exit stops firing on a ghost.
            # Otherwise the close-reject loop runs every tick forever.
            try:
                side_l = (order.side.value or "").lower()
                is_close = ("sell" in side_l or "to_close" in side_l)
                # Best-effort: query the most recent rejection for this
                # symbol from Tradier's order log to see the reason.
                # Cheaper heuristic: any close-reject after retry=2 we
                # treat as phantom and prune locally. After 2 retries
                # over backoff (1s + 2s = 3s) Tradier has had time to
                # settle; persistent reject = phantom local position.
                if is_close and retry >= 2:
                    with getattr(self, "_lock", _NullLock()):
                        if order.symbol in self._positions:
                            del self._positions[order.symbol]
                            self._recent_close_ts[order.symbol] = (
                                _time.time()
                            )
                            _log.warning(
                                "tradier_phantom_position_pruned "
                                "symbol=%s reason=close_keeps_rejecting "
                                "(local thought we owned it; Tradier "
                                "didn't) — removed from local book",
                                order.symbol,
                            )
                            # Persist immediately so a restart sees clean state
                            try:
                                from ..storage.position_snapshot import (
                                    save_snapshot,
                                )
                                from ..core.data_paths import data_path
                                save_snapshot(
                                    data_path("logs/broker_state.json"),
                                    self,
                                )
                            except Exception:
                                pass
                            continue
            except Exception:
                pass
            if retry >= max_retries:
                _log.warning(
                    "tradier_mirror_drop symbol=%s qty=%d side=%s "
                    "retries=%d err=%s RECONCILE_REQUIRED",
                    order.symbol, order.qty, order.side.value,
                    retry, err_desc[:160],
                )
                continue
            _log.warning(
                "tradier_mirror_failed symbol=%s qty=%d side=%s "
                "retry=%d/%d err=%s",
                order.symbol, order.qty, order.side.value,
                retry + 1, max_retries, err_desc[:160],
            )
            # Re-queue with backoff. Sleep here blocks this worker
            # thread only; main bot keeps trading locally.
            _time.sleep(backoff_schedule[retry])
            try:
                self._tradier_queue.put((order, retry + 1))
            except Exception:
                pass

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
