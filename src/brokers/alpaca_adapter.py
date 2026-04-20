"""AlpacaBroker — routes orders to Alpaca's REST API.

Imports alpaca-py lazily; raises a clear error if the package or credentials
are missing. Paper endpoint is default.

GUARDRAIL: requires settings.live_trading == True to route anything other
than paper orders. This is checked one level up in the orchestrator.

Every network call is wrapped in exponential-backoff retry so that
transient Alpaca 429 / 5xx responses don't kill the main loop. After the
max retry budget we log and surface the error to the caller.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, List, Optional

from ..core.types import Order, Fill, Position, Side
from .base import BrokerAdapter, AccountState


_log = logging.getLogger(__name__)

# Retry tuning. Alpaca defaults to ~200 req/min; conservative backoff here.
_RETRY_MAX_ATTEMPTS = 5
_RETRY_BASE_DELAY = 1.0   # seconds
_RETRY_MAX_DELAY = 30.0


def _is_retriable(exc: BaseException) -> bool:
    """Decide whether an Alpaca/REST exception is worth retrying.

    Retriable: 429 (rate limit), 5xx, connection errors, timeouts.
    NOT retriable: 4xx other than 429 (bad request, auth failure, etc.).
    """
    # alpaca-py raises APIError subclass with status_code
    status = getattr(exc, "status_code", None)
    if status is None:
        msg = str(exc).lower()
        if any(x in msg for x in ("timeout", "connection", "reset by peer",
                                   "429", "service unavailable", "bad gateway")):
            return True
        return False
    try:
        status = int(status)
    except Exception:
        return False
    if status == 429:
        return True
    if 500 <= status < 600:
        return True
    return False


def _with_retry(fn: Callable[[], Any], *, op: str,
                 max_attempts: int = _RETRY_MAX_ATTEMPTS) -> Any:
    """Exponential-backoff wrapper with full jitter.

    A FINAL failure (all attempts exhausted OR non-retriable error) is
    pushed to the Discord alerts channel once per unique (op, error)
    pair via the issue_reporter throttle. Retriable attempts stay in
    the log only to avoid spam.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:                           # noqa: BLE001
            last_exc = e
            if not _is_retriable(e) or attempt == max_attempts:
                _log.warning("alpaca_%s_failed_final attempt=%d err=%s",
                              op, attempt, e)
                try:
                    from ..notify.issue_reporter import report_issue
                    report_issue(
                        scope=f"alpaca.{op}",
                        message=f"Alpaca {op} failed after {attempt} attempts: {type(e).__name__}: {e}",
                        exc=e,
                    )
                except Exception:
                    pass
                raise
            delay = min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            delay *= random.uniform(0.5, 1.5)           # full jitter
            _log.info("alpaca_%s_retry attempt=%d delay=%.1fs err=%s",
                      op, attempt, delay, e)
            time.sleep(delay)
    # should not reach here, but defensively re-raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("alpaca_retry_unreachable")


class AlpacaBroker(BrokerAdapter):
    def __init__(self, api_key: str, api_secret: str,
                 paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as e:
            raise ImportError(
                "alpaca-py is required for AlpacaBroker. Run: pip install alpaca-py"
            ) from e
        self._client = TradingClient(api_key, api_secret, paper=paper)
        self._paper = paper

    def account(self) -> AccountState:
        def _call():
            a = self._client.get_account()
            return AccountState(
                equity=float(a.equity),
                cash=float(a.cash),
                buying_power=float(a.buying_power),
                day_pnl=float(getattr(a, "day_pnl", 0.0) or 0.0),
                total_pnl=0.0,
            )
        return _with_retry(_call, op="account")

    def positions(self) -> List[Position]:
        from ..core.types import Position as Pos

        def _call():
            out: List[Position] = []
            for p in self._client.get_all_positions():
                # Alpaca reports positions with `qty` always positive and
                # a separate `side` enum (LONG or SHORT). Our downstream
                # code (reconcile, etc.) relies on sign-of-qty to
                # determine direction. Coerce shorts to negative.
                raw_qty = int(float(p.qty))
                side = getattr(p, "side", None)
                side_str = str(side).lower() if side is not None else ""
                if "short" in side_str:
                    raw_qty = -abs(raw_qty)
                # Options detection: Alpaca symbols are OCC for options
                # (contain a date + C/P + strike) — leverage length.
                sym = str(p.symbol)
                is_option = len(sym) > 8 and (
                    "C" in sym[-10:] or "P" in sym[-10:]
                )
                out.append(Pos(
                    symbol=sym,
                    qty=raw_qty,
                    avg_price=float(p.avg_entry_price),
                    is_option=is_option,
                    multiplier=100 if is_option else 1,
                ))
            return out
        return _with_retry(_call, op="positions")

    def submit(self, order: Order, **_ignored) -> Optional[Fill]:
        """Submit to Alpaca. `**_ignored` absorbs the PaperBroker's
        extended kwargs (`contract=`, `auto_profit_target=`,
        `auto_stop_loss=`) so this adapter can be used as a drop-in
        target by MirrorAlpacaBroker without signature drift."""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        # Alpaca options trading rejects IOC / GTC / OPG / CLS with
        # error 42210000 "order_time_in_force provided not supported
        # for options trading". Only DAY is accepted. Our exit engine
        # and flatten_all submit orders with tif=IOC (intentional for
        # fast cancellation on the paper side) — coerce to DAY for
        # the Alpaca mirror so closes aren't silently rejected.
        if order.is_option:
            tif = TimeInForce.DAY
        else:
            tif = {
                "DAY": TimeInForce.DAY, "IOC": TimeInForce.IOC, "GTC": TimeInForce.GTC,
            }.get(order.tif, TimeInForce.DAY)
        side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL
        req = LimitOrderRequest(
            symbol=order.symbol, qty=order.qty,
            side=side, time_in_force=tif,
            limit_price=float(order.limit_price or 0),
        )

        def _call():
            self._client.submit_order(req)

        _with_retry(_call, op="submit")
        return None  # rely on webhook/poll for fills

    def submit_combo(self, combo) -> list:
        """Submit a multi-leg option order via Alpaca's native mleg API.

        Uses order_class="mleg" + OptionLegRequest[] if the installed
        alpaca-py supports it (added 2024). Falls back to leg-by-leg
        submission on older SDKs or when the mleg class raises.

        Returns a list of Fills (empty — Alpaca fills resolve async).
        """
        from alpaca.trading.enums import OrderSide, TimeInForce

        # Detect mleg support at call time — some environments pin an
        # older alpaca-py that predates multi-leg support.
        try:
            from alpaca.trading.requests import (
                LimitOrderRequest, OptionLegRequest,
            )
            try:
                from alpaca.trading.enums import OrderClass, PositionIntent
                has_mleg = True
            except Exception:
                has_mleg = False
        except Exception:
            has_mleg = False

        if not has_mleg:
            # Fall back to base-class leg-by-leg path.
            from .base import BrokerAdapter
            return BrokerAdapter.submit_combo(self, combo)

        legs = []
        for leg in combo.legs:
            legs.append(OptionLegRequest(
                symbol=leg.contract.symbol,
                side=(OrderSide.BUY if leg.side == Side.BUY else OrderSide.SELL),
                ratio_qty=leg.ratio,
            ))
        # Net limit price. Alpaca expects a positive number representing
        # the per-combo debit; a credit spread is expressed as a negative.
        req = LimitOrderRequest(
            qty=combo.qty,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            legs=legs,
            limit_price=round(float(combo.net_limit), 2),
        )

        def _call():
            self._client.submit_order(req)

        try:
            _with_retry(_call, op="submit_combo")
        except Exception:
            # On any failure, fall back to leg-by-leg; the base
            # implementation will log + alert per leg.
            from .base import BrokerAdapter
            return BrokerAdapter.submit_combo(self, combo)
        return []  # filled async via webhook/poll

    def cancel_all(self) -> None:
        _with_retry(lambda: self._client.cancel_orders(), op="cancel_all")

    def close_all_paper_positions(self) -> dict:
        """Nuclear reset: close every open position on the Alpaca paper
        account using Alpaca's `close_all_positions()` endpoint. Used
        when zombie / bug-contaminated positions need to be cleaned
        before starting fresh. Returns a summary dict."""
        def _call():
            return self._client.close_all_positions(cancel_orders=True)
        try:
            result = _with_retry(_call, op="close_all_paper_positions")
            return {
                "closed": len(result) if result else 0,
                "ok": True,
            }
        except Exception as e:                               # noqa: BLE001
            return {"closed": 0, "ok": False, "error": str(e)}

    def flatten_all(self, mark_prices=None) -> None:
        # `mark_prices` accepted for adapter-interface compatibility but
        # unused here — Alpaca closes at market on its side.
        _with_retry(
            lambda: self._client.close_all_positions(cancel_orders=True),
            op="flatten_all",
        )
