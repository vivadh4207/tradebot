"""Tradier broker adapter — paper (sandbox) and live trading via the
official Tradier brokerage API.

Uses the SAME TRADIER_TOKEN as the TradierProvider (data layer), plus:

  TRADIER_ACCOUNT_ID=<your_account>    # from Tradier dashboard
  TRADIER_SANDBOX=1                     # default on; unset for live

Broker is paper-first. `live=False` (default) routes to sandbox; only
when the operator sets `live=True` AND settings.yaml live_trading=true
does it hit the real endpoint.

Implements the BrokerAdapter interface so main.py can treat Tradier
identically to AlpacaBroker. Supports:
  - Equity orders (stocks/ETFs)
  - Option orders (OCC format)
  - Positions list
  - Cancel all
  - Combo orders declined (use multi-leg builder separately if needed)

Not built: complex order types (bracket, OCO), streaming fills, margin
accounts. Paper sandbox doesn't support them anyway.
"""
from __future__ import annotations

import json
import logging
import os
import re as _re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib import parse, request, error

from ..core.types import Fill, Order, Position, Side


_log = logging.getLogger(__name__)

_LIVE = "https://api.tradier.com"
_SANDBOX = "https://sandbox.tradier.com"


class TradierBroker:
    """Tradier brokerage adapter. Paper (sandbox) by default."""

    def __init__(
        self,
        *, token: Optional[str] = None,
        account_id: Optional[str] = None,
        sandbox: bool = True,
        timeout_sec: float = 8.0,
    ):
        # Accept any of several common env-var names so the operator
        # can paste whatever Tradier's dashboard gave them without
        # renaming. Priority: explicit arg > TRADIER_TOKEN >
        # TRADIER_ACCOUNT_API_KEY > TRADIER_API_KEY.
        self._token = (token
                        or os.getenv("TRADIER_TOKEN")
                        or os.getenv("TRADIER_ACCOUNT_API_KEY")
                        or os.getenv("TRADIER_API_KEY")
                        or "").strip()
        self._account = (account_id
                          or os.getenv("TRADIER_ACCOUNT_ID")
                          or os.getenv("TRADIER_ACCOUNT_NO")
                          or os.getenv("TRADIER_SANDBOX_ACCOUNT")
                          or os.getenv("TRADIER_SANDBOX_ACCOUNT_NO")
                          or "").strip()
        self._base = _SANDBOX if sandbox else _LIVE
        self._timeout = float(timeout_sec)
        if not self._token:
            raise RuntimeError("TRADIER_TOKEN missing")
        # Auto-discover account_id if not provided.
        if not self._account:
            self._account = self._discover_account_id()
        if not self._account:
            raise RuntimeError(
                "TRADIER_ACCOUNT_ID missing and could not auto-discover. "
                "Check https://dash.tradier.com/profile",
            )
        _log.info("tradier_broker_initialized sandbox=%s account=%s",
                  sandbox, self._account[:6] + "...")

    # ------------------------------------------------ positions

    def positions(self) -> List[Position]:
        from datetime import date as _d
        from ..core.types import OptionRight as _OR
        data = self._get(f"/v1/accounts/{self._account}/positions") or {}
        # Tradier quirk: when the account has no positions, this is
        # the literal string "null" rather than JSON null or an empty
        # dict. Handle all three shapes defensively.
        positions_obj = data.get("positions")
        if not positions_obj or not isinstance(positions_obj, dict):
            return []
        rows = positions_obj.get("position") or []
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            return []
        out: List[Position] = []
        for r in rows:
            try:
                sym = str(r.get("symbol", ""))
                qty = int(r.get("quantity", 0))
                # OCC-format options: 15+ chars
                is_option = len(sym) >= 15 and any(c.isdigit() for c in sym)
                multiplier = 100 if is_option else 1
                # Parse OCC symbol into underlying / expiry / right /
                # strike. Without this, fast_exit can't classify the
                # position (no `right` → can't tell call vs put, no
                # `expiry` → dte() returns 9999, no auto-exit logic
                # fires). Format:
                #   SPY260515C00280000
                #   = ticker (1-6 chars) + YYMMDD + C|P + 8-digit
                #     strike × 1000 (e.g. 280000 = $280.00)
                underlying = sym
                strike = None
                expiry = None
                right = None
                if is_option:
                    try:
                        # The 6-digit date precedes a single C or P.
                        m = _re.search(
                            r"^([A-Z]+)(\d{6})([CP])(\d{8})$", sym
                        )
                        if m:
                            underlying = m.group(1)
                            yy = int(m.group(2)[0:2])
                            mm = int(m.group(2)[2:4])
                            dd = int(m.group(2)[4:6])
                            full_year = 2000 + yy
                            expiry = _d(full_year, mm, dd)
                            right = (_OR.CALL if m.group(3) == "C"
                                       else _OR.PUT)
                            strike = float(m.group(4)) / 1000.0
                    except Exception:
                        pass
                # Tradier returns `cost_basis` as TOTAL dollars paid
                # (qty × price × multiplier). Divide for per-share px.
                cost_basis = float(r.get("cost_basis", 0))
                avg = cost_basis / max(1, abs(qty) * multiplier)
                out.append(Position(
                    symbol=sym,
                    qty=qty,
                    avg_price=round(avg, 4),
                    is_option=is_option,
                    underlying=underlying,
                    strike=strike,
                    expiry=expiry,
                    right=right,
                    multiplier=multiplier,
                    entry_ts=0.0,
                ))
            except Exception:
                continue
        return out

    # ------------------------------------------------ submit

    def submit(self, order: Order, **_ignored) -> Optional[Fill]:
        """Submit an order to Tradier. Returns a Fill ONLY if Tradier
        confirms terminal status `filled` (or partial fill with non-zero
        exec_quantity). Returns None on rejection, expiry, or network
        error. Caller treats None as failure and may retry / log.

        Critical pre-submit step for sell_to_close orders: cancel any
        existing pending order for the SAME option_symbol first.
        Tradier counts pending qty against your long position, so a
        stale pending sell makes the new sell get rejected with
        'Sell order is for more shares than current long position.'
        Today's bug-storm root cause."""
        # Cancel pending sells on the same contract before submitting
        # a new sell_to_close. Buy_to_open paths skip this — opens
        # don't conflict.
        is_close = (
            order.is_option
            and self._map_side(order.side, is_option=True)
            in ("sell_to_close", "buy_to_close")
        )
        if is_close:
            try:
                self.cancel_pending_for_symbol(order.symbol)
            except Exception:
                pass

        params: Dict[str, str] = {
            "class": "option" if order.is_option else "equity",
            "symbol": (self._underlying_from_occ(order.symbol)
                       if order.is_option else order.symbol),
            "side": self._map_side(order.side, is_option=order.is_option),
            "quantity": str(int(order.qty)),
            "duration": "day",
            "type": ("limit" if order.limit_price else "market"),
        }
        if order.is_option:
            params["option_symbol"] = order.symbol
        if order.limit_price:
            params["price"] = f"{float(order.limit_price):.2f}"
        resp = self._post(
            f"/v1/accounts/{self._account}/orders", params,
        )
        if not resp:
            return None
        o = resp.get("order") or {}
        if o.get("status") == "rejected" or "error" in o:
            _log.warning(
                "tradier_submit_rejected symbol=%s err=%s",
                order.symbol, o.get("error") or o,
            )
            return None
        order_id = o.get("id")
        if not order_id:
            _log.warning(
                "tradier_submit_no_id symbol=%s resp=%s",
                order.symbol, str(o)[:200],
            )
            return None

        # POLL until terminal (filled / rejected / cancelled / expired
        # / timeout). Sandbox fills usually within 1-3 seconds; give
        # 8s to be safe. This is THE critical change: previously the
        # bot returned a placeholder Fill on accept (status=ok) and
        # never re-checked. If Tradier later rejected at routing,
        # PaperBroker still thought the trade filled → phantom
        # positions → reconcile loop → close storm. Now we wait.
        final = self.poll_until_terminal(str(order_id), timeout_sec=8.0)
        final_status = (final.get("status") or "").lower()

        if final_status == "filled":
            fill_price = float(final.get("avg_fill_price")
                                 or order.limit_price or 0.0)
            exec_qty = int(float(final.get("exec_quantity") or order.qty))
            _log.info(
                "tradier_filled order_id=%s symbol=%s side=%s "
                "qty=%d price=%.4f",
                order_id, order.symbol, order.side.value,
                exec_qty, fill_price,
            )
            try:
                order.tag = (getattr(order, "tag", "") or "") + \
                    f"|tradier_oid={order_id}"
            except Exception:
                pass
            return Fill(order=order, price=fill_price, qty=exec_qty)

        if final_status in ("rejected", "canceled", "cancelled",
                              "expired", "error"):
            _log.warning(
                "tradier_terminal_reject order_id=%s symbol=%s "
                "side=%s qty=%d status=%s reason=%s",
                order_id, order.symbol, order.side.value, order.qty,
                final_status,
                str(final.get("reason_description") or "")[:160],
            )
            return None

        # Still open / pending at timeout — treat as soft-failure so
        # the caller retries instead of accruing a phantom fill.
        # Sandbox occasionally takes >8s on options; the order ID is
        # stashed so a follow-up reconcile can pick up the late fill.
        _log.warning(
            "tradier_pending_at_timeout order_id=%s symbol=%s status=%s",
            order_id, order.symbol, final_status,
        )
        return None

    # ------------------------------------------------ cancel

    def cancel(self, order_id: str) -> bool:
        try:
            resp = self._delete(
                f"/v1/accounts/{self._account}/orders/{order_id}",
            )
            return resp is not None
        except Exception:
            return False

    def cancel_all(self) -> None:
        data = self._get(f"/v1/accounts/{self._account}/orders") or {}
        orders_obj = data.get("orders")
        if not orders_obj or not isinstance(orders_obj, dict):
            return
        rows = orders_obj.get("order") or []
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            return
        for o in rows:
            oid = o.get("id")
            if oid and o.get("status") in ("open", "pending"):
                self.cancel(str(oid))

    def cancel_pending_for_symbol(self, option_symbol: str) -> int:
        """Cancel every open/pending order for `option_symbol`.

        This is critical before sending a sell_to_close: Tradier
        counts pending sell qty against your long position, so a stale
        pending sell makes new sells get rejected with "Sell order is
        for more shares than current long position." Returns # cancels."""
        if not option_symbol:
            return 0
        data = self._get(f"/v1/accounts/{self._account}/orders") or {}
        orders_obj = data.get("orders")
        if not orders_obj or not isinstance(orders_obj, dict):
            return 0
        rows = orders_obj.get("order") or []
        if isinstance(rows, dict):
            rows = [rows]
        n = 0
        for o in rows:
            sym = (o.get("option_symbol") or o.get("symbol") or "")
            if (sym == option_symbol
                    and o.get("status") in ("open", "pending")):
                if self.cancel(str(o.get("id"))):
                    n += 1
        if n:
            _log.info("tradier_cancelled_pending symbol=%s n=%d",
                        option_symbol, n)
        return n

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Returns the current order record (status, fill price, etc.)
        or None on error."""
        data = self._get(
            f"/v1/accounts/{self._account}/orders/{order_id}"
        )
        if not data:
            return None
        return data.get("order") or {}

    def poll_until_terminal(self, order_id: str,
                              timeout_sec: float = 8.0,
                              interval_sec: float = 1.0) -> Dict[str, Any]:
        """Poll order status until it reaches a terminal state
        (filled / rejected / canceled / expired) or timeout.
        Returns the final order dict (with status field set, possibly
        'open' / 'pending' if still in flight at timeout)."""
        import time as _t
        deadline = _t.time() + float(timeout_sec)
        last: Dict[str, Any] = {}
        while _t.time() < deadline:
            o = self.get_order_status(order_id) or {}
            last = o
            status = (o.get("status") or "").lower()
            if status in ("filled", "rejected", "canceled",
                            "cancelled", "expired", "error"):
                return o
            _t.sleep(interval_sec)
        return last

    # ------------------------------------------------ account

    def history(self, *, from_date: Optional[str] = None,
                   to_date: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch filled-order history. Falls back to /orders endpoint
        when /history returns nothing (common on Tradier sandbox).
        Normalized dicts: symbol, side, qty, fill_price, ts, amount."""
        from datetime import datetime as _dt, timedelta as _td
        if not from_date:
            from_date = (_dt.now() - _td(days=7)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = _dt.now().strftime("%Y-%m-%d")

        # Try /history first
        params = {
            "start": from_date, "end": to_date,
            "type": "trade", "limit": int(limit),
        }
        data = self._get(
            f"/v1/accounts/{self._account}/history", params
        ) or {}
        hist = data.get("history") or {}
        events = []
        if isinstance(hist, dict):
            ev = hist.get("event")
            events = [ev] if isinstance(ev, dict) else (ev or [])

        out: List[Dict[str, Any]] = []
        for ev in events:
            trade = ev.get("trade") or {}
            if not trade:
                continue
            out.append({
                "ts": ev.get("date"),
                "symbol": trade.get("symbol", ""),
                "side": trade.get("trade_type") or trade.get("side", ""),
                "qty": abs(float(trade.get("quantity", 0))),
                "fill_price": float(trade.get("price", 0)),
                "commission": float(trade.get("commission", 0) or 0),
                "description": ev.get("description", ""),
                "amount": float(ev.get("amount", 0) or 0),
            })
        if out:
            return out

        # Fallback: /orders endpoint — includes filled orders on sandbox
        _log.info("tradier_history_empty_try_orders")
        data2 = self._get(
            f"/v1/accounts/{self._account}/orders",
            {"includeTags": "true"},
        ) or {}
        orders_obj = data2.get("orders") or {}
        if not isinstance(orders_obj, dict):
            return []
        orders = orders_obj.get("order")
        if orders is None:
            return []
        if isinstance(orders, dict):
            orders = [orders]
        for o in orders:
            status = (o.get("status") or "").lower()
            if status not in ("filled", "partially_filled"):
                continue
            avg_fill = float(o.get("avg_fill_price", 0) or 0)
            if avg_fill <= 0:
                continue
            qty = float(o.get("exec_quantity") or o.get("quantity") or 0)
            side = str(o.get("side", "")).lower()
            sym = o.get("option_symbol") or o.get("symbol", "")
            out.append({
                "ts": o.get("transaction_date") or o.get("create_date"),
                "symbol": sym,
                "side": side,
                "qty": abs(qty),
                "fill_price": avg_fill,
                "commission": 0.0,
                "description": o.get("type", ""),
                "amount": -(avg_fill * qty * 100) if "buy" in side
                            else (avg_fill * qty * 100),
            })
        return out

    def account(self) -> Dict[str, Any]:
        """Account snapshot for dashboard / reconcile. Returns dict with:
        cash · equity · buying_power · day_pnl (today's realized +
        unrealized vs last close) · open_pl (total unrealized on open
        positions)."""
        bal = self._get(f"/v1/accounts/{self._account}/balances") or {}
        b = bal.get("balances") or {}
        # Tradier balance payload contains `open_pl`, `close_pl`, `total_cash`,
        # `total_equity`. day_pnl = open_pl + close_pl (today's realized +
        # unrealized combined).
        open_pl = float(b.get("open_pl", 0) or 0)
        close_pl = float(b.get("close_pl", 0) or 0)
        total_equity = float(b.get("total_equity", 0) or 0)
        cash = float(
            b.get("cash", {}).get("cash_available", 0)
            or b.get("total_cash", 0)
            or b.get("cash_available", 0)
            or 0
        )
        return {
            "account_id": self._account,
            "cash": cash,
            "equity": total_equity,
            "open_pl": open_pl,
            "close_pl": close_pl,
            "day_pnl": open_pl + close_pl,
            "buying_power": float(
                b.get("margin", {}).get("stock_buying_power", 0)
                or b.get("stock_buying_power", 0)
                or 0
            ),
        }

    # ------------------------------------------------ submit_combo (not supported)

    def submit_combo(self, combo) -> list:
        """Tradier supports multi-leg option orders but with a more
        complex body than single orders. Keeping it as a stub until
        the operator explicitly needs vertical / iron-condor orders
        via Tradier — for now the existing spreads path stays on
        Alpaca."""
        _log.warning("tradier_submit_combo_not_implemented — use alpaca for spreads")
        return []

    # ------------------------------------------------ helpers

    @staticmethod
    def _underlying_from_occ(occ: str) -> str:
        """Strip the date+right+strike suffix to get 'SPY' from
        'SPY260427P00705000'. Tradier wants the underlying as
        `symbol` and the full OCC in `option_symbol`."""
        import re as _re
        m = _re.match(r"^([A-Z]{1,5})", occ)
        return m.group(1) if m else occ

    @staticmethod
    def _map_side(side: Side, *, is_option: bool) -> str:
        """Tradier uses different side strings for equity vs options.
        Equity: 'buy' | 'sell' | 'sell_short' | 'buy_to_cover'
        Option: 'buy_to_open' | 'sell_to_close' | 'sell_to_open' | 'buy_to_close'

        Heuristic: treat our BUY as buy/buy_to_open and SELL as
        sell/sell_to_close. Short options would need richer metadata
        to distinguish open-short from close-long; not supported here.
        """
        if is_option:
            return "buy_to_open" if side == Side.BUY else "sell_to_close"
        return "buy" if side == Side.BUY else "sell"

    def _discover_account_id(self) -> str:
        """Hit /v1/user/profile to get the first account id."""
        data = self._get("/v1/user/profile") or {}
        try:
            profile = data.get("profile") or {}
            account = profile.get("account") or {}
            if isinstance(account, list):
                account = account[0]
            return str(account.get("account_number", ""))
        except Exception:
            return ""

    # ------------------------------------------------ http

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        url = f"{self._base}{path}"
        if params:
            url += "?" + parse.urlencode(params)
        return self._http("GET", url, None)

    def _post(self, path: str, params: dict) -> Optional[dict]:
        return self._http("POST", f"{self._base}{path}",
                           parse.urlencode(params).encode("utf-8"))

    def _delete(self, path: str) -> Optional[dict]:
        return self._http("DELETE", f"{self._base}{path}", None)

    def _http(self, method: str, url: str, data: Optional[bytes]
              ) -> Optional[dict]:
        try:
            req = request.Request(
                url, data=data, method=method,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept": "application/json",
                    **({"Content-Type": "application/x-www-form-urlencoded"}
                         if data else {}),
                },
            )
            with request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except error.HTTPError as e:
            _log.warning("tradier_http_%s path=%s code=%s body=%s",
                         method, url, e.code,
                         e.read().decode("utf-8", errors="replace")[:300])
            return None
        except Exception as e:                          # noqa: BLE001
            _log.info("tradier_network_err method=%s err=%s", method, e)
            return None
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            return None


def build_tradier_broker(*, sandbox: Optional[bool] = None,
                          ) -> Optional["TradierBroker"]:
    """Factory. Returns None when TRADIER_TOKEN isn't set.

    `sandbox` param overrides the env. If both None, checks
    TRADIER_SANDBOX env (default: sandbox mode).
    """
    if sandbox is None:
        sandbox = os.getenv("TRADIER_SANDBOX", "1").strip().lower() not in (
            "0", "false", "no", "off",
        )
    token = (os.getenv("TRADIER_TOKEN")
              or os.getenv("TRADIER_ACCOUNT_API_KEY")
              or os.getenv("TRADIER_API_KEY") or "").strip()
    if not token:
        return None
    try:
        return TradierBroker(sandbox=sandbox)
    except Exception as e:                              # noqa: BLE001
        _log.warning("tradier_broker_build_failed err=%s", e)
        return None
