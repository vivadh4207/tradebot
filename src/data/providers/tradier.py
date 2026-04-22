"""Tradier data provider — free options chain + quotes when you have
a funded Tradier brokerage account.

When no TRADIER_TOKEN is set, provider reports is_enabled()=False and
every method returns None. Aggregator moves on to the next source.

Coverage:
  - Equities quotes (real-time with funded paid data, or 15-min delayed
    on sandbox): /v1/markets/quotes
  - Options chain with greeks: /v1/markets/options/chains
  - Option expirations per symbol: /v1/markets/options/expirations

Enable via .env:
    TRADIER_TOKEN=<bearer_token>
    TRADIER_ACCOUNT=<account_id>       # optional, not used for data
    TRADIER_SANDBOX=1                  # use sandbox endpoint (delayed)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional
from urllib import parse, request, error

from .base import (
    MarketDataProvider, ProviderNewsItem, ProviderOptionRow, ProviderQuote,
)


_log = logging.getLogger(__name__)

_LIVE = "https://api.tradier.com"
_SANDBOX = "https://sandbox.tradier.com"


class TradierProvider:
    name = "tradier"

    def __init__(self, token: Optional[str] = None,
                 *, sandbox: Optional[bool] = None,
                 timeout_sec: float = 6.0):
        self._token = (token
                        or os.getenv("TRADIER_TOKEN")
                        or os.getenv("TRADIER_ACCOUNT_API_KEY")
                        or os.getenv("TRADIER_API_KEY")
                        or "").strip()
        use_sandbox = (sandbox if sandbox is not None
                       else os.getenv("TRADIER_SANDBOX", "").strip()
                             in ("1", "true", "yes"))
        self._base = _SANDBOX if use_sandbox else _LIVE
        self._timeout = float(timeout_sec)

    def is_enabled(self) -> bool:
        return bool(self._token)

    # ------------------------------------------------ quotes

    def latest_quote(self, symbol: str) -> Optional[ProviderQuote]:
        if not self.is_enabled():
            return None
        data = self._get("/v1/markets/quotes", {"symbols": symbol.upper()})
        if not data:
            return None
        try:
            q = ((data.get("quotes") or {}).get("quote"))
            if isinstance(q, list):
                q = q[0] if q else None
            if not q:
                return None
            bid = float(q.get("bid", 0))
            ask = float(q.get("ask", 0))
            last = float(q.get("last", 0))
            if bid <= 0 or ask <= 0:
                if last <= 0:
                    return None
                bid = ask = last
            return ProviderQuote(
                symbol=symbol.upper(),
                bid=bid, ask=ask, mid=(bid + ask) / 2.0,
                ts=datetime.now(tz=timezone.utc).isoformat(),
                source=self.name,
            )
        except Exception:
            return None

    # ------------------------------------------------ options chain

    def option_chain(self, underlying: str, expiry: Optional[str] = None
                     ) -> Optional[List[ProviderOptionRow]]:
        if not self.is_enabled():
            return None
        if expiry is None:
            # Pull nearest expiry if not specified.
            expiry = self._nearest_expiry(underlying)
            if expiry is None:
                return None
        data = self._get("/v1/markets/options/chains",
                          {"symbol": underlying.upper(),
                           "expiration": expiry,
                           "greeks": "true"})
        if not data:
            return None
        try:
            rows = ((data.get("options") or {}).get("option")) or []
            if isinstance(rows, dict):
                rows = [rows]
            out: List[ProviderOptionRow] = []
            for r in rows:
                greeks = r.get("greeks") or {}
                out.append(ProviderOptionRow(
                    symbol=str(r.get("symbol", "")),
                    underlying=underlying.upper(),
                    strike=float(r.get("strike", 0)),
                    expiry=str(r.get("expiration_date", expiry)),
                    right=str(r.get("option_type", "")).lower(),
                    bid=float(r.get("bid", 0)) or None,
                    ask=float(r.get("ask", 0)) or None,
                    last=float(r.get("last", 0)) or None,
                    volume=int(r.get("volume", 0)) or None,
                    open_interest=int(r.get("open_interest", 0)) or None,
                    implied_vol=(float(greeks.get("mid_iv", 0))
                                 or float(greeks.get("bid_iv", 0))
                                 or None),
                    delta=float(greeks.get("delta", 0)) or None,
                    gamma=float(greeks.get("gamma", 0)) or None,
                    vega=float(greeks.get("vega", 0)) or None,
                    theta=float(greeks.get("theta", 0)) or None,
                    source=self.name,
                ))
            return out or None
        except Exception:
            return None

    def _nearest_expiry(self, underlying: str) -> Optional[str]:
        data = self._get("/v1/markets/options/expirations",
                          {"symbol": underlying.upper()})
        if not data:
            return None
        try:
            exps = ((data.get("expirations") or {}).get("date")) or []
            if isinstance(exps, str):
                exps = [exps]
            # Pick the earliest future expiry.
            today = datetime.now(tz=timezone.utc).date().isoformat()
            future = [e for e in exps if e >= today]
            return future[0] if future else None
        except Exception:
            return None

    # ------------------------------------------------ news
    # Tradier doesn't publish a news endpoint on free tier — we decline
    # and let Finnhub/Alpaca/Polygon cover that.

    def news(self, symbol: Optional[str] = None, limit: int = 20
             ) -> Optional[List[ProviderNewsItem]]:
        return None

    # ------------------------------------------------ http

    def _get(self, path: str, params: dict) -> Optional[dict]:
        url = f"{self._base}{path}?{parse.urlencode(params)}"
        try:
            req = request.Request(url, headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
            })
            with request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except error.HTTPError as e:
            if e.code in (401, 403):
                _log.warning("tradier_auth_err path=%s -- check TRADIER_TOKEN", path)
            elif e.code == 429:
                _log.warning("tradier_rate_limited path=%s", path)
            else:
                _log.warning("tradier_http_err path=%s code=%s", path, e.code)
            return None
        except Exception as e:                          # noqa: BLE001
            _log.info("tradier_network_err path=%s err=%s", path, e)
            return None
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            return None
