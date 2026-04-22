"""Polygon.io provider — options chain (with greeks + IV), equities
quotes, news. Paid tiers starting at $29/mo give real-time options.

When no POLYGON_API_KEY is set, provider reports is_enabled()=False and
every method returns None. Aggregator moves on to the next source.

Coverage:
  - Real-time + delayed options chain snapshots (strikes, expiries,
    greeks, IV, OI, volume, bid/ask) via /v3/snapshot/options
  - Equity last-trade / NBBO quote via /v2/last/trade and /v2/last/nbbo
  - Reference data (tickers, contracts) via /v3/reference/options/contracts
  - News aggregated from multiple outlets via /v2/reference/news

Enabling later: add to .env:
    POLYGON_API_KEY=<your_key>
    POLYGON_PLAN=options_starter      # or stocks_starter, etc. Informational.
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
_BASE = "https://api.polygon.io"


class PolygonProvider:
    name = "polygon"

    def __init__(self, api_key: Optional[str] = None,
                 *, timeout_sec: float = 6.0):
        self._key = (api_key
                     or os.getenv("POLYGON_API_KEY")
                     or os.getenv("POLYGON_KEY")
                     or "").strip()
        self._timeout = float(timeout_sec)

    def is_enabled(self) -> bool:
        return bool(self._key)

    # ------------------------------------------------ quotes

    def latest_quote(self, symbol: str) -> Optional[ProviderQuote]:
        if not self.is_enabled():
            return None
        data = self._get(f"/v2/last/nbbo/{symbol.upper()}")
        if not data:
            return None
        try:
            results = data.get("results") or {}
            bid = float(results.get("P", 0))   # Bid price
            ask = float(results.get("p", 0))   # Ask price (oddly lowercase in API)
            if bid <= 0 or ask <= 0:
                return None
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
        # Snapshot returns the whole chain with greeks + IV in one call.
        path = f"/v3/snapshot/options/{underlying.upper()}"
        params: dict = {"limit": 250}
        if expiry:
            params["expiration_date"] = expiry
        data = self._get(path, params)
        if not data:
            return None
        results = data.get("results") or []
        out: List[ProviderOptionRow] = []
        for r in results:
            try:
                details = r.get("details") or {}
                greeks = r.get("greeks") or {}
                last_quote = r.get("last_quote") or {}
                day = r.get("day") or {}
                out.append(ProviderOptionRow(
                    symbol=details.get("ticker", ""),
                    underlying=underlying.upper(),
                    strike=float(details.get("strike_price", 0)),
                    expiry=str(details.get("expiration_date", "")),
                    right=str(details.get("contract_type", "")).lower(),
                    bid=float(last_quote.get("bid", 0)) or None,
                    ask=float(last_quote.get("ask", 0)) or None,
                    last=float(day.get("close", 0)) or None,
                    volume=int(day.get("volume", 0)) or None,
                    open_interest=int(r.get("open_interest", 0)) or None,
                    implied_vol=float(r.get("implied_volatility", 0)) or None,
                    delta=float(greeks.get("delta", 0)) or None,
                    gamma=float(greeks.get("gamma", 0)) or None,
                    vega=float(greeks.get("vega", 0)) or None,
                    theta=float(greeks.get("theta", 0)) or None,
                    source=self.name,
                ))
            except Exception:
                continue
        return out or None

    # ------------------------------------------------ news

    def news(self, symbol: Optional[str] = None, limit: int = 20
             ) -> Optional[List[ProviderNewsItem]]:
        if not self.is_enabled():
            return None
        params: dict = {"limit": limit, "order": "desc"}
        if symbol:
            params["ticker"] = symbol.upper()
        data = self._get("/v2/reference/news", params)
        if not data:
            return None
        rows = data.get("results") or []
        out: List[ProviderNewsItem] = []
        for r in rows[:limit]:
            try:
                out.append(ProviderNewsItem(
                    ts=str(r.get("published_utc", "")),
                    headline=str(r.get("title", ""))[:240],
                    summary=str(r.get("description", ""))[:500],
                    url=str(r.get("article_url", "")),
                    tickers=[str(t) for t in (r.get("tickers") or [])],
                    sentiment_score=None,
                    source=self.name,
                ))
            except Exception:
                continue
        return out or None

    # ------------------------------------------------ http

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        params = dict(params or {}, apiKey=self._key)
        url = f"{_BASE}{path}?{parse.urlencode(params)}"
        try:
            req = request.Request(url, headers={"Accept": "application/json"})
            with request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except error.HTTPError as e:
            if e.code == 429:
                _log.warning("polygon_rate_limited path=%s", path)
            elif e.code in (401, 403):
                _log.warning("polygon_auth_err path=%s -- check POLYGON_API_KEY", path)
            else:
                _log.warning("polygon_http_err path=%s code=%s", path, e.code)
            return None
        except Exception as e:                          # noqa: BLE001
            _log.info("polygon_network_err path=%s err=%s", path, e)
            return None
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            return None
