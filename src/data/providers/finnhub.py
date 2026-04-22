"""Finnhub data provider — news + sentiment + quote cross-check.

Finnhub strengths:
  - Free-tier-viable (60 req/min) company news with tickers already tagged
  - News-sentiment score (-1..+1) per item
  - Decent real-time equity quotes as a cross-check against Alpaca
  - Earnings calendar (useful for catalysts module eventually)

Weak spots (skip these — other providers are better):
  - Options chain: Finnhub's options API is limited + behind paid tiers
  - Level-2 quotes: paid only

Enable with FINNHUB_KEY in .env. If the key is missing the provider
silently reports is_enabled()=False and returns None from every call.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib import parse, request, error

from .base import (
    MarketDataProvider, ProviderNewsItem, ProviderOptionRow, ProviderQuote,
)


_log = logging.getLogger(__name__)
_BASE = "https://finnhub.io/api/v1"


class FinnhubProvider:
    name = "finnhub"

    def __init__(self, api_key: Optional[str] = None,
                 *, timeout_sec: float = 6.0):
        self._key = (api_key
                     or os.getenv("FINNHUB_KEY")
                     or os.getenv("FINNHUB_API_KEY")
                     or "").strip()
        self._timeout = float(timeout_sec)

    def is_enabled(self) -> bool:
        return bool(self._key)

    # ------------------------------------------------ quotes

    def latest_quote(self, symbol: str) -> Optional[ProviderQuote]:
        if not self.is_enabled():
            return None
        # Finnhub's /quote returns OHLC + current price — no bid/ask on free
        # tier, so we report price for both sides. Caller treats it as
        # cross-check, not execution source.
        q = self._get("/quote", {"symbol": symbol.upper()})
        if not q:
            return None
        try:
            c = float(q.get("c", 0))
            if c <= 0:
                return None
            return ProviderQuote(
                symbol=symbol.upper(),
                bid=c, ask=c, mid=c,
                ts=datetime.now(tz=timezone.utc).isoformat(),
                source=self.name,
            )
        except Exception:
            return None

    # ------------------------------------------------ chain
    # Finnhub options are gated behind paid tiers + their options endpoint
    # has flaky data quality. We intentionally decline here so the
    # aggregator skips us for chain and asks Polygon / Alpaca.

    def option_chain(self, underlying: str, expiry: Optional[str] = None
                     ) -> Optional[List[ProviderOptionRow]]:
        return None

    # ------------------------------------------------ news

    def news(self, symbol: Optional[str] = None, limit: int = 20
             ) -> Optional[List[ProviderNewsItem]]:
        if not self.is_enabled():
            return None
        if symbol:
            # Company-specific news with sentiment tagging.
            now = datetime.now(tz=timezone.utc).date()
            frm = (now - timedelta(days=3)).isoformat()
            to = now.isoformat()
            rows = self._get("/company-news",
                              {"symbol": symbol.upper(), "from": frm, "to": to})
        else:
            # Market-wide general news.
            rows = self._get("/news", {"category": "general"})
        if not rows or not isinstance(rows, list):
            return None

        out: List[ProviderNewsItem] = []
        for r in rows[:limit]:
            try:
                ts = datetime.fromtimestamp(
                    int(r.get("datetime", 0)), tz=timezone.utc,
                ).isoformat()
            except Exception:
                ts = ""
            tickers = []
            if symbol:
                tickers = [symbol.upper()]
            out.append(ProviderNewsItem(
                ts=ts,
                headline=str(r.get("headline", ""))[:240],
                summary=str(r.get("summary", ""))[:500],
                url=str(r.get("url", "")),
                tickers=tickers,
                sentiment_score=None,   # separate endpoint; fetch on demand
                source=self.name,
            ))
        return out

    def news_sentiment(self, symbol: str) -> Optional[float]:
        """One-shot aggregate sentiment for a symbol [-1..+1].
        Convenience wrapper over /news-sentiment."""
        if not self.is_enabled():
            return None
        data = self._get("/news-sentiment", {"symbol": symbol.upper()})
        if not data:
            return None
        try:
            # Finnhub returns companyNewsScore in [0..1]; map to [-1..+1].
            s = float(data.get("companyNewsScore", 0.5))
            return max(-1.0, min(1.0, (s - 0.5) * 2.0))
        except Exception:
            return None

    # ------------------------------------------------ http

    def _get(self, path: str, params: dict) -> Optional[dict]:
        params = dict(params, token=self._key)
        url = f"{_BASE}{path}?{parse.urlencode(params)}"
        try:
            req = request.Request(url, headers={"Accept": "application/json"})
            with request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except error.HTTPError as e:
            if e.code == 429:
                _log.warning("finnhub_rate_limited path=%s", path)
            else:
                _log.warning("finnhub_http_err path=%s code=%s", path, e.code)
            return None
        except Exception as e:                          # noqa: BLE001
            _log.info("finnhub_network_err path=%s err=%s", path, e)
            return None
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            return None
