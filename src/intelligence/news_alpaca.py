"""AlpacaNewsProvider — free news stream bundled with any Alpaca account.

Pulls recent news items tagged to a symbol via alpaca-py's news REST API.
Lazy-imports the SDK so the package works without it installed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .news import NewsProvider, NewsItem


class AlpacaNewsProvider(NewsProvider):
    def __init__(self, api_key: str, api_secret: str,
                 default_lookback_hours: int = 2,
                 page_limit: int = 20):
        self._api_key = api_key
        self._api_secret = api_secret
        self._lookback = default_lookback_hours
        self._page_limit = page_limit
        self._client = None
        try:
            from alpaca.data.historical.news import NewsClient
            self._client = NewsClient(api_key, api_secret)
        except Exception:
            self._client = None

    def fetch(self, symbol: str) -> List[NewsItem]:
        if not self._client:
            return []
        try:
            from alpaca.data.requests import NewsRequest
            since = datetime.now(tz=timezone.utc) - timedelta(hours=self._lookback)
            req = NewsRequest(
                symbols=symbol, start=since,
                limit=self._page_limit, include_content=False,
            )
            resp = self._client.get_news(req)
        except Exception:
            return []

        # alpaca-py returns a NewsSet-like object with .data or a list
        rows = getattr(resp, "data", None) or resp
        out: List[NewsItem] = []
        for r in rows or []:
            try:
                out.append(NewsItem(
                    symbol=symbol,
                    headline=str(getattr(r, "headline", "") or ""),
                    source=str(getattr(r, "source", "alpaca") or "alpaca"),
                    published_at=getattr(r, "created_at", None) or datetime.now(tz=timezone.utc),
                    url=str(getattr(r, "url", "") or ""),
                ))
            except Exception:
                continue
        return out
