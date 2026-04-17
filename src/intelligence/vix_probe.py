"""Live VIX fetcher with a 60-second cache.

Source priority:
  1. Alpaca — if we already have credentials in env, use the same stock
     data client we use for equities ($VIX is available as an index).
  2. yfinance — `^VIX` works with no API key.
  3. Static fallback — returns the `fallback_vix` (default 15.0) when
     neither data source is available. This keeps the bot operational
     even when the network is down.

The cache prevents hammering Yahoo / Alpaca on every 3-minute loop tick.
Typical cadence: ~6 calls per trading hour with TTL=60s.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class VixReading:
    value: float
    source: str           # 'alpaca' | 'yfinance' | 'fallback'
    ts: float             # epoch seconds when fetched


class VixProbe:
    def __init__(self,
                 ttl_seconds: int = 60,
                 fallback_vix: float = 15.0,
                 prefer: str = "auto"):
        """prefer: 'auto' | 'alpaca' | 'yfinance' | 'fallback'."""
        self.ttl = ttl_seconds
        self.fallback_vix = fallback_vix
        self.prefer = prefer
        self._lock = threading.Lock()
        self._cached: Optional[VixReading] = None

    def get(self) -> VixReading:
        with self._lock:
            now = time.time()
            if self._cached and (now - self._cached.ts) < self.ttl:
                return self._cached
            reading = self._fetch()
            self._cached = reading
            return reading

    def value(self) -> float:
        return self.get().value

    def _fetch(self) -> VixReading:
        order = []
        if self.prefer == "auto":
            order = ["alpaca", "yfinance"]
        elif self.prefer in ("alpaca", "yfinance"):
            order = [self.prefer]
        # fallback is always last
        for src in order:
            if src == "alpaca":
                v = self._try_alpaca()
                if v is not None:
                    return VixReading(value=v, source="alpaca", ts=time.time())
            elif src == "yfinance":
                v = self._try_yfinance()
                if v is not None:
                    return VixReading(value=v, source="yfinance", ts=time.time())
        return VixReading(value=self.fallback_vix, source="fallback",
                          ts=time.time())

    @staticmethod
    def _try_alpaca() -> Optional[float]:
        key = os.getenv("ALPACA_API_KEY_ID", "").strip()
        secret = os.getenv("ALPACA_API_SECRET_KEY", "").strip()
        if not (key and secret) or key.startswith("your_"):
            return None
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            client = StockHistoricalDataClient(key, secret)
            # Alpaca surfaces VIX under ^VIX via some feeds; try VIX plain too.
            for sym in ("^VIX", "VIX"):
                try:
                    resp = client.get_stock_latest_quote(
                        StockLatestQuoteRequest(symbol_or_symbols=sym)
                    )
                    q = resp.get(sym) if isinstance(resp, dict) else None
                    if q is None:
                        continue
                    bid = float(getattr(q, "bid_price", 0) or 0)
                    ask = float(getattr(q, "ask_price", 0) or 0)
                    mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)
                    if mid > 0:
                        return mid
                except Exception:
                    continue
            return None
        except Exception:
            return None

    @staticmethod
    def _try_yfinance() -> Optional[float]:
        try:
            import yfinance as yf
            t = yf.Ticker("^VIX")
            # yfinance exposes the last price via .fast_info or history()
            try:
                fi = t.fast_info
                v = float(getattr(fi, "last_price", 0) or 0)
                if v > 0:
                    return v
            except Exception:
                pass
            # fallback: most-recent 1-day bar
            hist = t.history(period="5d", interval="1d", auto_adjust=False)
            if hist is None or hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception:
            return None
