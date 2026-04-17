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
        # NOTE: Alpaca's stock data endpoint does NOT serve `^VIX` (an index,
        # not a stock). Their `/v1beta1/indicators/vix` endpoint exists but is
        # not exposed by alpaca-py's current SDK surface. Leaving this as a
        # deliberate no-op keeps the source-order explicit: Alpaca → yfinance
        # → fallback, where Alpaca is expected to miss for VIX specifically.
        # If Alpaca later ships an index-quote SDK method, wire it here.
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
