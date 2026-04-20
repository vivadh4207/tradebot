"""Historical daily bars for walk-forward. Cached to disk to avoid re-fetching.

Uses Alpaca when credentials are present; falls back to SyntheticDataAdapter
at daily granularity otherwise.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional

from ..core.types import Bar
from ..core.clock import ET
from ..data.market_data import SyntheticDataAdapter


class HistoricalDataProvider:
    def __init__(self, cache_dir: str = "data_cache",
                 alpaca_key: Optional[str] = None,
                 alpaca_secret: Optional[str] = None):
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._alpaca_key = alpaca_key or os.getenv("ALPACA_API_KEY_ID", "").strip()
        self._alpaca_secret = alpaca_secret or os.getenv("ALPACA_API_SECRET_KEY", "").strip()

    def daily_bars(self, symbol: str, years: int = 3) -> List[Bar]:
        """Return up to `years` of daily bars. Checks cache first."""
        cache_file = self._cache / f"{symbol}_daily_{years}y.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                return [Bar(symbol=symbol,
                             ts=datetime.fromisoformat(d["ts"]),
                             open=d["o"], high=d["h"], low=d["l"],
                             close=d["c"], volume=d["v"]) for d in data]
            except Exception:
                pass
        bars = self._fetch(symbol, years)
        try:
            cache_file.write_text(json.dumps(
                [{"ts": b.ts.isoformat(), "o": b.open, "h": b.high,
                   "l": b.low, "c": b.close, "v": b.volume} for b in bars]
            ))
        except Exception:
            pass
        return bars

    def _fetch(self, symbol: str, years: int) -> List[Bar]:
        if self._alpaca_key and self._alpaca_secret:
            try:
                return self._fetch_alpaca(symbol, years)
            except Exception:
                pass
        return self._fetch_synthetic(symbol, years)

    def _fetch_alpaca(self, symbol: str, years: int) -> List[Bar]:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        client = StockHistoricalDataClient(self._alpaca_key, self._alpaca_secret)
        end = datetime.now(tz=ET)
        start = end - timedelta(days=years * 365)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                start=start, end=end)
        resp = client.get_stock_bars(req)
        rows = resp.data.get(symbol, []) if hasattr(resp, "data") else []
        return [Bar(symbol=symbol, ts=b.timestamp,
                     open=float(b.open), high=float(b.high),
                     low=float(b.low), close=float(b.close),
                     volume=float(b.volume)) for b in rows]

    def _fetch_synthetic(self, symbol: str, years: int) -> List[Bar]:
        d = SyntheticDataAdapter()
        days = years * 252
        return d.get_bars(symbol, limit=days, timeframe_minutes=24 * 60)
