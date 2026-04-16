"""HistoricalMarketDataAdapter — real historical bars for backtests.

Three free sources in order of preference:

  1. Alpaca historical bars — uses existing API keys, best quality, no rate
     limit worth worrying about for 10 symbols / daily or minute data.
  2. yfinance — completely free, no key, up to ~2 years of 1-minute data.
     Slow (per-symbol HTTP) but fine for a one-shot backtest.
  3. Synthetic fallback — if neither source produces data.

Every fetch is cached to `data_cache/*.json` keyed by (symbol, timeframe,
start, end) so re-running a backtest doesn't re-hit the network.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from ..core.types import Bar, Quote
from ..core.clock import ET
from .market_data import MarketDataAdapter, SyntheticDataAdapter


class HistoricalMarketDataAdapter(MarketDataAdapter):
    """Drop-in MarketDataAdapter backed by real historical bars."""

    def __init__(self,
                 symbols: Optional[List[str]] = None,
                 start: Optional[datetime] = None,
                 end: Optional[datetime] = None,
                 timeframe_minutes: int = 1,
                 cache_dir: str = "data_cache",
                 alpaca_key: Optional[str] = None,
                 alpaca_secret: Optional[str] = None,
                 prefer: str = "auto"):
        """prefer: 'alpaca' | 'yfinance' | 'auto'."""
        self.timeframe_minutes = timeframe_minutes
        self.end = end or datetime.now(tz=ET)
        # Default: 30 calendar days backward
        self.start = start or (self.end - timedelta(days=30))
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._ak = alpaca_key or os.getenv("ALPACA_API_KEY_ID", "").strip()
        self._as = alpaca_secret or os.getenv("ALPACA_API_SECRET_KEY", "").strip()
        self._prefer = prefer
        self._synth = SyntheticDataAdapter()
        self._bars: dict[str, List[Bar]] = {}
        if symbols:
            for s in symbols:
                self._bars[s] = self._load(s)

    # ---- MarketDataAdapter interface ----
    def get_bars(self, symbol: str, *, limit: int = 200,
                 timeframe_minutes: int = 1,
                 end: Optional[datetime] = None) -> List[Bar]:
        if symbol not in self._bars:
            self._bars[symbol] = self._load(symbol)
        bars = self._bars[symbol]
        if end is not None:
            bars = [b for b in bars if b.ts <= end]
        return bars[-limit:]

    def latest_quote(self, symbol: str) -> Optional[Quote]:
        bars = self.get_bars(symbol, limit=1)
        if not bars:
            return None
        p = bars[-1].close
        spread = max(0.01, p * 0.0002)
        return Quote(symbol=symbol, ts=bars[-1].ts,
                     bid=p - spread / 2, ask=p + spread / 2,
                     bid_size=100, ask_size=100)

    # ---- cache + fetch ----
    def _cache_key(self, symbol: str) -> Path:
        h = hashlib.sha1(
            f"{symbol}|{self.timeframe_minutes}|"
            f"{self.start.date()}|{self.end.date()}".encode()
        ).hexdigest()[:12]
        return self._cache / f"hist_{symbol}_{self.timeframe_minutes}m_{h}.json"

    def _load(self, symbol: str) -> List[Bar]:
        cache_file = self._cache_key(symbol)
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                return [Bar(symbol=symbol,
                             ts=datetime.fromisoformat(d["ts"]),
                             open=d["o"], high=d["h"], low=d["l"],
                             close=d["c"], volume=d["v"],
                             vwap=d.get("vwap")) for d in data]
            except Exception:
                pass

        bars: List[Bar] = []
        if self._prefer in ("alpaca", "auto") and self._ak and self._as:
            bars = self._fetch_alpaca(symbol)
        if not bars and self._prefer in ("yfinance", "auto"):
            bars = self._fetch_yfinance(symbol)
        if not bars:
            bars = self._synth.get_bars(symbol, limit=2000,
                                         timeframe_minutes=self.timeframe_minutes,
                                         end=self.end)

        try:
            cache_file.write_text(json.dumps(
                [{"ts": b.ts.isoformat(), "o": b.open, "h": b.high,
                   "l": b.low, "c": b.close, "v": b.volume,
                   "vwap": b.vwap} for b in bars]
            ))
        except Exception:
            pass
        return bars

    def _fetch_alpaca(self, symbol: str) -> List[Bar]:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            client = StockHistoricalDataClient(self._ak, self._as)
            tf = TimeFrame(self.timeframe_minutes, TimeFrameUnit.Minute)
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                                    start=self.start, end=self.end)
            resp = client.get_stock_bars(req)
            rows = resp.data.get(symbol, []) if hasattr(resp, "data") else []
            out: List[Bar] = []
            for b in rows:
                out.append(Bar(
                    symbol=symbol, ts=b.timestamp,
                    open=float(b.open), high=float(b.high),
                    low=float(b.low), close=float(b.close),
                    volume=float(b.volume),
                    vwap=float(getattr(b, "vwap", 0.0)) or None,
                ))
            return out
        except Exception:
            return []

    def _fetch_yfinance(self, symbol: str) -> List[Bar]:
        try:
            import yfinance as yf
            # yfinance limits 1-minute to last 7 days; use 5m for >7d windows
            span_days = max(1, (self.end - self.start).days)
            if self.timeframe_minutes == 1 and span_days > 7:
                interval = "5m"
            elif self.timeframe_minutes >= 60:
                interval = "1h"
            elif self.timeframe_minutes >= 1440:
                interval = "1d"
            else:
                interval = f"{self.timeframe_minutes}m"
            df = yf.download(
                symbol, start=self.start.date().isoformat(),
                end=(self.end + timedelta(days=1)).date().isoformat(),
                interval=interval, progress=False, auto_adjust=False,
            )
            if df is None or df.empty:
                return []
            out: List[Bar] = []
            for ts, row in df.iterrows():
                try:
                    if hasattr(ts, "to_pydatetime"):
                        ts = ts.to_pydatetime()
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    out.append(Bar(
                        symbol=symbol, ts=ts,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                        vwap=None,
                    ))
                except Exception:
                    continue
            return out
        except Exception:
            return []
