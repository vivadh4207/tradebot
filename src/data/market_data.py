"""Market data adapters.

SyntheticDataAdapter: generates realistic OHLCV bars for dry-run / backtest.
AlpacaDataAdapter:    wraps alpaca-py REST (bars) — lazy import so the core
                      package works without alpaca-py installed.
"""
from __future__ import annotations

import abc
import math
import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import numpy as np

from ..core.types import Bar, Quote
from ..core.clock import ET


class MarketDataAdapter(abc.ABC):
    """Base class for any market-data source."""

    @abc.abstractmethod
    def get_bars(self, symbol: str, *, limit: int = 200,
                 timeframe_minutes: int = 1,
                 end: Optional[datetime] = None) -> List[Bar]: ...

    @abc.abstractmethod
    def latest_quote(self, symbol: str) -> Optional[Quote]: ...

    def latest_price(self, symbol: str) -> Optional[float]:
        q = self.latest_quote(symbol)
        if q and q.mid > 0:
            return q.mid
        bars = self.get_bars(symbol, limit=1)
        return bars[-1].close if bars else None


class SyntheticDataAdapter(MarketDataAdapter):
    """Deterministic GBM-ish bar generator. Used by backtests + dry runs.

    No network required. Reproducible via seed per symbol.
    """

    def __init__(self, seed: int = 42, annualized_vol: float = 0.20,
                 drift: float = 0.0, start_price: float = 500.0):
        self._seed = seed
        self._vol = annualized_vol
        self._drift = drift
        self._start_price = start_price
        self._cache: Dict[str, List[Bar]] = {}

    def _symbol_seed(self, symbol: str) -> int:
        return self._seed + (sum(ord(c) for c in symbol) % 1000)

    def get_bars(self, symbol: str, *, limit: int = 200,
                 timeframe_minutes: int = 1,
                 end: Optional[datetime] = None) -> List[Bar]:
        key = f"{symbol}:{timeframe_minutes}"
        if key in self._cache and len(self._cache[key]) >= limit:
            return self._cache[key][-limit:]

        rng = np.random.default_rng(self._symbol_seed(symbol))
        dt = timeframe_minutes / (60 * 24 * 252)          # in trading years
        sigma_step = self._vol * math.sqrt(dt)
        mu_step = (self._drift - 0.5 * self._vol * self._vol) * dt

        n = max(limit, 200)
        shocks = rng.standard_normal(n)
        log_returns = mu_step + sigma_step * shocks
        prices = self._start_price * np.exp(np.cumsum(log_returns))

        end = end or datetime.now(tz=ET)
        bars: List[Bar] = []
        for i, p in enumerate(prices):
            ts = end - timedelta(minutes=timeframe_minutes * (n - i - 1))
            open_p = prices[i - 1] if i > 0 else p
            high = max(open_p, float(p)) * (1 + abs(rng.normal(0, 0.0005)))
            low = min(open_p, float(p)) * (1 - abs(rng.normal(0, 0.0005)))
            vol = max(100, int(abs(rng.normal(50000, 20000))))
            bars.append(Bar(
                symbol=symbol, ts=ts,
                open=float(open_p), high=float(high), low=float(low),
                close=float(p), volume=float(vol),
                vwap=float((open_p + high + low + p) / 4),
            ))
        self._cache[key] = bars
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


class AlpacaDataAdapter(MarketDataAdapter):
    """Live Alpaca market data. Imports alpaca-py lazily.

    Falls back to SyntheticDataAdapter if credentials or SDK are missing.
    """

    def __init__(self, api_key: str, api_secret: str,
                 fallback: Optional[MarketDataAdapter] = None):
        self._api_key = api_key
        self._api_secret = api_secret
        self._fallback = fallback or SyntheticDataAdapter()
        self._client = None
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(api_key, api_secret)
        except Exception:
            self._client = None

    def get_bars(self, symbol: str, *, limit: int = 200,
                 timeframe_minutes: int = 1,
                 end: Optional[datetime] = None) -> List[Bar]:
        if not self._client:
            return self._fallback.get_bars(symbol, limit=limit,
                                           timeframe_minutes=timeframe_minutes, end=end)
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)
            end = end or datetime.now(tz=ET)
            start = end - timedelta(minutes=timeframe_minutes * (limit + 10))
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                                   start=start, end=end, limit=limit)
            resp = self._client.get_stock_bars(req)
            bars_raw = resp.data.get(symbol, []) if hasattr(resp, "data") else []
            out: List[Bar] = []
            for b in bars_raw:
                out.append(Bar(
                    symbol=symbol, ts=b.timestamp,
                    open=float(b.open), high=float(b.high),
                    low=float(b.low), close=float(b.close),
                    volume=float(b.volume),
                    vwap=float(getattr(b, "vwap", 0.0)) or None,
                ))
            return out[-limit:]
        except Exception:
            return self._fallback.get_bars(symbol, limit=limit,
                                           timeframe_minutes=timeframe_minutes, end=end)

    def latest_quote(self, symbol: str) -> Optional[Quote]:
        if not self._client:
            return self._fallback.latest_quote(symbol)
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            resp = self._client.get_stock_latest_quote(req)
            q = resp.get(symbol) if isinstance(resp, dict) else None
            if q is None:
                return self._fallback.latest_quote(symbol)
            return Quote(
                symbol=symbol, ts=q.timestamp,
                bid=float(q.bid_price), ask=float(q.ask_price),
                bid_size=float(getattr(q, "bid_size", 0) or 0),
                ask_size=float(getattr(q, "ask_size", 0) or 0),
            )
        except Exception:
            return self._fallback.latest_quote(symbol)
