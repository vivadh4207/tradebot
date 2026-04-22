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


# Realistic per-symbol starting prices so the synthetic fallback produces
# sensible spots when Alpaca is unreachable. Values approximate 2026 mid-year
# closes; updated when the universe changes. Missing symbols fall through to
# a hash-derived default so we don't concentrate everything at $500.
_SYMBOL_START_PRICE: Dict[str, float] = {
    # Index + broad-market ETFs (2026 levels — updated per operator
    # feedback that SPY synthetic was ~$140 below real market)
    "SPY": 720.0, "QQQ": 640.0, "IWM": 290.0, "DIA": 575.0,
    "VOO": 680.0, "VTI": 360.0, "IVV": 720.0,
    # Sector SPDRs
    "XLF": 48.0, "XLE": 95.0, "XLK": 230.0, "XLV": 150.0,
    "XLY": 205.0, "XLC": 95.0, "XLI": 140.0, "XLB": 95.0,
    "XLU": 78.0, "XLP": 80.0, "XBI": 96.0, "XHE": 120.0,
    "XME": 70.0, "XRT": 80.0, "XSD": 265.0, "XOP": 150.0,
    "XHB": 115.0, "XUR": 48.0,
    # Mega-cap tech
    "AAPL": 230.0, "MSFT": 440.0, "NVDA": 900.0, "GOOGL": 180.0,
    "META": 570.0, "AMZN": 195.0, "TSLA": 260.0,
    # Other single names (rough 2026 estimates)
    "AAOI": 40.0, "AMD": 170.0, "APLD": 12.0, "INTC": 22.0,
    "AVGO": 1700.0, "ASML": 950.0, "LRCX": 900.0, "MU": 120.0,
    "TSM": 205.0, "BABA": 90.0, "BIDU": 95.0, "PDD": 140.0,
    "JD": 35.0, "SHOP": 105.0, "CRM": 310.0, "ORCL": 160.0,
    "IBM": 225.0, "CSCO": 55.0, "NFLX": 720.0, "DIS": 105.0,
    "BAC": 45.0, "JPM": 240.0, "WFC": 70.0, "C": 75.0,
    "GS": 520.0, "MS": 115.0, "VZ": 42.0, "T": 22.0,
    "V": 305.0, "MA": 510.0, "PYPL": 72.0, "ADBE": 480.0,
    "NOW": 970.0, "ZM": 80.0, "DOCU": 70.0, "TWLO": 80.0,
    "UBER": 78.0, "LYFT": 15.0, "ABNB": 150.0, "BKNG": 5000.0,
    "EXPE": 190.0, "MAR": 280.0, "RCL": 245.0, "CCL": 25.0,
    "NCLH": 22.0, "DAL": 55.0, "LUV": 32.0, "AAL": 13.0,
    "UAL": 85.0, "AXP": 290.0, "HOOD": 48.0,
}


class SyntheticDataAdapter(MarketDataAdapter):
    """Deterministic GBM-ish bar generator. Used by backtests + dry runs,
    and as the fallback when AlpacaDataAdapter can't reach the network.

    Per-symbol starting prices come from _SYMBOL_START_PRICE so every
    ticker ends up in the right order of magnitude. Missing symbols fall
    through to a hash-derived default. Without this, every symbol
    drifted near the same 500.0 default and the bot thought NFLX and
    LUV both traded at $500 — broke strike selection completely.
    """

    def __init__(self, seed: int = 42, annualized_vol: float = 0.20,
                 drift: float = 0.0, start_price: float = 500.0):
        self._seed = seed
        self._vol = annualized_vol
        self._drift = drift
        self._default_start_price = start_price
        self._cache: Dict[str, List[Bar]] = {}

    def _start_for(self, symbol: str) -> float:
        sym = (symbol or "").upper()
        mapped = _SYMBOL_START_PRICE.get(sym)
        if mapped is not None:
            return mapped
        # Hash-derived fallback: spread unknown symbols across [$20, $200]
        # deterministically so they don't all collide near the default.
        h = sum(ord(c) for c in sym) % 180
        return 20.0 + float(h)

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
        # Per-symbol realistic starting price (not a flat 500 default).
        prices = self._start_for(symbol) * np.exp(np.cumsum(log_returns))

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
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "alpaca_data_client_init_failed: %s", _e
            )
            self._client = None

    def get_bars(self, symbol: str, *, limit: int = 200,
                 timeframe_minutes: int = 1,
                 end: Optional[datetime] = None) -> List[Bar]:
        if not self._client:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "alpaca_bars_no_client_falling_back symbol=%s", symbol
            )
            return self._fallback.get_bars(symbol, limit=limit,
                                           timeframe_minutes=timeframe_minutes, end=end)
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)
            end = end or datetime.now(tz=ET)
            start = end - timedelta(minutes=timeframe_minutes * (limit + 10))
            # Free / paper Alpaca accounts get IEX data, not SIP.
            # Without `feed="iex"` the default is SIP which gets rejected
            # with "subscription does not permit querying recent SIP data".
            # That fallback was silently dropping us to synthetic bars
            # (with stale 2024 prices → SPY thought to be $560 when
            # actual was ~$700 → bought deep-ITM puts by mistake).
            # Override via ALPACA_DATA_FEED env if you have a SIP sub.
            import os as _os
            feed = _os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()
            kw = dict(symbol_or_symbols=symbol, timeframe=tf,
                       start=start, end=end, limit=limit)
            try:
                from alpaca.data.enums import DataFeed as _DataFeed
                kw["feed"] = _DataFeed(feed)
            except Exception:
                # Older alpaca-py: just pass the string
                kw["feed"] = feed
            req = StockBarsRequest(**kw)
            resp = self._client.get_stock_bars(req)
            bars_raw = resp.data.get(symbol, []) if hasattr(resp, "data") else []
            if not bars_raw:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "alpaca_bars_empty_falling_back symbol=%s start=%s end=%s "
                    "(usually means: market data subscription missing, outside "
                    "RTH for this symbol, or symbol not covered by IEX feed)",
                    symbol, start, end,
                )
                return self._fallback.get_bars(symbol, limit=limit,
                                               timeframe_minutes=timeframe_minutes, end=end)
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
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "alpaca_bars_error_falling_back symbol=%s err=%s", symbol, e,
            )
            # Before giving up to the synthetic fallback, try
            # MultiProvider (Tradier / Yahoo / Finnhub) for latest bars
            # — they're real market data; synthetic is last resort.
            alt = _try_multi_provider_bars(symbol, limit=limit,
                                             timeframe_minutes=timeframe_minutes)
            if alt is not None:
                return alt
            # No Discord alert here — the fallback is BY DESIGN and
            # used to be per-tick spam. The log warning above is enough
            # for post-mortem; actual outages surface as repeated log
            # lines, not per-tick Discord embeds. Operator can grep
            # "alpaca_bars_error" if curious.
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
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "alpaca_quote_error_falling_back symbol=%s err=%s",
                symbol, e,
            )
            # Try Tradier / Finnhub / Yahoo before accepting synthetic
            alt = _try_multi_provider_quote(symbol)
            if alt is not None:
                return alt
            # No Discord alert — same rationale as bars path.
            return self._fallback.latest_quote(symbol)


# =============================================================================
# Multi-provider fallback helpers
# =============================================================================
# When Alpaca bars or quotes fail, try Tradier / Yahoo / Finnhub before
# surrendering to synthetic. Fail-open — any exception returns None so
# the caller uses the synthetic fallback.

def _try_multi_provider_quote(symbol: str):
    """Return a Quote from any real provider, else None."""
    try:
        from .multi_provider import MultiProvider
        from ..core.types import Quote
        mp = MultiProvider.from_env()
        if not mp.active_providers():
            return None
        q = mp.latest_quote(symbol)
        if q is None or q.mid <= 0:
            return None
        return Quote(
            symbol=symbol,
            ts=None,
            bid=float(q.bid), ask=float(q.ask),
            bid_size=0.0, ask_size=0.0,
        )
    except Exception:
        return None


def _try_multi_provider_bars(symbol: str, *, limit: int,
                              timeframe_minutes: int):
    """Tradier (and Polygon, when configured) expose bars; Yahoo has
    intraday history via yfinance. Build a list of Bar from whichever
    provider responds first. Returns None if none do, so the caller
    falls back to synthetic."""
    try:
        from .multi_provider import MultiProvider
        from ..core.types import Bar
        from datetime import datetime, timezone
        mp = MultiProvider.from_env()
        if not mp.active_providers():
            return None
        # Yahoo's yfinance can pull intraday bars cleanly.
        for p in getattr(mp, "_providers", []):
            if p.name != "yahoo" or not p.is_enabled():
                continue
            try:
                import yfinance as yf
                t = yf.Ticker(symbol.upper())
                interval = f"{max(1, int(timeframe_minutes))}m"
                period = "1d" if timeframe_minutes <= 15 else "5d"
                h = t.history(period=period, interval=interval)
                if h is None or h.empty:
                    continue
                bars = []
                for ts, row in h.tail(int(limit)).iterrows():
                    # Ensure TZ-aware UTC ts
                    ts_utc = ts.tz_convert("UTC") if ts.tzinfo else \
                        ts.tz_localize("UTC")
                    bars.append(Bar(
                        symbol=symbol,
                        ts=ts_utc.to_pydatetime(),
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                        vwap=None,
                    ))
                if bars:
                    return bars
            except Exception:
                continue
        return None
    except Exception:
        return None
