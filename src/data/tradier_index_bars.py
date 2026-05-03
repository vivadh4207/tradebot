"""Bars + quotes for cash-settled indexes (SPX, VIX, NDX, RUT).
Alpaca doesn't carry these. Tradier *production* does, but sandbox
returns 'unmatched_symbols' on `$SPX`. So:

  Primary: yfinance  (free, no key, has ^GSPC / ^VIX / ^NDX / ^RUT)
  Fallback: Tradier  (production tier — auto-engages when we upgrade)

Provides `TradierIndexBarsAdapter` (kept name for backwards-compat;
internally routes to yfinance first, Tradier second). Pair with
`RoutedDataAdapter` to send index symbols here, everything else to
the primary stock adapter.

yfinance symbology for indexes:
  SPX → ^GSPC     (S&P 500 cash index)
  VIX → ^VIX      (CBOE Volatility)
  NDX → ^NDX      (Nasdaq 100 cash)
  RUT → ^RUT      (Russell 2000 cash)
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from ..core.types import Bar, Quote
from .market_data import MarketDataAdapter

import logging
_log = logging.getLogger(__name__)


# Symbols this adapter handles. Both bare (SPX) and dollar-prefixed
# ($SPX) forms accepted from callers; we always send $-prefixed to
# Tradier internally.
INDEX_SYMBOLS = {"SPX", "VIX", "NDX", "RUT"}


def _is_index(symbol: str) -> bool:
    s = (symbol or "").upper().lstrip("$").strip()
    return s in INDEX_SYMBOLS


def _to_tradier_symbol(symbol: str) -> str:
    """Tradier expects $SPX, $VIX, etc. for indexes."""
    s = (symbol or "").upper().lstrip("$").strip()
    return f"${s}" if s in INDEX_SYMBOLS else symbol


# yfinance ticker mapping (^-prefix). yfinance is the primary source
# because the Tradier sandbox doesn't carry index symbols (and most
# operators are on the sandbox tier).
_YF_INDEX_MAP = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "NDX": "^NDX",
    "RUT": "^RUT",
}


def _to_yf_symbol(symbol: str) -> str:
    s = (symbol or "").upper().lstrip("$").strip()
    return _YF_INDEX_MAP.get(s, symbol)


class TradierIndexBarsAdapter(MarketDataAdapter):
    """Pulls bars + quotes from Tradier for cash indexes only.

    Production: api.tradier.com
    Sandbox:    sandbox.tradier.com (set TRADIER_SANDBOX=1 — default)
    """

    def __init__(self, token: Optional[str] = None,
                  base_url: Optional[str] = None,
                  timeout_sec: float = 8.0):
        self._token = (token or os.getenv("TRADIER_TOKEN") or "").strip()
        if base_url:
            self._base = base_url
        else:
            sandbox = os.getenv("TRADIER_SANDBOX", "1").strip() in ("1", "true", "yes")
            self._base = ("https://sandbox.tradier.com"
                            if sandbox else "https://api.tradier.com")
        self._timeout = float(timeout_sec)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict) -> Optional[dict]:
        if not self._token:
            return None
        qs = urllib.parse.urlencode(params)
        url = f"{self._base}{path}?{qs}"
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status != 200:
                    _log.info("tradier_idx_http_%s path=%s",
                                resp.status, path)
                    return None
                return json.loads(resp.read())
        except Exception as e:
            _log.info("tradier_idx_http_err path=%s err=%s",
                        path, str(e)[:160])
            return None

    def get_bars(self, symbol: str, *, limit: int = 200,
                  timeframe_minutes: int = 1,
                  end: Optional[datetime] = None) -> List[Bar]:
        """yfinance primary, Tradier fallback. Both return Bar list."""
        if not _is_index(symbol):
            return []
        # --- Primary: yfinance ---
        bars = self._yf_bars(symbol, limit, timeframe_minutes, end)
        if bars:
            return bars
        # --- Fallback: Tradier (production carries indexes) ---
        return self._tradier_bars(symbol, limit, timeframe_minutes, end)

    def _yf_bars(self, symbol: str, limit: int,
                  timeframe_minutes: int,
                  end: Optional[datetime]) -> List[Bar]:
        try:
            import yfinance as yf
        except Exception:
            return []
        yf_sym = _to_yf_symbol(symbol)
        if timeframe_minutes <= 1:
            interval, period = "1m", "1d"
        elif timeframe_minutes <= 5:
            interval, period = "5m", "5d"
        else:
            interval, period = "15m", "5d"
        try:
            df = yf.Ticker(yf_sym).history(
                period=period, interval=interval, auto_adjust=False,
            )
        except Exception as e:                          # noqa: BLE001
            _log.info("yf_idx_err sym=%s err=%s", yf_sym, str(e)[:120])
            return []
        if df is None or df.empty:
            return []
        df = df.tail(limit)
        out: List[Bar] = []
        for ts, row in df.iterrows():
            try:
                out.append(Bar(
                    symbol=symbol,
                    ts=(ts.to_pydatetime() if hasattr(ts, "to_pydatetime")
                         else datetime.now(tz=timezone.utc)),
                    open=float(row.get("Open", 0)),
                    high=float(row.get("High", 0)),
                    low=float(row.get("Low", 0)),
                    close=float(row.get("Close", 0)),
                    volume=float(row.get("Volume", 0) or 0),
                ))
            except Exception:
                continue
        return out

    def _tradier_bars(self, symbol: str, limit: int,
                       timeframe_minutes: int,
                       end: Optional[datetime]) -> List[Bar]:
        if not _is_index(symbol):
            return []
        tradier_sym = _to_tradier_symbol(symbol)
        # Tradier intervals: 1min, 5min, 15min
        if timeframe_minutes <= 1:
            interval = "1min"
        elif timeframe_minutes <= 5:
            interval = "5min"
        else:
            interval = "15min"
        # Window — pull `limit * timeframe_minutes` worth of history,
        # plus a 30-min buffer for weekends / off-hours edges.
        end_dt = end or datetime.now(tz=timezone.utc)
        # Tradier expects ET timestamps in YYYY-MM-DD HH:MM
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
            end_et = end_dt.astimezone(et)
        except Exception:
            end_et = end_dt
        start_et = end_et - timedelta(minutes=limit * timeframe_minutes + 30)
        params = {
            "symbol": tradier_sym,
            "interval": interval,
            "start": start_et.strftime("%Y-%m-%d %H:%M"),
            "end":   end_et.strftime("%Y-%m-%d %H:%M"),
            "session_filter": "all",
        }
        data = self._get("/v1/markets/timesales", params)
        if not data:
            return []
        rows = (data.get("series") or {}).get("data") or []
        if isinstance(rows, dict):
            rows = [rows]
        out: List[Bar] = []
        for r in rows[-limit:]:
            try:
                ts_str = r.get("time") or r.get("timestamp") or ""
                ts = datetime.fromisoformat(ts_str)
            except Exception:
                ts = end_dt
            try:
                out.append(Bar(
                    symbol=symbol,
                    ts=ts,
                    open=float(r.get("open") or 0),
                    high=float(r.get("high") or 0),
                    low=float(r.get("low") or 0),
                    close=float(r.get("close") or 0),
                    volume=float(r.get("volume") or 0),
                    vwap=(float(r.get("vwap")) if r.get("vwap") else None),
                ))
            except Exception:
                continue
        return out

    def latest_quote(self, symbol: str) -> Optional[Quote]:
        if not _is_index(symbol):
            return None
        # --- Primary: yfinance ---
        q = self._yf_quote(symbol)
        if q is not None:
            return q
        # --- Fallback: Tradier (production tier) ---
        tradier_sym = _to_tradier_symbol(symbol)
        data = self._get("/v1/markets/quotes", {"symbols": tradier_sym})
        if not data:
            return None
        rows = (data.get("quotes") or {}).get("quote")
        if isinstance(rows, list):
            rows = rows[0] if rows else {}
        if not rows:
            return None
        last = float(rows.get("last") or 0)
        # Indexes don't have bid/ask — derive synthetic 0.05% spread
        # so the Quote's `mid` returns the actual last price.
        spread_pct = 0.0005
        bid = last * (1 - spread_pct / 2)
        ask = last * (1 + spread_pct / 2)
        return Quote(
            symbol=symbol,
            ts=datetime.now(tz=timezone.utc),
            bid=bid, ask=ask,
            bid_size=0.0, ask_size=0.0,
        )

    def _yf_quote(self, symbol: str) -> Optional[Quote]:
        try:
            import yfinance as yf
        except Exception:
            return None
        yf_sym = _to_yf_symbol(symbol)
        try:
            t = yf.Ticker(yf_sym)
            # `fast_info` is fast and doesn't hit /info; falls back to
            # the latest 1m bar close if fast_info is unavailable.
            last = None
            try:
                fi = t.fast_info
                last = float(getattr(fi, "last_price", 0)) or None
            except Exception:
                last = None
            if last is None:
                df = t.history(period="1d", interval="1m", auto_adjust=False)
                if df is not None and not df.empty:
                    last = float(df["Close"].iloc[-1])
            if not last or last <= 0:
                return None
            spread_pct = 0.0005
            return Quote(
                symbol=symbol,
                ts=datetime.now(tz=timezone.utc),
                bid=last * (1 - spread_pct / 2),
                ask=last * (1 + spread_pct / 2),
            )
        except Exception as e:                          # noqa: BLE001
            _log.info("yf_idx_quote_err sym=%s err=%s",
                        yf_sym, str(e)[:120])
            return None


class RoutedDataAdapter(MarketDataAdapter):
    """Wraps a primary adapter (Alpaca) and an index adapter (Tradier).
    Routes index symbols to the index adapter, everything else to
    primary."""

    def __init__(self, primary: MarketDataAdapter,
                  index_adapter: TradierIndexBarsAdapter):
        self.primary = primary
        self.index = index_adapter

    def get_bars(self, symbol: str, *, limit: int = 200,
                  timeframe_minutes: int = 1,
                  end: Optional[datetime] = None) -> List[Bar]:
        if _is_index(symbol):
            return self.index.get_bars(
                symbol, limit=limit,
                timeframe_minutes=timeframe_minutes, end=end,
            )
        return self.primary.get_bars(
            symbol, limit=limit,
            timeframe_minutes=timeframe_minutes, end=end,
        )

    def latest_quote(self, symbol: str) -> Optional[Quote]:
        if _is_index(symbol):
            return self.index.latest_quote(symbol)
        return self.primary.latest_quote(symbol)
