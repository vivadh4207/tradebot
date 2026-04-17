"""Per-symbol dividend yield probe.

Fetches trailing dividend yield from yfinance, caches on disk for 24h
(yields don't move intraday). Non-dividend-payers return 0.0; the BS
pricer's `q=0.0` path is correct for them.

Cache format: json `{symbol: {yield, fetched_at}}` at `data_cache/div_yields.json`.
Auto-refreshes when older than `max_age_hours`.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional


_log = logging.getLogger(__name__)


class DividendYieldProvider:
    """Thread-safe per-symbol dividend yield cache.

    Call .get(symbol) → returns annual yield as a decimal (0.013 = 1.30%).
    Non-dividend payers and lookup failures cache as 0.0 so we don't
    hammer the upstream on every miss.
    """

    # Large-cap defaults for our 10-symbol universe. Used as a last-resort
    # fallback if yfinance is unavailable. Values are trailing-12-month
    # dividend yields as of ~2026; keep conservative — outdated yields are
    # a ~2-5 bps pricing error on 3-month calls, which is tolerable.
    _HARDCODED_FALLBACKS: Dict[str, float] = {
        "SPY":   0.0125,       # S&P 500 ETF
        "QQQ":   0.0065,       # Nasdaq 100 ETF
        "IWM":   0.0115,       # Russell 2000 ETF
        "AAPL":  0.0045,
        "MSFT":  0.0072,
        "NVDA":  0.0002,
        "TSLA":  0.0,
        "META":  0.0040,
        "AMZN":  0.0,
        "GOOGL": 0.0050,
    }

    def __init__(self, cache_path: str = "data_cache/div_yields.json",
                 max_age_hours: int = 24):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_hours * 3600
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, float]] = self._load()

    def _load(self) -> Dict[str, Dict[str, float]]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text())
        except Exception as e:
            _log.warning("dividend_cache_parse_failed path=%s err=%s",
                          self.cache_path, e)
            return {}

    def _save(self) -> None:
        try:
            self.cache_path.write_text(json.dumps(self._cache, indent=0))
        except Exception as e:
            _log.warning("dividend_cache_write_failed path=%s err=%s",
                          self.cache_path, e)

    def get(self, symbol: str) -> float:
        sym = symbol.upper()
        with self._lock:
            hit = self._cache.get(sym)
            if hit and (time.time() - hit.get("fetched_at", 0)) < self.max_age_seconds:
                return float(hit.get("yield", 0.0))
        # cache miss or expired — fetch outside the lock
        y = self._fetch(sym)
        if y is None:
            y = self._HARDCODED_FALLBACKS.get(sym, 0.0)
        with self._lock:
            self._cache[sym] = {"yield": float(y), "fetched_at": time.time()}
            self._save()
        return float(y)

    def _fetch(self, symbol: str) -> Optional[float]:
        try:
            import yfinance as yf
        except ImportError:
            return None
        try:
            t = yf.Ticker(symbol)
            # Preferred: fast_info has dividendYield if present
            try:
                fi = t.fast_info
                v = getattr(fi, "dividend_yield", None) or getattr(fi, "dividendYield", None)
                if v is not None:
                    v = float(v)
                    # yfinance sometimes returns as percent (e.g. 1.25 → 1.25%)
                    # other times as decimal (0.0125). Normalize: >1 → divide by 100.
                    if v > 1.0:
                        v /= 100.0
                    if 0.0 <= v < 0.25:    # sanity: yields > 25% are suspect
                        return v
            except Exception:
                pass
            # Fallback: info dict
            info = getattr(t, "info", None) or {}
            v = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
            if v is None:
                return 0.0                  # known non-payer or missing
            v = float(v)
            if v > 1.0:
                v /= 100.0
            if 0.0 <= v < 0.25:
                return v
            return 0.0
        except Exception as e:
            _log.info("dividend_yield_fetch_failed symbol=%s err=%s", symbol, e)
            return None

    def prime(self, symbols) -> None:
        """Eagerly warm the cache for a list of symbols (e.g. universe)."""
        for s in symbols:
            self.get(s)
