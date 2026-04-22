"""Yahoo Finance provider — free equity quotes, historical bars,
fundamentals, news headlines. Uses the yfinance library.

Strengths:
  - FREE (no API key needed)
  - Works for almost any listed US equity + ETF + many indices
  - Quote delays: real-time for some, ~15 min for others
  - Decent news feed (though summaries are truncated by Yahoo)

Weaknesses:
  - Unofficial (scrapes Yahoo's public JSON endpoints — yfinance
    occasionally breaks when Yahoo changes layout)
  - Rate-limited silently (burst 10-15 req/sec OK; sustained will get
    you throttled)
  - No options chain (Yahoo has one but yfinance's coverage is flaky;
    we return None for option_chain — let Polygon/Tradier handle that)

Enable with YAHOO_ENABLED=1 in .env (or just install yfinance — the
provider auto-enables if the lib imports cleanly). If yfinance isn't
installed, the provider reports is_enabled()=False gracefully.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from .base import (
    MarketDataProvider, ProviderNewsItem, ProviderOptionRow, ProviderQuote,
)


_log = logging.getLogger(__name__)


class YahooProvider:
    name = "yahoo"

    def __init__(self, *, enabled_override: Optional[bool] = None):
        self._yf = None
        self._available = False
        # Allow explicit opt-out via YAHOO_ENABLED=0
        env_opt_out = os.getenv("YAHOO_ENABLED", "").strip().lower() in (
            "0", "false", "no", "off",
        )
        if enabled_override is False or env_opt_out:
            return
        try:
            import yfinance as _yf
            self._yf = _yf
            self._available = True
        except Exception as e:                          # noqa: BLE001
            _log.info("yahoo_yfinance_not_installed err=%s -- "
                      "pip install yfinance to enable", e)

    def is_enabled(self) -> bool:
        return self._available

    # ------------------------------------------------ quotes

    def latest_quote(self, symbol: str) -> Optional[ProviderQuote]:
        if not self.is_enabled():
            return None
        try:
            t = self._yf.Ticker(symbol.upper())
            # fast_info is lightweight (single HTTP call, no stats scrape)
            fi = getattr(t, "fast_info", None)
            last = None
            if fi is not None:
                last = (getattr(fi, "last_price", None)
                        or getattr(fi, "regular_market_price", None))
            if last is None:
                # Fallback: pull last 1m bar
                hist = t.history(period="1d", interval="1m")
                if not hist.empty:
                    last = float(hist["Close"].iloc[-1])
            if not last or last <= 0:
                return None
            last = float(last)
            # Yahoo doesn't expose bid/ask on fast_info reliably; use last
            # as a proxy on both sides (cross-check only, not exec).
            return ProviderQuote(
                symbol=symbol.upper(),
                bid=last, ask=last, mid=last,
                ts=datetime.now(tz=timezone.utc).isoformat(),
                source=self.name,
            )
        except Exception as e:                          # noqa: BLE001
            _log.info("yahoo_quote_failed symbol=%s err=%s", symbol, e)
            return None

    # ------------------------------------------------ options chain
    # yfinance has an options API that is flaky — greeks aren't
    # provided, OI can lag a day, strikes are sometimes stale. But in
    # the absence of Polygon/Tradier it's the only free options source,
    # so we enable it with a YAHOO_CHAIN_ENABLED=1 opt-in. Good enough
    # for after-hours research when the market is closed anyway.

    def option_chain(self, underlying: str, expiry: Optional[str] = None
                     ) -> Optional[List[ProviderOptionRow]]:
        if not self.is_enabled():
            return None
        if os.getenv("YAHOO_CHAIN_ENABLED", "1").strip().lower() in (
            "0", "false", "no",
        ):
            return None
        try:
            t = self._yf.Ticker(underlying.upper())
            exps = list(getattr(t, "options", []) or [])
            if not exps:
                return None
            pick_exp = expiry if (expiry and expiry in exps) else exps[0]
            oc = t.option_chain(pick_exp)
            rows: List[ProviderOptionRow] = []
            for df, right in ((oc.calls, "call"), (oc.puts, "put")):
                if df is None or df.empty:
                    continue
                for _, r in df.iterrows():
                    try:
                        rows.append(ProviderOptionRow(
                            symbol=str(r.get("contractSymbol", "")),
                            underlying=underlying.upper(),
                            strike=float(r.get("strike", 0)),
                            expiry=pick_exp,
                            right=right,
                            bid=float(r.get("bid", 0)) or None,
                            ask=float(r.get("ask", 0)) or None,
                            last=float(r.get("lastPrice", 0)) or None,
                            volume=int(r.get("volume", 0)) or None,
                            open_interest=int(r.get("openInterest", 0)) or None,
                            implied_vol=float(r.get("impliedVolatility", 0)) or None,
                            delta=None, gamma=None, vega=None, theta=None,
                            source=self.name,
                        ))
                    except Exception:
                        continue
            return rows or None
        except Exception as e:                          # noqa: BLE001
            _log.info("yahoo_chain_failed sym=%s err=%s", underlying, e)
            return None

    # ------------------------------------------------ VIX snapshot
    # Convenience so callers can fetch the live volatility index
    # without special-casing the "^VIX" Yahoo ticker.

    def latest_vix(self) -> Optional[float]:
        """Return current VIX index value, or None on failure."""
        if not self.is_enabled():
            return None
        try:
            t = self._yf.Ticker("^VIX")
            fi = getattr(t, "fast_info", None)
            v = None
            if fi is not None:
                v = (getattr(fi, "last_price", None)
                     or getattr(fi, "regular_market_price", None))
            if v is None:
                hist = t.history(period="1d", interval="1m")
                if not hist.empty:
                    v = float(hist["Close"].iloc[-1])
            return float(v) if v else None
        except Exception as e:                          # noqa: BLE001
            _log.info("yahoo_vix_failed err=%s", e)
            return None

    # ------------------------------------------------ news

    def news(self, symbol: Optional[str] = None, limit: int = 20
             ) -> Optional[List[ProviderNewsItem]]:
        if not self.is_enabled() or not symbol:
            return None
        try:
            t = self._yf.Ticker(symbol.upper())
            news = getattr(t, "news", None) or []
            out: List[ProviderNewsItem] = []
            for r in news[:limit]:
                # yfinance returns dicts with nested 'content' on newer
                # versions and flat on older — handle both.
                c = r.get("content") or r
                try:
                    ts_raw = c.get("pubDate") or c.get("providerPublishTime")
                    if isinstance(ts_raw, (int, float)):
                        ts = datetime.fromtimestamp(
                            int(ts_raw), tz=timezone.utc,
                        ).isoformat()
                    else:
                        ts = str(ts_raw or "")
                except Exception:
                    ts = ""
                out.append(ProviderNewsItem(
                    ts=ts,
                    headline=str(c.get("title", ""))[:240],
                    summary=str(c.get("summary", ""))[:500],
                    url=str(
                        (c.get("canonicalUrl") or {}).get("url")
                        if isinstance(c.get("canonicalUrl"), dict)
                        else (c.get("link") or "")
                    ),
                    tickers=[symbol.upper()],
                    sentiment_score=None,
                    source=self.name,
                ))
            return out or None
        except Exception as e:                          # noqa: BLE001
            _log.info("yahoo_news_failed symbol=%s err=%s", symbol, e)
            return None
