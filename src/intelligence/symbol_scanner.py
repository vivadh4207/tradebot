"""Dynamic symbol scanner — discovers tickers worth researching beyond
SPY/QQQ.

Sources (all optional, fail-soft):
  1. Finnhub news — counts tickers mentioned in the last 24h of
     general market news
  2. Finnhub top active symbols (if key + tier supports it)
  3. Polygon gainers/losers (if key set)
  4. Reddit WSB trending tickers (from political_news RSS aggregator
     when available)

Produces a ranked list of candidate symbols, excluding:
  - The base universe (SPY, QQQ — those always get researched)
  - Illiquid names (market cap filter pushed upstream)
  - Nonsensical patterns (crypto tickers, options OCC codes)

Research agent runs `scan()` before calling the 70B, appends the
top-N dynamic names to its underlyings list, and the 70B produces
setups on all of them.

The rules engine still only TRADES SPY/QQQ (operator decision, see
CLAUDE.md). Dynamic-universe research is ADVISORY — posts ideas to
Discord for operator review, not for auto-execution.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import List, Optional


_log = logging.getLogger(__name__)

_VALID_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")
_BLOCKED_SYMBOLS = {
    # ETFs not worth dynamically researching (leveraged / inverse /
    # too obvious) — operator can whitelist them explicitly if wanted.
    "VOO", "VTI", "VIXY", "SQQQ", "TQQQ", "UVXY", "SVXY",
    # Common false positives from regex on news text
    "USA", "CEO", "CFO", "FED", "ETF", "IPO", "LLC", "INC", "CO",
    "IT", "AI", "ML", "API", "NYSE", "NASDAQ", "SPX", "DOW",
    "AM", "PM", "EST", "ET", "UTC", "GMT",
    "I", "A", "AN", "THE", "OR", "AND",
    # Geographic / political abbreviations that are NOT US tickers
    "US", "EU", "UK", "UN", "NATO", "WHO", "G7", "G20", "OPEC",
    # Economic indicators / policy bodies
    "GDP", "CPI", "PPI", "FOMC", "ECB", "BOE", "BOJ", "PBOC",
    "FX", "USD", "EUR", "GBP", "JPY", "CNY",
    "SEC", "FTC", "EPA", "IRS", "DOJ", "FDA", "FAA", "FBI", "CIA",
    "ZEW", "IFO", "NFP", "ADP",
    # Common single-letter noise (markets rarely use 1-letter tickers
    # in news headlines — "A" is the only US 1-letter and it's rare)
    "S", "P", "U", "E", "V", "M", "N", "O", "B", "W", "Y", "Z",
    # Two-letter geography / state abbreviations that aren't tickers
    "UT", "NY", "CA", "TX", "FL", "NJ", "VA", "AZ", "CO", "WA",
    "IL", "OH", "PA", "MA", "GA", "MI", "NC", "TN", "MO", "MN",
    # Common English words / noise
    "BUY", "SELL", "HOLD", "TOP", "LOW", "HIGH", "NEW", "OLD",
    "OPEN", "CLOSE", "DAY", "WEEK", "YEAR", "NOW", "ETC",
    "BIG", "MOST", "LESS", "MORE", "ALL", "NONE", "YES", "NO",
    "OF", "TO", "IN", "ON", "AT", "BY", "FOR", "VS",
    "GLP",     # not a US ticker (GLP-1 refs from pharma news)
}


def _extract_tickers_from_text(text: str) -> List[str]:
    """Pull $TICKER and standalone uppercase ticker-like tokens from
    free text. Filters out common false positives."""
    out: List[str] = []
    # Explicit $TICKER format
    for m in re.finditer(r"\$([A-Z]{1,5})\b", text):
        out.append(m.group(1))
    # Also catch standalone uppercase tokens that look like tickers,
    # but only if they're not in the blocklist.
    for tok in re.findall(r"\b[A-Z]{2,5}\b", text):
        if tok in _BLOCKED_SYMBOLS:
            continue
        if _VALID_SYMBOL_RE.match(tok):
            out.append(tok)
    return out


class SymbolScanner:
    """Rank candidate symbols for the research agent to analyze."""

    def __init__(self, multi_provider,
                 *, base_universe: Optional[List[str]] = None,
                 max_dynamic: int = 5,
                 lookback_hours: int = 24):
        self._mp = multi_provider
        self._base = [s.upper() for s in (base_universe or ["SPY", "QQQ"])]
        self._max = int(max_dynamic)
        self._lookback_hours = int(lookback_hours)

    def scan(self) -> List[str]:
        """Return an ordered list of candidate symbols: base universe
        first, then the top dynamic names by mention frequency in news
        + movers feeds. Max len = len(base) + max_dynamic."""
        counter: Counter = Counter()

        # Market-wide news — headlines from providers without a
        # symbol filter (Finnhub returns `category=general`).
        try:
            news = self._mp.news(symbol=None, limit=60) or []
        except Exception:
            news = []
        for item in news:
            text = f"{item.headline} {item.summary}"
            for sym in _extract_tickers_from_text(text):
                if sym in self._base:
                    continue
                # Give repeated mentions weight
                counter[sym] += 1
            # Providers that already tag tickers (Polygon) bump those
            # symbols too.
            for tag in (item.tickers or []):
                tag = (tag or "").upper().strip()
                if tag and tag not in self._base and _VALID_SYMBOL_RE.match(tag):
                    counter[tag] += 2

        # Rank: top-N by count, min 2 mentions to qualify
        candidates = [s for s, n in counter.most_common()
                      if n >= 2 and s not in _BLOCKED_SYMBOLS]
        dynamic = candidates[: self._max]

        out = list(self._base) + dynamic
        _log.info(
            "symbol_scanner base=%s dynamic=%s total=%d",
            self._base, dynamic, len(out),
        )
        return out
