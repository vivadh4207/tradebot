"""MultiProvider — fan-out across Polygon/Tradier/Finnhub/Alpaca,
aggregate responses, provide quorum + fallback.

Design goals:

  1. NO vendor single-point-of-failure. If Alpaca is slow / down, the
     aggregator still returns a quote from Polygon or Tradier.
  2. Cross-check. When two sources disagree by more than `quote_disagreement_pct`
     the aggregator takes the MEDIAN price (not the fastest one) —
     protects against a stale or bad feed.
  3. Zero per-feature cost when a provider isn't configured. Every
     provider reports is_enabled() first; disabled ones are skipped.
  4. Thread-safe so Discord chat + brain + scheduled agents can all
     call it concurrently.

Usage:
    mp = MultiProvider.from_env()
    quote = mp.latest_quote("SPY")          # aggregated
    chain = mp.option_chain("SPY", "2026-04-24")
    news  = mp.news("SPY", limit=10)        # pooled across providers
"""
from __future__ import annotations

import logging
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .providers.base import (
    MarketDataProvider, ProviderNewsItem, ProviderOptionRow, ProviderQuote,
)


_log = logging.getLogger(__name__)


class MultiProvider:
    def __init__(
        self,
        providers: List[MarketDataProvider],
        *,
        quote_disagreement_pct: float = 0.005,     # 0.5% max bid/ask drift
        fan_out_timeout_sec: float = 8.0,
    ):
        # Filter to only ENABLED providers so we don't waste a thread on
        # a no-op.
        self._providers = [p for p in providers if p.is_enabled()]
        self._q_drift = float(quote_disagreement_pct)
        self._timeout = float(fan_out_timeout_sec)
        self._lock = threading.Lock()
        # Track recent failures per provider name → deprioritize flaky ones.
        self._recent_fail_count: Dict[str, int] = {}

    @classmethod
    def from_env(cls) -> "MultiProvider":
        """Build a MultiProvider from whatever providers the env enables.
        Missing keys are no-ops — no exception.

        Reloads .env on each call so runtime edits (operator rotates a
        key, fixes a duplicate) take effect without restarting the bot.
        """
        try:
            # override=True so a key added after process start wins
            from dotenv import load_dotenv
            from pathlib import Path as _P
            # Walk up to find .env; handles both bot-root and test-cwd calls.
            cur = _P(".").resolve()
            for _ in range(4):
                candidate = cur / ".env"
                if candidate.exists():
                    load_dotenv(candidate, override=True)
                    break
                cur = cur.parent
        except Exception:
            pass
        from .providers.polygon import PolygonProvider
        from .providers.tradier import TradierProvider
        from .providers.finnhub import FinnhubProvider
        from .providers.yahoo import YahooProvider
        providers: List[MarketDataProvider] = [
            PolygonProvider(),
            TradierProvider(),
            FinnhubProvider(),
            YahooProvider(),
        ]
        mp = cls(providers)
        _log.info(
            "multi_provider_initialized active=%s",
            [p.name for p in mp._providers],
        )
        return mp

    def active_providers(self) -> List[str]:
        return [p.name for p in self._providers]

    # ------------------------------------------------ quotes

    def latest_quote(self, symbol: str) -> Optional[ProviderQuote]:
        """Fan out. Take the median mid when 2+ providers respond and
        disagree beyond threshold; otherwise return the fastest."""
        results = self._fan_out(
            lambda p: p.latest_quote(symbol)
        )
        if not results:
            return None
        if len(results) == 1:
            return results[0]
        mids = [r.mid for r in results if r.mid > 0]
        if not mids:
            return results[0]
        max_mid = max(mids)
        min_mid = min(mids)
        spread = (max_mid - min_mid) / max(1e-9, statistics.median(mids))
        if spread > self._q_drift:
            # Sources disagree — take the median-priced source as truth.
            median_mid = statistics.median(mids)
            median_src = min(results, key=lambda r: abs(r.mid - median_mid))
            _log.info(
                "multi_provider_quote_disagreement symbol=%s spread=%.4f "
                "min=%.2f max=%.2f picked=%s",
                symbol, spread, min_mid, max_mid, median_src.source,
            )
            return median_src
        return results[0]

    def all_quotes(self, symbol: str) -> List[ProviderQuote]:
        """All quotes from all active providers — useful for the LLM
        research agent so the 70B can see the cross-check itself."""
        return self._fan_out(lambda p: p.latest_quote(symbol))

    # ------------------------------------------------ options chain

    def option_chain(self, underlying: str, expiry: Optional[str] = None
                     ) -> Optional[List[ProviderOptionRow]]:
        """First provider that returns a non-empty chain wins. We don't
        merge chains across providers (OCC symbols are globally unique
        anyway, so duplicates would dominate but bring no new info)."""
        results = self._fan_out(
            lambda p: p.option_chain(underlying, expiry)
        )
        # Take the chain with the most rows (most detail). Typically
        # Polygon > Tradier > Alpaca; this ordering survives without
        # hardcoding.
        non_empty = [r for r in results if r]
        if not non_empty:
            return None
        return max(non_empty, key=len)

    # ------------------------------------------------ news

    def news(self, symbol: Optional[str] = None, limit: int = 20
             ) -> List[ProviderNewsItem]:
        """Pool news across providers, de-dupe by headline hash,
        keep most recent `limit`. Unlike quotes and chain, news is
        additive — more sources = more signal for the 70B."""
        results = self._fan_out(
            lambda p: p.news(symbol, limit=limit)
        )
        pooled: List[ProviderNewsItem] = []
        seen: set = set()
        for batch in results:
            if not batch:
                continue
            for n in batch:
                key = (n.headline or "").lower()[:120]
                if key in seen or not key:
                    continue
                seen.add(key)
                pooled.append(n)
        # Sort by timestamp desc when available.
        pooled.sort(key=lambda n: n.ts or "", reverse=True)
        return pooled[:limit]

    def news_sentiment(self, symbol: str) -> Optional[float]:
        """Aggregate sentiment score across providers that support it
        (currently Finnhub). Returns mean in [-1..+1] or None."""
        scores: List[float] = []
        for p in self._providers:
            fn = getattr(p, "news_sentiment", None)
            if fn is None:
                continue
            try:
                s = fn(symbol)
                if s is not None:
                    scores.append(float(s))
            except Exception:
                continue
        if not scores:
            return None
        return statistics.mean(scores)

    # ------------------------------------------------ internals

    def _fan_out(self, call) -> List:
        """Call `call(provider)` for every enabled provider, in parallel,
        with a global timeout. Returns the successful non-None results."""
        if not self._providers:
            return []
        out: List = []
        with ThreadPoolExecutor(max_workers=len(self._providers)) as ex:
            futs = {ex.submit(self._safe, call, p): p for p in self._providers}
            try:
                for fut in as_completed(futs, timeout=self._timeout):
                    result = fut.result()
                    if result is not None and result != []:
                        out.append(result)
            except Exception:
                # Some providers timed out; use what we have.
                pass
        return out

    @staticmethod
    def _safe(call, provider):
        try:
            return call(provider)
        except Exception as e:                          # noqa: BLE001
            _log.info("multi_provider_call_failed name=%s err=%s",
                      provider.name, e)
            return None
