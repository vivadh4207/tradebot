"""FinnhubIntelligence — deep-fundamentals layer.

Wraps 20 Finnhub endpoints covering company financials, ownership,
institutional positioning, insider activity, filings, analyst views,
and event calendars. Each endpoint is:
  - cached in-memory with endpoint-specific TTL (price targets 1h,
    filings 6h, ownership 24h — rate-limit friendly)
  - fail-soft (network/HTTP errors return None; never raise)
  - typed to a simple dict the LLM can read directly

Used by OptionsResearchAgent snapshot builder + Discord `!intel SYM`
+ dashboard /api/intel endpoints. Adds "fundamentals context" the 70B
cites when recommending directional trades.

Rate limit notes (Finnhub free tier = 60 calls/min):
  - `bundle(symbol)` makes ~15 calls per invocation → use sparingly
  - Cache hit rate is typically 80%+ so one symbol costs ~3 calls
  - All data feeds are SEC/exchange public — no privacy concerns
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple
from urllib import request, parse, error


_log = logging.getLogger(__name__)
_BASE = "https://finnhub.io/api/v1"

# Per-endpoint TTL (seconds). Balances freshness vs. rate limit.
_TTL = {
    "price_target":         3600,       # 1 h
    "recommendation":       3600,       # 1 h
    "basic_financials":     21600,      # 6 h
    "financials":           21600,
    "financials_reported":  86400,      # daily
    "revenue_breakdown":    86400,
    "dividends":            86400,
    "insider_transactions": 10800,      # 3 h
    "insider_sentiment":    10800,
    "ownership":            86400,
    "fund_ownership":       86400,
    "institutional_profile": 86400,
    "institutional_portfolio": 86400,
    "institutional_ownership": 86400,
    "filings":              21600,
    "filings_sentiment":    21600,
    "similarity_index":     86400,
    "ipo_calendar":         21600,
    "fda_calendar":         21600,
    "sector_metrics":       43200,      # 12 h
}


@dataclass
class IntelSnapshot:
    """Full bundle returned by bundle(symbol). Every field is optional
    — some endpoints require a premium plan and will be None on free."""
    symbol: str
    as_of: float = field(default_factory=time.time)
    # Analyst views
    price_target: Optional[Dict[str, Any]] = None
    recommendation_trends: Optional[List[Dict[str, Any]]] = None
    # Financials
    basic_financials: Optional[Dict[str, Any]] = None
    financials_reported: Optional[List[Dict[str, Any]]] = None
    revenue_breakdown: Optional[Dict[str, Any]] = None
    # Ownership
    ownership: Optional[List[Dict[str, Any]]] = None
    fund_ownership: Optional[List[Dict[str, Any]]] = None
    institutional_ownership: Optional[List[Dict[str, Any]]] = None
    # Insider activity
    insider_transactions: Optional[List[Dict[str, Any]]] = None
    insider_sentiment: Optional[Dict[str, Any]] = None
    # Filings
    filings: Optional[List[Dict[str, Any]]] = None
    filings_sentiment: Optional[Dict[str, Any]] = None
    similarity_index: Optional[List[Dict[str, Any]]] = None
    # Events
    dividends: Optional[List[Dict[str, Any]]] = None
    errors: List[str] = field(default_factory=list)


class FinnhubIntelligence:
    """Thread-safe cached fundamentals client."""

    def __init__(self, api_key: Optional[str] = None,
                 timeout: float = 8.0):
        self._key = (api_key or os.getenv("FINNHUB_KEY") or "").strip()
        self._timeout = float(timeout)
        self._cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}
        self._lock = RLock()

    @property
    def is_enabled(self) -> bool:
        return bool(self._key)

    # ------------------------------------------------ low-level
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None,
              ttl_key: str = "") -> Optional[Any]:
        """GET with caching. ttl_key selects the cache lifetime from _TTL."""
        if not self._key:
            return None
        params = dict(params or {})
        params["token"] = self._key
        cache_key = (path, parse.urlencode(sorted(params.items())))
        ttl = _TTL.get(ttl_key, 300)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and (time.time() - cached[0]) < ttl:
                return cached[1]
        url = f"{_BASE}{path}?{parse.urlencode(params)}"
        try:
            req = request.Request(url, headers={"Accept": "application/json"})
            with request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except error.HTTPError as e:
            if e.code == 429:
                _log.warning("finnhub_intel_rate_limited path=%s", path)
            elif e.code in (401, 403):
                _log.info("finnhub_intel_forbidden path=%s (likely premium)",
                          path)
            else:
                _log.info("finnhub_intel_http_err path=%s code=%s", path, e.code)
            return None
        except Exception as e:                              # noqa: BLE001
            _log.info("finnhub_intel_network_err path=%s err=%s", path, e)
            return None
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            return None
        with self._lock:
            self._cache[cache_key] = (time.time(), data)
        return data

    # ------------------------------------------------ endpoints

    def price_target(self, symbol: str) -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/price-target"""
        return self._get("/stock/price-target", {"symbol": symbol},
                          ttl_key="price_target")

    def recommendation_trends(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """https://finnhub.io/docs/api/recommendation-trends"""
        return self._get("/stock/recommendation", {"symbol": symbol},
                          ttl_key="recommendation")

    def basic_financials(self, symbol: str, metric: str = "all"
                          ) -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/company-basic-financials"""
        return self._get("/stock/metric",
                          {"symbol": symbol, "metric": metric},
                          ttl_key="basic_financials")

    def financials(self, symbol: str, *, statement: str = "ic",
                    freq: str = "annual") -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/financials — statement in
        {ic, bs, cf} × freq in {annual, quarterly}."""
        return self._get("/stock/financials",
                          {"symbol": symbol, "statement": statement, "freq": freq},
                          ttl_key="financials")

    def financials_reported(self, symbol: str, *,
                             freq: str = "annual") -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/financials-reported — raw SEC
        filings data (10-K / 10-Q)."""
        return self._get("/stock/financials-reported",
                          {"symbol": symbol, "freq": freq},
                          ttl_key="financials_reported")

    def revenue_breakdown(self, symbol: str) -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/revenue-breakdown"""
        return self._get("/stock/revenue-breakdown", {"symbol": symbol},
                          ttl_key="revenue_breakdown")

    def dividends(self, symbol: str, *, from_date: str = "2023-01-01",
                   to_date: str = "2030-12-31"
                   ) -> Optional[List[Dict[str, Any]]]:
        """https://finnhub.io/docs/api/stock-dividends"""
        return self._get("/stock/dividend",
                          {"symbol": symbol, "from": from_date, "to": to_date},
                          ttl_key="dividends")

    # ---- ownership ----
    def ownership(self, symbol: str, *, limit: int = 20
                   ) -> Optional[List[Dict[str, Any]]]:
        """https://finnhub.io/docs/api/ownership — top holders."""
        data = self._get("/stock/ownership",
                          {"symbol": symbol, "limit": limit},
                          ttl_key="ownership")
        return data.get("ownership") if isinstance(data, dict) else data

    def fund_ownership(self, symbol: str, *, limit: int = 20
                        ) -> Optional[List[Dict[str, Any]]]:
        """https://finnhub.io/docs/api/fund-ownership"""
        data = self._get("/stock/fund-ownership",
                          {"symbol": symbol, "limit": limit},
                          ttl_key="fund_ownership")
        return data.get("ownership") if isinstance(data, dict) else data

    def institutional_profile(self, cik: str) -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/institutional-profile"""
        return self._get("/institutional/profile", {"cik": cik},
                          ttl_key="institutional_profile")

    def institutional_portfolio(self, cik: str, *,
                                  from_date: str = "2023-01-01",
                                  to_date: str = "2030-12-31"
                                  ) -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/institutional-portfolio-13f"""
        return self._get("/institutional/portfolio",
                          {"cik": cik, "from": from_date, "to": to_date},
                          ttl_key="institutional_portfolio")

    def institutional_ownership(self, symbol: str, *,
                                  from_date: str = "2023-01-01",
                                  to_date: str = "2030-12-31"
                                  ) -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/institutional-ownership"""
        return self._get("/institutional/ownership",
                          {"symbol": symbol,
                            "from": from_date, "to": to_date},
                          ttl_key="institutional_ownership")

    # ---- insider ----
    def insider_transactions(self, symbol: str, *,
                                from_date: str = "2024-01-01",
                                to_date: str = "2030-12-31"
                                ) -> Optional[List[Dict[str, Any]]]:
        """https://finnhub.io/docs/api/insider-transactions"""
        data = self._get("/stock/insider-transactions",
                          {"symbol": symbol,
                            "from": from_date, "to": to_date},
                          ttl_key="insider_transactions")
        return data.get("data") if isinstance(data, dict) else data

    def insider_sentiment(self, symbol: str, *,
                            from_date: str = "2024-01-01",
                            to_date: str = "2030-12-31"
                            ) -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/insider-sentiment"""
        return self._get("/stock/insider-sentiment",
                          {"symbol": symbol,
                            "from": from_date, "to": to_date},
                          ttl_key="insider_sentiment")

    # ---- filings ----
    def filings(self, symbol: str, *, from_date: str = "2024-01-01",
                 to_date: str = "2030-12-31"
                 ) -> Optional[List[Dict[str, Any]]]:
        """https://finnhub.io/docs/api/filings"""
        return self._get("/stock/filings",
                          {"symbol": symbol,
                            "from": from_date, "to": to_date},
                          ttl_key="filings")

    def filings_sentiment(self, *, access_number: str = "",
                            symbol: str = ""
                            ) -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/filings-sentiment — requires
        accessNumber OR symbol. Often premium-only."""
        p = {}
        if access_number:
            p["accessNumber"] = access_number
        if symbol:
            p["symbol"] = symbol
        return self._get("/stock/filings-sentiment", p,
                          ttl_key="filings_sentiment")

    def similarity_index(self, symbol: str, *,
                           freq: str = "annual"
                           ) -> Optional[List[Dict[str, Any]]]:
        """https://finnhub.io/docs/api/similarity-index"""
        data = self._get("/stock/similarity-index",
                          {"symbol": symbol, "freq": freq},
                          ttl_key="similarity_index")
        return data.get("similarity") if isinstance(data, dict) else data

    # ---- calendars ----
    def ipo_calendar(self, *, from_date: str = "",
                      to_date: str = "") -> Optional[List[Dict[str, Any]]]:
        """https://finnhub.io/docs/api/ipo-calendar"""
        from datetime import datetime, timedelta
        if not from_date:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        data = self._get("/calendar/ipo",
                          {"from": from_date, "to": to_date},
                          ttl_key="ipo_calendar")
        return data.get("ipoCalendar") if isinstance(data, dict) else data

    def fda_calendar(self) -> Optional[List[Dict[str, Any]]]:
        """https://finnhub.io/docs/api/fda-committee-meeting-calendar"""
        return self._get("/fda-advisory-committee-calendar", {},
                          ttl_key="fda_calendar")

    def sector_metrics(self, *, region: str = "NA"
                        ) -> Optional[Dict[str, Any]]:
        """https://finnhub.io/docs/api/sector-metrics"""
        return self._get("/sector/metrics", {"region": region},
                          ttl_key="sector_metrics")

    # ------------------------------------------------ aggregators

    def bundle(self, symbol: str, *, include_heavy: bool = False
                ) -> IntelSnapshot:
        """Pull ~10-15 endpoints in sequence and return a single
        IntelSnapshot. `include_heavy` adds financials_reported +
        similarity_index (those are biggest + least-often-useful)."""
        symbol = symbol.upper()
        snap = IntelSnapshot(symbol=symbol)

        def _safe(name, fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                snap.errors.append(f"{name}: {type(e).__name__}")
                return None

        snap.price_target          = _safe("price_target",
                                            self.price_target, symbol)
        snap.recommendation_trends = _safe("recommendation_trends",
                                            self.recommendation_trends,
                                            symbol)
        snap.basic_financials      = _safe("basic_financials",
                                            self.basic_financials, symbol)
        snap.revenue_breakdown     = _safe("revenue_breakdown",
                                            self.revenue_breakdown, symbol)
        snap.ownership             = _safe("ownership",
                                            self.ownership, symbol)
        snap.fund_ownership        = _safe("fund_ownership",
                                            self.fund_ownership, symbol)
        snap.institutional_ownership = _safe("institutional_ownership",
                                              self.institutional_ownership,
                                              symbol)
        snap.insider_transactions  = _safe("insider_transactions",
                                            self.insider_transactions,
                                            symbol)
        snap.insider_sentiment     = _safe("insider_sentiment",
                                            self.insider_sentiment, symbol)
        snap.filings               = _safe("filings",
                                            self.filings, symbol)
        snap.dividends             = _safe("dividends",
                                            self.dividends, symbol)
        if include_heavy:
            snap.financials_reported = _safe(
                "financials_reported",
                self.financials_reported, symbol,
            )
            snap.similarity_index    = _safe(
                "similarity_index",
                self.similarity_index, symbol,
            )
        return snap

    # ------------------------------------------------ LLM-friendly compaction

    def compact_snapshot(self, snap: IntelSnapshot) -> Dict[str, Any]:
        """Shrink a full bundle into a trim, LLM-friendly dict. Drops
        large arrays, summarizes ownership, and highlights only the
        most actionable signals (analyst upside, insider net, filings)."""
        out: Dict[str, Any] = {"symbol": snap.symbol}

        # Analyst
        if snap.price_target:
            pt = snap.price_target
            out["analyst"] = {
                "target_mean":   pt.get("targetMean"),
                "target_high":   pt.get("targetHigh"),
                "target_low":    pt.get("targetLow"),
                "n_analysts":    pt.get("numberOfAnalysts"),
                "last_updated":  pt.get("lastUpdated"),
            }
        if snap.recommendation_trends:
            latest = (snap.recommendation_trends or [{}])[0]
            out["recommendations"] = {
                "period":    latest.get("period"),
                "strong_buy": latest.get("strongBuy"),
                "buy":        latest.get("buy"),
                "hold":       latest.get("hold"),
                "sell":       latest.get("sell"),
                "strong_sell": latest.get("strongSell"),
            }

        # Basic financials — pluck the most-watched
        if snap.basic_financials and isinstance(snap.basic_financials, dict):
            m = snap.basic_financials.get("metric", {}) or {}
            out["fundamentals"] = {
                "pe_ttm":      m.get("peNormalizedAnnual") or m.get("peTTM"),
                "market_cap":  m.get("marketCapitalization"),
                "52w_high":    m.get("52WeekHigh"),
                "52w_low":     m.get("52WeekLow"),
                "52w_pct_change": m.get("52WeekPriceReturnDaily"),
                "beta":        m.get("beta"),
                "div_yield":   m.get("dividendYieldIndicatedAnnual"),
                "eps_growth":  m.get("epsGrowth5Y") or m.get("epsGrowthQuarterly"),
                "revenue_growth_ttm": m.get("revenueGrowth5Y"),
                "profit_margin": m.get("netProfitMarginTTM"),
                "debt_to_equity": m.get("totalDebt/totalEquityAnnual"),
            }

        # Insider: net sells vs buys in last 3 mo
        if snap.insider_transactions:
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            recent = [t for t in snap.insider_transactions
                        if (t.get("transactionDate") or "") >= cutoff]
            buys  = sum(1 for t in recent if (t.get("change") or 0) > 0)
            sells = sum(1 for t in recent if (t.get("change") or 0) < 0)
            net_shares = sum((t.get("change") or 0) for t in recent)
            out["insider_90d"] = {
                "buys": buys, "sells": sells,
                "net_shares": net_shares, "n_tx": len(recent),
            }
        if snap.insider_sentiment and isinstance(snap.insider_sentiment, dict):
            data = snap.insider_sentiment.get("data", []) or []
            if data:
                latest = data[-1]
                out["insider_sentiment"] = {
                    "mspr":     latest.get("mspr"),   # -100..100
                    "change":   latest.get("change"),
                    "month":    latest.get("month"),
                    "year":     latest.get("year"),
                }

        # Ownership — top 5 holders
        if snap.ownership:
            top = snap.ownership[:5]
            out["top_holders"] = [
                {"name": h.get("name"),
                 "share": h.get("share"),
                 "change": h.get("change")}
                for h in top
            ]
        if snap.institutional_ownership and isinstance(
                snap.institutional_ownership, dict):
            arr = snap.institutional_ownership.get("data", []) or []
            if arr:
                latest = arr[-1]
                out["institutional_ownership"] = {
                    "at_date": latest.get("atDate"),
                    "n_holders": (latest.get("ownership") or {}).get(
                        "numberOfOwners"
                    ),
                    "pct_held": (latest.get("ownership") or {}).get(
                        "percentage"
                    ),
                }

        # Filings — last 3
        if snap.filings:
            out["recent_filings"] = [
                {"form": f.get("form"),
                 "filed_at": f.get("filedDate"),
                 "accepted_at": f.get("acceptedDate")}
                for f in snap.filings[:3]
            ]

        # Dividends — next upcoming if any
        if snap.dividends:
            upcoming = [d for d in snap.dividends
                          if (d.get("date") or "")
                          >= time.strftime("%Y-%m-%d")]
            if upcoming:
                d = upcoming[0]
                out["next_dividend"] = {
                    "ex_date": d.get("date"),
                    "amount":  d.get("amount"),
                    "pay_date": d.get("payDate"),
                }

        # Revenue breakdown — top segments
        if snap.revenue_breakdown and isinstance(snap.revenue_breakdown, dict):
            data = snap.revenue_breakdown.get("data", []) or []
            if data:
                latest = data[0]
                segments = latest.get("breakdown", {}) or {}
                top_segs = sorted(segments.items(),
                                    key=lambda kv: -(kv[1] or 0))[:5]
                out["revenue_segments"] = [
                    {"segment": k, "revenue": v} for k, v in top_segs
                ]

        if snap.errors:
            out["_errors"] = snap.errors[:5]
        return out


# ---- module-level factory + cache ----------------------------


_singleton: Optional[FinnhubIntelligence] = None
_singleton_lock = RLock()


def build_finnhub_intelligence(
    api_key: Optional[str] = None,
) -> Optional[FinnhubIntelligence]:
    """Singleton — shared cache across all callers. Returns None when
    FINNHUB_KEY isn't set."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            fh = FinnhubIntelligence(api_key=api_key)
            if not fh.is_enabled:
                _log.info("finnhub_intel_skip: FINNHUB_KEY not set")
                return None
            _singleton = fh
        return _singleton
