"""CatalystCalendar — per-symbol earnings + FDA dates.

Aggregates three providers, in order of preference:
  1. StaticCatalystProvider   — user-editable YAML (config/catalysts.yaml).
                                 Always checked. Authoritative for FDA/PDUFA.
  2. FinnhubCalendarProvider  — if FINNHUB_KEY is set. Covers earnings +
                                 economic events including some FDA.
  3. YFinanceEarningsProvider — fallback for earnings when Finnhub isn't
                                 configured. No API key required.

The aggregator produces `CatalystEvent` objects and hydrates an
`EconomicCalendar` with full-day per-symbol blackouts.
"""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .econ_calendar import EconomicCalendar, ScheduledEvent


@dataclass
class CatalystEvent:
    symbol: str
    event_type: str                # 'earnings' | 'fda' | 'other'
    when: date
    details: str = ""
    timing: str = "unknown"        # 'bmo' | 'amc' | 'unknown'


class CatalystProvider(abc.ABC):
    @abc.abstractmethod
    def fetch(self, symbols: List[str], days: int = 14) -> List[CatalystEvent]: ...


# ------------------------------------------------------------------ Static
class StaticCatalystProvider(CatalystProvider):
    """Reads catalysts from a YAML file.

    Schema (either list is optional):
        fda:
          - symbol: MRNA
            date: 2026-05-15
            event: PDUFA decision
        earnings:
          - symbol: AAPL
            date: 2026-05-02
            timing: amc
            event: Q2 earnings
    """

    def __init__(self, path: str | Path = "config/catalysts.yaml"):
        self._path = Path(path)

    def fetch(self, symbols: List[str], days: int = 14) -> List[CatalystEvent]:
        if not self._path.exists():
            return []
        try:
            import yaml
            data = yaml.safe_load(self._path.read_text()) or {}
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "static_catalyst_parse_failed path=%s err=%s", self._path, _e
            )
            return []
        wanted = {s.upper() for s in symbols}
        out: List[CatalystEvent] = []
        today = date.today()
        horizon = today + timedelta(days=days)
        for item in data.get("fda", []) or []:
            ev = self._to_event(item, "fda")
            if ev and ev.symbol in wanted and today <= ev.when <= horizon:
                out.append(ev)
        for item in data.get("earnings", []) or []:
            ev = self._to_event(item, "earnings")
            if ev and ev.symbol in wanted and today <= ev.when <= horizon:
                out.append(ev)
        return out

    @staticmethod
    def _to_event(item: Dict, kind: str) -> Optional[CatalystEvent]:
        try:
            sym = str(item["symbol"]).upper()
            d = item["date"]
            if isinstance(d, str):
                d = date.fromisoformat(d)
            timing = str(item.get("timing", "unknown")).lower()
            details = str(item.get("event", "") or "")
            return CatalystEvent(symbol=sym, event_type=kind,
                                  when=d, details=details, timing=timing)
        except Exception:
            return None


# ------------------------------------------------------------------ Finnhub
class FinnhubCalendarProvider(CatalystProvider):
    """Uses Finnhub /calendar/earnings and /calendar/economic endpoints.

    Requires FINNHUB_KEY in env. Silently returns [] if missing or on error.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._key = api_key or os.getenv("FINNHUB_KEY", "").strip()

    def fetch(self, symbols: List[str], days: int = 14) -> List[CatalystEvent]:
        if not self._key:
            return []
        try:
            import requests
        except ImportError:
            return []
        today = date.today()
        horizon = today + timedelta(days=days)
        wanted = {s.upper() for s in symbols}
        out: List[CatalystEvent] = []

        # Earnings calendar per-symbol (the /calendar/earnings endpoint is
        # range-based so one call covers all requested names)
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/calendar/earnings",
                params={"from": today.isoformat(),
                        "to": horizon.isoformat(),
                        "token": self._key},
                timeout=6,
            )
            if r.status_code == 200:
                for row in (r.json() or {}).get("earningsCalendar", []) or []:
                    sym = str(row.get("symbol", "")).upper()
                    if sym not in wanted:
                        continue
                    try:
                        d = date.fromisoformat(row["date"])
                    except Exception:
                        continue
                    timing_raw = str(row.get("hour", "") or "").lower()
                    timing = ("bmo" if timing_raw == "bmo"
                              else "amc" if timing_raw == "amc" else "unknown")
                    out.append(CatalystEvent(
                        symbol=sym, event_type="earnings",
                        when=d, timing=timing,
                        details=f"EPS est {row.get('epsEstimate')}",
                    ))
        except Exception as e:                          # noqa: BLE001
            # Surface API-level Finnhub failures so a revoked/throttled
            # key doesn't silently zero out earnings data. Throttled to
            # once per 6h — catalyst refresh runs daily, so this fires
            # at most once per bad day.
            try:
                from ..notify.issue_reporter import report_issue
                report_issue(
                    scope="catalysts.finnhub",
                    message=f"Finnhub earnings calendar fetch failed: {type(e).__name__}: {e}",
                    exc=e,
                    throttle_sec=6 * 3600.0,
                )
            except Exception:
                pass
        return out


# ------------------------------------------------------------------ yfinance
class YFinanceEarningsProvider(CatalystProvider):
    """Free earnings dates via the `yfinance` package. Slow (one HTTP call per
    symbol) — cache results via CatalystCalendar, not per-tick.
    """

    def fetch(self, symbols: List[str], days: int = 14) -> List[CatalystEvent]:
        try:
            import yfinance as yf
        except ImportError:
            return []
        out: List[CatalystEvent] = []
        today = date.today()
        horizon = today + timedelta(days=days)
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                # `calendar` returns earnings date(s) in several shapes
                cal = getattr(t, "calendar", None)
                edate = None
                if cal is None:
                    continue
                # dict shape (newer yfinance)
                if isinstance(cal, dict):
                    raw = cal.get("Earnings Date")
                    if isinstance(raw, list) and raw:
                        edate = raw[0]
                    elif raw is not None:
                        edate = raw
                # DataFrame shape (older yfinance)
                else:
                    try:
                        edate = cal.loc["Earnings Date"].iloc[0]
                    except Exception:
                        edate = None
                if edate is None:
                    continue
                if hasattr(edate, "date"):
                    edate = edate.date()
                if isinstance(edate, str):
                    edate = date.fromisoformat(edate[:10])
                if not isinstance(edate, date):
                    continue
                if today <= edate <= horizon:
                    out.append(CatalystEvent(
                        symbol=sym.upper(), event_type="earnings",
                        when=edate, details="yfinance",
                    ))
            except Exception:
                continue
        return out


# ------------------------------------------------------------------ Aggregator
class CatalystCalendar:
    def __init__(self,
                 providers: Optional[List[CatalystProvider]] = None,
                 lookahead_days: int = 14):
        self.lookahead_days = lookahead_days
        self.providers: List[CatalystProvider] = providers or []
        self.events: List[CatalystEvent] = []
        self._last_refresh: Optional[datetime] = None

    def refresh(self, symbols: List[str]) -> List[CatalystEvent]:
        """Pull from every provider; dedupe by (symbol, date, event_type)."""
        seen = set()
        merged: List[CatalystEvent] = []
        for p in self.providers:
            try:
                got = p.fetch(symbols, days=self.lookahead_days) or []
            except Exception as err:                    # noqa: BLE001
                got = []
                # A whole provider failing means its data is missing
                # from today's catalyst set — could lead to entering
                # trades into unseen earnings. Alert.
                try:
                    from ..notify.issue_reporter import report_issue
                    report_issue(
                        scope=f"catalysts.provider.{type(p).__name__}",
                        message=f"catalyst provider {type(p).__name__} failed: {type(err).__name__}: {err}",
                        exc=err,
                        throttle_sec=6 * 3600.0,
                    )
                except Exception:
                    pass
            for e in got:
                key = (e.symbol, e.when, e.event_type)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(e)
        self.events = merged
        self._last_refresh = datetime.now(tz=timezone.utc)
        return merged

    def hydrate_econ_calendar(self, econ: EconomicCalendar) -> int:
        """Push this calendar's events into the EconomicCalendar as full-day
        symbol blackouts. Returns the count added.
        """
        econ.clear_symbol_blackouts()
        n = 0
        for e in self.events:
            econ.add_symbol_blackout(e.symbol, e.when)
            # ALSO add a high-impact ScheduledEvent at 09:35 ET on the event day
            # with a wide after-window. This is belt-and-suspenders: the
            # full-day blackout is the primary gate.
            dt = datetime.combine(e.when, datetime.min.time()).replace(
                hour=9, minute=35, tzinfo=timezone.utc,
            )
            econ.add(ScheduledEvent(
                name=f"{e.event_type}:{e.symbol}",
                when=dt, impact="high", symbol=e.symbol,
            ))
            n += 1
        return n


def build_default_catalyst_calendar(
    static_yaml_path: str | Path = "config/catalysts.yaml",
    lookahead_days: int = 14,
) -> CatalystCalendar:
    """Wire the three default providers based on environment + config."""
    providers: List[CatalystProvider] = [StaticCatalystProvider(static_yaml_path)]
    if os.getenv("FINNHUB_KEY", "").strip():
        providers.append(FinnhubCalendarProvider())
    providers.append(YFinanceEarningsProvider())   # always last
    return CatalystCalendar(providers=providers, lookahead_days=lookahead_days)
