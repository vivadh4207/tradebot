"""Economic + per-symbol catalyst calendar.

- BROAD events (FOMC, CPI, NFP, etc.) apply to the whole market and use
  a ±30-minute blackout window.
- SYMBOL blackouts apply to a single ticker for the entire trading day
  (earnings, FDA PDUFA dates, advisory-committee meetings). These are
  full-day because our bot can't time the announcement precisely and
  gap/vol risk spans the session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Set


@dataclass
class ScheduledEvent:
    name: str
    when: datetime
    impact: str = "high"              # 'high' | 'medium' | 'low'
    symbol: Optional[str] = None      # set for symbol-specific events


class EconomicCalendar:
    def __init__(self,
                 events: Optional[List[ScheduledEvent]] = None,
                 blackout_minutes_before: int = 30,
                 blackout_minutes_after: int = 30):
        self.events: List[ScheduledEvent] = events or []
        self.before = timedelta(minutes=blackout_minutes_before)
        self.after = timedelta(minutes=blackout_minutes_after)
        self._symbol_days: Dict[str, Set[date]] = {}     # full-day blackouts

    # ----- broad events -----
    def add(self, event: ScheduledEvent) -> None:
        self.events.append(event)

    def in_blackout(self, now: datetime, symbol: Optional[str] = None) -> bool:
        """Return True if `now` is within a blackout window.

        If `symbol` is given, also consult the per-symbol full-day blackouts.
        Tolerates mixed tz-aware / naive inputs by stripping tzinfo for the
        comparison only.
        """
        def _strip(dt: datetime) -> datetime:
            return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

        now_naive = _strip(now)
        for e in self.events:
            if e.impact != "high":
                continue
            if e.symbol and symbol and e.symbol != symbol:
                continue
            w = _strip(e.when)
            if w - self.before <= now_naive <= w + self.after:
                return True
        if symbol and symbol in self._symbol_days:
            if now_naive.date() in self._symbol_days[symbol]:
                return True
        return False

    # ----- per-symbol full-day blackouts -----
    def add_symbol_blackout(self, symbol: str, d: date) -> None:
        self._symbol_days.setdefault(symbol.upper(), set()).add(d)

    def symbol_blacked_out(self, symbol: str, d: date) -> bool:
        return d in self._symbol_days.get(symbol.upper(), set())

    def clear_symbol_blackouts(self) -> None:
        self._symbol_days.clear()

    def summary(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for sym, days in self._symbol_days.items():
            out[sym] = [d.isoformat() for d in sorted(days)]
        return out
