"""Market session clock. US Eastern time-aware.

All decisions use market time (America/New_York), not UTC or local.
"""
from __future__ import annotations

from datetime import datetime, time, date, timedelta
from typing import Optional
import pytz

ET = pytz.timezone("America/New_York")

# US market holidays (partial — extend for production). For backtest-level accuracy
# we rely on the bar stream omitting holidays rather than hard-coded dates.
_HARD_HOLIDAYS_2026 = {
    date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3),  date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3),  date(2026, 9, 7),  date(2026, 11, 26),
    date(2026, 12, 25),
}


class MarketClock:
    def __init__(
        self,
        market_open: str = "09:30",
        market_close: str = "16:00",
        no_new_entries_after: str = "15:30",
        eod_force_close: str = "15:45",
    ):
        self.market_open = time.fromisoformat(market_open + ":00")
        self.market_close = time.fromisoformat(market_close + ":00")
        self.no_new_entries_after = time.fromisoformat(no_new_entries_after + ":00")
        self.eod_force_close = time.fromisoformat(eod_force_close + ":00")

    @staticmethod
    def now_et() -> datetime:
        return datetime.now(tz=ET)

    @staticmethod
    def to_et(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return ET.localize(dt)
        return dt.astimezone(ET)

    def is_trading_day(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_et()
        dt = self.to_et(dt)
        if dt.weekday() >= 5:
            return False
        if dt.date() in _HARD_HOLIDAYS_2026:
            return False
        return True

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_et()
        dt = self.to_et(dt)
        if not self.is_trading_day(dt):
            return False
        t = dt.time()
        return self.market_open <= t < self.market_close

    def can_enter_new(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_et()
        dt = self.to_et(dt)
        if not self.is_market_open(dt):
            return False
        return dt.time() < self.no_new_entries_after

    def should_eod_force_close(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_et()
        dt = self.to_et(dt)
        if not self.is_trading_day(dt):
            return False
        return dt.time() >= self.eod_force_close

    def minutes_to_close(self, dt: Optional[datetime] = None) -> float:
        dt = dt or self.now_et()
        dt = self.to_et(dt)
        close_dt = dt.replace(
            hour=self.market_close.hour,
            minute=self.market_close.minute,
            second=0,
            microsecond=0,
        )
        return (close_dt - dt).total_seconds() / 60.0

    def opening_range_window(self, dt: Optional[datetime] = None, minutes: int = 30):
        """Return (start, end) datetimes for the opening-range window."""
        dt = dt or self.now_et()
        dt = self.to_et(dt)
        start = dt.replace(
            hour=self.market_open.hour,
            minute=self.market_open.minute,
            second=0,
            microsecond=0,
        )
        end = start + timedelta(minutes=minutes)
        return start, end
