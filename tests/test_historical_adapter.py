from datetime import datetime, timedelta
import pytest

from src.data.historical_adapter import HistoricalMarketDataAdapter
from src.core.clock import ET


def test_cache_path_is_deterministic(tmp_path):
    end = datetime(2026, 4, 16, 14, 30, tzinfo=ET)
    start = end - timedelta(days=5)
    a = HistoricalMarketDataAdapter(
        symbols=[], start=start, end=end, timeframe_minutes=1,
        cache_dir=str(tmp_path), prefer="yfinance",
    )
    p1 = a._cache_key("SPY")
    p2 = a._cache_key("SPY")
    assert p1 == p2
    assert p1 != a._cache_key("QQQ")


def test_falls_back_to_synthetic_when_sources_empty(tmp_path, monkeypatch):
    # Prevent any real network work by stripping creds and forcing yfinance
    # to return nothing via monkeypatching the adapter's fetch methods.
    monkeypatch.setenv("ALPACA_API_KEY_ID", "")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "")
    a = HistoricalMarketDataAdapter(
        symbols=[],
        start=datetime(2026, 4, 15, tzinfo=ET),
        end=datetime(2026, 4, 16, tzinfo=ET),
        timeframe_minutes=1,
        cache_dir=str(tmp_path),
        prefer="auto",
    )
    monkeypatch.setattr(a, "_fetch_alpaca", lambda s: [])
    monkeypatch.setattr(a, "_fetch_yfinance", lambda s: [])
    bars = a.get_bars("SPY", limit=50)
    assert len(bars) > 0                         # synthetic took over
    assert all(b.symbol == "SPY" for b in bars)


def test_latest_quote_from_last_bar(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "")
    a = HistoricalMarketDataAdapter(
        symbols=[],
        start=datetime(2026, 4, 15, tzinfo=ET),
        end=datetime(2026, 4, 16, tzinfo=ET),
        cache_dir=str(tmp_path), prefer="auto",
    )
    monkeypatch.setattr(a, "_fetch_alpaca", lambda s: [])
    monkeypatch.setattr(a, "_fetch_yfinance", lambda s: [])
    q = a.latest_quote("SPY")
    assert q is not None and q.bid < q.ask
