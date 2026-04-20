"""VixProbe tests: cache hit/miss, fallback, source preference."""
import time
from src.intelligence.vix_probe import VixProbe, VixReading


def test_fallback_returned_when_no_sources_succeed(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    p = VixProbe(ttl_seconds=30, fallback_vix=14.5, prefer="auto")
    # Force both sources to no-op
    monkeypatch.setattr(p, "_try_alpaca", lambda: None)
    monkeypatch.setattr(p, "_try_yfinance", lambda: None)
    r = p.get()
    assert isinstance(r, VixReading)
    assert r.value == 14.5
    assert r.source == "fallback"


def test_cache_returns_cached_reading_within_ttl(monkeypatch):
    p = VixProbe(ttl_seconds=30)
    calls = {"n": 0}
    def once():
        calls["n"] += 1
        return 23.7
    monkeypatch.setattr(p, "_try_yfinance", once)
    monkeypatch.setattr(p, "_try_alpaca", lambda: None)

    r1 = p.get()
    r2 = p.get()
    r3 = p.get()
    assert r1.value == 23.7 and r1.source == "yfinance"
    assert r2.value == 23.7 and r3.value == 23.7
    assert calls["n"] == 1, f"expected exactly one fetch, got {calls['n']}"


def test_cache_refreshes_after_ttl(monkeypatch):
    p = VixProbe(ttl_seconds=1)
    seq = iter([12.0, 18.0])
    monkeypatch.setattr(p, "_try_alpaca", lambda: None)
    monkeypatch.setattr(p, "_try_yfinance", lambda: next(seq))
    r1 = p.get()
    time.sleep(1.1)
    r2 = p.get()
    assert r1.value == 12.0
    assert r2.value == 18.0


def test_prefer_alpaca_tries_alpaca_only(monkeypatch):
    p = VixProbe(ttl_seconds=30, prefer="alpaca")
    called = {"alpaca": 0, "yf": 0}
    monkeypatch.setattr(p, "_try_alpaca", lambda: (called.__setitem__("alpaca", called["alpaca"] + 1) or None))
    monkeypatch.setattr(p, "_try_yfinance", lambda: (called.__setitem__("yf", called["yf"] + 1) or 99.9))
    r = p.get()
    # alpaca returned None, yfinance not tried (prefer=alpaca only)
    assert called["alpaca"] == 1
    assert called["yf"] == 0
    assert r.source == "fallback"
