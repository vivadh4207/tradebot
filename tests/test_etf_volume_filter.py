"""Regression tests for ETF-aware volume confirmation.

Locks in the fix for the real-session bug where SPY/IWM signals with
valid ensemble scores were blocked at f09_volume_confirmation because
the single 1.2× threshold was inappropriate for mega-liquid ETFs.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _ctx(symbol: str, *, is_etf: bool, cur_vol: float, avg_vol: float,
          settings_dict: dict | None = None):
    """Build a minimal ExecutionContext for the volume-filter test."""
    from src.core.types import Signal, Side
    from src.risk.execution_chain import ExecutionContext
    sig = Signal(
        source="ensemble", symbol=symbol, side=Side.BUY,
        confidence=1.0, rationale="test", meta={},
    )
    return ExecutionContext(
        signal=sig,
        now=datetime.now(tz=timezone.utc),
        account_equity=10_000.0, day_pnl=0.0, open_positions_count=0,
        current_bar_volume=cur_vol, avg_bar_volume=avg_vol,
        opening_range_high=100.0, opening_range_low=99.0,
        spot=99.5, vwap=99.5, vix=15.0,
        is_etf=is_etf,
    )


def _chain(settings_dict: dict):
    """Build an ExecutionChain with given nested settings dict."""
    from src.risk.execution_chain import ExecutionChain
    from src.core.clock import MarketClock
    clock = MarketClock(
        market_open="09:30", market_close="16:00",
        no_new_entries_after="15:30", eod_force_close="15:45",
    )
    return ExecutionChain(settings_dict, clock)


def test_etf_passes_volume_filter_at_1x():
    """SPY at 1.0× avg volume MUST pass (it was blocked before)."""
    ec = _chain({
        "execution": {
            "min_volume_confirmation": 1.20,
            "min_volume_confirmation_etf": 0.80,
            "min_volume_confirmation_stock": 1.20,
        },
    })
    ctx = _ctx("SPY", is_etf=True, cur_vol=1_000_000, avg_vol=1_000_000)
    r = ec.f09_volume_confirmation(ctx)
    assert r.passed is True
    assert "etf" in r.reason


def test_etf_blocked_when_deeply_below_etf_threshold():
    """SPY at 0.50× avg volume should still block (0.80 min for ETFs)."""
    ec = _chain({
        "execution": {
            "min_volume_confirmation_etf": 0.80,
            "min_volume_confirmation_stock": 1.20,
        },
    })
    ctx = _ctx("SPY", is_etf=True, cur_vol=500_000, avg_vol=1_000_000)
    r = ec.f09_volume_confirmation(ctx)
    assert r.passed is False
    assert "etf" in r.reason and "0.50<0.8" in r.reason


def test_stock_still_requires_1_2x_surge():
    """Stock at 1.0× avg volume MUST block (the pre-existing behavior)."""
    ec = _chain({
        "execution": {
            "min_volume_confirmation": 1.20,
            "min_volume_confirmation_etf": 0.80,
            "min_volume_confirmation_stock": 1.20,
        },
    })
    ctx = _ctx("AAPL", is_etf=False, cur_vol=1_000_000, avg_vol=1_000_000)
    r = ec.f09_volume_confirmation(ctx)
    assert r.passed is False
    assert "stock" in r.reason
    assert "1.00<1.2" in r.reason


def test_stock_passes_when_volume_surges():
    """Stock at 1.5× avg volume passes — this is what the filter was built for."""
    ec = _chain({
        "execution": {
            "min_volume_confirmation_stock": 1.20,
            "min_volume_confirmation_etf": 0.80,
        },
    })
    ctx = _ctx("AAPL", is_etf=False, cur_vol=1_500_000, avg_vol=1_000_000)
    r = ec.f09_volume_confirmation(ctx)
    assert r.passed is True


def test_legacy_single_threshold_still_honored():
    """If someone's old settings.yaml only has min_volume_confirmation,
    the stock path falls back to it cleanly."""
    ec = _chain({
        "execution": {
            "min_volume_confirmation": 1.50,   # stricter legacy value
            "min_volume_confirmation_etf": 0.80,
            # NOTE: min_volume_confirmation_stock deliberately omitted
        },
    })
    ctx = _ctx("AAPL", is_etf=False, cur_vol=1_300_000, avg_vol=1_000_000)
    r = ec.f09_volume_confirmation(ctx)
    # 1.30x < 1.50x legacy fallback → block
    assert r.passed is False


def test_is_etf_classifier_covers_sector_spdrs():
    """The _is_etf helper must flag the user's sector-ETF universe
    (XLF, XLE, XBI, XHB, XUR, etc.) — otherwise f09 uses the strict
    stock threshold and blocks them like regular equities did before."""
    from src.main import _is_etf
    assert _is_etf("SPY") is True
    assert _is_etf("QQQ") is True
    assert _is_etf("IWM") is True
    assert _is_etf("XLF") is True
    assert _is_etf("XLE") is True
    assert _is_etf("XBI") is True
    assert _is_etf("XHB") is True
    assert _is_etf("XUR") is True
    assert _is_etf("TLT") is True
    # Not ETFs:
    assert _is_etf("AAPL") is False
    assert _is_etf("NVDA") is False
    assert _is_etf("PDD") is False
    # Edge case: 5+ char symbols starting with X are not sector ETFs
    assert _is_etf("XYZCORP") is False
