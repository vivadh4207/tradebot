"""Tests for the ORB signal, covering both immediate and retest modes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

from src.core.types import Side, OptionRight
from src.signals.orb import OpeningRangeBreakout
from src.signals.base import SignalContext


def _ctx(spot, orh=580.5, orl=579.5, symbol="SPY"):
    return SignalContext(
        symbol=symbol,
        now=datetime(2026, 4, 17, 10, 30, tzinfo=timezone.utc),
        bars=[],
        spot=spot,
        opening_range_high=orh,
        opening_range_low=orl,
    )


# ---------- immediate mode (legacy) ----------


def test_immediate_fires_on_upside_breakout():
    s = OpeningRangeBreakout(retest_required=False)
    sig = s.emit(_ctx(spot=581.0))
    assert sig is not None
    assert sig.side == Side.BUY
    assert sig.option_right == OptionRight.CALL
    assert sig.confidence == 0.7


def test_immediate_fires_on_downside_breakout():
    s = OpeningRangeBreakout(retest_required=False)
    sig = s.emit(_ctx(spot=579.0))
    assert sig is not None
    assert sig.option_right == OptionRight.PUT


def test_immediate_no_signal_inside_range():
    s = OpeningRangeBreakout(retest_required=False)
    assert s.emit(_ctx(spot=580.0)) is None


# ---------- retest mode ----------


def test_retest_does_not_fire_on_first_breakout():
    """Retest mode must NOT fire on the initial break — must wait."""
    s = OpeningRangeBreakout(retest_required=True)
    # Breakout above
    assert s.emit(_ctx(spot=580.8)) is None
    # Still holding above, no pullback yet → still no fire
    assert s.emit(_ctx(spot=580.9)) is None


def test_retest_fires_after_pullback_and_hold_up():
    """Up breakout → pull back to within band of ORH → reclaim = fire."""
    s = OpeningRangeBreakout(retest_required=True, retest_band_pct=0.003)
    # 1. Initial breakout
    assert s.emit(_ctx(spot=580.9)) is None
    # 2. Pullback to within 0.3% of ORH (580.5)
    assert s.emit(_ctx(spot=580.55)) is None
    # 3. Reclaim above ORH → fire
    sig = s.emit(_ctx(spot=580.8))
    assert sig is not None
    assert sig.option_right == OptionRight.CALL
    assert sig.confidence == 0.85          # higher than immediate
    assert sig.meta.get("mode") == "retest"


def test_retest_fires_after_pullback_and_hold_down():
    s = OpeningRangeBreakout(retest_required=True, retest_band_pct=0.003)
    # Down breakout → pullback → break down again
    assert s.emit(_ctx(spot=579.0)) is None
    assert s.emit(_ctx(spot=579.45)) is None        # within band of ORL
    sig = s.emit(_ctx(spot=579.1))
    assert sig is not None
    assert sig.option_right == OptionRight.PUT


def test_retest_does_not_refire_after_emission():
    """Once fired, don't fire again from the same setup."""
    s = OpeningRangeBreakout(retest_required=True, retest_band_pct=0.003)
    s.emit(_ctx(spot=580.9))    # BROKE_UP
    s.emit(_ctx(spot=580.55))   # RETESTED_UP
    sig = s.emit(_ctx(spot=580.8))  # FIRED
    assert sig is not None
    # Subsequent tick at same-or-higher spot should NOT re-fire
    assert s.emit(_ctx(spot=580.9)) is None
    assert s.emit(_ctx(spot=581.0)) is None


def test_retest_resets_on_opposite_breakout():
    """Price regimes flip: up-break → pullback → price blows DOWN through
    ORL. State should flip to BROKE_DN, not fire an up signal."""
    s = OpeningRangeBreakout(retest_required=True, retest_band_pct=0.003)
    s.emit(_ctx(spot=580.9))     # BROKE_UP
    s.emit(_ctx(spot=580.55))    # RETESTED_UP
    # Price collapses through lower band
    assert s.emit(_ctx(spot=579.0)) is None
    # Now should be BROKE_DN; a pullback + reclaim should fire PUT
    assert s.emit(_ctx(spot=579.45)) is None
    sig = s.emit(_ctx(spot=579.0))
    assert sig is not None
    assert sig.option_right == OptionRight.PUT


def test_retest_state_is_per_symbol():
    """SPY and QQQ should have independent state machines."""
    s = OpeningRangeBreakout(retest_required=True, retest_band_pct=0.003)
    # SPY breakout
    s.emit(_ctx(symbol="SPY", spot=580.9))
    # QQQ is still idle — its first signal should be None (not inherit SPY state)
    r = s.emit(SignalContext(symbol="QQQ",
                               now=datetime.now(tz=timezone.utc),
                               bars=[], spot=641.0,
                               opening_range_high=640.0,
                               opening_range_low=639.0))
    assert r is None   # QQQ's first tick is just a breakout, not a retest-fire


def test_reset_clears_state():
    s = OpeningRangeBreakout(retest_required=True, retest_band_pct=0.003)
    s.emit(_ctx(spot=580.9))
    s.reset()
    # After reset, a breakout should start fresh (still no fire without retest)
    assert s.emit(_ctx(spot=580.95)) is None


def test_reset_single_symbol():
    s = OpeningRangeBreakout(retest_required=True, retest_band_pct=0.003)
    s.emit(_ctx(symbol="SPY", spot=580.9))
    s.emit(_ctx(symbol="QQQ", spot=641.0))
    s.reset(symbol="SPY")
    assert "SPY" not in s._state
    assert "QQQ" in s._state
