"""Regression test for the 0DTE scalp-window force-close.

User-reported: 0DTE positions were being bought but not sold until
15:45 ET EOD sweep, by which time theta had destroyed most of the
premium. The scalp-window caps the hold time to N minutes so the bot
takes the open P&L (win or loss) before theta eats the remaining value.

Both FastExitEvaluator (runs every 5s) and ExitEngine (runs with main
loop) must enforce the timeout; otherwise one path fires and the other
doesn't.
"""
from __future__ import annotations

import time
from datetime import date, datetime, time as dtime, timezone

import pytest

from src.core.types import Position, OptionRight


def _mk_pos(*, age_minutes: float, dte_days: int,
             avg_price: float = 1.00) -> Position:
    """Build a Position whose entry_ts is `age_minutes` ago and whose
    expiry is `dte_days` from today."""
    return Position(
        symbol="TEST260417C00100000",
        qty=1, avg_price=avg_price,
        is_option=True, multiplier=100,
        underlying="TEST", strike=100.0,
        expiry=date.today() + __import__("datetime").timedelta(days=dte_days),
        right=OptionRight.CALL,
        entry_ts=time.time() - age_minutes * 60.0,
    )


# ---------- FastExitEvaluator ----------
def test_fast_exit_0dte_scalp_timeout_fires_after_threshold():
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(zero_dte_max_hold_minutes=30.0))
    pos = _mk_pos(age_minutes=45, dte_days=0)  # held 45min on 0DTE
    # Price unchanged from entry: no PT/SL would fire on P&L alone
    d = ev.evaluate(pos, current_price=pos.avg_price)
    assert d is not None
    assert d.should_close is True
    assert "0dte_scalp_timeout" in d.reason


def test_fast_exit_0dte_under_threshold_stays_open():
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(zero_dte_max_hold_minutes=30.0))
    pos = _mk_pos(age_minutes=10, dte_days=0)  # only held 10min
    d = ev.evaluate(pos, current_price=pos.avg_price)
    # No exit — P&L flat, hold time under threshold
    assert d is None


def test_fast_exit_scalp_does_not_apply_to_1dte():
    """1DTE positions have overnight theta already priced in; the scalp
    cutoff intentionally only fires for same-day (DTE=0) contracts."""
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(zero_dte_max_hold_minutes=30.0))
    pos = _mk_pos(age_minutes=120, dte_days=1)
    d = ev.evaluate(pos, current_price=pos.avg_price)
    assert d is None  # 1DTE not subject to the scalp timeout


def test_fast_exit_scalp_disabled_when_zero():
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(zero_dte_max_hold_minutes=0.0))
    pos = _mk_pos(age_minutes=240, dte_days=0)
    # Held 4 hours on 0DTE, P&L flat — but timeout is disabled
    d = ev.evaluate(pos, current_price=pos.avg_price)
    assert d is None


# ---------- ExitEngine ----------
def test_exit_engine_layer_1_5_scalp_timeout():
    from src.exits.exit_engine import ExitEngine, ExitEngineConfig
    eng = ExitEngine(ExitEngineConfig(
        zero_dte_max_hold_minutes=30.0,
        # Force pre-EOD time so Layer 1 (15:45 force close) does NOT fire
        zero_dte_force_close_time=dtime(15, 45),
    ))
    pos = _mk_pos(age_minutes=45, dte_days=0)
    now = datetime(2026, 4, 17, 14, 0, tzinfo=timezone.utc)  # well before 15:45
    d = eng.decide(pos, current_price=pos.avg_price, now=now,
                    vix=15.0, spot=100.0, vwap=100.0, bars=[])
    assert d.should_close is True
    assert "scalp_timeout" in d.reason


def test_exit_engine_pt_still_fires_before_scalp_timeout():
    """If profit-target hits in the scalp window, that wins — the scalp
    window is a floor, not a cap on good exits."""
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(
        pt_short_pct=0.35, zero_dte_max_hold_minutes=30.0,
    ))
    pos = _mk_pos(age_minutes=10, dte_days=0, avg_price=1.00)
    # Price moved to +40% in 10 min — PT should fire
    d = ev.evaluate(pos, current_price=1.40)
    assert d is not None
    assert d.should_close is True
    assert "pt_hit" in d.reason  # not scalp_timeout
