"""MVP test suite for the wheel strategy (Path B).

Covers the three critical behaviors:
  1. WheelExitEvaluator fires PT at +50% pnl on short puts
  2. WheelExitEvaluator fires SL at -100% pnl (option doubled)
  3. WheelExitEvaluator fires time-stop at <= 21 DTE
  4. Skips long-option positions (those go through FastExitEvaluator)
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pytest

from src.core.types import Position, OptionRight


def _short_put(*, dte_days: int, entry_premium: float,
                qty: int = -1) -> Position:
    """Build a short-put Position (qty negative) with given entry premium."""
    return Position(
        symbol="SPY260518P00700000",
        qty=qty,                  # NEGATIVE = short
        avg_price=entry_premium,
        is_option=True, multiplier=100,
        underlying="SPY", strike=700.0,
        expiry=date.today() + timedelta(days=dte_days),
        right=OptionRight.PUT,
        entry_ts=time.time(),
    )


# ---------- PT ----------
def test_wheel_pt_fires_when_option_halves():
    from src.exits.wheel_exits import WheelExitEvaluator, WheelExitConfig
    ev = WheelExitEvaluator(WheelExitConfig(profit_target_pct=0.50,
                                              dte_roll_threshold=21))
    pos = _short_put(dte_days=35, entry_premium=4.00)
    # Option dropped from $4.00 → $2.00 = we captured 50% of premium
    d = ev.evaluate(pos, current_price=2.00)
    assert d is not None
    assert d.should_close is True
    assert "wheel_pt_50pct_capture" in d.reason


def test_wheel_pt_does_not_fire_at_40pct_capture():
    from src.exits.wheel_exits import WheelExitEvaluator, WheelExitConfig
    ev = WheelExitEvaluator(WheelExitConfig(profit_target_pct=0.50,
                                              dte_roll_threshold=21))
    pos = _short_put(dte_days=35, entry_premium=4.00)
    # $2.40 = 40% capture (not yet 50%)
    d = ev.evaluate(pos, current_price=2.40)
    assert d is None


# ---------- SL ----------
def test_wheel_sl_fires_when_option_doubles():
    from src.exits.wheel_exits import WheelExitEvaluator, WheelExitConfig
    ev = WheelExitEvaluator(WheelExitConfig(stop_loss_pct=1.00,
                                              dte_roll_threshold=21))
    pos = _short_put(dte_days=35, entry_premium=4.00)
    d = ev.evaluate(pos, current_price=8.00)   # 2× entry → -100% pnl
    assert d is not None
    assert d.should_close is True
    assert "wheel_sl_option_doubled" in d.reason


def test_wheel_sl_does_not_fire_at_50pct_loss():
    from src.exits.wheel_exits import WheelExitEvaluator, WheelExitConfig
    ev = WheelExitEvaluator(WheelExitConfig(stop_loss_pct=1.00,
                                              dte_roll_threshold=21))
    pos = _short_put(dte_days=35, entry_premium=4.00)
    d = ev.evaluate(pos, current_price=6.00)   # -50% pnl, under -100 threshold
    assert d is None


# ---------- DTE roll ----------
def test_wheel_dte_roll_fires_at_21():
    from src.exits.wheel_exits import WheelExitEvaluator, WheelExitConfig
    ev = WheelExitEvaluator(WheelExitConfig(dte_roll_threshold=21))
    pos = _short_put(dte_days=21, entry_premium=4.00)
    # Not hit PT or SL
    d = ev.evaluate(pos, current_price=3.50)
    assert d is not None
    assert d.should_close is True
    assert "wheel_dte_roll" in d.reason


def test_wheel_dte_roll_does_not_fire_at_22():
    from src.exits.wheel_exits import WheelExitEvaluator, WheelExitConfig
    ev = WheelExitEvaluator(WheelExitConfig(dte_roll_threshold=21))
    pos = _short_put(dte_days=22, entry_premium=4.00)
    d = ev.evaluate(pos, current_price=3.50)
    assert d is None


# ---------- Long positions ignored ----------
def test_wheel_exits_skip_long_positions():
    """WheelExitEvaluator MUST ignore long options (qty > 0). Those are
    handled by FastExitEvaluator."""
    from src.exits.wheel_exits import WheelExitEvaluator
    ev = WheelExitEvaluator()
    long_pos = _short_put(dte_days=10, entry_premium=4.00, qty=+3)
    d = ev.evaluate(long_pos, current_price=2.00)
    assert d is None   # skipped, not its job


def test_wheel_exits_skip_stock_positions():
    from src.exits.wheel_exits import WheelExitEvaluator
    from src.core.types import Position
    ev = WheelExitEvaluator()
    stock_pos = Position(
        symbol="AAPL", qty=-100, avg_price=200.0,
        is_option=False, multiplier=1,
    )
    d = ev.evaluate(stock_pos, current_price=150.0)
    assert d is None


# ---------- Build-close-order helper ----------
def test_build_close_order_flattens_short():
    from src.exits.wheel_exits import build_wheel_close_order
    pos = _short_put(dte_days=30, entry_premium=4.00, qty=-2)
    o = build_wheel_close_order(pos, limit_price=2.10)
    # To close a short, we BUY
    from src.core.types import Side
    assert o.side == Side.BUY
    assert o.qty == 2
    assert o.is_option is True
    assert o.tif == "DAY"
    assert o.tag == "wheel_close"


# ---------- WheelRunner pick ----------
def test_wheel_runner_picks_near_target_strike(monkeypatch):
    """The runner's put-picker must pick an OTM put near the target strike."""
    from src.signals.wheel_runner import WheelRunner, WheelRunnerConfig
    from src.core.types import OptionContract, OptionRight
    from datetime import date as _d, timedelta as _td

    chain = [
        OptionContract(symbol="SPY_670P", underlying="SPY",
                         strike=670, expiry=_d.today() + _td(days=35),
                         right=OptionRight.PUT, multiplier=100,
                         open_interest=1000, today_volume=500,
                         bid=3.00, ask=3.20, last=3.10, iv=0.22),
        OptionContract(symbol="SPY_680P", underlying="SPY",
                         strike=680, expiry=_d.today() + _td(days=35),
                         right=OptionRight.PUT, multiplier=100,
                         open_interest=1000, today_volume=500,
                         bid=4.50, ask=4.70, last=4.60, iv=0.22),
        OptionContract(symbol="SPY_720P", underlying="SPY",
                         strike=720, expiry=_d.today() + _td(days=35),
                         right=OptionRight.PUT, multiplier=100,
                         open_interest=1000, today_volume=500,
                         bid=9.00, ask=9.30, last=9.15, iv=0.22),
    ]
    # Fake bot — only needs the one attribute the picker uses
    class _FakeBot: pass
    cfg = WheelRunnerConfig(universe=["SPY"], target_delta=0.30)
    runner = WheelRunner(cfg, _FakeBot())
    # spot=700, target=spot*(1-0.3*0.1)=700*0.97=679. Nearest OTM put=680.
    pick = runner._pick_put(chain, 700.0)
    assert pick is not None
    assert pick.strike == 680   # nearest OTM to target
