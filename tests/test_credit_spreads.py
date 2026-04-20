"""Tests for the credit-spread runners and their exit engine."""
from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta
from typing import List, Optional, Dict

import pytest

from src.core.types import (
    OptionContract, OptionRight, Position, Side,
)
from src.exits.credit_spread_exits import (
    CreditSpreadExitConfig, group_spread_positions, evaluate_spread,
    build_close_combo,
)
from src.signals.credit_spread_runner import (
    _delta_of, _pick_by_delta, _rsi, _pivot_low,
)


TODAY = date.today()
FUTURE = TODAY + timedelta(days=35)


def _c(symbol, strike, right, bid=1.0, ask=1.1, iv=0.20, expiry=FUTURE):
    return OptionContract(
        symbol=symbol, underlying="SPY",
        strike=strike, expiry=expiry, right=right,
        bid=bid, ask=ask, iv=iv,
    )


def _p(symbol, qty, avg, right, strike, expiry=FUTURE, tag=""):
    return Position(
        symbol=symbol, qty=qty, avg_price=avg,
        is_option=True, underlying="SPY",
        strike=strike, expiry=expiry, right=right,
        entry_tags={"tag": tag} if tag else {},
    )


# ---------- pick_by_delta ----------


def test_pick_by_delta_selects_closest_to_target():
    """Build a thin synthetic chain with known deltas; verify picker
    lands on the ~0.20-delta put. For 35 DTE + 20% IV, verified deltas:
    460→0.07, 470→0.14, 480→0.22, 490→0.33. Target 0.20 → strike 480."""
    spot = 500.0
    chain = [
        _c("SPY...P460", 460, OptionRight.PUT, iv=0.20),   # |delta|=0.07
        _c("SPY...P470", 470, OptionRight.PUT, iv=0.20),   # |delta|=0.14
        _c("SPY...P480", 480, OptionRight.PUT, iv=0.20),   # |delta|=0.22 ← closest
        _c("SPY...P490", 490, OptionRight.PUT, iv=0.20),   # |delta|=0.33
    ]
    pick = _pick_by_delta(chain, spot, 0.20, OptionRight.PUT,
                          tolerance=0.08)
    assert pick is not None
    assert pick.strike == 480


def test_pick_by_delta_returns_none_when_chain_too_thin():
    """If every row is too far from target, picker refuses."""
    spot = 500.0
    # All deltas ~ -0.5 (ATM); target is 0.05 — too far
    chain = [_c(f"P{s}", s, OptionRight.PUT, iv=0.20) for s in (498, 499, 500, 501)]
    pick = _pick_by_delta(chain, spot, 0.05, OptionRight.PUT, tolerance=0.02)
    assert pick is None


# ---------- rsi + pivot ----------


def test_rsi_all_up_days_returns_100():
    closes = [100 + i for i in range(30)]
    assert _rsi(closes, period=14) == 100.0


def test_rsi_all_down_days_returns_0():
    closes = [100 - i for i in range(30)]
    # all losses, no gains → avg_gain=0 → rs=0 → rsi=0
    assert _rsi(closes, period=14) == pytest.approx(0, abs=0.5)


def test_rsi_too_few_bars_returns_none():
    assert _rsi([100, 101, 102], period=14) is None


def test_pivot_low_returns_min_over_window():
    closes = [105, 103, 99, 101, 104, 102, 100, 106, 107, 105,
              103, 101, 98, 102, 104, 105, 107, 108, 106, 104]
    assert _pivot_low(closes, 20) == 98.0


# ---------- group_spread_positions ----------


def test_group_by_matching_entry_tag():
    legs_qqq = [
        _p("QQQ...P450", -1, 0.85, OptionRight.PUT, 450, tag="weekly_pcs:QQQ:A1"),
        _p("QQQ...P440", +1, 0.25, OptionRight.PUT, 440, tag="weekly_pcs:QQQ:A1"),
    ]
    legs_spy = [
        _p("SPY...P590", -1, 0.90, OptionRight.PUT, 590, tag="0dte_pcs:SPY:B2"),
        _p("SPY...P585", +1, 0.30, OptionRight.PUT, 585, tag="0dte_pcs:SPY:B2"),
    ]
    # Random long-option, not a credit spread — should be ignored
    random_long = _p("AAPL...C200", +1, 3.5, OptionRight.CALL, 200,
                     tag="momentum:AAPL")
    groups = group_spread_positions(legs_qqq + legs_spy + [random_long])
    assert set(groups.keys()) == {"weekly_pcs:QQQ:A1", "0dte_pcs:SPY:B2"}
    assert len(groups["weekly_pcs:QQQ:A1"]) == 2


# ---------- evaluate_spread ----------


def test_profit_target_triggers_exit():
    """Entry credit $0.80. Current net cost $0.30 → 62.5% captured → PT hit."""
    legs = [
        _p("SPY...P450", -1, 1.10, OptionRight.PUT, 450, tag="weekly_pcs:SPY:1"),
        _p("SPY...P440", +1, 0.30, OptionRight.PUT, 440, tag="weekly_pcs:SPY:1"),
    ]
    # Current marks: spread narrowed
    marks = {"SPY...P450": 0.45, "SPY...P440": 0.15}
    cfg = CreditSpreadExitConfig(profit_target_pct=0.50)
    d = evaluate_spread(legs, marks, cfg)
    assert d is not None and d.should_close
    assert "pt_hit" in d.reason
    assert d.net_pnl_pct_of_credit > 0.50


def test_stop_loss_triggers_when_spread_widens_past_threshold():
    """Entry credit $0.80. Spread widened to $1.50 → net -87% of credit → SL hit."""
    legs = [
        _p("SPY...P450", -1, 1.10, OptionRight.PUT, 450, tag="weekly_pcs:SPY:1"),
        _p("SPY...P440", +1, 0.30, OptionRight.PUT, 440, tag="weekly_pcs:SPY:1"),
    ]
    marks = {"SPY...P450": 2.00, "SPY...P440": 0.50}  # close cost = 1.50
    cfg = CreditSpreadExitConfig(stop_loss_pct=0.80)
    d = evaluate_spread(legs, marks, cfg)
    assert d is not None and d.should_close
    assert "sl_hit" in d.reason


def test_no_exit_when_pnl_in_the_middle():
    legs = [
        _p("SPY...P450", -1, 1.10, OptionRight.PUT, 450, tag="weekly_pcs:SPY:1"),
        _p("SPY...P440", +1, 0.30, OptionRight.PUT, 440, tag="weekly_pcs:SPY:1"),
    ]
    # Entry credit 0.80; close cost 0.60 → 25% captured, below 50% PT
    marks = {"SPY...P450": 0.80, "SPY...P440": 0.20}
    cfg = CreditSpreadExitConfig(profit_target_pct=0.50, stop_loss_pct=1.50)
    assert evaluate_spread(legs, marks, cfg) is None


def test_dte_close_triggers_for_weekly_not_0dte():
    """At 20 DTE, weekly spread should close even if still profitable
    per PT/SL. 0DTE never triggers the DTE rule (it has its own
    force-close time)."""
    near_expiry = TODAY + timedelta(days=20)
    legs_weekly = [
        _p("SPY...P450", -1, 1.10, OptionRight.PUT, 450,
           expiry=near_expiry, tag="weekly_pcs:SPY:1"),
        _p("SPY...P440", +1, 0.30, OptionRight.PUT, 440,
           expiry=near_expiry, tag="weekly_pcs:SPY:1"),
    ]
    marks = {"SPY...P450": 0.90, "SPY...P440": 0.25}  # 0.65 close, ~19% captured
    cfg = CreditSpreadExitConfig(dte_close_threshold=21)
    d = evaluate_spread(legs_weekly, marks, cfg)
    assert d is not None and "dte_close" in d.reason

    # Same scenario but tagged 0dte — DTE rule should NOT fire
    legs_0dte = [
        _p("SPY...P450", -1, 1.10, OptionRight.PUT, 450,
           expiry=near_expiry, tag="0dte_pcs:SPY:1"),
        _p("SPY...P440", +1, 0.30, OptionRight.PUT, 440,
           expiry=near_expiry, tag="0dte_pcs:SPY:1"),
    ]
    assert evaluate_spread(legs_0dte, marks, cfg) is None


def test_0dte_force_close_at_end_of_day():
    """0DTE positions force-close after the configured ET time."""
    legs = [
        _p("SPY...P450", -1, 0.90, OptionRight.PUT, 450,
           expiry=TODAY, tag="0dte_pcs:SPY:1"),
        _p("SPY...P445", +1, 0.30, OptionRight.PUT, 445,
           expiry=TODAY, tag="0dte_pcs:SPY:1"),
    ]
    marks = {"SPY...P450": 0.50, "SPY...P445": 0.20}
    cfg = CreditSpreadExitConfig(zero_dte_force_close_et=dtime(15, 45))
    now = datetime.combine(TODAY, dtime(15, 50))
    d = evaluate_spread(legs, marks, cfg, now_et=now)
    assert d is not None and "0dte_force_close" in d.reason


def test_half_a_spread_does_not_trigger_exit():
    """If only one leg is detected (maybe the other hasn't filled yet),
    don't try to evaluate — too easy to misinterpret."""
    legs = [_p("SPY...P450", -1, 1.10, OptionRight.PUT, 450,
                tag="weekly_pcs:SPY:1")]
    marks = {"SPY...P450": 0.30}
    assert evaluate_spread(legs, marks, CreditSpreadExitConfig()) is None


# ---------- build_close_combo ----------


def test_build_close_combo_reverses_each_leg():
    """Closing a short put credit spread: BUY back the short, SELL back
    the long. Net is a debit (we pay to close)."""
    legs = [
        _p("SPY...P450", -1, 1.10, OptionRight.PUT, 450, tag="weekly_pcs:SPY:1"),
        _p("SPY...P440", +1, 0.30, OptionRight.PUT, 440, tag="weekly_pcs:SPY:1"),
    ]
    marks = {"SPY...P450": 0.40, "SPY...P440": 0.15}
    d = evaluate_spread(legs, marks,
                        CreditSpreadExitConfig(profit_target_pct=0.50))
    assert d is not None
    chain = {
        "SPY...P450": _c("SPY...P450", 450, OptionRight.PUT, bid=0.38, ask=0.42),
        "SPY...P440": _c("SPY...P440", 440, OptionRight.PUT, bid=0.13, ask=0.17),
    }
    combo = build_close_combo(d, chain)
    assert combo is not None
    assert combo.qty == 1
    sides = {leg.contract.symbol: leg.side for leg in combo.legs}
    assert sides["SPY...P450"] == Side.BUY        # close short → BUY
    assert sides["SPY...P440"] == Side.SELL       # close long  → SELL
    assert combo.is_debit      # paying to close
