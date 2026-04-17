"""Tests for the regime-aware multiplier on the position sizer."""
from datetime import date

import pytest

from src.core.types import OptionContract, OptionRight
from src.risk.position_sizer import PositionSizer, SizingInputs


def _inp(ask: float = 1.00, equity: float = 100_000.0, is_0dte: bool = True):
    c = OptionContract(
        symbol="SPY", underlying="SPY", strike=500.0,
        expiry=date.today(), right=OptionRight.CALL,
        bid=ask * 0.95, ask=ask, open_interest=5000, today_volume=1000,
    )
    return SizingInputs(
        equity=equity, contract=c,
        win_rate_est=0.58, avg_win=0.025, avg_loss=0.020,
        vix_today=18.0, vix_52w_low=10.0, vix_52w_high=40.0,
        vrp_zscore=0.5, is_0dte=is_0dte, is_long=True,
    )


def test_no_regime_behaves_like_before():
    s = PositionSizer()
    s_r = PositionSizer(regime_multipliers={"trend_lowvol": 1.5})
    n_default = s.contracts(_inp())
    n_no_regime = s_r.contracts(_inp())   # regime=None → multiplier skipped
    assert n_default == n_no_regime


def test_growing_regime_scales_up():
    base = PositionSizer()
    grow = PositionSizer(regime_multipliers={"trend_lowvol": 1.5})
    n_base = base.contracts(_inp())
    n_grow = grow.contracts(_inp(), regime="trend_lowvol")
    assert n_grow >= n_base


def test_shrinking_regime_scales_down():
    base = PositionSizer()
    shrink = PositionSizer(regime_multipliers={"closing": 0.30})
    n_base = base.contracts(_inp())
    n_shrink = shrink.contracts(_inp(), regime="closing")
    if n_base > 0:
        assert n_shrink <= n_base


def test_forbidding_regime_returns_zero():
    forb = PositionSizer(regime_multipliers={"closing": 0.0})
    assert forb.contracts(_inp(), regime="closing") == 0


def test_multipliers_clipped_to_safe_range():
    s = PositionSizer(regime_multipliers={"r": 99.0})
    # clipped to 2.0; passing through contracts() preserves that ceiling
    assert s.regime_multipliers["r"] == 2.0
    s2 = PositionSizer(regime_multipliers={"r": -5.0})
    assert s2.regime_multipliers["r"] == 0.0


def test_unknown_regime_defaults_to_one():
    s = PositionSizer(regime_multipliers={"trend_lowvol": 1.5})
    n_default = PositionSizer().contracts(_inp())
    n_unknown = s.contracts(_inp(), regime="totally_unknown_regime")
    assert n_default == n_unknown
