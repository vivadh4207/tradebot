"""Tests for the hedge-fund roadmap implementation batch.

One test (or small cluster) per module. Heavy math modules (joint Kelly,
Monte Carlo VaR, HMM, local vol) get sanity-check tests — we verify
shape, invariants, and expected-sign behavior, not exact numerical
identities.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from src.core.types import (
    Order, Side, Position, OptionContract, OptionRight, Bar,
)


# ------------------------------- slippage ----------------------------------

def test_slippage_grows_with_size_and_vix():
    from src.brokers.slippage_model import StochasticCostModel, MarketContext
    m = StochasticCostModel(seed=0)
    ctx_small = MarketContext(bid=99.99, ask=100.01, bid_size=10_000,
                                ask_size=10_000, vix=15.0)
    ctx_big = MarketContext(bid=99.99, ask=100.01, bid_size=100,
                              ask_size=100, vix=15.0)
    ctx_vix_hi = MarketContext(bid=99.99, ask=100.01, bid_size=10_000,
                                 ask_size=10_000, vix=45.0)
    o = Order(symbol="X", side=Side.BUY, qty=10, limit_price=100.0)
    small = m.fill(o, ctx_small).slippage_bps
    big = m.fill(o, ctx_big).slippage_bps
    vix = m.fill(o, ctx_vix_hi).slippage_bps
    assert big > small, "larger qty / smaller displayed_size must cost more"
    assert vix > small, "high-VIX regime must widen slippage"
    # Cost model never goes negative
    assert small >= 0 and big >= 0 and vix >= 0


def test_paper_broker_with_slippage_model(tmp_path):
    from src.brokers.paper import PaperBroker
    from src.brokers.slippage_model import StochasticCostModel, MarketContext
    b = PaperBroker(starting_equity=100_000, slippage_model=StochasticCostModel(seed=1))
    b.update_market_context(
        "SPY",
        MarketContext(bid=499.95, ask=500.05, bid_size=500, ask_size=500, vix=18.0),
    )
    fill = b.submit(Order(symbol="SPY", side=Side.BUY, qty=5,
                           is_option=False, limit_price=500.00))
    assert fill is not None
    # BUY at ~mid + slippage; executed > 500.00
    assert fill.price > 499.9, f"unexpected fill price {fill.price}"


# ------------------------------- joint Kelly -------------------------------

def test_joint_kelly_penalizes_correlation():
    from src.risk.joint_kelly import joint_kelly
    # Two highly-correlated returns: ρ = 0.95
    cov = np.array([[0.04, 0.038], [0.038, 0.04]])
    mu = np.array([0.02, 0.02])
    res = joint_kelly(["A", "B"], mu, cov, fractional=1.0, hard_cap=1.0)
    diag_sum = sum(res.diagonal_only.values())
    joint_sum = sum(res.fractions.values())
    # Joint allocation should be much smaller than the diagonal sum when
    # the two assets are near-perfect substitutes.
    assert joint_sum < diag_sum * 0.7
    assert all(v >= 0 for v in res.fractions.values())
    assert all(res.correlation_penalty[s] >= 0 for s in ("A", "B"))


def test_rolling_covariance_basic():
    from src.risk.joint_kelly import rolling_covariance
    rng = np.random.default_rng(0)
    rets = {"SPY": rng.normal(0, 0.01, 100).tolist(),
            "QQQ": rng.normal(0, 0.012, 100).tolist()}
    syms, cov = rolling_covariance(rets)
    assert syms == ["SPY", "QQQ"]
    assert cov.shape == (2, 2)
    # Symmetric + positive diagonal
    assert np.allclose(cov, cov.T)
    assert cov[0, 0] > 0 and cov[1, 1] > 0


# ------------------------------- vol scaling -------------------------------

def test_vol_scale_down_for_high_vol():
    from src.risk.vol_scaling import vol_scale
    t = datetime(2026, 4, 16, 10, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)
    # 60 bars of price with HIGH vol
    prices_hi = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 60)))
    bars_hi = [Bar(symbol="NVDA", ts=t + timedelta(minutes=i),
                    open=p, high=p*1.001, low=p*0.999, close=p, volume=1000)
                for i, p in enumerate(prices_hi)]
    # Low vol
    prices_lo = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.0005, 60)))
    bars_lo = [Bar(symbol="KO", ts=t + timedelta(minutes=i),
                    open=p, high=p*1.0005, low=p*0.9995, close=p, volume=1000)
                for i, p in enumerate(prices_lo)]
    hi = vol_scale(bars_hi, target_annual_vol=0.20)
    lo = vol_scale(bars_lo, target_annual_vol=0.20)
    assert hi.multiplier < 1.0, f"expected scale-down for high vol, got {hi.multiplier}"
    assert lo.multiplier > 1.0, f"expected scale-up for low vol, got {lo.multiplier}"


# ------------------------------- drawdown guard ----------------------------

def test_drawdown_guard_tiers():
    from src.risk.drawdown_guard import DrawdownGuard
    g = DrawdownGuard()
    # No drawdown → 1.0
    s = g.evaluate(current_equity=100_000, peak_equity=100_000)
    assert s.size_multiplier == 1.0 and not s.halted
    # 6% DD → 0.75
    s = g.evaluate(current_equity=94_000, peak_equity=100_000)
    assert s.size_multiplier == 0.75 and not s.halted
    # 10% DD → 0.50
    s = g.evaluate(current_equity=90_000, peak_equity=100_000)
    assert s.size_multiplier == 0.50 and not s.halted
    # 15% DD → halt
    s = g.evaluate(current_equity=85_000, peak_equity=100_000)
    assert s.halted and s.size_multiplier == 0.0


# ------------------------------- Monte Carlo VaR ---------------------------

def test_var_report_shape():
    from src.risk.monte_carlo_var import monte_carlo_var
    pos = Position(symbol="SPY", qty=100, avg_price=500.0, is_option=False,
                    multiplier=1)
    report = monte_carlo_var([pos], spots={"SPY": 500.0},
                              vols={"SPY": 0.18},
                              horizon_days=1.0, n_paths=2000, seed=7)
    assert report.n_paths == 2000
    # 95% VaR must be positive (loss)
    assert report.var_95 > 0
    # 99% VaR always >= 95% VaR
    assert report.var_99 >= report.var_95
    # CVaR >= VaR (tail average is deeper than the quantile)
    assert report.cvar_95 >= report.var_95
    assert report.cvar_99 >= report.var_99


# ------------------------------- HMM regime --------------------------------

def test_hmm_finds_high_vol_on_volatile_tail():
    from src.intelligence.regime_hmm import HMMRegimeClassifier
    rng = np.random.default_rng(0)
    # 150 quiet bars + 50 noisy bars
    quiet = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 150)))
    noisy = quiet[-1] * np.exp(np.cumsum(rng.normal(0, 0.01, 50)))
    closes = np.concatenate([quiet, noisy])
    cls = HMMRegimeClassifier()
    res = cls.classify(closes.tolist())
    assert res is not None
    # Most recent bars are in the noisy regime → should classify as high_vol
    assert res.current_label == "high_vol"


# ------------------------------- TWAP slicer -------------------------------

def test_twap_slicer_splits_and_submits():
    from src.brokers.paper import PaperBroker
    from src.brokers.slicer import TWAPSlicer
    b = PaperBroker(starting_equity=100_000, slippage_bps=0)
    parent = Order(symbol="SPY", side=Side.BUY, qty=10, is_option=False,
                    limit_price=500.0, tag="parent")
    slicer = TWAPSlicer(b, slices=5, interval_sec=0)   # 0s for test speed
    r = slicer.submit(parent, blocking=True)
    assert r.total_filled == 10
    # After 5 BUYs of 2 each, position is 10 @ ~500
    pos = b.positions()
    assert len(pos) == 1 and pos[0].qty == 10


# ------------------------------- feature drift -----------------------------

def test_feature_drift_detects_shift():
    from src.ml.feature_drift import check_drift
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (500, 3)).astype(np.float32)
    # Live: same distribution for col 0, shifted for col 1, same for col 2
    live_same = rng.normal(0, 1, 200).astype(np.float32)
    live_shift = rng.normal(2.0, 1, 200).astype(np.float32)
    live_same2 = rng.normal(0, 1, 200).astype(np.float32)
    live = np.stack([live_same, live_shift, live_same2], axis=1)
    r = check_drift(train, live, ["f0", "f1", "f2"],
                     alert_thresh=0.15, warn_thresh=0.08)
    # f1 should trip alert
    assert any(a.feature == "f1" and a.severity == "alert" for a in r.alerts)
    assert not any(a.feature == "f0" and a.severity == "alert" for a in r.alerts)


# ------------------------------- run registry ------------------------------

def test_run_registry_append_and_read(tmp_path):
    from src.backtest.run_registry import register_run, read_registry
    p = tmp_path / "runs.jsonl"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("a: 1\n")
    register_run(
        p, settings_path=cfg, seed=42, data_source="synthetic",
        window_days=30, total_bars=300, final_equity=10_000,
        metrics={"total_return_pct": 1.2, "sharpe": 0.9,
                  "max_drawdown_pct": -5.0, "n_trades": 42},
    )
    register_run(
        p, settings_path=cfg, seed=43, data_source="historical",
        window_days=60, total_bars=600, final_equity=11_000,
        metrics={"total_return_pct": 10.0, "sharpe": 1.5,
                  "max_drawdown_pct": -3.0, "n_trades": 88},
    )
    rows = read_registry(p)
    assert len(rows) == 2
    assert rows[0].seed == 42 and rows[1].seed == 43
    assert rows[0].config_sha256 == rows[1].config_sha256   # same config


# ------------------------------- pydantic schema ---------------------------

def test_config_schema_rejects_out_of_range():
    from src.core.config_schema import validate_settings
    import sys as _sys
    # Build a malformed settings dict
    bad = {
        "account": {"paper_starting_equity": 10_000,
                     "max_risk_per_trade_pct": 0.01,
                     "max_daily_loss_pct": 0.05,
                     "max_open_positions": 5},
        "sizing": {"kelly_fraction_cap": 2.5,   # INVALID: must be ≤ 1.0
                    "kelly_hard_cap_pct": 0.05,
                    "max_contracts_0dte": 5,
                    "max_contracts_multiday": 10},
    }
    with pytest.raises(Exception):
        validate_settings(bad)


# ------------------------------- PnL attribution ---------------------------

def test_pnl_attribution_residual_small_for_bs_consistent_move():
    from src.analytics.pnl_attribution import attribute_pnl
    # A 1-DTE ATM call: S 100→101, sigma unchanged, T shrinks by 1/365.
    from src.math_tools.pricer import bs_price
    K, T0, sigma, r, q = 100.0, 30/365, 0.22, 0.045, 0.015
    right = "call"
    px0 = bs_price(100.0, K, T0, r, sigma, q, right)
    pos = Position(
        symbol="SPY241218C100", qty=1, avg_price=px0,
        is_option=True, underlying="SPY", strike=K,
        expiry=date.today() + timedelta(days=30),
        right=OptionRight.CALL, multiplier=100,
    )
    report = attribute_pnl(
        pos, S_t0=100.0, S_t1=101.0,
        sigma_t0=sigma, sigma_t1=sigma,
        T_t0=T0, T_t1=T0 - 1/365,
    )
    assert abs(report.total_pnl) > 0.01
    # Residual should be a small fraction of total when move is small and
    # vol is unchanged — the Greek breakdown should explain most of it.
    assert report.residual_pct_of_total < 0.50, \
        f"residual {report.residual_pct_of_total} too large"
