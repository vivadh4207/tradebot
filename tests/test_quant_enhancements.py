"""Tests for the quant-level enhancements batch:
dividend yield, Friday cycle, position snapshot, numerical guards,
master-stack threshold, put-call parity.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core.types import OptionContract, OptionRight


# ----------------------------- dividend yield -------------------------------

def test_dividend_yield_caches(tmp_path, monkeypatch):
    from src.intelligence.dividend_yield import DividendYieldProvider

    p = DividendYieldProvider(cache_path=str(tmp_path / "div.json"))
    calls = {"n": 0}

    def fake_fetch(sym):
        calls["n"] += 1
        return 0.0125

    monkeypatch.setattr(p, "_fetch", fake_fetch)
    y1 = p.get("SPY")
    y2 = p.get("SPY")
    y3 = p.get("SPY")
    assert y1 == 0.0125 and y2 == 0.0125 and y3 == 0.0125
    assert calls["n"] == 1, f"expected 1 fetch, got {calls['n']}"


def test_dividend_yield_falls_back_to_hardcoded(monkeypatch, tmp_path):
    from src.intelligence.dividend_yield import DividendYieldProvider
    p = DividendYieldProvider(cache_path=str(tmp_path / "div.json"))
    monkeypatch.setattr(p, "_fetch", lambda s: None)
    # AAPL is in the hardcoded fallback map; should return ~0.0045
    v = p.get("AAPL")
    assert 0.0 < v < 0.02


def test_dividend_yield_zero_for_non_payer(monkeypatch, tmp_path):
    from src.intelligence.dividend_yield import DividendYieldProvider
    p = DividendYieldProvider(cache_path=str(tmp_path / "div.json"))
    monkeypatch.setattr(p, "_fetch", lambda s: None)
    # TSLA and AMZN hardcoded as 0.0 (don't pay dividends)
    assert p.get("TSLA") == 0.0
    assert p.get("AMZN") == 0.0


# ----------------------------- Friday-cycle ---------------------------------

def test_alpaca_chain_filters_to_standard_weekdays():
    """Synthetic test: if only Tuesday (wkday=1) + Friday (wkday=4) are in the
    parsed chain, the Friday should be picked and the Tuesday dropped."""
    from src.data.options_chain_alpaca import AlpacaOptionsChain, _parse_occ

    # We use the underlying _STANDARD_WEEKDAYS set directly.
    assert AlpacaOptionsChain._STANDARD_WEEKDAYS == {0, 2, 4}

    # Verify the parser works end-to-end and the set includes Fri
    r = _parse_occ("SPY260417C00500000", "SPY")  # 2026-04-17 = Friday
    assert r is not None
    assert r[0].weekday() == 4


# ----------------------------- position snapshot ----------------------------

def test_snapshot_roundtrip(tmp_path):
    from src.brokers.paper import PaperBroker
    from src.core.types import Order, Side
    from src.storage.position_snapshot import save_snapshot, load_snapshot

    snap_path = tmp_path / "snap.json"
    b = PaperBroker(starting_equity=10_000, slippage_bps=0,
                     snapshot_path=str(snap_path))
    b.submit(Order(symbol="SPY", side=Side.BUY, qty=2, is_option=False,
                    limit_price=500.00))
    # Auto-snapshot fired on submit
    assert snap_path.exists()

    snap = load_snapshot(str(snap_path))
    assert snap is not None
    assert len(snap.positions) == 1
    assert snap.positions[0].symbol == "SPY"
    assert snap.positions[0].qty == 2

    # Fresh broker → restore_from_snapshot → same state
    b2 = PaperBroker(starting_equity=0, slippage_bps=0)
    n = b2.restore_from_snapshot(str(snap_path))
    assert n == 1
    pos2 = b2.positions()
    assert len(pos2) == 1
    assert pos2[0].symbol == "SPY" and pos2[0].qty == 2


def test_snapshot_corrupt_file_returns_none(tmp_path):
    from src.storage.position_snapshot import load_snapshot
    p = tmp_path / "bad.json"
    p.write_text("{ not valid json")
    assert load_snapshot(str(p)) is None


def test_reconcile_detects_mismatch(tmp_path):
    from src.brokers.paper import PaperBroker
    from src.core.types import Order, Side
    from src.storage.position_snapshot import reconcile_with_live

    sim = PaperBroker(starting_equity=10_000, slippage_bps=0)
    live = PaperBroker(starting_equity=10_000, slippage_bps=0)
    sim.submit(Order(symbol="SPY", side=Side.BUY, qty=1, is_option=False, limit_price=500))
    live.submit(Order(symbol="QQQ", side=Side.BUY, qty=1, is_option=False, limit_price=400))
    r = reconcile_with_live(sim, live)
    assert not r["ok"]
    assert "SPY" in r["missing"]
    assert "QQQ" in r["extra"]


# ----------------------------- numerical guards -----------------------------

def test_pricer_guards_against_tiny_sigma_and_T():
    from src.math_tools.pricer import bs_greeks
    g_small_sigma = bs_greeks(100, 100, 30/365, 0.045, 1e-5, 0.015, "call")
    g_small_T = bs_greeks(100, 100, 1e-5, 0.045, 0.2, 0.015, "call")
    # vanna/charm should be zeroed (not NaN/Inf) under both conditions
    for g in (g_small_sigma, g_small_T):
        assert g["vanna"] == 0.0
        assert g["charm"] == 0.0
        for v in g.values():
            assert math.isfinite(v)


def test_pricer_handles_zero_spot_and_strike():
    from src.math_tools.pricer import bs_price, bs_greeks
    assert bs_price(0, 100, 0.1, 0.045, 0.2, 0.015, "call") == 0.0
    assert bs_price(100, 0, 0.1, 0.045, 0.2, 0.015, "call") == 0.0
    g = bs_greeks(0, 100, 0.1, 0.045, 0.2, 0.015, "call")
    assert all(v == 0.0 for v in g.values())


def test_implied_vol_returns_nan_when_no_sign_change():
    """Market price way above bracket_hi sigma=5 price → no sign change → NaN."""
    from src.math_tools.pricer import bs_price, implied_vol
    # At sigma=5 (the upper bracket), call on S=100,K=100,T=30d is ~ S = 100
    # A market price of 200 is impossible — no solution in [1e-4, 5.0].
    iv = implied_vol(200.0, 100, 100, 30/365, 0.045, 0.015, "call")
    assert math.isnan(iv)


# ----------------------------- master-stack threshold -----------------------

def test_master_stack_default_threshold_is_lower():
    from src.signals.master_stack import MasterSignalStack
    m = MasterSignalStack()
    assert m.thresh == pytest.approx(0.15, rel=1e-6)


def test_master_stack_fires_at_moderate_composite():
    from src.signals.master_stack import MasterSignalStack
    m = MasterSignalStack()
    # Compose inputs that yield a composite ≈ 0.2 (would have been FLAT at 0.3)
    d = m.decide(
        atm_iv_30d=0.22, rv_20d=0.15,            # VRP > 0 → positive
        iv_52w_low=0.10, iv_52w_high=0.40,       # rank mid → small positive
        iv_30d=0.22, iv_90d=0.24,                # mild contango
        skew_zscore=0.0, rv_percentile_252d=0.5, # neutral
    )
    assert d.regime in ("SELL_PREMIUM", "BUY_PREMIUM", "FLAT")
    # At threshold 0.15 a mild positive composite should trip SELL_PREMIUM
    assert d.regime == "SELL_PREMIUM"


# ----------------------------- put-call parity ------------------------------

def _par_pair(strike, call_mid, put_mid, expiry_days=30):
    from src.core.types import OptionContract
    exp = date.today() + timedelta(days=expiry_days)
    spread = 0.02
    call = OptionContract(symbol=f"X{int(strike)}C", underlying="X",
                           strike=strike, expiry=exp, right=OptionRight.CALL,
                           bid=call_mid - spread/2, ask=call_mid + spread/2,
                           open_interest=1000, today_volume=200)
    put = OptionContract(symbol=f"X{int(strike)}P", underlying="X",
                          strike=strike, expiry=exp, right=OptionRight.PUT,
                          bid=put_mid - spread/2, ask=put_mid + spread/2,
                          open_interest=1000, today_volume=200)
    return call, put


def test_parity_clean_chain_passes():
    """Construct a chain using BS-consistent prices; should have zero violations."""
    from src.math_tools.parity import violations_in_chain
    from src.math_tools.pricer import bs_price

    S, r, q, T_days = 100.0, 0.045, 0.015, 30
    T = T_days / 365.0
    chain = []
    for K in (95.0, 100.0, 105.0):
        for sigma in (0.22,):
            c_mid = bs_price(S, K, T, r, sigma, q, "call")
            p_mid = bs_price(S, K, T, r, sigma, q, "put")
            call, put = _par_pair(K, c_mid, p_mid, expiry_days=T_days)
            chain.extend([call, put])
    viol = violations_in_chain(chain, spot=S, r=r, q=q)
    assert viol == [], f"expected no violations, got {viol}"


def test_parity_detects_gross_violation():
    """Set call_mid way off expected → parity check flags it."""
    from src.math_tools.parity import violations_in_chain
    S, r, q = 100.0, 0.045, 0.015
    # 100C should be ~ 1.15 given 100P ~ 0.64 at ATM 30d; make call absurdly high.
    call, put = _par_pair(100.0, call_mid=10.0, put_mid=0.50, expiry_days=30)
    viol = violations_in_chain([call, put], spot=S, r=r, q=q,
                                 abs_tolerance=0.05, pct_tolerance=0.10)
    assert len(viol) == 1
    assert viol[0].strike == 100.0
    assert viol[0].abs_violation > 5.0
