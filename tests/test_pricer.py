import math
import pytest
from src.math_tools.pricer import bs_price, bs_greeks, implied_vol


def test_bs_price_atm_positive():
    p = bs_price(100, 100, 30/365, 0.045, 0.20, 0.015, "call")
    assert p > 0


def test_call_put_parity():
    # C - P = S*exp(-qT) - K*exp(-rT)
    S, K, T, r, sigma, q = 100.0, 95.0, 0.25, 0.045, 0.25, 0.01
    c = bs_price(S, K, T, r, sigma, q, "call")
    p = bs_price(S, K, T, r, sigma, q, "put")
    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-6


def test_greeks_shape():
    g = bs_greeks(100, 100, 30/365, 0.045, 0.20, 0.015, "call")
    assert 0 < g["delta"] < 1
    assert g["gamma"] > 0
    assert g["vega"] > 0
    assert g["theta"] < 0           # long call decays


def test_implied_vol_roundtrip():
    S, K, T, r, sigma, q = 100.0, 100.0, 30/365, 0.045, 0.23, 0.015
    p = bs_price(S, K, T, r, sigma, q, "call")
    iv = implied_vol(p, S, K, T, r, q, "call")
    assert abs(iv - sigma) < 1e-4


def test_implied_vol_below_intrinsic_returns_nan():
    iv = implied_vol(0.0, 100, 90, 30/365, 0.045, 0.015, "call")  # intrinsic=10
    assert math.isnan(iv)
