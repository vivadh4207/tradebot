"""Black-Scholes pricer, full Greeks, and Brent-method IV solver.

Lifted directly from the playbook Section 1. Back-solve IV from mid-quote
on every recalc; do not trust vendor IV.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


_MIN_SIGMA = 1e-4      # below this, second-order Greeks are meaningless
_MIN_T = 1e-4          # ~1 hour — below, time decay is a discontinuity
_IV_BRACKET_LO = 1e-4
_IV_BRACKET_HI = 5.0


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, option_type: str = "call") -> float:
    """Black-Scholes-Merton price for a European option.

    Returns intrinsic value when T<=0 or sigma<=0. Also handles S<=0 and
    K<=0 edge cases (log(S/K) would otherwise be -inf / nan).
    """
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option_type == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              q: float = 0.0, option_type: str = "call") -> dict:
    """Delta, Gamma, Vega, Theta, Rho, Vanna, Charm.

    Guards: second-order Greeks (vanna, charm) depend on 1/sigma and
    1/(T * sqrt(T)), which blow up at sigma→0 or T→0. We return 0 for
    vanna/charm in those regimes rather than letting them pollute
    downstream risk aggregation with NaN/Inf.
    """
    if S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                "theta": 0.0, "rho": 0.0, "vanna": 0.0, "charm": 0.0}
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                "theta": 0.0, "rho": 0.0, "vanna": 0.0, "charm": 0.0}
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1)) / 365.0
        rho = K * T * disc_r * norm.cdf(d2) / 100.0
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1)) / 365.0
        rho = -K * T * disc_r * norm.cdf(-d2) / 100.0

    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT / 100.0  # per 1% vol move

    # Second-order Greeks: guard against sigma→0 and T→0 singularities.
    # If we're below the numerical floor we return 0 — these Greeks are
    # meaningless near the asymptote and should not drive risk decisions.
    if sigma < _MIN_SIGMA or T < _MIN_T:
        vanna = 0.0
        charm = 0.0
    else:
        vanna = -disc_q * pdf_d1 * d2 / sigma
        charm_core = (q * disc_q * norm.cdf(d1) if option_type == "call"
                      else -q * disc_q * norm.cdf(-d1))
        charm = charm_core - disc_q * pdf_d1 * (2 * (r - q) * T - d2 * sigma * sqrtT) / (2 * T * sigma * sqrtT)
        charm /= 365.0

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta), "rho": float(rho),
            "vanna": float(vanna), "charm": float(charm)}


def implied_vol(market_price: float, S: float, K: float, T: float,
                r: float, q: float = 0.0, option_type: str = "call") -> float:
    """Brent's method IV solver. Returns NaN if no solution exists.

    Pre-check: if the objective function doesn't change sign across the
    [lo, hi] bracket, brentq would fail internally. We detect this
    explicitly and return NaN early (faster + avoids scipy warnings).
    """
    if S <= 0 or K <= 0 or T <= 0 or market_price < 0:
        return float("nan")
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price < intrinsic - 1e-6:
        return float("nan")
    obj = lambda sigma: bs_price(S, K, T, r, sigma, q, option_type) - market_price
    try:
        f_lo = obj(_IV_BRACKET_LO)
        f_hi = obj(_IV_BRACKET_HI)
    except (ValueError, FloatingPointError):
        return float("nan")
    # No sign change → no root in the bracket. Returning NaN is more honest
    # than letting brentq silently fail.
    if f_lo * f_hi > 0:
        return float("nan")
    try:
        return float(brentq(obj, _IV_BRACKET_LO, _IV_BRACKET_HI,
                             maxiter=100, xtol=1e-6))
    except (ValueError, RuntimeError):
        return float("nan")
