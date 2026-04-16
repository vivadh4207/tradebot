"""Black-Scholes pricer, full Greeks, and Brent-method IV solver.

Lifted directly from the playbook Section 1. Back-solve IV from mid-quote
on every recalc; do not trust vendor IV.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, option_type: str = "call") -> float:
    """Black-Scholes-Merton price for a European option."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              q: float = 0.0, option_type: str = "call") -> dict:
    """Delta, Gamma, Vega, Theta, Rho, Vanna, Charm."""
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
    """Brent's method IV solver. Returns NaN if no solution exists."""
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price < intrinsic - 1e-6:
        return float("nan")
    obj = lambda sigma: bs_price(S, K, T, r, sigma, q, option_type) - market_price
    try:
        return float(brentq(obj, 1e-6, 5.0, maxiter=100, xtol=1e-6))
    except (ValueError, RuntimeError):
        return float("nan")
