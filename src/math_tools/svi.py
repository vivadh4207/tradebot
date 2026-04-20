"""Stochastic Volatility Inspired (SVI) surface fit (Gatheral 2004).

Industry-standard parameterization for arbitrage-free implied-vol surfaces.
Enables surface-arb, skew and term-structure signals.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def svi_total_variance(k, a, b, rho, m, sigma):
    """w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    km = k - m
    return a + b * (rho * km + np.sqrt(km * km + sigma * sigma))


def fit_svi_slice(strikes, forward: float, T: float, market_ivs):
    """Fit SVI to one expiry slice. Returns (a, b, rho, m, sigma)."""
    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)
    log_moneyness = np.log(strikes / forward)
    market_w = (market_ivs ** 2) * T

    def objective(params):
        a, b, rho, m, sigma = params
        if b < 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        model_w = svi_total_variance(log_moneyness, a, b, rho, m, sigma)
        weights = np.exp(-log_moneyness ** 2 * 2)   # ATM-weighted
        return float(np.sum(weights * (model_w - market_w) ** 2))

    x0 = [float(np.mean(market_w)) * 0.5, 0.1, -0.3, 0.0, 0.1]
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000})
    return tuple(result.x)
