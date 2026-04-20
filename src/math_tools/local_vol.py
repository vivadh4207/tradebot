"""Local volatility (Dupire) on top of a fitted SVI surface.

The Dupire formula gives local vol σ(K, T) from call prices c(K, T):

    σ²(K, T) = (∂c/∂T + (r - q)·K·∂c/∂K + q·c) / (0.5·K²·∂²c/∂K²)

We derive partial derivatives numerically from the SVI total-variance
surface (which is arbitrage-free by construction), reprice with BS at
the local vol, and use that for strikes 3-5% OTM where BS(ATM IV)
misprices.

HONEST LIMITATIONS:
  - This is a 2D stencil — noisy on sparse chains.
  - For a production MM desk you'd use a Markov-functional or SLV
    calibration; this is a retail-grade approximation.
  - Requires a FITTED SVI slice per expiry. Wire `math_tools/svi.py`
    first; this module assumes those params are available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .svi import svi_total_variance


@dataclass
class LocalVolPoint:
    strike: float
    T: float
    local_vol: float
    implied_vol_bs: float


def total_variance_from_svi(log_moneyness: float, T: float,
                             svi_params: Sequence[float]) -> float:
    """w(k) from the SVI slice at total-variance level."""
    a, b, rho, m, sigma = svi_params
    return svi_total_variance(log_moneyness, a, b, rho, m, sigma)


def dupire_local_vol(
    spot: float, strike: float, T: float, r: float, q: float,
    svi_params_T: Sequence[float],
    svi_params_T_minus: Optional[Sequence[float]] = None,
    svi_params_T_plus: Optional[Sequence[float]] = None,
    dT: float = 1.0 / 252.0,
    dK_rel: float = 0.005,
) -> LocalVolPoint:
    """Dupire local vol at (strike, T) derived from neighboring SVI slices.

    If neighboring slices are not supplied, we approximate ∂w/∂T as
    total_variance(T) / T (i.e. constant-vol extrapolation). This is a
    pragmatic fallback that degrades gracefully at the short end.
    """
    forward = spot * np.exp((r - q) * T)
    k = float(np.log(strike / forward))
    w = total_variance_from_svi(k, T, svi_params_T)

    # ∂w/∂K at fixed T, central difference in log-moneyness
    dk = dK_rel
    k_plus = k + dk
    k_minus = k - dk
    w_kplus = total_variance_from_svi(k_plus, T, svi_params_T)
    w_kminus = total_variance_from_svi(k_minus, T, svi_params_T)
    dw_dk = (w_kplus - w_kminus) / (2 * dk)
    d2w_dk2 = (w_kplus - 2 * w + w_kminus) / (dk * dk)

    # ∂w/∂T at fixed k, forward difference using a supplied neighbor slice
    if svi_params_T_plus is not None:
        w_tplus = total_variance_from_svi(k, T + dT, svi_params_T_plus)
        dw_dT = (w_tplus - w) / dT
    else:
        # constant-vol extrapolation: w ∝ T at the short end
        dw_dT = w / max(T, 1e-6)

    # Dupire in parameterized form (Gatheral 2011 p. 9):
    # σ²_LV = (∂w/∂T) / (1 - (k/w)·(∂w/∂k) + 0.25·(-0.25 - 1/w + k²/w²)·(∂w/∂k)² + 0.5·∂²w/∂k²)
    denom = (
        1.0
        - (k / (w + 1e-12)) * dw_dk
        + 0.25 * (-0.25 - 1.0 / (w + 1e-12) + k * k / (w * w + 1e-24)) * dw_dk * dw_dk
        + 0.5 * d2w_dk2
    )
    # Guard: denom can go negative on sparsely-calibrated data → fall back
    # to implied-vol at this point.
    if denom <= 1e-6 or dw_dT <= 0:
        iv = float(np.sqrt(w / max(T, 1e-6))) if w > 0 else 0.0
        return LocalVolPoint(strike=strike, T=T, local_vol=iv, implied_vol_bs=iv)
    lv_sq = dw_dT / denom
    if lv_sq <= 0:
        iv = float(np.sqrt(w / max(T, 1e-6))) if w > 0 else 0.0
        return LocalVolPoint(strike=strike, T=T, local_vol=iv, implied_vol_bs=iv)
    iv = float(np.sqrt(w / max(T, 1e-6)))
    return LocalVolPoint(strike=strike, T=T,
                          local_vol=float(np.sqrt(lv_sq)),
                          implied_vol_bs=iv)
