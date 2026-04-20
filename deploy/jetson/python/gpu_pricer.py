"""Vectorized Black-Scholes pricer + Greeks on GPU via CuPy.

Only used when we need to price thousands of contracts at once (e.g. fitting
an SVI surface across a full chain, or re-pricing the whole book after a
spot move). For single-contract pricing stick to the CPU path in
`src/math_tools/pricer.py` — the kernel launch overhead is not worth it.

Import is lazy so the rest of the package works without CuPy installed.
"""
from __future__ import annotations

from typing import Dict, Optional

try:
    import cupy as cp                                       # type: ignore
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    cp = None  # type: ignore


def _norm_cdf(x):
    return 0.5 * (1.0 + cp.erf(x / cp.sqrt(2.0)))


def bs_price_batch(S, K, T, r, sigma, q=0.0, option_type: str = "call"):
    """Batch Black-Scholes price. All inputs can be scalars or CuPy arrays.

    Shapes broadcast numpy-style. Returns a CuPy array.
    """
    if not _HAS_CUPY:
        raise ImportError("cupy is not installed; run on CPU pricer instead.")
    S = cp.asarray(S, dtype=cp.float32)
    K = cp.asarray(K, dtype=cp.float32)
    T = cp.asarray(T, dtype=cp.float32)
    sigma = cp.asarray(sigma, dtype=cp.float32)
    r = cp.asarray(r, dtype=cp.float32)
    q = cp.asarray(q, dtype=cp.float32)

    # intrinsic where T<=0 or sigma<=0
    intrinsic = cp.where(option_type == "call", cp.maximum(S - K, 0),
                          cp.maximum(K - S, 0))
    valid = (T > 0) & (sigma > 0)
    d1 = (cp.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * cp.sqrt(T))
    d2 = d1 - sigma * cp.sqrt(T)
    disc_q = cp.exp(-q * T)
    disc_r = cp.exp(-r * T)
    if option_type == "call":
        price = S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    else:
        price = K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)
    return cp.where(valid, price, intrinsic)


def gamma_batch(S, K, T, r, sigma, q=0.0) -> "cp.ndarray":
    """Batch gamma. Same-shape inputs as bs_price_batch."""
    if not _HAS_CUPY:
        raise ImportError("cupy is not installed.")
    S = cp.asarray(S, dtype=cp.float32)
    K = cp.asarray(K, dtype=cp.float32)
    T = cp.asarray(T, dtype=cp.float32)
    sigma = cp.asarray(sigma, dtype=cp.float32)
    r = cp.asarray(r, dtype=cp.float32)
    q = cp.asarray(q, dtype=cp.float32)
    d1 = (cp.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * cp.sqrt(T))
    pdf = cp.exp(-0.5 * d1 * d1) / cp.sqrt(2 * cp.pi)
    return cp.exp(-q * T) * pdf / (S * sigma * cp.sqrt(T))


def available() -> bool:
    return _HAS_CUPY
