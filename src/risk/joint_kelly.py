"""Joint Kelly sizing with correlation — book-level rather than per-trade.

Motivation: three bullish-SPY trades across SPY / QQQ / IWM are NOT three
independent bets. Their returns share >90% of variance. Per-trade Kelly
lets us lever up 3× on one underlying factor.

Formula (continuous Kelly, multi-asset):
    f* = Σ⁻¹ · μ
where Σ is the return covariance matrix and μ is the vector of expected
per-period log-returns per unit exposure.

Apply fractional Kelly (0.25×) and a hard per-trade cap just like the
single-asset version.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class JointKellyResult:
    fractions: Dict[str, float]        # symbol → fraction of equity
    diagonal_only: Dict[str, float]    # per-trade baseline for comparison
    correlation_penalty: Dict[str, float]
    notes: str = ""


def _pseudo_inverse(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Tikhonov-regularized inverse; adds a tiny ridge for numerical stability."""
    n = cov.shape[0]
    return np.linalg.inv(cov + ridge * np.eye(n))


def joint_kelly(
    symbols: Sequence[str],
    expected_returns: Sequence[float],
    cov_matrix: np.ndarray,
    *,
    fractional: float = 0.25,
    hard_cap: float = 0.05,
    floor: float = 0.0,
) -> JointKellyResult:
    """Solve f* = Σ⁻¹ · μ with fractional Kelly and per-position caps.

    `expected_returns` should be already scaled by the probability-of-win
    and average-win/avg-loss of your prior estimate, i.e. the same
    win_rate × avg_win − (1-win_rate) × avg_loss quantity that single-asset
    Kelly uses. Units: fraction of equity.

    `cov_matrix` is the *per-trade* return covariance (annualized is fine,
    as long as μ matches the same time unit).
    """
    n = len(symbols)
    if n == 0:
        return JointKellyResult({}, {}, {}, notes="empty")
    mu = np.asarray(expected_returns, dtype=np.float64)
    if cov_matrix.shape != (n, n):
        raise ValueError(f"cov_matrix must be ({n},{n}), got {cov_matrix.shape}")

    # Symmetrize (guard against FP drift)
    cov = 0.5 * (cov_matrix + cov_matrix.T)
    cov_inv = _pseudo_inverse(cov)
    raw = cov_inv @ mu

    # Apply fractional Kelly + clip
    scaled = raw * fractional
    clipped = np.clip(scaled, floor, hard_cap)

    # Diagonal-only (per-trade) for comparison diagnostics
    var = np.diag(cov).clip(min=1e-12)
    diag = (mu / var) * fractional
    diag = np.clip(diag, floor, hard_cap)

    return JointKellyResult(
        fractions={s: float(clipped[i]) for i, s in enumerate(symbols)},
        diagonal_only={s: float(diag[i]) for i, s in enumerate(symbols)},
        correlation_penalty={
            s: float(diag[i] - clipped[i]) for i, s in enumerate(symbols)
        },
        notes=f"n={n} ridge=1e-6 fractional={fractional} cap={hard_cap}",
    )


def rolling_covariance(
    returns_by_symbol: Dict[str, Sequence[float]],
    *,
    min_samples: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> "tuple[list, np.ndarray]":
    """Compute sample covariance of aligned return series per symbol.

    Returns (symbols_in_order, cov). Drops symbols with < min_samples.
    """
    cleaned = {}
    for s, rets in returns_by_symbol.items():
        arr = np.asarray(list(rets), dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size >= min_samples:
            cleaned[s] = arr
    if not cleaned:
        return [], np.zeros((0, 0))
    # Align on the shortest length so we have a common sample
    length = min(len(v) for v in cleaned.values())
    mat = np.stack([v[-length:] for v in cleaned.values()], axis=0)   # (n, T)
    cov = np.cov(mat, ddof=1)
    if annualize:
        cov = cov * periods_per_year
    # cov can be 0-dim if only one symbol
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return list(cleaned.keys()), cov
