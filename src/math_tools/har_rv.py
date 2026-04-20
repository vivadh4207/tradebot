"""HAR-RV realized-volatility forecaster (Corsi 2009).

Industry standard for vol forecasting at vol desks. Used by the gamma-scalp
edge signal and the master signal stack.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np

try:
    import pandas as pd
    import statsmodels.api as sm
    _HAS_STATS = True
except ImportError:
    _HAS_STATS = False


def realized_vol(returns: Sequence[float], annualize: bool = True) -> float:
    """Standard sample-stdev realized vol."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sigma = float(np.std(r, ddof=1))
    return sigma * np.sqrt(252.0) if annualize else sigma


def har_rv_forecast(rv_daily_series: Sequence[float]) -> float:
    """HAR-RV(1): RV_{t+1} = c + b_d·RV_t + b_w·RV_{5d_avg} + b_m·RV_{22d_avg}.

    Fits OLS on the series and predicts one step ahead. Returns ANNUALIZED.
    Falls back to mean RV if statsmodels isn't available.
    """
    rv = np.asarray(rv_daily_series, dtype=float)
    if rv.size < 30:
        return float(np.mean(rv[-5:])) * np.sqrt(252.0) if rv.size else 0.0

    if not _HAS_STATS:
        # fallback: simple blend of daily/weekly/monthly means
        d = rv[-1]
        w = np.mean(rv[-5:])
        m = np.mean(rv[-22:])
        return float(0.3 * d + 0.4 * w + 0.3 * m) * np.sqrt(252.0)

    df = pd.DataFrame({"rv": rv})
    df["rv_d"] = df["rv"].shift(1)
    df["rv_w"] = df["rv"].rolling(5).mean().shift(1)
    df["rv_m"] = df["rv"].rolling(22).mean().shift(1)
    df = df.dropna()
    if len(df) < 10:
        return float(df["rv"].mean()) * np.sqrt(252.0)
    X = sm.add_constant(df[["rv_d", "rv_w", "rv_m"]])
    model = sm.OLS(df["rv"], X).fit()
    last = pd.DataFrame({
        "const": [1.0],
        "rv_d": [rv[-1]],
        "rv_w": [np.mean(rv[-5:])],
        "rv_m": [np.mean(rv[-22:])],
    })
    return float(model.predict(last).iloc[0]) * np.sqrt(252.0)
