"""Feature drift monitoring — KS-test live features vs. training distribution.

If a production feature's distribution drifts > some threshold away from
what the model was trained on, the model is seeing out-of-sample data and
its predictions may not be reliable.

We store the training distribution (per-feature percentiles at fit time)
into the checkpoint meta, then run the KS test on recent live features.
Alert on max drift > threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from scipy.stats import ks_2samp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class DriftAlert:
    feature: str
    ks_statistic: float
    p_value: float
    severity: str            # 'ok' | 'warn' | 'alert'


@dataclass
class DriftReport:
    n_live_samples: int
    n_train_samples: int
    alerts: List[DriftAlert]
    max_ks: float
    features_checked: List[str]


def ks_drift(train_samples: np.ndarray, live_samples: np.ndarray,
             alert_thresh: float = 0.15, warn_thresh: float = 0.08) -> DriftAlert:
    """KS-test one feature's training distribution against live samples."""
    if not _HAS_SCIPY:
        # Manual Kolmogorov-Smirnov (empirical CDF difference)
        a = np.sort(train_samples.astype(np.float64))
        b = np.sort(live_samples.astype(np.float64))
        all_vals = np.concatenate([a, b])
        # ECDFs
        cdf_a = np.searchsorted(a, all_vals, side="right") / a.size
        cdf_b = np.searchsorted(b, all_vals, side="right") / b.size
        ks = float(np.max(np.abs(cdf_a - cdf_b)))
        p = None
    else:
        stat, p = ks_2samp(train_samples, live_samples)
        ks = float(stat)
        p = float(p)
    sev = "ok"
    if ks >= alert_thresh:
        sev = "alert"
    elif ks >= warn_thresh:
        sev = "warn"
    return DriftAlert(feature="", ks_statistic=ks,
                      p_value=float(p) if p is not None else float("nan"),
                      severity=sev)


def check_drift(
    train_feature_matrix: np.ndarray,
    live_feature_matrix: np.ndarray,
    feature_names: Sequence[str],
    *,
    alert_thresh: float = 0.15,
    warn_thresh: float = 0.08,
) -> DriftReport:
    """Both matrices shape (N, F). feature_names has length F."""
    if train_feature_matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match training matrix width")
    if live_feature_matrix.shape[1] != len(feature_names):
        raise ValueError("live matrix width must match feature_names")
    alerts: List[DriftAlert] = []
    max_ks = 0.0
    for i, name in enumerate(feature_names):
        a = train_feature_matrix[:, i]
        b = live_feature_matrix[:, i]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 20 or b.size < 20:
            continue
        alert = ks_drift(a, b, alert_thresh=alert_thresh, warn_thresh=warn_thresh)
        alert.feature = name
        if alert.severity != "ok":
            alerts.append(alert)
        if alert.ks_statistic > max_ks:
            max_ks = alert.ks_statistic
    return DriftReport(
        n_live_samples=int(live_feature_matrix.shape[0]),
        n_train_samples=int(train_feature_matrix.shape[0]),
        alerts=alerts, max_ks=float(max_ks),
        features_checked=list(feature_names),
    )
