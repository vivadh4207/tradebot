"""Build training datasets from historical bars."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..core.types import Bar
from .features import (
    build_feature_matrix, build_sequences, FeatureStats, FEATURE_COLS,
)


@dataclass
class SymbolDataset:
    symbol: str
    X_seq: np.ndarray      # (N, seq_len, F)
    y: np.ndarray          # (N,)
    bar_count: int


def build_dataset(
    bars_by_symbol: Dict[str, List[Bar]],
    seq_len: int = 30,
    horizon: int = 5,
    up_thr: float = 0.001,
    down_thr: float = -0.001,
    min_bars: int = 200,
) -> Tuple[np.ndarray, np.ndarray, FeatureStats, List[SymbolDataset]]:
    """Stack all symbols into one flat (X, y) dataset and fit normalization.

    Returns (X_norm, y, stats, per_symbol).
    Each symbol is required to have >= min_bars after warm-up, else skipped.
    """
    per_symbol: List[SymbolDataset] = []
    for sym, bars in bars_by_symbol.items():
        if len(bars) < max(min_bars, seq_len + horizon + 25):
            continue
        F = build_feature_matrix(bars)
        closes = np.array([b.close for b in bars], dtype=np.float64)
        X_seq, y = build_sequences(F, closes, seq_len=seq_len, horizon=horizon,
                                    up_thr=up_thr, down_thr=down_thr)
        if X_seq.shape[0] == 0:
            continue
        per_symbol.append(SymbolDataset(
            symbol=sym, X_seq=X_seq, y=y, bar_count=len(bars),
        ))
    if not per_symbol:
        empty_X = np.zeros((0, seq_len, len(FEATURE_COLS)), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        return empty_X, empty_y, FeatureStats(means=[0.0] * len(FEATURE_COLS),
                                                stds=[1.0] * len(FEATURE_COLS)), []
    X = np.concatenate([d.X_seq for d in per_symbol], axis=0)
    y = np.concatenate([d.y for d in per_symbol], axis=0)
    stats = FeatureStats.fit(X)
    X_norm = stats.transform(X).astype(np.float32)
    return X_norm, y, stats, per_symbol


def time_ordered_split(X: np.ndarray, y: np.ndarray,
                        val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """No shuffling — take the first (1-val_frac) for train, rest for val.

    This is the right split for time-series forecasting.
    """
    n = X.shape[0]
    if n == 0:
        return X, y, X, y
    cut = max(1, int(n * (1.0 - val_frac)))
    return X[:cut], y[:cut], X[cut:], y[cut:]
