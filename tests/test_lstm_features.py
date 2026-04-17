"""Pure-numpy feature tests. No torch required."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.core.types import Bar
from src.ml.features import (
    FEATURE_COLS, build_feature_matrix, build_sequences, FeatureStats,
)


def _synth_bars(n: int = 100, seed: int = 7) -> list:
    rng = np.random.default_rng(seed)
    base = 100.0
    bars = []
    ts0 = datetime(2026, 4, 16, 9, 30, tzinfo=timezone.utc)
    price = base
    for i in range(n):
        shock = rng.normal(0, 0.002)
        price *= (1 + shock)
        high = price * (1 + abs(rng.normal(0, 0.0008)))
        low = price * (1 - abs(rng.normal(0, 0.0008)))
        vol = abs(rng.normal(10000, 2000))
        bars.append(Bar(
            symbol="TST",
            ts=ts0 + timedelta(minutes=i),
            open=price, high=high, low=low, close=price, volume=vol,
            vwap=price,
        ))
    return bars


def test_feature_matrix_shape():
    bars = _synth_bars(100)
    X = build_feature_matrix(bars)
    assert X.shape == (100, len(FEATURE_COLS))
    assert X.dtype == np.float32


def test_feature_values_are_bounded():
    bars = _synth_bars(200)
    X = build_feature_matrix(bars)
    # cap log_return to ±0.2
    assert np.all(np.abs(X[:, 0]) <= 0.2 + 1e-6)
    # rsi centered to [-1, 1] (warm-up zero is fine)
    assert np.all(np.abs(X[:, 4]) <= 1.0 + 1e-6)
    # minute sin/cos bounded
    assert np.all(np.abs(X[:, 5]) <= 1.0 + 1e-6)
    assert np.all(np.abs(X[:, 6]) <= 1.0 + 1e-6)


def test_sequences_build_labels_correctly():
    bars = _synth_bars(200)
    X = build_feature_matrix(bars)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    X_seq, y = build_sequences(X, closes, seq_len=20, horizon=5,
                                up_thr=0.001, down_thr=-0.001)
    assert X_seq.ndim == 3
    assert X_seq.shape[1] == 20
    assert X_seq.shape[2] == len(FEATURE_COLS)
    assert X_seq.shape[0] == y.shape[0]
    assert set(np.unique(y).tolist()).issubset({0, 1, 2})


def test_sequences_handle_insufficient_data():
    bars = _synth_bars(10)
    X = build_feature_matrix(bars)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    X_seq, y = build_sequences(X, closes, seq_len=30, horizon=5)
    assert X_seq.shape[0] == 0
    assert y.shape[0] == 0


def test_feature_stats_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 20, 7)).astype(np.float32)
    stats = FeatureStats.fit(X)
    X2 = stats.transform(X)
    # normalized: mean ≈ 0, std ≈ 1 per feature
    assert np.abs(X2.reshape(-1, 7).mean(axis=0)).max() < 1e-3
    assert np.abs(X2.reshape(-1, 7).std(axis=0) - 1.0).max() < 0.02
    d = stats.to_dict()
    stats2 = FeatureStats.from_dict(d)
    assert stats2.means == stats.means
    assert stats2.stds == stats.stds
