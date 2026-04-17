"""Per-bar feature engineering for the LSTM signal.

Pure numpy — no torch here so it's testable without the GPU stack.
Everything is computed causally (no look-ahead): each feature at bar t
uses data up to and including bar t.

Features (7):
  1. log_return           log(close_t / close_{t-1})
  2. log_range            log((high - low) / close + eps)
  3. log_volume_ratio     log(volume_t / rolling_mean_20)
  4. vwap_dev             (close - vwap) / vwap (0 if vwap missing)
  5. rsi_14               standard RSI, Wilder smoothing, centered to [-1, 1]
  6. minute_sin           sin(2π · minute_of_day / 390)
  7. minute_cos           cos(2π · minute_of_day / 390)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Sequence

import numpy as np

from ..core.types import Bar


FEATURE_COLS = [
    "log_return", "log_range", "log_volume_ratio",
    "vwap_dev", "rsi_14", "minute_sin", "minute_cos",
]
EPS = 1e-8


def feature_columns() -> List[str]:
    return list(FEATURE_COLS)


def _rsi_14(closes: np.ndarray) -> np.ndarray:
    """Wilder RSI. Returns array same length as closes; first 14 are zero."""
    n = closes.size
    out = np.zeros(n, dtype=np.float32)
    if n < 15:
        return out
    deltas = np.diff(closes)
    gains = np.clip(deltas, 0.0, None)
    losses = -np.clip(deltas, None, 0.0)
    avg_gain = gains[:14].mean()
    avg_loss = losses[:14].mean()
    for i in range(14, n):
        if i > 14:
            avg_gain = (avg_gain * 13 + gains[i - 1]) / 14
            avg_loss = (avg_loss * 13 + losses[i - 1]) / 14
        rs = avg_gain / (avg_loss + EPS)
        rsi = 100 - 100 / (1 + rs)
        # center to [-1, 1]
        out[i] = (rsi - 50) / 50.0
    return out


def _minute_of_day(bars: Sequence[Bar]) -> np.ndarray:
    """Minutes since market open (09:30 ET). Wraps cleanly via sin/cos."""
    out = np.zeros(len(bars), dtype=np.float32)
    for i, b in enumerate(bars):
        ts = b.ts
        hour = getattr(ts, "hour", 10)
        minute = getattr(ts, "minute", 0)
        total = (hour - 9) * 60 + minute - 30
        total = max(0, min(389, total))
        out[i] = float(total)
    return out


def build_feature_matrix(bars: Sequence[Bar]) -> np.ndarray:
    """Convert a list of Bars to a (T, F) float32 matrix.

    Returns a zero matrix if fewer than 2 bars are provided. Caller should
    slice off the first 20 rows (RSI warm-up + rolling stats) before use.
    """
    n = len(bars)
    if n < 2:
        return np.zeros((n, len(FEATURE_COLS)), dtype=np.float32)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    highs = np.array([b.high for b in bars], dtype=np.float64)
    lows = np.array([b.low for b in bars], dtype=np.float64)
    vols = np.array([b.volume for b in bars], dtype=np.float64)
    vwaps = np.array([(b.vwap if b.vwap is not None else b.close) for b in bars],
                      dtype=np.float64)

    log_ret = np.zeros(n, dtype=np.float32)
    log_ret[1:] = np.log(np.clip(closes[1:] / closes[:-1], 1e-6, 1e6))
    log_ret = np.clip(log_ret, -0.2, 0.2)   # cap outliers (news halts etc.)

    rng = (highs - lows) / (closes + EPS)
    log_rng = np.log(rng + 1e-6).astype(np.float32)

    roll_v = np.zeros(n, dtype=np.float64)
    w = 20
    for i in range(n):
        lo = max(0, i - w + 1)
        seg = vols[lo: i + 1]
        roll_v[i] = seg.mean() if seg.size else 1.0
    log_vol_ratio = np.log((vols + 1.0) / (roll_v + 1.0)).astype(np.float32)
    log_vol_ratio = np.clip(log_vol_ratio, -3.0, 3.0)

    vwap_dev = ((closes - vwaps) / (vwaps + EPS)).astype(np.float32)
    vwap_dev = np.clip(vwap_dev, -0.05, 0.05)

    rsi = _rsi_14(closes.astype(np.float32))

    minutes = _minute_of_day(bars)
    angle = 2.0 * np.pi * minutes / 390.0
    minute_sin = np.sin(angle).astype(np.float32)
    minute_cos = np.cos(angle).astype(np.float32)

    mat = np.stack([log_ret, log_rng, log_vol_ratio,
                    vwap_dev, rsi, minute_sin, minute_cos], axis=1)
    return mat.astype(np.float32)


@dataclass
class FeatureStats:
    """Per-feature mean/std computed on the training set. Used to normalize
    at both training time and inference time so the model sees the same
    distribution.
    """
    means: List[float]
    stds: List[float]

    @classmethod
    def fit(cls, X: np.ndarray) -> "FeatureStats":
        means = X.reshape(-1, X.shape[-1]).mean(axis=0)
        stds = X.reshape(-1, X.shape[-1]).std(axis=0) + EPS
        return cls(means=means.tolist(), stds=stds.tolist())

    def transform(self, X: np.ndarray) -> np.ndarray:
        m = np.asarray(self.means, dtype=X.dtype)
        s = np.asarray(self.stds, dtype=X.dtype)
        return (X - m) / s

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStats":
        return cls(means=list(d["means"]), stds=list(d["stds"]))


def build_sequences(X: np.ndarray, closes: np.ndarray,
                    seq_len: int = 30, horizon: int = 5,
                    up_thr: float = 0.001, down_thr: float = -0.001):
    """Turn a (T, F) feature matrix + close prices into (X_seq, y) arrays.

    Label at index i = forward return from close[i+seq_len-1] to
    close[i+seq_len-1+horizon]:
        >  up_thr   → 2 (bullish)
        <  down_thr → 0 (bearish)
        else         → 1 (neutral)

    Returns (X_seq shape (N, seq_len, F), y shape (N,) int64).
    """
    T = X.shape[0]
    last = T - seq_len - horizon
    if last <= 0:
        return (np.zeros((0, seq_len, X.shape[1]), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    xs = np.stack([X[i: i + seq_len] for i in range(last + 1)], axis=0)
    base = closes[seq_len - 1: seq_len - 1 + last + 1]
    fwd = closes[seq_len - 1 + horizon: seq_len - 1 + horizon + last + 1]
    ret = (fwd - base) / (base + EPS)
    y = np.where(ret > up_thr, 2, np.where(ret < down_thr, 0, 1)).astype(np.int64)
    return xs.astype(np.float32), y
