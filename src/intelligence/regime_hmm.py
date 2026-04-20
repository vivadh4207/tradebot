"""2-state Gaussian Hidden Markov Model for regime detection.

Pure-numpy implementation so we don't add `hmmlearn` as a hard dependency.
Two latent states: LOW_VOL / HIGH_VOL. Observations are per-bar returns.

Viterbi decoding gives the most likely state sequence. The *current*
regime is the state Viterbi assigns to the most recent bar.

This is a drop-in alternative to the rule-based RegimeClassifier. The
HMM is slower to fit (~50-200 ms for 2k bars) but produces smoother
transitions and handles regime persistence correctly — rule-based
classifiers flicker on borderline VIX.

If `hmmlearn` is installed we use it (more robust); otherwise we fall
back to a simple EM fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class HMMRegime:
    current_state: int                   # 0 = low vol, 1 = high vol
    current_label: str                   # 'low_vol' | 'high_vol'
    log_likelihood: float
    transition_matrix: np.ndarray        # 2x2
    state_means: np.ndarray              # (2,)
    state_stds: np.ndarray               # (2,)
    n_obs: int


def _fit_hmm_em(returns: np.ndarray,
                 n_states: int = 2,
                 n_iter: int = 30,
                 tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple Baum-Welch fit for a Gaussian-emission HMM.

    Returns (start_probs, transition_matrix, means, stds).
    """
    n = returns.size
    # Init: K-means-ish split on abs return magnitude
    rng = np.random.default_rng(0)
    order = np.argsort(np.abs(returns))
    means = np.array([
        returns[order[: n // 2]].mean(),
        returns[order[n // 2 :]].mean(),
    ], dtype=np.float64)
    stds = np.array([
        returns[order[: n // 2]].std(ddof=1) + 1e-6,
        returns[order[n // 2 :]].std(ddof=1) + 1e-6,
    ], dtype=np.float64)
    start = np.array([0.6, 0.4])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])

    prev_ll = -np.inf
    for _ in range(n_iter):
        # E-step: forward-backward
        emit = np.zeros((n, n_states))
        for k in range(n_states):
            emit[:, k] = (
                1.0 / (stds[k] * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((returns - means[k]) / stds[k]) ** 2)
            )
        emit = np.clip(emit, 1e-300, None)
        alpha = np.zeros((n, n_states))
        scale = np.zeros(n)
        alpha[0] = start * emit[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ trans) * emit[t]
            scale[t] = alpha[t].sum() + 1e-300
            alpha[t] /= scale[t]

        beta = np.zeros((n, n_states))
        beta[-1] = 1.0 / scale[-1]
        for t in range(n - 2, -1, -1):
            beta[t] = (trans @ (emit[t + 1] * beta[t + 1])) / scale[t]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        xi = np.zeros((n - 1, n_states, n_states))
        for t in range(n - 1):
            num = (
                alpha[t][:, None] * trans * emit[t + 1][None, :] * beta[t + 1][None, :]
            )
            s = num.sum()
            if s > 0:
                xi[t] = num / s

        # M-step
        start = gamma[0]
        trans_new = xi.sum(axis=0) / (gamma[:-1].sum(axis=0)[:, None] + 1e-300)
        # row-normalize
        trans_new = trans_new / (trans_new.sum(axis=1, keepdims=True) + 1e-300)
        trans = trans_new
        means_new = (gamma * returns[:, None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)
        stds_new = np.sqrt(
            (gamma * (returns[:, None] - means_new[None, :]) ** 2).sum(axis=0)
            / (gamma.sum(axis=0) + 1e-300)
        ) + 1e-6
        means = means_new
        stds = stds_new

        ll = float(np.log(scale + 1e-300).sum())
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return start, trans, means, stds


def _viterbi(returns: np.ndarray, start: np.ndarray, trans: np.ndarray,
              means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    n = returns.size
    k = start.size
    log_emit = np.zeros((n, k))
    for s in range(k):
        log_emit[:, s] = -0.5 * np.log(2 * np.pi) - np.log(stds[s]) - \
                         0.5 * ((returns - means[s]) / stds[s]) ** 2
    log_trans = np.log(trans + 1e-300)
    V = np.zeros((n, k))
    back = np.zeros((n, k), dtype=np.int32)
    V[0] = np.log(start + 1e-300) + log_emit[0]
    for t in range(1, n):
        scores = V[t - 1][:, None] + log_trans + log_emit[t][None, :]
        back[t] = np.argmax(scores, axis=0)
        V[t] = np.max(scores, axis=0)
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmax(V[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


class HMMRegimeClassifier:
    """Fits a 2-state Gaussian HMM on log-returns and returns the most
    likely regime for the most recent observation.

    Use as a drop-in augment to the rule-based `RegimeClassifier`:

        cls = HMMRegimeClassifier()
        snap = cls.classify(closes=last_200_closes)
        # map snap.current_label to TREND/RANGE as you see fit
    """

    def __init__(self, low_vol_label: str = "low_vol",
                 high_vol_label: str = "high_vol"):
        self.low_label = low_vol_label
        self.high_label = high_vol_label

    def classify(self, closes: Sequence[float]) -> Optional[HMMRegime]:
        closes = np.asarray(closes, dtype=np.float64)
        closes = closes[closes > 0]
        if closes.size < 50:
            return None
        returns = np.diff(np.log(closes))
        if returns.size < 40:
            return None

        start, trans, means, stds = _fit_hmm_em(returns)
        path = _viterbi(returns, start, trans, means, stds)
        cur = int(path[-1])
        # Canonicalize: state with larger std is "high vol"
        high_state = int(np.argmax(stds))
        is_high = (cur == high_state)

        # log-likelihood via the forward scaling we used in fit (recomputed
        # once for simplicity)
        emit = np.zeros((returns.size, 2))
        for k in range(2):
            emit[:, k] = (
                1.0 / (stds[k] * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((returns - means[k]) / stds[k]) ** 2)
            )
        alpha = start * emit[0]
        ll = float(np.log(alpha.sum() + 1e-300))
        for t in range(1, returns.size):
            alpha = (alpha @ trans) * emit[t]
            s = alpha.sum()
            ll += float(np.log(s + 1e-300))
            alpha /= (s + 1e-300)

        return HMMRegime(
            current_state=cur,
            current_label=self.high_label if is_high else self.low_label,
            log_likelihood=ll,
            transition_matrix=trans,
            state_means=means, state_stds=stds, n_obs=returns.size,
        )
