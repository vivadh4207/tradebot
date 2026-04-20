"""Master Signal Stack — weighted composite of VRP, IV rank, term, skew, RV regime.

Playbook Section 8. Returns a single decision: SELL_PREMIUM | BUY_PREMIUM | FLAT.
Use this as an OVERLAY on individual strategy signals: strategies may fire,
but if the regime is against them (e.g. ORB call during BUY_PREMIUM regime
when volatility is surging) we can de-weight.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MasterDecision:
    regime: str          # SELL_PREMIUM | BUY_PREMIUM | FLAT
    score: float         # composite in [-1, 1]
    components: dict


class MasterSignalStack:
    def __init__(self,
                 weights=None,
                 decision_threshold: float = 0.15):
        self.weights = weights or {
            "vrp": 0.35, "iv_rank": 0.20, "term": 0.15,
            "skew": 0.15, "rv_regime": 0.15,
        }
        self.thresh = decision_threshold

    def decide(self, atm_iv_30d: float, rv_20d: float,
               iv_52w_low: float, iv_52w_high: float,
               iv_30d: float, iv_90d: float,
               skew_zscore: float, rv_percentile_252d: float) -> MasterDecision:
        vrp = atm_iv_30d - rv_20d
        s_vrp = float(np.tanh(vrp / 0.05))

        rng = max(iv_52w_high - iv_52w_low, 1e-9)
        iv_rank = (atm_iv_30d - iv_52w_low) / rng
        s_iv = 2 * iv_rank - 1

        ts_slope = (iv_90d / iv_30d - 1) if iv_30d > 0 else 0.0
        s_term = float(np.tanh(ts_slope * 10))

        s_skew = float(-np.tanh(skew_zscore / 2))
        s_rv = 0.5 - rv_percentile_252d

        comps = {"vrp": s_vrp, "iv_rank": s_iv, "term": s_term,
                 "skew": s_skew, "rv_regime": s_rv}
        composite = sum(comps[k] * self.weights[k] for k in comps)
        if composite > self.thresh:
            regime = "SELL_PREMIUM"
        elif composite < -self.thresh:
            regime = "BUY_PREMIUM"
        else:
            regime = "FLAT"
        return MasterDecision(regime=regime, score=float(composite), components=comps)
