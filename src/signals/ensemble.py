"""EnsembleCoordinator — collect per-signal emissions, weight by regime,
emit (at most) one consolidated Signal.

Why this matters:
- Each SignalSource emits independently. Without coordination, three
  strategies firing on SPY would queue three entry attempts.
- Different signals work in different regimes. A trending regime favors
  momentum + LSTM; a range-bound regime favors VWAP reversion + ORB;
  a high-IV regime favors premium harvest (VRP / Wheel).
- This coordinator multiplies each signal's raw confidence by a
  regime-specific weight, sums per-direction, and emits ONE signal IF
  the dominant direction clears both a minimum weighted-confidence AND a
  dominance ratio over the opposing direction.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.types import Signal
from ..intelligence.regime import Regime


# Default regime × signal-source weights. 1.0 = neutral; >1 = favored, <1 = down-weighted.
# Confidence multiplier applied to each SignalSource.emit() output.
DEFAULT_WEIGHTS: Dict[Regime, Dict[str, float]] = {
    Regime.TREND_LOWVOL: {
        "momentum": 1.30, "orb": 0.80, "vwap_reversion": 0.60,
        "vrp": 0.70, "wheel": 0.70, "lstm": 0.00, "claude_ai": 1.10,
        "candle_patterns": 1.00,
    },
    Regime.TREND_HIGHVOL: {
        "momentum": 1.10, "orb": 0.70, "vwap_reversion": 0.50,
        "vrp": 0.80, "wheel": 0.60, "lstm": 0.00, "claude_ai": 0.90,
        "candle_patterns": 1.10,
    },
    Regime.RANGE_LOWVOL: {
        "momentum": 0.60, "orb": 1.00, "vwap_reversion": 1.30,
        "vrp": 1.00, "wheel": 1.00, "lstm": 0.00, "claude_ai": 0.90,
        "candle_patterns": 1.30,           # patterns matter most in range
    },
    Regime.RANGE_HIGHVOL: {
        "momentum": 0.50, "orb": 0.90, "vwap_reversion": 1.10,
        "vrp": 1.40, "wheel": 1.30, "lstm": 0.00, "claude_ai": 0.80,
        "candle_patterns": 1.20,
    },
    Regime.OPENING: {
        "momentum": 0.80, "orb": 1.50, "vwap_reversion": 0.80,
        "vrp": 0.60, "wheel": 0.60, "lstm": 0.00, "claude_ai": 0.90,
        "candle_patterns": 0.90,
    },
    Regime.CLOSING: {
        # Effectively shut down new entries in the last 30 min — let the
        # session filter + EOD sweep own this window.
        "momentum": 0.30, "orb": 0.30, "vwap_reversion": 0.30,
        "vrp": 0.20, "wheel": 0.20, "lstm": 0.00, "claude_ai": 0.30,
        "candle_patterns": 0.30,
    },
}


@dataclass
class Contribution:
    source: str
    direction: str
    raw_confidence: float
    weight: float

    @property
    def weighted(self) -> float:
        return self.raw_confidence * self.weight


@dataclass
class EnsembleDecision:
    """Full record of what the coordinator decided and why."""
    emitted: bool
    signal: Optional[Signal]
    regime: Regime
    dominant_direction: Optional[str]
    dominant_score: float
    opposing_score: float
    n_inputs: int
    reason: str
    contributions: List[Contribution] = field(default_factory=list)


class EnsembleCoordinator:
    """Given a batch of raw signals for one symbol, decide what to enter.

    Parameters:
      min_weighted_confidence — sum of weighted confidences for the
        dominant direction must clear this. Default 0.70.
      dominance_ratio — dominant / opposing score ratio needed. Default 1.5.
        Set to 1.0 to allow emission whenever there's a dominant direction.
    """

    def __init__(self,
                 weights: Optional[Dict[Regime, Dict[str, float]]] = None,
                 min_weighted_confidence: float = 0.70,
                 dominance_ratio: float = 1.5,
                 max_confidence: float = 1.0):
        self.weights = weights or DEFAULT_WEIGHTS
        self.min_weighted = float(min_weighted_confidence)
        self.dominance_ratio = float(dominance_ratio)
        self.max_confidence = float(max_confidence)

    def aggregate(self, signals: List[Signal], regime: Regime) -> EnsembleDecision:
        if not signals:
            return EnsembleDecision(
                emitted=False, signal=None, regime=regime,
                dominant_direction=None, dominant_score=0.0,
                opposing_score=0.0, n_inputs=0, reason="no_signals",
            )

        reg_weights = self.weights.get(regime, {})
        contribs: List[Contribution] = []
        by_direction: Dict[str, List[Tuple[Signal, float]]] = defaultdict(list)

        directionless = 0
        for sig in signals:
            direction = str(sig.meta.get("direction", "")).strip()
            w = float(reg_weights.get(sig.source, 1.0))
            # Record every signal as a contribution for observability, even
            # those without a usable direction. Only directed ones contribute
            # to by_direction scoring.
            contribs.append(Contribution(
                source=sig.source, direction=direction or "(none)",
                raw_confidence=float(sig.confidence), weight=w,
            ))
            if not direction:
                directionless += 1
                continue
            weighted = sig.confidence * w
            by_direction[direction].append((sig, weighted))

        if not by_direction:
            return EnsembleDecision(
                emitted=False, signal=None, regime=regime,
                dominant_direction=None, dominant_score=0.0,
                opposing_score=0.0, n_inputs=len(signals),
                reason="no_directed_signals", contributions=contribs,
            )

        scores = {d: sum(w for _, w in lst) for d, lst in by_direction.items()}
        dom_dir, dom_score = max(scores.items(), key=lambda kv: kv[1])
        # opposing = sum of everything else; treat premium_harvest specially
        # (not directly bullish/bearish opposites, but still competing attention)
        opposing = sum(v for d, v in scores.items() if d != dom_dir)

        if dom_score < self.min_weighted:
            return EnsembleDecision(
                emitted=False, signal=None, regime=regime,
                dominant_direction=dom_dir, dominant_score=dom_score,
                opposing_score=opposing, n_inputs=len(signals),
                reason=f"below_threshold:{dom_score:.3f}<{self.min_weighted:.2f}",
                contributions=contribs,
            )

        if opposing > 0 and dom_score / opposing < self.dominance_ratio:
            return EnsembleDecision(
                emitted=False, signal=None, regime=regime,
                dominant_direction=dom_dir, dominant_score=dom_score,
                opposing_score=opposing, n_inputs=len(signals),
                reason=f"conflict:{dom_score:.3f}/{opposing:.3f}="
                        f"{(dom_score / opposing):.2f}<{self.dominance_ratio:.2f}",
                contributions=contribs,
            )

        # Use the strongest constituent as the template for the aggregated signal.
        strongest_sig, _ = max(by_direction[dom_dir], key=lambda x: x[1])
        contributors = [s.source for s, _ in by_direction[dom_dir]]
        final = Signal(
            source="ensemble",
            symbol=strongest_sig.symbol,
            side=strongest_sig.side,
            option_right=strongest_sig.option_right,
            strike=strongest_sig.strike,
            expiry=strongest_sig.expiry,
            confidence=min(dom_score, self.max_confidence),
            rationale=f"regime={regime.value} "
                      f"dir={dom_dir} score={dom_score:.2f} "
                      f"n_contrib={len(contributors)}",
            meta={
                "direction": dom_dir,
                "regime": regime.value,
                "ensemble": True,
                "contributors": contributors,
                "dominant_score": dom_score,
                "opposing_score": opposing,
                "entry_tag": strongest_sig.meta.get("entry_tag", "ensemble"),
                "mi_edge_score": strongest_sig.meta.get("mi_edge_score", 0),
                "premium_action": strongest_sig.meta.get("premium_action"),
            },
        )
        return EnsembleDecision(
            emitted=True, signal=final, regime=regime,
            dominant_direction=dom_dir, dominant_score=dom_score,
            opposing_score=opposing, n_inputs=len(signals),
            reason=f"emit:{dom_dir}:{dom_score:.3f}",
            contributions=contribs,
        )
