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
# Ensemble weights — original distribution restored (operator
# decision 2026-04-23: "old strategy might be right"). sr_bounce_break
# added as a PEER signal (weight 1.0, not dominant) so its votes
# count alongside the rest. Can upweight later if it proves out.
DEFAULT_WEIGHTS: Dict[Regime, Dict[str, float]] = {
    Regime.TREND_LOWVOL: {
        "sr_bounce_break": 1.00,   # peer weight
        "momentum": 1.30, "orb": 0.80, "vwap_reversion": 0.60,
        "vrp": 0.70, "wheel": 0.70, "lstm": 0.00, "claude_ai": 1.10,
        "candle_patterns": 1.00, "technical_analysis": 1.10,
        "long_put_dip": 1.00, "llm_origination": 1.10,
    },
    Regime.TREND_HIGHVOL: {
        "sr_bounce_break": 1.00,
        "momentum": 1.10, "orb": 0.70, "vwap_reversion": 0.50,
        "vrp": 0.80, "wheel": 0.60, "lstm": 0.00, "claude_ai": 0.90,
        "candle_patterns": 1.10, "technical_analysis": 1.20,
        "long_put_dip": 1.30, "llm_origination": 1.20,
    },
    Regime.RANGE_LOWVOL: {
        "sr_bounce_break": 1.00,
        "momentum": 0.60, "orb": 1.00, "vwap_reversion": 1.30,
        "vrp": 1.00, "wheel": 1.00, "lstm": 0.00, "claude_ai": 0.90,
        "candle_patterns": 1.30, "technical_analysis": 1.40,
        "long_put_dip": 1.10, "llm_origination": 1.10,
    },
    Regime.RANGE_HIGHVOL: {
        "sr_bounce_break": 1.00,
        "momentum": 0.50, "orb": 0.90, "vwap_reversion": 1.10,
        "vrp": 1.40, "wheel": 1.30, "lstm": 0.00, "claude_ai": 0.80,
        "candle_patterns": 1.20, "technical_analysis": 1.30,
        "long_put_dip": 1.40, "llm_origination": 1.30,
    },
    Regime.OPENING: {
        "sr_bounce_break": 1.00,
        "momentum": 0.80, "orb": 1.50, "vwap_reversion": 0.80,
        "vrp": 0.60, "wheel": 0.60, "lstm": 0.00, "claude_ai": 0.90,
        "candle_patterns": 0.90, "technical_analysis": 0.80,
        "long_put_dip": 1.20, "llm_origination": 0.90,
    },
    Regime.CLOSING: {
        # Effectively shut down new entries in the last 30 min — let the
        # session filter + EOD sweep own this window.
        "sr_bounce_break": 0.50,   # rare late-session bounces still valid
        "momentum": 0.30, "orb": 0.30, "vwap_reversion": 0.30,
        "vrp": 0.20, "wheel": 0.20, "lstm": 0.00, "claude_ai": 0.30,
        "candle_patterns": 0.30, "technical_analysis": 0.30, "long_put_dip": 0.30, "llm_origination": 0.30,
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

        # Session-aware score floor — afternoon directional moves are
        # weaker than morning, so afternoon entries need higher
        # conviction. Operator: "afternoon has less directional
        # conviction than morning — don't take same risk as mornings."
        # Schedule (US Eastern):
        #   09:30-11:30  morning   = base threshold (e.g. 0.70)
        #   11:30-13:30  midday    = base + 0.30 (lunch chop, harder)
        #   13:30-15:00  afternoon = base + 0.60 (less conviction)
        #   15:00-15:30  late      = base + 1.00 (only takes monsters)
        #   15:30+       no_new_entries (already enforced elsewhere)
        # Override via env TRADEBOT_DISABLE_SESSION_FLOOR=1 to flatten.
        try:
            import os as _os
            if (_os.getenv("TRADEBOT_DISABLE_SESSION_FLOOR", "")
                    .strip() not in ("1", "true", "yes")):
                from datetime import datetime as _dt, timezone as _tz
                from zoneinfo import ZoneInfo as _ZI
                _now_et = _dt.now(_tz.utc).astimezone(_ZI("America/New_York"))
                _hm = _now_et.hour * 100 + _now_et.minute
                _bonus = 0.0
                _band = "morning"
                if 1130 <= _hm < 1330:
                    _bonus, _band = 0.30, "midday"
                elif 1330 <= _hm < 1500:
                    _bonus, _band = 0.60, "afternoon"
                elif 1500 <= _hm < 1530:
                    _bonus, _band = 1.00, "late_session"
                # Runtime override hook (lets operator dial it live).
                try:
                    from ..core.runtime_overrides import get_override
                    _bonus = float(get_override(
                        f"score_floor_bonus_{_band}", _bonus,
                    ))
                except Exception:
                    pass
                # ---- Option B: vol-aware auto-loosen ----
                # If the regime detector says we're in a high-vol
                # regime (TREND_HIGHVOL or RANGE_HIGHVOL), the market
                # IS moving and afternoon "chop" assumption is wrong.
                # Loosen the bonus by `vol_loosen_factor` (default 0.30)
                # so real afternoon breakouts aren't blocked. Operator:
                # "if a strong move comes in afternoon, don't block."
                try:
                    from ..core.types import OptionRight as _ignored  # noqa: F401
                    _regime_str = str(getattr(regime, "value", regime) or "").lower()
                    _is_high_vol = (
                        "highvol" in _regime_str
                        or "trend_high" in _regime_str
                    )
                    if _is_high_vol and _bonus > 0:
                        try:
                            from ..core.runtime_overrides import get_override
                            _loosen = float(get_override(
                                "vol_loosen_factor", 0.30,
                            ))
                        except Exception:
                            _loosen = 0.30
                        _orig_bonus = _bonus
                        _bonus = _bonus * max(0.0, min(1.0, _loosen))
                        _band = f"{_band}_volloose"
                        # Note: log via structlog inside the rejection
                        # path only — keep this branch silent on accept
                        # so we don't spam logs on every signal.
                except Exception:
                    pass
                _eff_threshold = self.min_weighted + _bonus
                if dom_score < _eff_threshold:
                    return EnsembleDecision(
                        emitted=False, signal=None, regime=regime,
                        dominant_direction=dom_dir,
                        dominant_score=dom_score,
                        opposing_score=opposing, n_inputs=len(signals),
                        reason=(f"below_session_threshold[{_band}]"
                                 f":{dom_score:.3f}<{_eff_threshold:.2f}"
                                 f"_(base+{_bonus:.2f})"),
                        contributions=contribs,
                    )
        except Exception:
            pass
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

        # ---- minimum-contributors quality gate ----
        # Operator: "quality > quantity — only take trades like META
        # (4 contributors, score 2.58) and NVDA (3 contributors, 2.39)."
        # Single-strategy 1.0 signals are noise that costs spread.
        # Default = 2 (any agreement); raise via override to 3 for
        # high-conviction-only mode.
        #
        # CRASH/RUSH override: when SPY's intraday regime is in a
        # confirmed crash or rush state, the move IS the confirmation.
        # Drop the contributor requirement to 2 so the bot reacts fast.
        # Operator: "morning had volatility, bot didn't execute puts."
        try:
            from ..core.runtime_overrides import get_override
            min_contrib = int(get_override("ensemble_min_contributors", 2))
            # Detect regime via ensemble's regime parameter.
            # Loosen on ANY of: highvol regime, crash, rush, vol_expansion.
            # Operator: morning fast moves often fire with only 2
            # contributors before slower strategies catch up.
            _reg = str(getattr(regime, "value", regime) or "").lower()
            if ("highvol" in _reg or "crash" in _reg
                    or "rush" in _reg or "vol_expansion" in _reg):
                crash_min = int(get_override(
                    "ensemble_min_contributors_crash", 2
                ))
                if crash_min < min_contrib:
                    min_contrib = crash_min
        except Exception:
            min_contrib = 2
        n_dom_contrib = len(by_direction.get(dom_dir, []))
        if n_dom_contrib < min_contrib:
            return EnsembleDecision(
                emitted=False, signal=None, regime=regime,
                dominant_direction=dom_dir, dominant_score=dom_score,
                opposing_score=opposing, n_inputs=len(signals),
                reason=(f"min_contributors:{n_dom_contrib}"
                         f"<{min_contrib}_(noise_filter)"),
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
                "n_contributors": len(contributors),     # for conviction sizing
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
