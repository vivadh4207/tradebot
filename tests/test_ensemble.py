"""Tests for EnsembleCoordinator."""
from datetime import date

import pytest

from src.core.types import Signal, Side, OptionRight
from src.intelligence.regime import Regime
from src.signals.ensemble import EnsembleCoordinator, DEFAULT_WEIGHTS


def _sig(source: str, direction: str, confidence: float = 0.7,
         symbol: str = "SPY") -> Signal:
    right = OptionRight.CALL if direction == "bullish" else OptionRight.PUT
    return Signal(
        source=source, symbol=symbol, side=Side.BUY,
        option_right=right, confidence=confidence,
        rationale=f"{source}-test",
        meta={"direction": direction, "entry_tag": source},
    )


def test_no_signals_returns_no_emit():
    c = EnsembleCoordinator()
    d = c.aggregate([], Regime.TREND_LOWVOL)
    assert d.emitted is False
    assert d.signal is None
    assert d.reason == "no_signals"


def test_single_strong_momentum_emits_in_trending_regime():
    c = EnsembleCoordinator(min_weighted_confidence=0.5, dominance_ratio=1.0)
    # momentum in TREND_LOWVOL has weight 1.30, conf 0.7 → weighted 0.91
    signals = [_sig("momentum", "bullish", confidence=0.7)]
    d = c.aggregate(signals, Regime.TREND_LOWVOL)
    assert d.emitted is True
    assert d.signal is not None
    assert d.signal.source == "ensemble"
    assert d.signal.meta["direction"] == "bullish"
    assert d.signal.meta["regime"] == "trend_lowvol"
    assert "momentum" in d.signal.meta["contributors"]


def test_range_regime_downweights_momentum():
    # In RANGE_LOWVOL momentum has weight 0.6. Confidence 0.7 → 0.42.
    # With default min_weighted=0.70, this should NOT emit.
    c = EnsembleCoordinator()
    signals = [_sig("momentum", "bullish", confidence=0.7)]
    d = c.aggregate(signals, Regime.RANGE_LOWVOL)
    assert d.emitted is False
    assert "below_threshold" in d.reason


def test_two_aligned_signals_stack_confidence():
    c = EnsembleCoordinator(min_weighted_confidence=1.0, dominance_ratio=1.0)
    signals = [
        _sig("momentum", "bullish", confidence=0.7),       # 0.7 * 1.30 = 0.91
        _sig("lstm", "bullish", confidence=0.6),           # 0.6 * 1.20 = 0.72
        _sig("vwap_reversion", "bearish", confidence=0.5), # 0.5 * 0.60 = 0.30
    ]
    d = c.aggregate(signals, Regime.TREND_LOWVOL)
    assert d.emitted is True
    assert d.dominant_direction == "bullish"
    assert d.dominant_score == pytest.approx(0.91 + 0.72, rel=1e-3)
    assert d.opposing_score == pytest.approx(0.30, rel=1e-3)


def test_conflicting_signals_blocked_by_dominance_ratio():
    c = EnsembleCoordinator(min_weighted_confidence=0.5, dominance_ratio=2.0)
    # strong bullish and strong bearish → ratio below 2.0 → block
    signals = [
        _sig("momentum", "bullish", confidence=0.7),       # 0.91 weighted
        _sig("lstm", "bearish", confidence=0.7),           # 0.84 weighted
    ]
    d = c.aggregate(signals, Regime.TREND_LOWVOL)
    assert d.emitted is False
    assert "conflict" in d.reason


def test_range_highvol_favors_premium_harvest():
    c = EnsembleCoordinator(min_weighted_confidence=0.5, dominance_ratio=1.0)
    signals = [
        _sig("vrp", "premium_harvest", confidence=0.7),    # weight 1.40 → 0.98
        _sig("momentum", "bullish", confidence=0.7),       # weight 0.50 → 0.35
    ]
    d = c.aggregate(signals, Regime.RANGE_HIGHVOL)
    assert d.emitted is True
    assert d.dominant_direction == "premium_harvest"


def test_closing_regime_suppresses_everything():
    c = EnsembleCoordinator()
    # even a high-confidence signal should be below threshold because
    # closing weights are ~0.30.
    signals = [_sig("momentum", "bullish", confidence=0.9)]
    d = c.aggregate(signals, Regime.CLOSING)
    assert d.emitted is False


def test_contributions_record_metadata():
    c = EnsembleCoordinator(min_weighted_confidence=0.2, dominance_ratio=1.0)
    signals = [
        _sig("momentum", "bullish", confidence=0.5),
        _sig("orb", "bullish", confidence=0.6),
    ]
    d = c.aggregate(signals, Regime.OPENING)
    assert d.emitted is True
    sources = {c.source for c in d.contributions}
    assert sources == {"momentum", "orb"}
    for c_ in d.contributions:
        assert c_.weight > 0
        assert c_.raw_confidence > 0
        assert c_.weighted == pytest.approx(c_.raw_confidence * c_.weight, rel=1e-6)
