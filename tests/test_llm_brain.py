"""LLMBrain review-layer tests."""
from __future__ import annotations

import time
from typing import List

import pytest

from src.intelligence.llm_brain import (
    CandidateDecision, ReviewContext, ReviewedDecision,
    LLMBrain, LLMBrainConfig,
    _build_prompt, _parse_response, _fingerprint, _FAIL_OPEN,
)


# ---------- fixtures ----------


def _candidate(**over):
    base = dict(symbol="SPY", action="enter_long", direction="bullish",
                 source="ensemble", confidence=0.75, rationale="momentum+orb")
    base.update(over)
    return CandidateDecision(**base)


def _ctx(**over):
    base = dict(spot=580.0, vwap=579.5, regime="trend_lowvol", vix=16.0,
                 breadth_score=0.2, breadth_is_risk_off=False, rsi_14=55.0,
                 nearest_support=578.0, nearest_resistance=582.5,
                 five_bar_move_pct=0.004, volume_vs_avg_20=1.3,
                 iv_rank=0.45, open_positions=1, position_on_symbol=0,
                 day_pnl_usd=120.0, catalyst_in_24h=False, news_score=0.1,
                 contributing_signals=["momentum", "orb"])
    base.update(over)
    return ReviewContext(**base)


class _StubBrain(LLMBrain):
    """LLMBrain whose inference is pre-scripted — no model file needed."""
    def __init__(self, raw_responses: List[str], cfg=None):
        super().__init__(cfg or LLMBrainConfig(enabled=True,
                                                  model_path="/tmp/fake.gguf"))
        self._scripted = list(raw_responses)
        self._call_count = 0
        self._model = object()     # skip _ensure_model entirely

    def _ensure_model(self):
        return True

    def _infer(self, prompt):
        self._call_count += 1
        return self._scripted.pop(0) if self._scripted else ""


# ---------- prompt + parse ----------


def test_build_prompt_includes_candidate_and_context_as_json():
    p = _build_prompt(_candidate(), _ctx())
    assert '"symbol":"SPY"' in p
    assert '"regime":"trend_lowvol"' in p
    assert "confidence_multiplier" in p      # schema reminder for the model


def test_parse_response_accepts_bare_json():
    d = _parse_response('{"action": "confirm", '
                         '"confidence_multiplier": 1.0, "reason": "ok"}')
    assert d is not None
    assert d.action == "confirm"
    assert d.confidence_multiplier == 1.0


def test_parse_response_tolerates_preamble_and_trailing_text():
    """Models sometimes prepend 'Here is the JSON:' or <json>. We should
    extract the first balanced object and ignore the rest."""
    raw = ('Sure, here is my review:\n\n'
           '{"action": "adjust", "confidence_multiplier": 0.75, '
           '"reason": "breadth softening"}\n\n'
           'I hope this helps!')
    d = _parse_response(raw)
    assert d is not None
    assert d.action == "adjust"
    assert d.confidence_multiplier == 0.75


def test_parse_response_rejects_invalid_action():
    raw = '{"action": "buy", "confidence_multiplier": 1.0, "reason": "x"}'
    assert _parse_response(raw) is None


def test_parse_response_rejects_non_json():
    assert _parse_response("Just some text.") is None
    assert _parse_response("") is None


def test_parse_response_clamps_runaway_reason_length():
    raw = '{"action": "confirm", "confidence_multiplier": 1.0, ' \
          '"reason": "' + ("x" * 500) + '"}'
    d = _parse_response(raw)
    assert d is not None
    assert len(d.reason) <= 120


# ---------- cache fingerprint ----------


def test_fingerprint_stable_for_near_identical_inputs():
    """Tiny price / confidence perturbations hit the same fingerprint
    so the cache isn't defeated by 0.01 differences."""
    a = _fingerprint(_candidate(confidence=0.75), _ctx(spot=580.0, vix=16.0))
    b = _fingerprint(_candidate(confidence=0.76), _ctx(spot=580.1, vix=16.2))
    assert a == b


def test_fingerprint_differs_on_regime_change():
    a = _fingerprint(_candidate(), _ctx(regime="trend_lowvol"))
    b = _fingerprint(_candidate(), _ctx(regime="range_highvol"))
    assert a != b


def test_fingerprint_differs_on_direction_flip():
    a = _fingerprint(_candidate(direction="bullish"), _ctx())
    b = _fingerprint(_candidate(direction="bearish"), _ctx())
    assert a != b


# ---------- review happy-path ----------


def test_review_confirm_roundtrip():
    raw = '{"action":"confirm","confidence_multiplier":1.0,"reason":"clean setup"}'
    brain = _StubBrain([raw])
    d = brain.review(_candidate(), _ctx())
    assert d.action == "confirm"
    assert d.confidence_multiplier == 1.0
    assert d.latency_ms >= 0


def test_review_adjust_applies_clamp():
    """LLM asks for 2.5x — must be clamped to max_clamp (1.3)."""
    raw = '{"action":"adjust","confidence_multiplier":2.5,"reason":"extra juice"}'
    brain = _StubBrain([raw])
    d = brain.review(_candidate(), _ctx())
    assert d.action == "adjust"
    assert d.confidence_multiplier == 1.3     # default max_clamp


def test_review_adjust_floor_clamp():
    raw = '{"action":"adjust","confidence_multiplier":0.05,"reason":"weak"}'
    brain = _StubBrain([raw])
    d = brain.review(_candidate(), _ctx())
    assert d.confidence_multiplier == 0.30    # default min_clamp


# ---------- veto soft/hard modes ----------


def test_veto_is_demoted_to_adjust_in_soft_mode():
    """Default mode: LLM can't block trades. A veto is demoted to
    adjust@0.5x confidence at most."""
    raw = '{"action":"veto","confidence_multiplier":1.0,"reason":"contradictory"}'
    brain = _StubBrain([raw], cfg=LLMBrainConfig(enabled=True,
                                                   hard_gate=False,
                                                   model_path="/tmp/fake"))
    d = brain.review(_candidate(), _ctx())
    assert d.action == "adjust"
    assert d.confidence_multiplier <= 0.5
    assert d.is_veto is False


def test_veto_is_honored_in_hard_gate_mode():
    raw = '{"action":"veto","confidence_multiplier":1.0,"reason":"contradictory"}'
    brain = _StubBrain([raw], cfg=LLMBrainConfig(enabled=True,
                                                   hard_gate=True,
                                                   model_path="/tmp/fake"))
    d = brain.review(_candidate(), _ctx())
    assert d.action == "veto"
    assert d.is_veto is True


# ---------- fail-open behavior ----------


def test_disabled_brain_always_fail_opens():
    brain = LLMBrain(LLMBrainConfig(enabled=False))
    d = brain.review(_candidate(), _ctx())
    assert d is _FAIL_OPEN or d.action == "confirm"
    assert d.confidence_multiplier == 1.0


def test_malformed_json_fail_opens():
    brain = _StubBrain(["the model rambled and produced no json"])
    d = brain.review(_candidate(), _ctx())
    assert d.confidence_multiplier == 1.0
    assert d.action == "confirm"


def test_infer_exception_fail_opens():
    class _Crashing(_StubBrain):
        def _infer(self, prompt):
            raise RuntimeError("model crashed")
    brain = _Crashing([])
    d = brain.review(_candidate(), _ctx())
    assert d.confidence_multiplier == 1.0


def test_missing_model_file_fail_opens():
    """Real model path doesn't exist on disk → lazy load fails →
    every review() returns fail-open."""
    brain = LLMBrain(LLMBrainConfig(enabled=True,
                                      model_path="/does/not/exist.gguf"))
    d = brain.review(_candidate(), _ctx())
    assert d.confidence_multiplier == 1.0


# ---------- rate limiting + caching ----------


def test_cache_returns_same_decision_within_ttl():
    raw = '{"action":"adjust","confidence_multiplier":0.8,"reason":"x"}'
    brain = _StubBrain([raw])
    d1 = brain.review(_candidate(), _ctx())
    # Second call with identical fingerprint should NOT re-infer
    d2 = brain.review(_candidate(), _ctx())
    assert brain._call_count == 1
    assert d2.from_cache is True
    assert d2.action == d1.action


def test_rate_limit_per_symbol_fails_open_between_calls():
    """Second call on the same symbol too soon after the first falls
    through to fail-open (caller sees no adjustment)."""
    # Give the stub only ONE response to prove the 2nd call never
    # reached _infer.
    brain = _StubBrain(
        ['{"action":"adjust","confidence_multiplier":0.8,"reason":"a"}'],
        cfg=LLMBrainConfig(enabled=True,
                             model_path="/tmp/fake",
                             rate_limit_sec_per_symbol=60.0,
                             cache_ttl_sec=0.001),  # defeat the cache
    )
    _ = brain.review(_candidate(), _ctx())
    time.sleep(0.01)  # cache expires but rate limit still active
    d2 = brain.review(_candidate(symbol="SPY", confidence=0.72),
                       _ctx(spot=585.0))          # new fingerprint
    assert d2.confidence_multiplier == 1.0   # fail-open
    assert brain._call_count == 1            # still only one inference
