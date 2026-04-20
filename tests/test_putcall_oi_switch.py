"""Unit tests for the put/call OI risk switch."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.risk.putcall_oi_switch import (
    PutCallOIConfig, compute_risk_state, read_state, current_size_multiplier,
)


def _hist(values, start_date="2026-04-01"):
    """Build a sorted-oldest-first list of {date, ratio} records."""
    base = datetime.fromisoformat(start_date).date()
    return [{"date": (base + timedelta(days=i)).isoformat(),
             "ratio": float(v)}
            for i, v in enumerate(values)]


# ---------- compute_risk_state ----------


def test_no_trigger_when_ratio_below_threshold():
    cfg = PutCallOIConfig(risk_off_ratio_threshold=2.0, lookback_days=5)
    # Stable 1.2 ratio — normal market
    state = compute_risk_state(_hist([1.1, 1.2, 1.3, 1.2, 1.2]), cfg)
    assert state["risk_off"] is False
    assert state["reason"] == "no_trigger"
    assert state["size_multiplier"] == 1.0


def test_trigger_when_latest_above_threshold_and_avg_rising():
    cfg = PutCallOIConfig(risk_off_ratio_threshold=2.0, lookback_days=5)
    # Prior 5d avg ~1.2, then ratios spike above 2.0 and 5d avg rises
    hist = _hist([1.1, 1.2, 1.3, 1.2, 1.2,     # prior window: avg 1.2
                  2.1, 2.3, 2.0, 2.2, 2.4])    # last window:  avg 2.2
    state = compute_risk_state(hist, cfg)
    assert state["risk_off"] is True
    assert state["size_multiplier"] == pytest.approx(cfg.risk_off_size_multiplier)
    assert state["ratio_latest"] == 2.4
    assert state["is_rising"] is True


def test_no_trigger_when_ratio_above_but_falling():
    """Elevated ratio alone isn't enough — has to be intensifying."""
    cfg = PutCallOIConfig(risk_off_ratio_threshold=2.0, lookback_days=5)
    # Latest is 2.1 but 5d avg is BELOW prior 5d avg (hedging easing)
    hist = _hist([2.5, 2.6, 2.5, 2.4, 2.5,     # prior window: avg 2.5
                  2.3, 2.2, 2.2, 2.1, 2.1])    # last window:  avg 2.18
    state = compute_risk_state(hist, cfg)
    assert state["risk_off"] is False
    assert state["is_rising"] is False


def test_insufficient_history_returns_safe_default():
    cfg = PutCallOIConfig(lookback_days=5)
    state = compute_risk_state(_hist([1.2, 1.3]), cfg)
    assert state["risk_off"] is False
    assert state["reason"] == "insufficient_history"


# ---------- read_state ----------


def test_read_state_returns_default_when_file_missing(tmp_path):
    s = read_state(tmp_path / "missing.json")
    assert s["risk_off"] is False
    assert s["size_multiplier"] == 1.0
    assert s["reason"] == "no_state_file"


def test_read_state_survives_malformed_json(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("this isn't json {{{ ")
    s = read_state(p)
    assert s["risk_off"] is False
    assert s["size_multiplier"] == 1.0


def test_read_state_expires_old_windows(tmp_path):
    """A state written yesterday with expires_at in the past should
    self-clear on read."""
    p = tmp_path / "state.json"
    past = (datetime.now(tz=timezone.utc) - timedelta(days=5)).isoformat()
    p.write_text(json.dumps({
        "risk_off": True,
        "size_multiplier": 0.7,
        "entered_at": past,
        "expires_at": past,
    }))
    s = read_state(p)
    assert s["risk_off"] is False
    assert s["reason"] == "window_expired"
    assert s["size_multiplier"] == 1.0


def test_read_state_preserves_active_risk_off(tmp_path):
    p = tmp_path / "state.json"
    future = (datetime.now(tz=timezone.utc) + timedelta(days=2)).isoformat()
    p.write_text(json.dumps({
        "risk_off": True,
        "size_multiplier": 0.7,
        "expires_at": future,
    }))
    s = read_state(p)
    assert s["risk_off"] is True
    assert s["size_multiplier"] == 0.7


# ---------- current_size_multiplier ----------


def test_current_size_multiplier_clamps_to_sane_range(tmp_path):
    """Guard against a malformed state file giving us an absurd number."""
    p = tmp_path / "state.json"
    future = (datetime.now(tz=timezone.utc) + timedelta(days=2)).isoformat()
    p.write_text(json.dumps({
        "risk_off": True,
        "size_multiplier": 999.9,
        "expires_at": future,
    }))
    cfg = PutCallOIConfig(state_path=str(p))
    # current_size_multiplier clamps at 1.5 upper bound
    m = current_size_multiplier(state_path=p, cfg=cfg)
    assert m == 1.5


def test_current_size_multiplier_defaults_to_one_on_missing_file(tmp_path):
    cfg = PutCallOIConfig()
    m = current_size_multiplier(state_path=tmp_path / "absent.json", cfg=cfg)
    assert m == 1.0
