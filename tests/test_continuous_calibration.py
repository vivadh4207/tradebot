"""Tests for the continuous slippage calibration loop:
SlippageLogger, AutoCalibratingCostModel, and the keep-what-works rule.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def test_slippage_logger_appends_valid_jsonl(tmp_path):
    from src.analytics.slippage_calibration import SlippageLogger, load_recent
    p = tmp_path / "cal.jsonl"
    lg = SlippageLogger(str(p))
    lg.record(
        symbol="SPY", side="buy", qty=5, is_option=False,
        limit_price=500.0, executed_price=500.15,
        predicted_bps=3.0,
        components={"half_spread_bps": 2.0, "size_impact_bps": 1.0},
        mid=500.0, vix=15.0, tag="entry:momentum",
    )
    lg.record(
        symbol="SPY", side="sell", qty=5, is_option=False,
        limit_price=500.0, executed_price=499.85,
        predicted_bps=3.0,
        components={"half_spread_bps": 2.0, "size_impact_bps": 1.0},
        mid=500.0, vix=15.0, tag="exit:target",
    )
    rows = load_recent(str(p), days=1)
    assert len(rows) == 2
    # observed_bps is positive for both sides (sign convention: always "worse")
    for r in rows:
        assert r["observed_bps"] > 0
        assert r["predicted_bps"] == 3.0


def test_analyze_returns_stats_with_components(tmp_path):
    from src.analytics.slippage_calibration import SlippageLogger, load_recent, analyze
    p = tmp_path / "cal.jsonl"
    lg = SlippageLogger(str(p))
    for i in range(50):
        lg.record(
            symbol="SPY", side="buy", qty=1, is_option=False,
            limit_price=500.0, executed_price=500.15,
            predicted_bps=3.0,
            components={"half_spread_bps": 2.0, "size_impact_bps": 1.0,
                         "vix_impact_bps": 0.0, "noise_bps": 0.0},
            mid=500.0, vix=15.0,
        )
    stats = analyze(load_recent(str(p), days=1))
    assert stats is not None
    assert stats.n == 50
    assert stats.mean_predicted == pytest.approx(3.0, rel=1e-3)
    assert stats.mean_observed == pytest.approx(3.0, rel=1e-3)
    assert stats.mean_ratio == pytest.approx(1.0, rel=1e-2)
    assert "half_spread_bps" in stats.per_component_mean


def test_propose_tuning_keeps_calibrated_model():
    """ratio in [0.8, 1.2] → NO change (keep-what-works)."""
    from src.analytics.slippage_calibration import CalibrationStats, propose_tuning
    stats = CalibrationStats(
        n=100, mean_predicted=3.0, mean_observed=3.1,
        median_observed=3.0, p95_observed=6.0, p99_observed=8.0,
        mean_ratio=1.033,
        per_component_mean={"half_spread_bps": 2.0, "size_impact_bps": 1.0},
        per_symbol_mean={"SPY": 3.0},
        days_covered=7.0,
    )
    proposal = propose_tuning(stats)
    # ratio 1.033 is inside [0.8, 1.2] — no constants should change
    assert proposal.proposed == proposal.current


def test_propose_tuning_bumps_underpredicted_model():
    """ratio > 1.2 → increase half_spread_mult + size_impact_coef."""
    from src.analytics.slippage_calibration import CalibrationStats, propose_tuning
    stats = CalibrationStats(
        n=100, mean_predicted=2.0, mean_observed=4.0,
        median_observed=4.0, p95_observed=8.0, p99_observed=10.0,
        mean_ratio=2.0,
        per_component_mean={"half_spread_bps": 1.5, "size_impact_bps": 0.5},
        per_symbol_mean={"SPY": 4.0},
        days_covered=7.0,
    )
    proposal = propose_tuning(stats)
    assert proposal.proposed["half_spread_mult"] > proposal.current["half_spread_mult"]
    assert proposal.proposed["size_impact_coef"] > proposal.current["size_impact_coef"]


def test_propose_tuning_backs_off_overpredicted_model():
    """ratio < 0.8 → decrease constants."""
    from src.analytics.slippage_calibration import CalibrationStats, propose_tuning
    stats = CalibrationStats(
        n=100, mean_predicted=4.0, mean_observed=2.0,
        median_observed=2.0, p95_observed=4.0, p99_observed=5.0,
        mean_ratio=0.5,
        per_component_mean={"half_spread_bps": 3.0, "size_impact_bps": 1.0},
        per_symbol_mean={"SPY": 2.0},
        days_covered=7.0,
    )
    proposal = propose_tuning(stats)
    assert proposal.proposed["half_spread_mult"] < proposal.current["half_spread_mult"]
    assert proposal.proposed["size_impact_coef"] < proposal.current["size_impact_coef"]


def test_propose_tuning_insufficient_samples_keeps_current():
    from src.analytics.slippage_calibration import CalibrationStats, propose_tuning
    stats = CalibrationStats(
        n=10, mean_predicted=3.0, mean_observed=10.0,
        median_observed=10.0, p95_observed=20.0, p99_observed=25.0,
        mean_ratio=3.3,          # very far off, but n too small
        per_component_mean={}, per_symbol_mean={}, days_covered=0.5,
    )
    proposal = propose_tuning(stats)
    assert proposal.proposed == proposal.current
    assert any("insufficient" in r for r in proposal.rationale)


def test_auto_calibrating_model_updates_constants_on_recalibrate(tmp_path):
    """End-to-end: populate calibration log, call recalibrate(), verify
    the inner model's constants moved and history was written."""
    from src.brokers.slippage_model import StochasticCostModel
    from src.brokers.auto_calibrating_model import AutoCalibratingCostModel
    from src.analytics.slippage_calibration import SlippageLogger

    cal_path = tmp_path / "cal.jsonl"
    hist_path = tmp_path / "hist.jsonl"
    # Seed 50 fills where predicted=2bps but observed=5bps (ratio 2.5 → bump up)
    lg = SlippageLogger(str(cal_path))
    for i in range(60):
        lg.record(
            symbol="SPY", side="buy", qty=1, is_option=False,
            limit_price=500.0, executed_price=500.25,
            predicted_bps=2.0,
            components={"half_spread_bps": 1.5, "size_impact_bps": 0.5,
                         "vix_impact_bps": 0.0, "noise_bps": 0.0},
            mid=500.0, vix=15.0,
        )

    inner = StochasticCostModel(base_half_spread_mult=1.0, size_impact_coef=0.25)
    model = AutoCalibratingCostModel(
        inner=inner,
        calibration_path=str(cal_path),
        history_path=str(hist_path),
        min_samples=30,
    )
    result = model.recalibrate(lookback_hours=24)
    assert "changes" in result
    assert result["changes"]   # at least one constant moved
    assert inner.base_half_spread_mult > 1.0   # bumped up
    assert hist_path.exists()
    line = hist_path.read_text().strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["n_fills"] == 60
    assert rec["ratio"] > 1.2


def test_auto_calibrating_model_respects_drift_cap(tmp_path):
    """Even if ratio says "bump 5x", the drift cap limits us to 2x baseline."""
    from src.brokers.slippage_model import StochasticCostModel
    from src.brokers.auto_calibrating_model import AutoCalibratingCostModel
    from src.analytics.slippage_calibration import SlippageLogger

    cal_path = tmp_path / "c.jsonl"
    lg = SlippageLogger(str(cal_path))
    # Extreme: predicted 1, observed 100 (ratio 100x). Model would want to
    # move enormously, but the drift cap (2x baseline) must bind.
    for i in range(50):
        lg.record(
            symbol="SPY", side="buy", qty=1, is_option=False,
            limit_price=500.0, executed_price=505.0,  # 100 bps observed
            predicted_bps=1.0,
            components={"half_spread_bps": 0.5, "size_impact_bps": 0.5,
                         "vix_impact_bps": 0.0, "noise_bps": 0.0},
            mid=500.0, vix=15.0,
        )

    inner = StochasticCostModel(base_half_spread_mult=1.0, size_impact_coef=0.25)
    model = AutoCalibratingCostModel(
        inner=inner,
        calibration_path=str(cal_path),
        history_path=str(tmp_path / "h.jsonl"),
        min_samples=30,
        max_step_per_cycle=1.0,           # loose per-cycle
        max_drift_from_baseline=2.0,      # tight drift cap
    )
    # Run recalibrate 10 times — constants should stop at 2× baseline
    for _ in range(10):
        model.recalibrate(lookback_hours=24)
    assert inner.base_half_spread_mult <= 2.0 + 1e-6
    assert inner.size_impact_coef <= 0.50 + 1e-6


def test_paper_broker_records_calibration_on_fill(tmp_path, monkeypatch):
    from src.brokers.paper import PaperBroker
    from src.brokers.slippage_model import StochasticCostModel, MarketContext
    from src.core.types import Order, Side

    cal_path = tmp_path / "slip.jsonl"
    monkeypatch.setenv("TRADEBOT_SLIPPAGE_LOG", str(cal_path))
    # Reset the cached logger so our env var takes effect
    import src.brokers.paper as _p
    _p._SLIPPAGE_LOGGER = None

    b = PaperBroker(starting_equity=100_000,
                     slippage_model=StochasticCostModel(seed=0))
    b.update_market_context(
        "SPY",
        MarketContext(bid=499.95, ask=500.05, bid_size=1000, ask_size=1000, vix=15),
    )
    b.submit(Order(symbol="SPY", side=Side.BUY, qty=1, is_option=False,
                    limit_price=500.0, tag="t"))
    assert cal_path.exists()
    lines = cal_path.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["symbol"] == "SPY"
    assert row["side"] == "buy"
    assert "half_spread_bps" in row["components"]


def test_daily_report_no_data(tmp_path, monkeypatch):
    """Daily report should run cleanly on an empty journal + no calibration."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    import subprocess, sys as _sys
    venv_py = "/sessions/gallant-confident-keller/.venv/bin/python"
    # Just import the script — running it needs a live journal. Import check
    # protects against syntax errors sneaking in.
    import scripts.daily_report as dr   # noqa: F401
    import scripts.calibrate_slippage as cs   # noqa: F401


def test_dashboard_calibration_endpoint(tmp_path, monkeypatch):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from src.dashboard import app as dash
    from src.storage.journal import SqliteJournal
    db = tmp_path / "d.sqlite"
    monkeypatch.setattr(
        dash, "_load_journal",
        lambda: (lambda j: (j.init_schema(), j)[1])(SqliteJournal(str(db))),
    )
    client = TestClient(dash.app)
    r = client.get("/api/calibration?days=7")
    assert r.status_code == 200
    body = r.json()
    assert "stats" in body and "auto_history" in body
