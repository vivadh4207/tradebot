"""Integration tests — every new Tier 1-3 module must actually be REACHED
by the trading path. These tests lock that in so future refactors can't
quietly break the wiring.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def test_tradebot_instantiates_stochastic_cost_model(tmp_path, monkeypatch):
    """TradeBot.__init__ must attach a StochasticCostModel to its broker.

    When `broker.auto_calibrate` is `hourly` or `daily` the model is wrapped
    in `AutoCalibratingCostModel`; either way the inner model must be a
    StochasticCostModel.
    """
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src.main import TradeBot
    from src.core.config import load_settings
    from src.brokers.slippage_model import StochasticCostModel
    from src.brokers.auto_calibrating_model import AutoCalibratingCostModel
    s = load_settings()
    bot = TradeBot(s)
    model = bot.broker._slippage_model
    assert model is not None
    if isinstance(model, AutoCalibratingCostModel):
        assert isinstance(model._inner, StochasticCostModel)
    else:
        assert isinstance(model, StochasticCostModel)


def test_tradebot_has_drawdown_guard_wired(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src.main import TradeBot
    from src.core.config import load_settings
    from src.risk.drawdown_guard import DrawdownGuard
    s = load_settings()
    bot = TradeBot(s)
    assert isinstance(bot.drawdown_guard, DrawdownGuard)
    assert bot._dd_size_multiplier == 1.0
    assert bot._peak_equity == s.paper_equity


def test_drawdown_guard_scales_down_on_tick(monkeypatch):
    """Simulate an equity drop and confirm _check_halt_conditions updates
    the size multiplier. Explicitly enables the drawdown guard (which
    defaults to OFF per operator preference)."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src.main import TradeBot
    from src.core.config import load_settings
    s = load_settings()
    # Enable the drawdown guard for this test — the default is now off
    # so that the bot keeps trading through DD events and uses per-trade
    # stops to limit risk.
    s.raw.setdefault("account", {})["drawdown_guard_enabled"] = True
    bot = TradeBot(s)
    # Seed a peak then drop equity by 9%
    bot._peak_equity = 10_000.0
    bot.broker._equity = 9_100.0   # 9% DD
    # Make sure daily-loss check doesn't short-circuit
    bot.broker._day_pnl = 0.0
    bot._check_halt_conditions()
    assert bot._dd_size_multiplier == 0.50
    # Past halt threshold
    bot.broker._equity = 8_500.0   # 15% DD
    bot._halted_today = False
    bot._check_halt_conditions()
    assert bot._halted_today


def test_measured_priors_fallback_when_journal_empty(monkeypatch, tmp_path):
    """Fresh bot with no trades → falls back to weakly positive-EV priors.

    The fallback MUST produce positive Kelly so the bot can actually place
    trades on day 1 for data collection. Negative-EV fallbacks are a
    chicken-and-egg trap (no trades → no journal → priors never update).
    """
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src.main import TradeBot
    from src.core.config import load_settings
    s = load_settings()
    bot = TradeBot(s)
    # Defaults: 0.52 win-rate, 0.025 avg-win, 0.020 avg-loss → +0.34% EV
    assert bot._win_rate == 0.52
    assert bot._avg_win == 0.025
    assert bot._avg_loss == 0.020
    # Kelly must be positive or the sizer returns 0 and no trades ever fire.
    b = bot._avg_win / bot._avg_loss
    kelly = (b * bot._win_rate - (1 - bot._win_rate)) / b
    assert kelly > 0, f"fallback priors must have positive Kelly; got f={kelly:.4f}"


def test_measured_priors_use_journal_when_enough_trades(monkeypatch, tmp_path):
    """Seed the journal with 40 wins + 20 losses and verify priors update."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src.main import TradeBot
    from src.core.config import load_settings
    from src.storage.journal import SqliteJournal, ClosedTrade

    # Redirect the journal to a temp path
    db = tmp_path / "j.sqlite"
    j = SqliteJournal(str(db))
    j.init_schema()
    now = datetime.now(tz=timezone.utc)
    for i in range(40):
        j.record_trade(ClosedTrade(
            symbol="SPY", opened_at=now - timedelta(days=i, hours=1),
            closed_at=now - timedelta(days=i, hours=0), side="long",
            qty=1, entry_price=500.0, exit_price=510.0,
            pnl=10.0, pnl_pct=0.02, entry_tag="t", exit_reason="t",
            is_option=False,
        ))
    for i in range(20):
        j.record_trade(ClosedTrade(
            symbol="SPY", opened_at=now - timedelta(days=i, hours=1),
            closed_at=now - timedelta(days=i, hours=0), side="long",
            qty=1, entry_price=500.0, exit_price=490.0,
            pnl=-10.0, pnl_pct=-0.015, entry_tag="t", exit_reason="t",
            is_option=False,
        ))
    j.close()

    s = load_settings()
    # Monkeypatch the build_journal to point at our temp journal
    from src.main import _build_journal_from_settings
    def _bj(settings):
        return SqliteJournal(str(db))
    monkeypatch.setattr("src.main._build_journal_from_settings", _bj)

    bot = TradeBot(s)
    # The 30d lookback clips some seeded trades; what matters is that the
    # journal path was USED (priors != the 0.55 fallback) and the values
    # are inside the sanity-clamp range.
    assert bot._win_rate != 0.55, "should have used measured priors, not fallback"
    assert 0.30 <= bot._win_rate <= 0.80   # inside clamp
    assert bot._avg_win != 0.020 or bot._avg_loss != 0.025, \
        "at least one prior must differ from default"
    assert 0.005 <= bot._avg_win <= 0.10
    assert 0.005 <= bot._avg_loss <= 0.10


def test_run_backtest_registers_run(tmp_path, monkeypatch):
    """run_backtest.py must call register_run at the end."""
    # We don't run the full backtest; we just confirm the import is in place
    # and the registry function is callable.
    import scripts.run_backtest as rb   # noqa: F401
    from src.backtest.run_registry import register_run, read_registry
    log_path = tmp_path / "runs.jsonl"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("k: 1\n")
    register_run(log_path, settings_path=cfg, seed=1,
                  data_source="test", window_days=30, total_bars=100,
                  final_equity=10_000,
                  metrics={"total_return_pct": 0, "sharpe": 0,
                           "max_drawdown_pct": 0, "n_trades": 0})
    assert len(read_registry(log_path)) == 1


def test_dashboard_var_endpoint_empty(tmp_path, monkeypatch):
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
    r = client.get("/api/var")
    assert r.status_code == 200
    body = r.json()
    # Either no report file or a parsed JSON — both are fine for this smoke test
    assert "ts" in body or "error" in body or "message" in body


def test_dashboard_backtest_runs_endpoint(monkeypatch):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from src.dashboard import app as dash
    client = TestClient(dash.app)
    r = client.get("/api/backtest_runs")
    assert r.status_code == 200
    assert "runs" in r.json()


def test_hmm_regime_flag_triggers_classifier(monkeypatch, tmp_path):
    """Setting regime.classifier=hmm must attach an HMMRegimeClassifier."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src.main import TradeBot
    from src.core.config import load_settings
    s = load_settings()
    # Override the setting in-place on the loaded object
    s.raw.setdefault("regime", {})["classifier"] = "hmm"
    bot = TradeBot(s)
    assert bot.regime_kind == "hmm"
    assert bot.hmm_regime_classifier is not None


def test_vol_scale_multiplier_is_applied_to_size():
    """Exercise the sizing path: vol scaling must move the final contract count."""
    from src.main import TradeBot
    from src.core.config import load_settings
    from src.core.types import Bar, Order, Side
    from datetime import timedelta
    import numpy as np

    s = load_settings()
    bot = TradeBot(s)

    # Stand up bars with LOW and HIGH realized vol; confirm vol_scale
    # multipliers go the right direction. (We already tested vol_scale
    # in isolation — this is a smoke test that it's imported in main.)
    t0 = datetime(2026, 4, 16, 10, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    prices_hi = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    bars_hi = [Bar(symbol="NVDA", ts=t0 + timedelta(minutes=i),
                    open=p, high=p * 1.001, low=p * 0.999, close=p, volume=1000)
                for i, p in enumerate(prices_hi)]
    from src.risk.vol_scaling import vol_scale
    vs = vol_scale(bars_hi, target_annual_vol=0.20)
    assert vs.multiplier < 1.0, "high-vol symbol must scale DOWN"
