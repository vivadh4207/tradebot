import importlib
import pytest


def test_dashboard_imports():
    """Smoke test: the FastAPI app imports cleanly with fastapi installed."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    m = importlib.import_module("src.dashboard.app")
    assert hasattr(m, "app")


def test_dashboard_endpoints_exist(tmp_path, monkeypatch):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from src.dashboard import app as dash
    from src.storage.journal import SqliteJournal

    db = tmp_path / "dash.sqlite"

    def fake_load_journal():
        j = SqliteJournal(str(db))
        j.init_schema()
        return j

    monkeypatch.setattr(dash, "_load_journal", fake_load_journal)

    client = TestClient(dash.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "tradebot" in r.text.lower()
    r = client.get("/api/equity?days=7")
    assert r.status_code == 200
    assert r.json() == {"points": []}
    r = client.get("/api/trades?days=7")
    assert r.status_code == 200
    assert r.json() == {"trades": []}
    r = client.get("/api/metrics?days=7")
    assert r.status_code == 200
    body = r.json()
    assert body["n_trades"] == 0
    assert "win_rate" in body
    # ensemble endpoint exists and returns a well-formed empty shape
    r = client.get("/api/ensemble?days=7")
    assert r.status_code == 200
    body = r.json()
    assert body["n_decisions"] == 0
    assert body["by_regime"] == {}
    assert body["recent"] == []


def test_dashboard_ensemble_endpoint_with_data(tmp_path, monkeypatch):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        import pytest; pytest.skip("fastapi not installed")
    from datetime import datetime, timedelta, timezone
    from src.dashboard import app as dash
    from src.storage.journal import (
        SqliteJournal, EnsembleRecord, ClosedTrade,
    )
    import json as _json

    db = tmp_path / "ens.sqlite"
    j = SqliteJournal(str(db))
    j.init_schema()

    now = datetime.now(tz=timezone.utc)
    # seed 2 decisions (1 emit, 1 block) + 1 matching closed trade
    j.record_ensemble_decision(EnsembleRecord(
        id=None, ts=now, symbol="SPY", regime="trend_lowvol",
        emitted=True, dominant_direction="bullish",
        dominant_score=1.2, opposing_score=0.3, n_inputs=2,
        reason="emit:bullish:1.200",
        contributors=_json.dumps([
            {"source": "momentum", "direction": "bullish", "raw": 0.7, "weight": 1.3},
            {"source": "lstm", "direction": "bullish", "raw": 0.6, "weight": 1.2},
        ]),
    ))
    j.record_ensemble_decision(EnsembleRecord(
        id=None, ts=now - timedelta(minutes=5), symbol="QQQ",
        regime="range_lowvol", emitted=False,
        dominant_direction="bullish",
        dominant_score=0.42, opposing_score=0.0, n_inputs=1,
        reason="below_threshold:0.420<0.700",
        contributors=_json.dumps([
            {"source": "momentum", "direction": "bullish", "raw": 0.7, "weight": 0.6},
        ]),
    ))
    j.record_trade(ClosedTrade(
        symbol="SPY",
        opened_at=now + timedelta(seconds=5),
        closed_at=now + timedelta(minutes=30),
        side="long", qty=1,
        entry_price=500.0, exit_price=503.0,
        pnl=3.0, pnl_pct=0.006,
        entry_tag="ensemble", exit_reason="layer4_target_hit", is_option=False,
    ))

    monkeypatch.setattr(dash, "_load_journal",
                        lambda: SqliteJournal(str(db)))

    client = TestClient(dash.app)
    r = client.get("/api/ensemble?days=1")
    assert r.status_code == 200
    body = r.json()
    assert body["n_decisions"] == 2
    assert body["n_emitted"] == 1
    assert "trend_lowvol" in body["by_regime"]
    assert "range_lowvol" in body["by_regime"]
    # SPY emitted and matched against the trade → win_rate should be 1.0
    trend = body["by_regime"]["trend_lowvol"]
    assert trend["emits"] == 1
    assert trend["matched_trades"] == 1
    assert trend["win_rate"] == 1.0
    # recent list preserves both
    assert len(body["recent"]) == 2
    emitted_row = [x for x in body["recent"] if x["emitted"]][0]
    assert emitted_row["symbol"] == "SPY"
    assert emitted_row["direction"] == "bullish"
    assert any(c["source"] == "momentum" for c in emitted_row["contributors"])
