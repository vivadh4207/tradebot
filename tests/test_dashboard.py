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
    # NEW v2 endpoints — all return well-formed shapes even with no data
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/positions_open").status_code == 200
    r = client.get("/api/ml_recent?days=7")
    assert r.status_code == 200 and "n_resolved" in r.json()
    assert client.get("/api/regime_now").status_code == 200
    assert client.get("/api/attribution?days=7").status_code == 200
    r = client.get("/api/logs_tail?lines=10")
    assert r.status_code == 200


def test_bot_status_endpoint_returns_state(tmp_path, monkeypatch):
    """Sanity: /api/bot/status returns a well-formed shape. tradebotctl
    may or may not be reachable in test env — we just assert the shape."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from src.dashboard import app as dash

    def fake_run_ctl(action, timeout=25.0):
        return {"ok": True, "rc": 0, "stdout": "stopped", "stderr": "", "action": action}

    monkeypatch.setattr(dash, "_run_ctl", fake_run_ctl)
    monkeypatch.setattr(dash, "_dashboard_controls_enabled", lambda: False)

    client = TestClient(dash.app)
    r = client.get("/api/bot/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "stopped"
    assert body["controls_enabled"] is False


def test_bot_controls_disabled_by_default(monkeypatch):
    """Dashboard is unauthenticated; start/stop/restart MUST return 403
    unless TRADEBOT_DASHBOARD_CONTROLS=1 is set."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from src.dashboard import app as dash

    monkeypatch.setattr(dash, "_dashboard_controls_enabled", lambda: False)

    client = TestClient(dash.app)
    for action in ("start", "stop", "restart"):
        r = client.post(f"/api/bot/{action}")
        assert r.status_code == 403, f"{action} must require opt-in"
        assert r.json()["ok"] is False


def test_bot_controls_enabled_invokes_ctl(monkeypatch):
    """With the env flag set, start/stop/restart shell out to tradebotctl
    and return its stdout."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from src.dashboard import app as dash

    calls = []

    def fake_run_ctl(action, timeout=25.0):
        calls.append(action)
        return {"ok": True, "rc": 0, "stdout": f"{action} ok", "stderr": "", "action": action}

    monkeypatch.setattr(dash, "_run_ctl", fake_run_ctl)
    monkeypatch.setattr(dash, "_dashboard_controls_enabled", lambda: True)

    client = TestClient(dash.app)
    for action in ("start", "stop", "restart"):
        r = client.post(f"/api/bot/{action}")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert action in body["stdout"]
    assert calls == ["start", "stop", "restart"]


def test_run_ctl_rejects_unsafe_actions():
    """The whitelist guards against an attacker crafting a path like
    'start;rm -rf /' even if they bypass routing."""
    from src.dashboard.app import _run_ctl
    r = _run_ctl("delete")
    assert r["ok"] is False and "not allowed" in r["error"]
    r = _run_ctl("start; rm -rf /")
    assert r["ok"] is False


def test_run_ctl_whitelist_includes_flow_actions():
    """The expanded whitelist must include the dashboard flow actions
    (reset-paper, walkforward, putcall-oi). Non-existent ones still
    rejected."""
    from src.dashboard.app import _run_ctl
    for allowed in ("walkforward", "putcall-oi", "reset-paper"):
        # These resolve — whether they succeed depends on the env, but
        # they must NOT be rejected as "not allowed".
        r = _run_ctl(allowed, timeout=1.0)
        # Either it worked (ok=True), or failed for real reasons (timeout,
        # script crash). The one thing it MUST NOT do is return the
        # whitelist-rejection message.
        assert "not allowed" not in (r.get("error") or "")


def test_flow_endpoints_gated_by_controls_env(monkeypatch):
    """reset_paper, walkforward, refresh_risk_switch — all require the
    same TRADEBOT_DASHBOARD_CONTROLS gate as start/stop."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from src.dashboard import app as dash

    monkeypatch.setattr(dash, "_dashboard_controls_enabled", lambda: False)

    client = TestClient(dash.app)
    for path in ("reset_paper", "walkforward", "refresh_risk_switch"):
        r = client.post(f"/api/bot/{path}")
        assert r.status_code == 403, f"{path} must require opt-in"
        assert r.json()["ok"] is False


def test_flow_endpoints_call_ctl_when_enabled(monkeypatch):
    """With the env flag set, flow endpoints shell out with the correct
    action strings (reset-paper gets --yes so it skips interactive
    confirm)."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from src.dashboard import app as dash

    calls = []

    def fake_run_ctl(action, timeout=25.0, extra_args=None):
        calls.append((action, list(extra_args) if extra_args else []))
        return {"ok": True, "rc": 0, "stdout": f"{action} ok",
                "stderr": "", "action": action}

    monkeypatch.setattr(dash, "_run_ctl", fake_run_ctl)
    monkeypatch.setattr(dash, "_dashboard_controls_enabled", lambda: True)

    client = TestClient(dash.app)
    for path, expected_action in (("reset_paper",          "reset-paper"),
                                    ("walkforward",          "walkforward"),
                                    ("refresh_risk_switch",  "putcall-oi")):
        r = client.post(f"/api/bot/{path}")
        assert r.status_code == 200
        assert r.json()["ok"] is True
    # Verify the actions that were routed, + reset-paper got --yes
    actions = {c[0] for c in calls}
    assert {"reset-paper", "walkforward", "putcall-oi"}.issubset(actions)
    reset_call = next(c for c in calls if c[0] == "reset-paper")
    assert "--yes" in reset_call[1]


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
