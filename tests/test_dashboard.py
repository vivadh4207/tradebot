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
