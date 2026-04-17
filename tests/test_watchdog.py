"""Unit tests for scripts/watchdog_run.py.

Strategy: don't actually spawn a Python child. Instead, monkeypatch
`_spawn_child` to return a FakeChild whose behavior we control — exit
code, timing, whether it writes a heartbeat, whether it responds to
SIGTERM. That gives us deterministic coverage of every watchdog branch
(clean exit, crash, stale heartbeat, shutdown forwarding) in <1s.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def wd(tmp_path, monkeypatch):
    """Import watchdog_run with paths pointed at tmp_path/logs."""
    # Redirect log paths so tests never touch the real logs/ dir.
    monkeypatch.setenv("WATCHDOG_HEARTBEAT_STALE_SEC", "0.3")
    monkeypatch.setenv("WATCHDOG_CHECK_INTERVAL_SEC", "0.05")
    monkeypatch.setenv("WATCHDOG_STARTUP_GRACE_SEC", "0.0")
    # Fresh import so the env vars above are read at module import time.
    if "watchdog_run" in sys.modules:
        del sys.modules["watchdog_run"]
    mod = importlib.import_module("watchdog_run")
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(mod, "LOG_DIR", logs)
    monkeypatch.setattr(mod, "HEARTBEAT_FILE", logs / "heartbeat.txt")
    monkeypatch.setattr(mod, "EVENTS_FILE", logs / "watchdog_events.jsonl")
    monkeypatch.setattr(mod, "STDERR_CAPTURE", logs / "tradebot.err")
    # Silence the notifier — the webhook path would try to POST for real.
    monkeypatch.setattr(mod, "_notify", lambda *a, **k: None)
    return mod


class FakeChild:
    """Quacks like subprocess.Popen for the three methods the watchdog uses."""

    def __init__(self, *, exit_after: float, exit_code: int = 0,
                 write_heartbeat: bool = True,
                 heartbeat_path: Path = None, respond_to_term: bool = True):
        self.pid = 4242
        self._started = time.time()
        self._exit_after = exit_after
        self._exit_code = exit_code
        self._write_hb = write_heartbeat
        self._hb_path = heartbeat_path
        self._respond_to_term = respond_to_term
        self._term_seen_at = None
        self.returncode = None

    def _maybe_update(self) -> None:
        age = time.time() - self._started
        if self._write_hb and self._hb_path is not None:
            try:
                self._hb_path.write_text("hb")
            except Exception:
                pass
        # If SIGTERM'd and we're a good citizen, exit promptly.
        if self._term_seen_at is not None and self._respond_to_term:
            self.returncode = -15
            return
        if age >= self._exit_after:
            self.returncode = self._exit_code

    def poll(self):
        self._maybe_update()
        return self.returncode

    def terminate(self):
        self._term_seen_at = time.time()

    def kill(self):
        self.returncode = -9

    def send_signal(self, sig):
        self._term_seen_at = time.time()


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_clean_exit_returns_zero(wd, monkeypatch):
    child = FakeChild(exit_after=0.1, exit_code=0,
                       write_heartbeat=True, heartbeat_path=wd.HEARTBEAT_FILE)
    monkeypatch.setattr(wd, "_spawn_child", lambda: child)
    rc = wd.main()
    # Clean exit (rc=0) that wasn't requested by us still returns 0 so
    # launchd can restart via KeepAlive without treating it as failure.
    assert rc == 0
    events = _read_events(wd.EVENTS_FILE)
    kinds = [e["kind"] for e in events]
    assert "start" in kinds
    assert "exit" in kinds
    exit_evt = next(e for e in events if e["kind"] == "exit")
    assert exit_evt["exit_code"] == 0


def test_crash_exit_returns_child_code(wd, monkeypatch):
    child = FakeChild(exit_after=0.1, exit_code=137,
                       write_heartbeat=True, heartbeat_path=wd.HEARTBEAT_FILE)
    monkeypatch.setattr(wd, "_spawn_child", lambda: child)
    rc = wd.main()
    assert rc == 137    # propagate so launchd's crash-loop detector fires
    events = _read_events(wd.EVENTS_FILE)
    exit_evt = next(e for e in events if e["kind"] == "exit")
    assert exit_evt["exit_code"] == 137


def test_stale_heartbeat_kills_child_and_returns_nonzero(wd, monkeypatch):
    """Child is alive but heartbeat never refreshed → watchdog terminates
    it and exits 97 so launchd restarts the whole thing."""
    # Child runs for a long time and does NOT write heartbeat.
    child = FakeChild(exit_after=60.0, exit_code=0,
                       write_heartbeat=False,
                       heartbeat_path=wd.HEARTBEAT_FILE,
                       respond_to_term=True)
    monkeypatch.setattr(wd, "_spawn_child", lambda: child)

    # Seed a stale heartbeat so the watchdog has something to measure age on.
    wd.HEARTBEAT_FILE.write_text("stale")
    stale_ts = time.time() - 60
    os.utime(wd.HEARTBEAT_FILE, (stale_ts, stale_ts))

    rc = wd.main()
    assert rc == 97
    events = _read_events(wd.EVENTS_FILE)
    assert any(e["kind"] == "heartbeat_stale" for e in events)


def test_fresh_heartbeat_does_not_trip_staleness(wd, monkeypatch):
    """If the child writes a heartbeat on every poll, we never kill it."""
    child = FakeChild(exit_after=0.5, exit_code=0,
                       write_heartbeat=True, heartbeat_path=wd.HEARTBEAT_FILE)
    monkeypatch.setattr(wd, "_spawn_child", lambda: child)
    rc = wd.main()
    assert rc == 0
    events = _read_events(wd.EVENTS_FILE)
    assert not any(e["kind"] == "heartbeat_stale" for e in events)


def test_missing_heartbeat_during_startup_grace_does_not_trip(wd, monkeypatch):
    """With STARTUP_GRACE_SEC set, a missing heartbeat in early life must
    not be flagged. We simulate this by re-importing with grace=5s and
    a child that exits before grace is up."""
    monkeypatch.setenv("WATCHDOG_STARTUP_GRACE_SEC", "5")
    monkeypatch.setenv("WATCHDOG_HEARTBEAT_STALE_SEC", "0.1")
    monkeypatch.setenv("WATCHDOG_CHECK_INTERVAL_SEC", "0.05")
    if "watchdog_run" in sys.modules:
        del sys.modules["watchdog_run"]
    mod = importlib.import_module("watchdog_run")
    logs = wd.LOG_DIR
    monkeypatch.setattr(mod, "LOG_DIR", logs)
    monkeypatch.setattr(mod, "HEARTBEAT_FILE", logs / "heartbeat2.txt")
    monkeypatch.setattr(mod, "EVENTS_FILE", logs / "watchdog_events2.jsonl")
    monkeypatch.setattr(mod, "STDERR_CAPTURE", logs / "tradebot2.err")
    monkeypatch.setattr(mod, "_notify", lambda *a, **k: None)

    child = FakeChild(exit_after=0.2, exit_code=0,
                       write_heartbeat=False, heartbeat_path=mod.HEARTBEAT_FILE)
    monkeypatch.setattr(mod, "_spawn_child", lambda: child)
    rc = mod.main()
    assert rc == 0
    events = _read_events(mod.EVENTS_FILE)
    # Nothing stale-flagged because we were still in grace
    assert not any(e["kind"] == "heartbeat_stale" for e in events)


def test_event_log_is_jsonl_append_only(wd, monkeypatch):
    """Two runs in a row append to the same file without clobbering."""
    c1 = FakeChild(exit_after=0.05, exit_code=0,
                    write_heartbeat=True, heartbeat_path=wd.HEARTBEAT_FILE)
    monkeypatch.setattr(wd, "_spawn_child", lambda: c1)
    wd.main()
    c2 = FakeChild(exit_after=0.05, exit_code=1,
                    write_heartbeat=True, heartbeat_path=wd.HEARTBEAT_FILE)
    monkeypatch.setattr(wd, "_spawn_child", lambda: c2)
    wd.main()
    events = _read_events(wd.EVENTS_FILE)
    # Two starts + two exits
    assert sum(1 for e in events if e["kind"] == "start") == 2
    assert sum(1 for e in events if e["kind"] == "exit") == 2


def test_stderr_tail_truncation(wd):
    """_tail_stderr returns the last n lines, never the full file."""
    wd.STDERR_CAPTURE.write_text("\n".join(f"line{i}" for i in range(500)))
    out = wd._tail_stderr(n=5)
    lines = out.splitlines()
    assert len(lines) == 5
    assert lines[-1] == "line499"


def test_reset_heartbeat_removes_stale_file(wd):
    wd.HEARTBEAT_FILE.write_text("old")
    assert wd.HEARTBEAT_FILE.exists()
    wd._reset_heartbeat()
    assert not wd.HEARTBEAT_FILE.exists()
    # idempotent
    wd._reset_heartbeat()


# --- dashboard endpoint ---------------------------------------------------
def test_dashboard_watchdog_endpoint_empty(tmp_path, monkeypatch):
    """Verify /api/watchdog returns a well-formed response even when no
    heartbeat or events file exists yet (fresh install)."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from src.dashboard import app as dash

    # Repoint _settings so the endpoint reads tmp_path/logs (which is empty).
    def _fake_settings():
        from src.core.config import load_settings
        root = ROOT
        return load_settings(root / "config" / "settings.yaml"), tmp_path
    monkeypatch.setattr(dash, "_settings", _fake_settings)
    (tmp_path / "logs").mkdir()

    client = TestClient(dash.app)
    r = client.get("/api/watchdog")
    assert r.status_code == 200
    body = r.json()
    assert body["heartbeat_status"] == "missing"
    assert body["heartbeat_age_sec"] is None
    assert body["recent_events"] == []
    assert body["counts"] == {}


def test_dashboard_watchdog_endpoint_with_events(tmp_path, monkeypatch):
    """Seed a heartbeat + event log and confirm the endpoint surfaces them."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from src.dashboard import app as dash

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "heartbeat.txt").write_text("2026-04-16T12:00:00+00:00")
    evs = [
        {"ts": "2026-04-16T10:00:00+00:00", "kind": "start", "pid": 1234},
        {"ts": "2026-04-16T11:00:00+00:00", "kind": "exit",
         "exit_code": 1, "duration_sec": 42.0},
        {"ts": "2026-04-16T11:05:00+00:00", "kind": "start", "pid": 1300},
    ]
    (logs / "watchdog_events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in evs) + "\n"
    )

    def _fake_settings():
        from src.core.config import load_settings
        return load_settings(ROOT / "config" / "settings.yaml"), tmp_path
    monkeypatch.setattr(dash, "_settings", _fake_settings)

    client = TestClient(dash.app)
    r = client.get("/api/watchdog?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["heartbeat_status"] in ("fresh", "stale")  # depends on wallclock
    assert body["heartbeat_age_sec"] is not None
    # Newest-first ordering:
    assert body["recent_events"][0]["kind"] == "start"
    assert body["recent_events"][0]["pid"] == 1300
    # Counts aggregate over the full file:
    assert body["counts"]["start"] == 2
    assert body["counts"]["exit"] == 1
