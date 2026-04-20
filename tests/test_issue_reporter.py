"""Tests for the centralized error → Discord alerts pipeline.

Verifies:
  - report_issue throttles duplicate fingerprints within the window
  - report_issue sends level=error + title=error (routes to alerts)
  - alert_on_crash catches + notifies + re-raises (default)
  - alert_on_crash with rethrow=False swallows + returns non-zero
  - install_excepthooks wires sys.excepthook + threading.excepthook
  - notifier failure inside report_issue never raises
"""
from __future__ import annotations

import sys
import threading
import time

import pytest

from src.notify import issue_reporter as ir
from src.notify.base import Notifier


class _CapturingNotifier(Notifier):
    def __init__(self):
        self.calls = []

    def notify(self, text, *, level="info", title="", meta=None):
        self.calls.append({"text": text, "level": level, "title": title, "meta": meta or {}})


@pytest.fixture(autouse=True)
def _reset_notifier(monkeypatch):
    """Replace the shared notifier with a capturing one for each test."""
    cap = _CapturingNotifier()
    monkeypatch.setattr(ir, "_notifier", cap)
    # Clear throttle cache so tests don't pollute each other
    ir._last_sent.clear()
    yield cap


def test_report_issue_routes_to_alerts(_reset_notifier):
    cap = _reset_notifier
    ir.report_issue(scope="unit_test", message="something broke")
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call["level"] == "error"
    # title="error" triggers MultiChannelNotifier routing to alerts
    assert call["title"] == "error"
    assert "something broke" in call["text"]
    assert call["meta"]["Scope"] == "unit_test"


def test_report_issue_includes_traceback_when_exc_passed(_reset_notifier):
    cap = _reset_notifier
    try:
        raise ValueError("boom at line 42")
    except ValueError as e:
        ir.report_issue(scope="unit_test", message="ctx", exc=e)
    assert len(cap.calls) == 1
    assert "Traceback" in cap.calls[0]["meta"]
    assert "ValueError" in cap.calls[0]["meta"]["Traceback"]
    assert "boom at line 42" in cap.calls[0]["meta"]["Traceback"]


def test_report_issue_throttles_duplicate_fingerprints(_reset_notifier):
    cap = _reset_notifier
    ir.report_issue(scope="loop", message="bad things happened")
    ir.report_issue(scope="loop", message="bad things happened")
    ir.report_issue(scope="loop", message="bad things happened")
    # Only first hits the notifier
    assert len(cap.calls) == 1


def test_report_issue_different_scopes_not_throttled(_reset_notifier):
    cap = _reset_notifier
    ir.report_issue(scope="scope_a", message="same message")
    ir.report_issue(scope="scope_b", message="same message")
    assert len(cap.calls) == 2


def test_report_issue_zero_throttle_disables_dedup(_reset_notifier):
    cap = _reset_notifier
    ir.report_issue(scope="s", message="m", throttle_sec=0.0)
    ir.report_issue(scope="s", message="m", throttle_sec=0.0)
    assert len(cap.calls) == 2


def test_report_issue_never_raises_on_notifier_crash(monkeypatch):
    """If the underlying Discord send explodes, report_issue must not
    propagate — otherwise a broken webhook takes down the trade loop."""
    class _BrokenNotifier(Notifier):
        def notify(self, text, *, level="info", title="", meta=None):
            raise RuntimeError("Discord 500")

    monkeypatch.setattr(ir, "_notifier", _BrokenNotifier())
    ir._last_sent.clear()
    # Must NOT raise
    ir.report_issue(scope="anywhere", message="whatever")


def test_fingerprint_strips_decimal_numbers():
    """Line numbers, PIDs, timestamps — all decimal — shouldn't
    fragment throttle keys. (Hex pointers aren't normalized; the
    helper only strips isdigit() chars, which covers the common case
    of `line 42` vs `line 99`.)"""
    a = ir._fingerprint("s", "error at line 42 pid 1234")
    b = ir._fingerprint("s", "error at line 99 pid 5678")
    assert a == b


# ---------- alert_on_crash decorator ----------


def test_alert_on_crash_default_rethrows(_reset_notifier):
    cap = _reset_notifier

    @ir.alert_on_crash("fn_under_test")
    def fails():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        fails()

    assert len(cap.calls) == 1
    assert "fn_under_test" in cap.calls[0]["meta"]["Scope"]
    assert "RuntimeError" in cap.calls[0]["meta"]["Traceback"]


def test_alert_on_crash_rethrow_false_swallows(_reset_notifier):
    cap = _reset_notifier

    @ir.alert_on_crash("cron_script", rethrow=False)
    def fails():
        raise RuntimeError("cron fail")
        return 0  # unreachable

    rv = fails()
    assert rv == 1                       # returns non-zero sentinel
    assert len(cap.calls) == 1


def test_alert_on_crash_passes_through_normal_return(_reset_notifier):
    cap = _reset_notifier

    @ir.alert_on_crash("ok")
    def succeeds():
        return 42

    assert succeeds() == 42
    assert len(cap.calls) == 0           # no alert when fn returns cleanly


def test_alert_on_crash_never_intercepts_systemexit(_reset_notifier):
    """SystemExit from `sys.exit(2)` is a normal control-flow
    mechanism — decorator must not treat it as a crash."""
    cap = _reset_notifier

    @ir.alert_on_crash("s")
    def exits_cleanly():
        raise SystemExit(2)

    with pytest.raises(SystemExit):
        exits_cleanly()
    assert len(cap.calls) == 0


# ---------- excepthooks ----------


@pytest.fixture
def _clean_excepthooks():
    """Save + restore BOTH sys.excepthook and threading.excepthook so
    tests that install hooks don't leak state into later tests (which
    is how we got duplicate alerts before the idempotency guard)."""
    prev_sys = sys.excepthook
    prev_thr = getattr(threading, "excepthook", None)
    yield
    sys.excepthook = prev_sys
    if prev_thr is not None:
        threading.excepthook = prev_thr


def test_install_excepthooks_sets_sys_excepthook(_reset_notifier, _clean_excepthooks):
    """After install, an unhandled exception raised on the main thread
    should push an alert when sys.excepthook is invoked manually."""
    cap = _reset_notifier
    ir.install_excepthooks(scope_prefix="pytest")
    try:
        raise ValueError("stray")
    except ValueError:
        exc_type, exc, tb = sys.exc_info()
        sys.excepthook(exc_type, exc, tb)
    assert len(cap.calls) == 1
    assert "pytest.main" in cap.calls[0]["meta"]["Scope"]


def test_install_excepthooks_ignores_keyboard_interrupt(_reset_notifier, _clean_excepthooks):
    """Ctrl-C is a normal shutdown path; don't alert."""
    cap = _reset_notifier
    ir.install_excepthooks(scope_prefix="pytest")
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        exc_type, exc, tb = sys.exc_info()
        sys.excepthook(exc_type, exc, tb)
    assert len(cap.calls) == 0


def test_install_excepthooks_is_idempotent(_reset_notifier, _clean_excepthooks):
    """Calling install_excepthooks twice must not chain-wrap. A single
    crash should produce ONE alert, not two."""
    cap = _reset_notifier
    ir.install_excepthooks(scope_prefix="a")
    ir.install_excepthooks(scope_prefix="b")     # no-op
    try:
        raise RuntimeError("stray")
    except RuntimeError:
        exc_type, exc, tb = sys.exc_info()
        sys.excepthook(exc_type, exc, tb)
    assert len(cap.calls) == 1
    # scope_prefix from the first (successful) install
    assert "a.main" in cap.calls[0]["meta"]["Scope"]


def test_install_excepthooks_catches_thread_crash(_reset_notifier, _clean_excepthooks):
    cap = _reset_notifier
    if not hasattr(threading, "excepthook"):
        pytest.skip("threading.excepthook unavailable on this Python")
    ir.install_excepthooks(scope_prefix="pytest")

    def explode():
        raise RuntimeError("thread bomb")

    t = threading.Thread(target=explode, name="testworker")
    t.start()
    t.join(timeout=2.0)
    time.sleep(0.05)                        # let the hook land
    assert len(cap.calls) == 1
    assert "testworker" in cap.calls[0]["meta"]["Scope"]
    assert "RuntimeError" in cap.calls[0]["meta"]["Traceback"]
