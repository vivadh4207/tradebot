"""Per-signal audit log tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from src.core.types import Bar, Side, OptionRight, Signal
from src.signals.base import SignalSource, SignalContext
from src.core import signal_audit


class _FakeSource(SignalSource):
    """Emits a fixed signal (or None) for testing."""
    name = "fake"
    def __init__(self, sig: Optional[Signal] = None):
        self._sig = sig
    def emit(self, ctx):
        return self._sig


def _ctx(symbol="SPY"):
    return SignalContext(
        symbol=symbol,
        now=datetime(2026, 4, 17, 10, 30, tzinfo=timezone.utc),
        bars=[], spot=580.0,
    )


@pytest.fixture
def audit_env(tmp_path, monkeypatch):
    log = tmp_path / "audit.jsonl"
    monkeypatch.setenv("TRADEBOT_SIGNAL_AUDIT", "1")
    monkeypatch.setenv("TRADEBOT_SIGNAL_AUDIT_PATH", str(log))
    return log


# ---------- log_emit direct ----------


def test_log_emit_writes_record(audit_env):
    signal_audit.log_emit(
        source="momentum", symbol="SPY",
        emitted=True, confidence=0.82,
        rationale="slope>0.01", side="buy", option_right="call",
    )
    lines = audit_env.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["source"] == "momentum"
    assert rec["symbol"] == "SPY"
    assert rec["emitted"] is True
    assert rec["confidence"] == 0.82
    assert rec["side"] == "buy"
    assert rec["right"] == "call"


def test_log_emit_silent_when_audit_disabled(tmp_path, monkeypatch):
    """Without the env flag, log_emit should write nothing."""
    monkeypatch.delenv("TRADEBOT_SIGNAL_AUDIT", raising=False)
    monkeypatch.setenv("TRADEBOT_SIGNAL_AUDIT_PATH", str(tmp_path / "x.jsonl"))
    signal_audit.log_emit(source="m", symbol="SPY", emitted=True)
    assert not (tmp_path / "x.jsonl").exists()


def test_log_emit_handles_non_json_meta(audit_env):
    """Meta values that aren't JSON-serializable should be coerced to
    str instead of crashing."""
    class _Weird:
        def __repr__(self): return "<weird>"
    signal_audit.log_emit(
        source="x", symbol="SPY", emitted=False,
        meta={"thing": _Weird(), "ok": 42},
    )
    rec = json.loads(audit_env.read_text().splitlines()[0])
    assert rec["meta"]["thing"] == "<weird>"
    assert rec["meta"]["ok"] == 42


def test_log_emit_never_raises_on_io_failure(monkeypatch):
    """A read-only filesystem or full disk must NOT crash the bot."""
    monkeypatch.setenv("TRADEBOT_SIGNAL_AUDIT", "1")
    monkeypatch.setenv("TRADEBOT_SIGNAL_AUDIT_PATH",
                        "/root/nonexistent/forbidden/audit.jsonl")
    # Must NOT raise
    signal_audit.log_emit(source="m", symbol="SPY", emitted=True)


# ---------- audit_source wrapper ----------


def test_audit_source_logs_non_emit(audit_env):
    src = signal_audit.audit_source(_FakeSource(None))
    src.emit(_ctx("SPY"))
    lines = audit_env.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["source"] == "fake"
    assert rec["emitted"] is False


def test_audit_source_logs_emission_with_meta(audit_env):
    sig = Signal(
        source="fake", symbol="SPY", side=Side.BUY,
        option_right=OptionRight.CALL, confidence=0.7,
        rationale="mock", meta={"direction": "bullish"},
    )
    src = signal_audit.audit_source(_FakeSource(sig))
    result = src.emit(_ctx("SPY"))
    assert result is sig
    rec = json.loads(audit_env.read_text().splitlines()[0])
    assert rec["emitted"] is True
    assert rec["confidence"] == 0.7
    assert rec["meta"]["direction"] == "bullish"


def test_audit_source_preserves_signal_passthrough(audit_env):
    """The wrapper must not alter the returned Signal — it's audit
    only, not a transform."""
    sig_in = Signal(source="fake", symbol="QQQ", side=Side.SELL,
                     confidence=0.5, rationale="r")
    src = signal_audit.audit_source(_FakeSource(sig_in))
    sig_out = src.emit(_ctx("QQQ"))
    assert sig_out is sig_in
    assert sig_out.symbol == "QQQ"


def test_audit_source_noop_when_disabled(tmp_path, monkeypatch):
    """If audit is disabled, the wrapper returns the source unchanged
    (zero overhead path)."""
    monkeypatch.delenv("TRADEBOT_SIGNAL_AUDIT", raising=False)
    src_in = _FakeSource(None)
    src_out = signal_audit.audit_source(src_in)
    assert src_out is src_in


def test_audit_source_swallows_wrapper_errors(audit_env, monkeypatch):
    """If the audit write itself crashes, the wrapped source must
    still return the original emit() result — nothing lost."""
    def boom(*a, **kw): raise RuntimeError("disk full")
    monkeypatch.setattr(signal_audit, "log_emit", boom)
    sig = Signal(source="fake", symbol="SPY", side=Side.BUY,
                  confidence=0.5, rationale="x")
    src = signal_audit.audit_source(_FakeSource(sig))
    # MUST not raise despite log_emit exploding
    assert src.emit(_ctx("SPY")) is sig


# ---------- read_tail ----------


def test_read_tail_returns_last_n(audit_env):
    for i in range(50):
        signal_audit.log_emit(source=f"s{i}", symbol="SPY", emitted=False)
    tail = signal_audit.read_tail(10)
    assert len(tail) == 10
    assert tail[-1]["source"] == "s49"


def test_read_tail_returns_empty_when_no_log(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEBOT_SIGNAL_AUDIT_PATH",
                        str(tmp_path / "absent.jsonl"))
    assert signal_audit.read_tail(10) == []


def test_read_tail_skips_malformed_lines(audit_env):
    signal_audit.log_emit(source="good", symbol="SPY", emitted=True)
    # Inject garbage
    with audit_env.open("a") as f:
        f.write("this is not json\n")
    signal_audit.log_emit(source="good2", symbol="QQQ", emitted=False)
    tail = signal_audit.read_tail(100)
    assert len(tail) == 2    # malformed line dropped
