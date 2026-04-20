"""LSTMSignal smoke tests — verify the signal no-ops gracefully when
no checkpoint is available, which is the default state on a fresh repo.
"""
from datetime import datetime, timedelta, timezone

from src.core.types import Bar
from src.signals.lstm_signal import LSTMSignal
from src.signals.base import SignalContext


def _bars(n=60):
    ts0 = datetime(2026, 4, 16, 10, 0, tzinfo=timezone.utc)
    return [Bar(symbol="SPY", ts=ts0 + timedelta(minutes=i),
                 open=500 + i * 0.01, high=500 + i * 0.01 + 0.1,
                 low=500 + i * 0.01 - 0.1, close=500 + i * 0.01,
                 volume=10000, vwap=500 + i * 0.01) for i in range(n)]


def test_emit_returns_none_when_no_checkpoint():
    sig = LSTMSignal(checkpoint_path="/nonexistent/lstm_best.pt")
    assert sig._model is None
    ctx = SignalContext(symbol="SPY", now=datetime.now(), bars=_bars(),
                         spot=500.0, vwap=500.0,
                         opening_range_high=501, opening_range_low=499)
    assert sig.emit(ctx) is None


def test_emit_returns_none_with_not_enough_bars_even_if_model_present(tmp_path):
    # Write a bogus file so _load attempts to open it (will fail, which is fine).
    p = tmp_path / "bogus.pt"
    p.write_bytes(b"not a real checkpoint")
    sig = LSTMSignal(checkpoint_path=str(p))
    assert sig._model is None    # failed to load — the signal is disabled
    ctx = SignalContext(symbol="SPY", now=datetime.now(), bars=_bars(5),
                         spot=500.0, vwap=500.0,
                         opening_range_high=501, opening_range_low=499)
    assert sig.emit(ctx) is None
