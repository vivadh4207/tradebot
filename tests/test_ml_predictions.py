"""Tests for ml_predictions storage + resolver classification logic."""
from datetime import datetime, timedelta, timezone

import pytest

from src.storage.journal import SqliteJournal, MLPrediction


def _tmp(tmp_path_factory, name="ml.sqlite"):
    return str(tmp_path_factory.mktemp("ml") / name)


def _pred(ts: datetime, symbol: str = "SPY") -> MLPrediction:
    return MLPrediction(
        id=None, ts=ts, symbol=symbol, model="lstm-v1",
        pred_class=2, confidence=0.62,
        p_bearish=0.12, p_neutral=0.26, p_bullish=0.62,
        horizon_minutes=25, up_thr=0.0015, down_thr=-0.0015,
        entry_price=500.0,
    )


def test_record_and_fetch_unresolved(tmp_path_factory):
    j = SqliteJournal(_tmp(tmp_path_factory, "a.sqlite"))
    j.init_schema()
    t = datetime(2026, 4, 16, 14, 30, tzinfo=timezone.utc)
    pid = j.record_ml_prediction(_pred(t, "SPY"))
    assert pid > 0

    # ts is before the "older_than" cutoff → should appear
    later = t + timedelta(minutes=30)
    rows = j.unresolved_ml_predictions(older_than=later)
    assert len(rows) == 1
    assert rows[0].symbol == "SPY" and rows[0].id == pid
    j.close()


def test_resolve_updates_row(tmp_path_factory):
    j = SqliteJournal(_tmp(tmp_path_factory, "b.sqlite"))
    j.init_schema()
    t = datetime(2026, 4, 16, 14, 30, tzinfo=timezone.utc)
    pid = j.record_ml_prediction(_pred(t))
    j.resolve_ml_prediction(pid, forward_return=0.0021, true_class=2)

    rows = j.resolved_ml_predictions(model="lstm-v1")
    assert len(rows) == 1
    r = rows[0]
    assert r.id == pid
    assert r.forward_return == pytest.approx(0.0021, rel=1e-4)
    assert r.true_class == 2
    assert r.resolved_at is not None

    # and it's no longer unresolved
    assert len(j.unresolved_ml_predictions(older_than=t + timedelta(minutes=30))) == 0
    j.close()


def test_resolver_classification_thresholds():
    from scripts.resolve_ml_predictions import _classify
    assert _classify(0.01, 0.001, -0.001) == 2          # bullish
    assert _classify(-0.01, 0.001, -0.001) == 0         # bearish
    assert _classify(0.0005, 0.001, -0.001) == 1        # neutral (below up_thr)
    assert _classify(-0.0005, 0.001, -0.001) == 1       # neutral (above down_thr)
    assert _classify(0.0, 0.001, -0.001) == 1


def test_lstm_signal_records_to_journal_when_model_loads(tmp_path_factory,
                                                           monkeypatch):
    """Verify: if the LSTM signal were to successfully infer, it WOULD log.

    We can't cheaply train a real model in-process, but we CAN exercise the
    journal-write path by constructing the MLPrediction directly and
    checking our schema accepts it — which is also what the real signal
    writes.
    """
    j = SqliteJournal(_tmp(tmp_path_factory, "c.sqlite"))
    j.init_schema()

    # Simulate what LSTMSignal.emit() writes
    now = datetime.now(tz=timezone.utc)
    for cls, conf in [(0, 0.80), (1, 0.48), (2, 0.66)]:
        j.record_ml_prediction(MLPrediction(
            id=None, ts=now, symbol="NVDA", model="lstm-v1",
            pred_class=cls, confidence=conf,
            p_bearish=(0.8 if cls == 0 else 0.1),
            p_neutral=(0.8 if cls == 1 else 0.15),
            p_bullish=(0.8 if cls == 2 else 0.05),
            horizon_minutes=25, up_thr=0.0015, down_thr=-0.0015,
            entry_price=800.0,
        ))
    rows = j.unresolved_ml_predictions(older_than=now + timedelta(minutes=1))
    assert len(rows) == 3
    confidences = sorted(r.confidence for r in rows)
    assert confidences == pytest.approx([0.48, 0.66, 0.80], rel=1e-3)
    j.close()
