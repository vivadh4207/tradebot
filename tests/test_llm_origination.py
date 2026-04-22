"""LLMOriginationSignal + queue safety rails."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest

from src.intelligence.llm_autotrade_queue import (
    LLMAutotradeQueue, QueuedIdea, write_ideas,
)
from src.signals.llm_origination import LLMOriginationSignal
from src.signals.base import SignalContext
from src.core.types import Bar, OptionRight


def _fresh_idea(**kw):
    ts = kw.pop("ts", datetime.now(tz=timezone.utc).isoformat())
    symbol = kw.pop("symbol", "SPY")
    direction = kw.pop("direction", "put")
    strike = kw.pop("strike", 705.0)
    expiry = kw.pop("expiry", "2026-05-02")
    return QueuedIdea(
        id=QueuedIdea.make_id(symbol, direction, strike, expiry),
        ts=ts, symbol=symbol, direction=direction,
        strike=strike, expiry=expiry,
        confidence=kw.pop("confidence", "high"),
        entry=kw.pop("entry", 3.0),
        profit_target=kw.pop("profit_target", 4.8),
        stop_loss=kw.pop("stop_loss", 1.8),
        time_horizon=kw.pop("time_horizon", "1-3d"),
        rationale=kw.pop("rationale", "VIX spike + breadth deterioration"),
    )


def _ctx(symbol="SPY"):
    now = datetime.now(tz=timezone.utc)
    bars = [Bar(symbol=symbol, ts=now, open=100, high=101, low=99,
                 close=100, volume=1000) for _ in range(20)]
    return SignalContext(symbol=symbol, now=now, bars=bars,
                          spot=100.0, vwap=100.0)


# ----------------------------------------------------------- queue-level


def test_queue_write_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    idea = _fresh_idea()
    n = write_ideas([idea], data_root=tmp_path)
    assert n == 1
    q = LLMAutotradeQueue(data_root=tmp_path)
    got = q.next_idea_for("SPY")
    assert got is not None
    assert got.id == idea.id
    # Second call must NOT return the same idea (consumed)
    assert q.next_idea_for("SPY") is None


def test_queue_kill_switch(tmp_path):
    (tmp_path / "logs").mkdir()
    write_ideas([_fresh_idea()], data_root=tmp_path)
    q = LLMAutotradeQueue(data_root=tmp_path)
    q.set_killed(True)
    assert q.is_killed()
    assert q.next_idea_for("SPY") is None
    q.set_killed(False)
    assert not q.is_killed()
    # Now it should pop
    assert q.next_idea_for("SPY") is not None


def test_queue_stale_ideas_skipped(tmp_path):
    (tmp_path / "logs").mkdir()
    old_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).isoformat()
    idea = _fresh_idea(ts=old_ts)
    write_ideas([idea], data_root=tmp_path)
    q = LLMAutotradeQueue(data_root=tmp_path, max_age_min=30)
    assert q.next_idea_for("SPY") is None


def test_queue_fractional_strike_rejected(tmp_path):
    """SPY strike $705.40 doesn't exist — queue must reject."""
    (tmp_path / "logs").mkdir()
    write_ideas([_fresh_idea(strike=705.4)], data_root=tmp_path)
    q = LLMAutotradeQueue(data_root=tmp_path)
    assert q.next_idea_for("SPY") is None


def test_queue_low_confidence_skipped(tmp_path):
    (tmp_path / "logs").mkdir()
    write_ideas([_fresh_idea(confidence="low")], data_root=tmp_path)
    q = LLMAutotradeQueue(data_root=tmp_path)
    assert q.next_idea_for("SPY") is None


def test_queue_daily_cap(tmp_path):
    (tmp_path / "logs").mkdir()
    # write 5 distinct ideas
    ideas = [_fresh_idea(strike=705 + i, symbol="SPY") for i in range(5)]
    write_ideas(ideas, data_root=tmp_path)
    q = LLMAutotradeQueue(data_root=tmp_path, max_trades_per_day=2)
    assert q.next_idea_for("SPY") is not None
    assert q.next_idea_for("SPY") is not None
    # 3rd call blocked by daily cap
    assert q.next_idea_for("SPY") is None


def test_queue_peek_state(tmp_path):
    (tmp_path / "logs").mkdir()
    write_ideas([_fresh_idea(), _fresh_idea(strike=710)], data_root=tmp_path)
    q = LLMAutotradeQueue(data_root=tmp_path)
    s = q.peek_state()
    assert s["queue_fresh"] == 2
    assert s["killed"] is False
    assert s["daily_count"] == 0


# ----------------------------------------------------------- signal-level


def test_signal_disabled_without_env(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    monkeypatch.delenv("LLM_AUTOTRADE", raising=False)
    # Even with an idea in the queue, signal must NOT emit.
    write_ideas([_fresh_idea()], data_root=tmp_path)
    monkeypatch.chdir(tmp_path)
    sig = LLMOriginationSignal()
    assert sig.emit(_ctx("SPY")) is None


def test_signal_emits_when_env_and_fresh_idea(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    monkeypatch.setenv("LLM_AUTOTRADE", "1")
    write_ideas([_fresh_idea(direction="put", strike=705)], data_root=tmp_path)
    monkeypatch.chdir(tmp_path)
    sig = LLMOriginationSignal()
    s = sig.emit(_ctx("SPY"))
    assert s is not None
    assert s.option_right == OptionRight.PUT
    assert s.meta["source"] == "llm_origination"
    assert s.meta["proposed_strike"] == 705


def test_signal_no_double_emit(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    monkeypatch.setenv("LLM_AUTOTRADE", "1")
    write_ideas([_fresh_idea()], data_root=tmp_path)
    monkeypatch.chdir(tmp_path)
    sig = LLMOriginationSignal()
    first = sig.emit(_ctx("SPY"))
    second = sig.emit(_ctx("SPY"))
    assert first is not None
    assert second is None                 # consumed; doesn't re-emit
