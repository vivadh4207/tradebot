"""Log digest — parses tradebot.out and builds a Discord-sized summary."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.reports.log_digest import build_digest


def _fmt(dt, level, event, rest=""):
    return f"{dt.isoformat().replace('+00:00','Z')} [{level:<8}] {event} {rest}"


def _write_sample(tmp_path: Path, minutes_ago_events):
    """Write a log file with events at specific minute offsets from now."""
    now = datetime.now(tz=timezone.utc)
    lines = []
    for offset, level, event, rest in minutes_ago_events:
        ts = now - timedelta(minutes=offset)
        lines.append(_fmt(ts, level, event, rest))
    p = tmp_path / "tradebot.out"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_digest_empty_log_returns_empty_digest(tmp_path):
    p = tmp_path / "tradebot.out"
    p.write_text("")
    d = build_digest(p, window_minutes=60)
    assert d.n_parsed == 0
    assert "no parseable events" in d.to_markdown()


def test_digest_missing_file_is_safe(tmp_path):
    d = build_digest(tmp_path / "missing.out", window_minutes=60)
    assert d.n_parsed == 0


def test_digest_counts_entries_exits_skips(tmp_path):
    p = _write_sample(tmp_path, [
        (40, "info",    "exec_chain_pass",   "signal=ensemble"),
        (35, "info",    "exit_placed",       "symbol=SPY"),
        (30, "info",    "ensemble_skip",
         "reason=below_threshold:0.55<0.60 regime=range_lowvol symbol=SPY"),
        (25, "info",    "ensemble_skip",
         "reason=stale_quote regime=range_lowvol symbol=QQQ"),
    ])
    d = build_digest(p, window_minutes=60)
    assert d.entries_fired == 1
    assert d.exits_fired == 1
    assert sum(d.skips_by_reason.values()) == 2
    assert "below_threshold" in d.skips_by_reason


def test_digest_highest_score_tracked(tmp_path):
    p = _write_sample(tmp_path, [
        (30, "info", "ensemble_skip",
         "reason=below_threshold:0.55<0.60 regime=range_lowvol symbol=SPY"),
        (20, "info", "ensemble_skip",
         "reason=below_threshold:0.71<0.60 regime=range_lowvol symbol=QQQ"),
        (10, "info", "ensemble_skip",
         "reason=below_threshold:0.48<0.60 regime=range_lowvol symbol=SPY"),
    ])
    d = build_digest(p, window_minutes=60)
    assert abs(d.highest_score - 0.71) < 1e-6
    assert d.highest_score_symbol == "QQQ"


def test_digest_counts_warnings_and_errors(tmp_path):
    p = _write_sample(tmp_path, [
        (20, "warning", "notifier_post_network_error", "title=FOO"),
        (15, "warning", "alpaca_bars_error_falling_back", "symbol=SPY"),
        (10, "error",   "some_crash", "err=boom"),
    ])
    d = build_digest(p, window_minutes=60)
    assert d.n_errors == 1
    assert d.n_warnings == 2
    # alpaca_bars warning should bump the alpaca bucket
    assert d.alpaca_network_errors >= 1


def test_digest_respects_time_window(tmp_path):
    p = _write_sample(tmp_path, [
        (90, "info", "exec_chain_pass", ""),           # outside window
        (20, "info", "exec_chain_pass", ""),           # inside window
    ])
    d = build_digest(p, window_minutes=60)
    assert d.entries_fired == 1


def test_digest_tracks_shutdowns(tmp_path):
    p = _write_sample(tmp_path, [
        (30, "warning", "shutdown_signal", "signal=SIGTERM"),
        (20, "warning", "shutdown_signal", "signal=SIGTERM"),
    ])
    d = build_digest(p, window_minutes=60)
    assert d.shutdown_signals == 2


def test_digest_markdown_is_under_1800_chars(tmp_path):
    """Discord cap is 2000. Keep digest well under."""
    events = []
    for i in range(200):
        events.append((i % 60, "info", "ensemble_skip",
                       f"reason=below_threshold:0.5<0.60 regime=range_lowvol symbol=SPY"))
    p = _write_sample(tmp_path, events)
    d = build_digest(p, window_minutes=60)
    assert len(d.to_markdown()) < 1800
