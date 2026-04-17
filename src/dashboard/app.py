"""FastAPI read-only dashboard: equity curve, trades, open positions.

Reads the configured journal (SQLite or CockroachDB). No authentication —
bind to localhost and reach it through an SSH tunnel or a reverse-proxy
with auth. Do NOT expose to the public internet.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import json
import os
from collections import Counter, defaultdict

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from ..core.config import load_settings
from ..storage.journal import build_journal
from ..storage.position_snapshot import load_snapshot


def _load_journal():
    root = Path(__file__).resolve().parents[2]
    s = load_settings(root / "config" / "settings.yaml")
    return build_journal(
        backend=s.get("storage.backend", "sqlite"),
        sqlite_path=s.get("storage.sqlite_path", str(root / "logs" / "tradebot.sqlite")),
        dsn_env_var=s.get("storage.cockroach_dsn_env", "COCKROACH_DSN"),
        cockroach_schema=s.get("storage.cockroach_schema", "tradebot"),
    )


app = FastAPI(title="tradebot dashboard", docs_url=None, redoc_url=None)


@app.get("/api/equity", response_class=JSONResponse)
def equity(days: int = Query(30, ge=1, le=365)):
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        rows = j.equity_series(since=since, limit=20000)
        return {"points": [{"ts": r[0], "equity": r[1], "cash": r[2], "day_pnl": r[3]}
                            for r in rows]}
    finally:
        j.close()


@app.get("/api/trades", response_class=JSONResponse)
def trades(days: int = Query(30, ge=1, le=365), limit: int = Query(500, le=5000)):
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        ts = j.closed_trades(since=since)[-limit:]
        return {"trades": [
            {"symbol": t.symbol,
             "opened_at": t.opened_at.isoformat() if t.opened_at else None,
             "closed_at": t.closed_at.isoformat() if t.closed_at else None,
             "side": t.side, "qty": t.qty,
             "entry_price": t.entry_price, "exit_price": t.exit_price,
             "pnl": t.pnl, "pnl_pct": t.pnl_pct,
             "entry_tag": t.entry_tag, "exit_reason": t.exit_reason,
             "is_option": t.is_option}
            for t in ts
        ]}
    finally:
        j.close()


@app.get("/api/metrics", response_class=JSONResponse)
def metrics(days: int = Query(30, ge=1, le=365)):
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        ts = j.closed_trades(since=since)
        wins = [t for t in ts if (t.pnl or 0) > 0]
        losses = [t for t in ts if (t.pnl or 0) < 0]
        n_decided = len(wins) + len(losses)
        win_rate = (len(wins) / n_decided) if n_decided else 0.0
        avg_win = (sum(t.pnl_pct or 0 for t in wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(abs(t.pnl_pct or 0) for t in losses) / len(losses)) if losses else 0.0
        total_pnl = sum((t.pnl or 0) for t in ts)
        return {"n_trades": len(ts), "n_wins": len(wins), "n_losses": len(losses),
                "win_rate": round(win_rate, 4),
                "avg_win_pct": round(avg_win, 4),
                "avg_loss_pct": round(avg_loss, 4),
                "total_pnl": round(total_pnl, 2),
                "expected_value_pct":
                    round(win_rate * avg_win - (1 - win_rate) * avg_loss, 4)}
    finally:
        j.close()


@app.get("/api/ensemble", response_class=JSONResponse)
def ensemble(days: int = Query(14, ge=1, le=365),
             recent_limit: int = Query(50, ge=1, le=500)):
    """Per-regime stats + recent decisions.

    Joins ensemble_decisions with closed_trades on approximate time
    (decision_ts ≤ opened_at ≤ decision_ts + 60min) for a per-regime
    win-rate readout.
    """
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        decisions = j.ensemble_decisions(since=since)
        trades = j.closed_trades(since=since)
    finally:
        j.close()

    if not decisions:
        return {
            "n_decisions": 0,
            "n_emitted": 0,
            "global_emit_rate": 0.0,
            "by_regime": {},
            "recent": [],
            "contributors_overall": [],
        }

    # Index trades by symbol → sorted by opened_at for fast matching.
    tr_by_symbol: Dict[str, List] = defaultdict(list)
    for t in trades:
        if t.opened_at is None:
            continue
        tr_by_symbol[t.symbol].append(t)
    for k in tr_by_symbol:
        tr_by_symbol[k].sort(key=lambda t: t.opened_at)

    def _match(decision):
        """Nearest closed trade opened within 60 min of the decision."""
        ts = decision.ts if decision.ts.tzinfo else decision.ts.replace(tzinfo=timezone.utc)
        window = ts + timedelta(minutes=60)
        for t in tr_by_symbol.get(decision.symbol, []):
            ot = t.opened_at
            if ot.tzinfo is None:
                ot = ot.replace(tzinfo=timezone.utc)
            if ts <= ot <= window:
                return t
            if ot > window:
                break
        return None

    by_regime: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "n": 0, "emits": 0, "matched": 0, "wins": 0,
        "pnl_pcts": [], "contributors": Counter(),
    })
    emitted_count = 0
    for d in decisions:
        b = by_regime[d.regime]
        b["n"] += 1
        if d.emitted:
            b["emits"] += 1
            emitted_count += 1
            t = _match(d)
            if t is not None:
                b["matched"] += 1
                if (t.pnl or 0) > 0:
                    b["wins"] += 1
                if t.pnl_pct is not None:
                    b["pnl_pcts"].append(float(t.pnl_pct))
        if d.contributors:
            try:
                for c in json.loads(d.contributors):
                    b["contributors"][c.get("source", "?")] += 1
            except Exception:
                pass

    by_regime_out: Dict[str, Any] = {}
    overall_contrib: Counter = Counter()
    for regime, b in by_regime.items():
        n = b["n"]
        matched = b["matched"]
        pnl_list = b["pnl_pcts"]
        by_regime_out[regime] = {
            "n": n,
            "emits": b["emits"],
            "emit_rate": round(b["emits"] / n, 4) if n else 0.0,
            "matched_trades": matched,
            "win_rate": round(b["wins"] / matched, 4) if matched else None,
            "mean_pnl_pct": round(sum(pnl_list) / len(pnl_list), 6) if pnl_list else None,
            "contributors": [{"source": s, "count": c}
                              for s, c in b["contributors"].most_common(8)],
        }
        overall_contrib.update(b["contributors"])

    recent = []
    for d in decisions[-recent_limit:][::-1]:
        contributors = []
        if d.contributors:
            try:
                contributors = json.loads(d.contributors)
            except Exception:
                contributors = []
        recent.append({
            "ts": d.ts.isoformat() if hasattr(d.ts, "isoformat") else str(d.ts),
            "symbol": d.symbol, "regime": d.regime,
            "emitted": bool(d.emitted),
            "direction": d.dominant_direction,
            "score": round(d.dominant_score, 4) if d.dominant_score is not None else None,
            "opposing": round(d.opposing_score, 4) if d.opposing_score is not None else None,
            "n_inputs": d.n_inputs,
            "reason": d.reason,
            "contributors": contributors,
        })

    return {
        "n_decisions": len(decisions),
        "n_emitted": emitted_count,
        "global_emit_rate": round(emitted_count / len(decisions), 4) if decisions else 0.0,
        "by_regime": by_regime_out,
        "recent": recent,
        "contributors_overall": [{"source": s, "count": c}
                                  for s, c in overall_contrib.most_common(12)],
    }


def _settings():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    return load_settings(root / "config" / "settings.yaml"), root


@app.get("/api/health", response_class=JSONResponse)
def health():
    """Liveness + last-tick freshness. Essential for ops."""
    s, root = _settings()
    j = _load_journal()
    try:
        snap = load_snapshot(s.get("broker.snapshot_path",
                                     str(root / "logs" / "broker_state.json")))
        # Derive "last tick" from the most recent equity_curve entry if present
        from datetime import datetime, timedelta, timezone
        since = datetime.now(tz=timezone.utc) - timedelta(hours=48)
        eq = j.equity_series(since=since, limit=1)
        last_tick = eq[-1][0] if eq else None
        trades = j.closed_trades(since=since)
        last_trade = trades[-1].closed_at.isoformat() if trades and trades[-1].closed_at else None
    finally:
        j.close()

    log_path = root / "logs" / "tradebot.out"
    log_size = log_path.stat().st_size if log_path.exists() else 0

    snap_saved = snap.saved_at if snap else None
    snap_positions = len(snap.positions) if snap else 0
    snap_cash = snap.cash if snap else None
    snap_day_pnl = snap.day_pnl if snap else None

    return {
        "backend": s.get("storage.backend", "sqlite"),
        "universe": s.get("universe", []),
        "ensemble_enabled": s.get("ensemble.enabled", True),
        "lstm_enabled": s.get("ml.lstm_enabled", True),
        "live_trading": s.live_trading,
        "last_equity_snapshot": last_tick,
        "last_trade_close": last_trade,
        "broker_snapshot_saved_at": snap_saved,
        "broker_open_positions": snap_positions,
        "broker_cash": snap_cash,
        "broker_day_pnl": snap_day_pnl,
        "log_size_bytes": log_size,
    }


@app.get("/api/positions_open", response_class=JSONResponse)
def positions_open():
    """Currently open positions (from snapshot) with estimated unrealized P&L."""
    s, root = _settings()
    snap = load_snapshot(s.get("broker.snapshot_path",
                                 str(root / "logs" / "broker_state.json")))
    if snap is None:
        return {"positions": [], "saved_at": None}
    positions = []
    for p in snap.positions:
        positions.append({
            "symbol": p.symbol, "qty": p.qty, "avg_price": p.avg_price,
            "is_option": p.is_option, "underlying": p.underlying,
            "strike": p.strike, "expiry": p.expiry_iso, "right": p.right,
            "entry_tag": p.entry_tag,
            "auto_profit_target": p.auto_profit_target,
            "auto_stop_loss": p.auto_stop_loss,
            "consecutive_holds": p.consecutive_holds,
        })
    return {"positions": positions, "saved_at": snap.saved_at,
            "cash": snap.cash, "day_pnl": snap.day_pnl,
            "total_pnl": snap.total_pnl}


@app.get("/api/ml_recent", response_class=JSONResponse)
def ml_recent(days: int = Query(7, ge=1, le=90),
               limit: int = Query(100, ge=1, le=1000)):
    """Recent LSTM predictions + resolved accuracy summary."""
    from datetime import datetime, timedelta, timezone
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        resolved = j.resolved_ml_predictions(since=since, limit=limit)
        unresolved = j.unresolved_ml_predictions(older_than=datetime.now(tz=timezone.utc),
                                                   limit=limit)
    finally:
        j.close()
    total = resolved
    correct = sum(1 for t in total if t.pred_class == t.true_class)
    wins = sum(1 for t in total if t.pred_class != 1 and t.pred_class == t.true_class)
    losses = sum(1 for t in total if t.pred_class != 1 and t.pred_class != t.true_class
                 and t.true_class != 1)
    directional = [t for t in total if t.pred_class != 1]
    dir_correct = sum(1 for t in directional if t.pred_class == t.true_class)
    dir_wrong_side = sum(
        1 for t in directional
        if (t.pred_class == 2 and t.true_class == 0)
        or (t.pred_class == 0 and t.true_class == 2)
    )
    accuracy = correct / len(total) if total else 0.0
    return {
        "n_resolved": len(total),
        "n_unresolved": len(unresolved),
        "overall_accuracy": round(accuracy, 4),
        "n_directional": len(directional),
        "directional_hit_rate": round(dir_correct / len(directional), 4) if directional else 0.0,
        "directional_wrong_side_rate": round(dir_wrong_side / len(directional), 4) if directional else 0.0,
        "recent": [
            {"ts": t.ts.isoformat(), "symbol": t.symbol,
             "pred": ["bearish", "neutral", "bullish"][t.pred_class],
             "confidence": round(t.confidence, 3),
             "true": (["bearish", "neutral", "bullish"][t.true_class]
                      if t.true_class is not None else None),
             "fwd_return": round(t.forward_return, 4) if t.forward_return is not None else None,
             "horizon_min": t.horizon_minutes}
            for t in total[-limit:][::-1]
        ],
    }


@app.get("/api/regime_now", response_class=JSONResponse)
def regime_now():
    """Most recent regime label the bot saw (from ensemble_decisions)."""
    from datetime import datetime, timedelta, timezone
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=2)
        decisions = j.ensemble_decisions(since=since, limit=5000)
    finally:
        j.close()
    if not decisions:
        return {"regime": None, "as_of": None, "recent_distribution": {}}
    last = decisions[-1]
    # distribution of last 200 decisions
    recent = decisions[-200:]
    dist = Counter(d.regime for d in recent)
    return {
        "regime": last.regime,
        "as_of": last.ts.isoformat() if hasattr(last.ts, "isoformat") else str(last.ts),
        "symbol": last.symbol,
        "emitted": bool(last.emitted),
        "recent_distribution": dict(dist),
    }


@app.get("/api/catalysts_upcoming", response_class=JSONResponse)
def catalysts_upcoming():
    """Upcoming earnings / FDA events in the next 14 days (auto-populated)."""
    from pathlib import Path
    import json as _json
    s, root = _settings()
    # Read the most recent refresh log if available (cron writes these)
    logs_dir = root / "logs"
    out = {"events": [], "source": None}
    if not logs_dir.exists():
        return out
    candidates = sorted(logs_dir.glob("catalysts.*.log"))
    if not candidates:
        return out
    last = candidates[-1]
    try:
        text = last.read_text()
        # the refresh script writes plain lines unless --json; try to parse
        # simple lines: SYMBOL DATE TYPE TIMING DETAILS
        events = []
        for line in text.splitlines():
            parts = line.split(None, 4)
            if len(parts) >= 4 and len(parts[0]) <= 8 and "-" in parts[1]:
                events.append({
                    "symbol": parts[0], "date": parts[1],
                    "type": parts[2], "timing": parts[3],
                    "details": parts[4] if len(parts) > 4 else "",
                })
        out["events"] = events
        out["source"] = last.name
    except Exception:
        pass
    return out


@app.get("/api/logs_tail", response_class=JSONResponse)
def logs_tail(lines: int = Query(200, ge=10, le=5000),
               grep: str = Query("", max_length=256)):
    """Tail the bot's primary log file. Optional grep filter (plain substring)."""
    from pathlib import Path
    s, root = _settings()
    log_path = root / "logs" / "tradebot.out"
    if not log_path.exists():
        return {"lines": [], "path": str(log_path), "missing": True}
    # Efficient-ish tail for small files; OK up to ~100 MB
    try:
        with log_path.open("rb") as f:
            # seek to end, read backward in chunks, collect last N lines
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = 64 * 1024
            buf = b""
            pos = size
            collected = 0
            while pos > 0 and collected <= lines * 2:
                read_size = min(chunk, pos)
                pos -= read_size
                f.seek(pos)
                buf = f.read(read_size) + buf
                collected = buf.count(b"\n")
            text = buf.decode("utf-8", errors="replace")
            all_lines = text.splitlines()
    except Exception as e:
        return {"lines": [], "path": str(log_path), "error": str(e)}
    if grep:
        g = grep.lower()
        all_lines = [ln for ln in all_lines if g in ln.lower()]
    tail = all_lines[-lines:]
    return {"lines": tail, "path": str(log_path), "missing": False,
            "total_matched": len(all_lines)}


@app.get("/api/var", response_class=JSONResponse)
def var_report():
    """Read the most recent Monte Carlo VaR report written by daily_var.py."""
    from pathlib import Path
    s, root = _settings()
    p = root / "logs" / "var_report.json"
    if not p.exists():
        return {"ts": None, "message": "no VaR report yet — run scripts/daily_var.py"}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/backtest_runs", response_class=JSONResponse)
def backtest_runs(limit: int = Query(50, ge=1, le=500)):
    """Read the JSONL run registry (git SHA + config hash + metrics)."""
    from pathlib import Path
    s, root = _settings()
    p = root / "logs" / "backtest_runs.jsonl"
    if not p.exists():
        return {"runs": []}
    runs = []
    for line in p.read_text().splitlines()[-limit:][::-1]:
        try:
            runs.append(json.loads(line))
        except Exception:
            continue
    return {"runs": runs}


@app.get("/api/calibration", response_class=JSONResponse)
def calibration(days: int = Query(7, ge=1, le=90)):
    """Slippage calibration stats + most recent auto-calibration cycles."""
    from pathlib import Path
    from src.analytics.slippage_calibration import load_recent, analyze
    s, root = _settings()
    cal_path = s.get("broker.calibration_path",
                      str(root / "logs" / "slippage_calibration.jsonl"))
    hist_path = s.get("broker.calibration_history",
                       str(root / "logs" / "calibration_history.jsonl"))
    if not Path(cal_path).is_absolute():
        cal_path = str(root / cal_path)
    if not Path(hist_path).is_absolute():
        hist_path = str(root / hist_path)

    rows = load_recent(cal_path, days=days)
    stats = analyze(rows)
    stats_dict = None
    if stats is not None:
        stats_dict = {
            "n": stats.n,
            "mean_predicted_bps": round(stats.mean_predicted, 3),
            "mean_observed_bps": round(stats.mean_observed, 3),
            "median_observed_bps": round(stats.median_observed, 3),
            "p95_observed_bps": round(stats.p95_observed, 3),
            "p99_observed_bps": round(stats.p99_observed, 3),
            "ratio": round(stats.mean_ratio, 3),
            "per_component_mean": {k: round(v, 3) for k, v in stats.per_component_mean.items()},
            "per_symbol_mean": {k: round(v, 3) for k, v in stats.per_symbol_mean.items()},
            "days_covered": round(stats.days_covered, 2),
            "keep_or_tune": (
                "keep" if stats.n >= 30 and 0.8 <= stats.mean_ratio <= 1.2
                else ("tune_up" if stats.mean_ratio > 1.2
                      else ("tune_down" if 0 < stats.mean_ratio < 0.8 else "insufficient"))
            ),
        }
    # Auto-calibration history tail
    history = []
    p = Path(hist_path)
    if p.exists():
        try:
            for line in p.read_text().splitlines()[-50:][::-1]:
                try:
                    history.append(json.loads(line))
                except Exception:
                    continue
        except Exception:
            pass
    return {"stats": stats_dict, "auto_history": history, "lookback_days": days}


@app.get("/api/daily_report", response_class=JSONResponse)
def daily_report():
    """Read the most recent daily EOD snapshot (written by scripts/daily_report.py)."""
    from pathlib import Path
    s, root = _settings()
    p = root / "logs" / "daily_report.json"
    if not p.exists():
        return {"message": "no daily report yet — run scripts/daily_report.py"}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/watchdog", response_class=JSONResponse)
def watchdog(limit: int = Query(25, ge=1, le=500)):
    """Watchdog + heartbeat status for the Loop Insights panel.

    Returns:
      - heartbeat_age_sec:  seconds since bot's main loop last wrote
                            logs/heartbeat.txt. None if never written.
                            >300s during market hours means the loop is
                            wedged (watchdog about to recycle it).
      - heartbeat_status:   "fresh" | "stale" | "missing"
      - recent_events:      last N entries from logs/watchdog_events.jsonl,
                            newest first. Kinds: start, exit,
                            clean_shutdown, heartbeat_stale.
      - counts:             aggregate event counts over the file's lifetime
                            (cheap lookback for the panel's sparkline).
    """
    import time
    from pathlib import Path
    _, root = _settings()
    hb_path = root / "logs" / "heartbeat.txt"
    ev_path = root / "logs" / "watchdog_events.jsonl"

    heartbeat_age_sec = None
    heartbeat_status = "missing"
    try:
        if hb_path.exists():
            heartbeat_age_sec = max(0.0, time.time() - hb_path.stat().st_mtime)
            heartbeat_status = "fresh" if heartbeat_age_sec < 300 else "stale"
    except Exception:
        pass

    recent: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    if ev_path.exists():
        try:
            lines = ev_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    counts[json.loads(line).get("kind", "?")] += 1
                except Exception:
                    continue
            for line in reversed(lines[-limit:]):
                if not line.strip():
                    continue
                try:
                    recent.append(json.loads(line))
                except Exception:
                    continue
        except Exception:
            pass

    return {
        "heartbeat_age_sec": round(heartbeat_age_sec, 1) if heartbeat_age_sec is not None else None,
        "heartbeat_status": heartbeat_status,
        "stale_threshold_sec": 300,
        "recent_events": recent,
        "counts": dict(counts),
    }


@app.get("/api/attribution", response_class=JSONResponse)
def attribution(days: int = Query(30, ge=1, le=365)):
    """Per-entry-tag performance attribution. Maps entry_tag → win rate, mean pnl,
    count, total PnL — so you can see which signal source is carrying the book."""
    from datetime import datetime, timedelta, timezone
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        ts = j.closed_trades(since=since)
    finally:
        j.close()

    buckets: dict = defaultdict(lambda: {"n": 0, "wins": 0,
                                           "pnl_sum": 0.0, "pnl_pct_sum": 0.0})
    for t in ts:
        tag = t.entry_tag or "(none)"
        b = buckets[tag]
        b["n"] += 1
        if (t.pnl or 0) > 0:
            b["wins"] += 1
        b["pnl_sum"] += float(t.pnl or 0)
        b["pnl_pct_sum"] += float(t.pnl_pct or 0)
    out = []
    for tag, b in sorted(buckets.items(), key=lambda kv: -kv[1]["pnl_sum"]):
        n = b["n"]
        out.append({
            "entry_tag": tag,
            "n": n,
            "win_rate": round(b["wins"] / n, 4) if n else 0.0,
            "mean_pnl_pct": round(b["pnl_pct_sum"] / n, 6) if n else 0.0,
            "total_pnl": round(b["pnl_sum"], 2),
        })
    return {"by_entry_tag": out}


_INDEX_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>tradebot</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #0b1020;
         color: #e6e9f5; margin: 0; padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 16px; }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .08em;
       color: #9aa3c7; margin: 24px 0 10px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .card { background: #141a33; border: 1px solid #223; border-radius: 10px;
          padding: 14px; }
  .card .k { color: #9aa3c7; font-size: 11px; text-transform: uppercase; }
  .card .v { font-size: 22px; font-weight: 600; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #1e2540; }
  th { color: #9aa3c7; font-weight: 500; font-size: 11px; text-transform: uppercase; }
  tr:hover { background: #131a35; }
  .pos { color: #6ee7b7; } .neg { color: #fca5a5; } .mut { color: #9aa3c7; }
  canvas { max-height: 280px; }
  .row { display: flex; gap: 12px; align-items: center; margin-bottom: 6px; }
  select { background: #141a33; color: #e6e9f5; border: 1px solid #2b3360;
           border-radius: 6px; padding: 4px 8px; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 10px;
          font-size: 11px; font-weight: 500; }
  .pill.emit  { background: #14291e; color: #6ee7b7; }
  .pill.block { background: #2a1a1f; color: #fca5a5; }
  .pill.regime { background: #182045; color: #a2b7ff; font-family: ui-monospace, monospace; }
  .chip { display: inline-block; background: #1a2142; color: #cbd5ff;
          border-radius: 4px; padding: 1px 6px; font-size: 11px; margin-right: 4px; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head><body>
<h1>tradebot &nbsp;·&nbsp; <span class="mut">executive view</span></h1>

<h2>health</h2>
<div class="grid" id="health"></div>

<div class="row">
  <label>lookback</label>
  <select id="days">
    <option value="1">1d</option><option value="7">7d</option>
    <option value="30" selected>30d</option>
    <option value="90">90d</option><option value="365">365d</option>
  </select>
</div>
<div class="grid" id="metrics"></div>

<h2>open positions</h2>
<div class="card"><table id="open-pos"><thead><tr>
  <th>symbol</th><th>qty</th><th>avg px</th><th>type</th>
  <th>strike</th><th>expiry</th><th>auto PT</th><th>auto SL</th>
  <th>holds</th><th>entry tag</th>
</tr></thead><tbody></tbody></table></div>

<h2>regime + vix</h2>
<div class="grid" id="regime-now"></div>

<h2>performance attribution by entry tag</h2>
<div class="card"><table id="attribution"><thead><tr>
  <th>entry tag</th><th>n</th><th>win rate</th>
  <th>mean pnl %</th><th>total pnl</th>
</tr></thead><tbody></tbody></table></div>

<h2>LSTM calibration (resolved)</h2>
<div class="grid" id="ml-summary"></div>
<div class="card"><table id="ml-recent"><thead><tr>
  <th>time</th><th>symbol</th><th>predicted</th><th>conf</th>
  <th>actual</th><th>fwd return</th><th>horizon</th>
</tr></thead><tbody></tbody></table></div>

<h2>upcoming catalysts</h2>
<div class="card"><table id="catalysts"><thead><tr>
  <th>symbol</th><th>date</th><th>type</th><th>timing</th><th>details</th>
</tr></thead><tbody></tbody></table></div>
<h2>equity curve</h2>
<div class="card"><canvas id="chart"></canvas></div>
<h2>closed trades</h2>
<div class="card"><table id="trades"><thead><tr>
  <th>closed</th><th>symbol</th><th>side</th><th>qty</th>
  <th>entry</th><th>exit</th><th>pnl</th><th>pnl %</th>
  <th>entry tag</th><th>exit reason</th>
</tr></thead><tbody></tbody></table></div>

<h2>ensemble &nbsp;·&nbsp; <span class="mut">regime-weighted signal coordinator</span></h2>
<div class="grid" id="ens-summary"></div>
<h2>regime breakdown</h2>
<div class="card"><table id="ens-regimes"><thead><tr>
  <th>regime</th><th>decisions</th><th>emits</th><th>emit rate</th>
  <th>matched trades</th><th>win rate</th><th>mean pnl %</th><th>top contributors</th>
</tr></thead><tbody></tbody></table></div>
<h2>recent ensemble decisions</h2>
<div class="card"><table id="ens-recent"><thead><tr>
  <th>time</th><th>symbol</th><th>regime</th><th>result</th><th>direction</th>
  <th>score</th><th>opposing</th><th>n inputs</th><th>reason</th>
</tr></thead><tbody></tbody></table></div>

<h2>loop insights &nbsp;·&nbsp; <span class="mut">watchdog + heartbeat + slippage calibration</span></h2>
<div class="grid" id="loop-summary"></div>
<div class="card" style="margin-top:10px;">
  <div style="display:flex;gap:24px;flex-wrap:wrap;">
    <div style="flex:1;min-width:320px;">
      <div class="mut" style="margin-bottom:6px;">recent watchdog events</div>
      <table id="wd-events"><thead><tr>
        <th>ts</th><th>kind</th><th>details</th>
      </tr></thead><tbody></tbody></table>
    </div>
    <div style="flex:1;min-width:320px;">
      <div class="mut" style="margin-bottom:6px;">auto-calibration history (most recent first)</div>
      <table id="cal-history"><thead><tr>
        <th>ts</th><th>n</th><th>ratio</th><th>changes</th>
      </tr></thead><tbody></tbody></table>
    </div>
  </div>
</div>

<h2>logs &nbsp;·&nbsp; <span class="mut">last 200 lines from tradebot.out</span></h2>
<div class="row">
  <label>filter</label>
  <input id="log-grep" type="text" placeholder="substring filter — e.g. ensemble_emit, fill, HALT"
         style="background:#141a33;color:#e6e9f5;border:1px solid #2b3360;
                border-radius:6px;padding:4px 8px;flex:1;max-width:500px;"/>
  <button id="log-refresh" style="background:#2b3360;color:#e6e9f5;border:0;
          padding:5px 14px;border-radius:6px;cursor:pointer;">refresh</button>
</div>
<div class="card" style="max-height:320px;overflow:auto;padding:10px;">
  <pre id="log-view" style="white-space:pre-wrap;margin:0;font-size:12px;
       font-family:ui-monospace,monospace;color:#cbd5ff;"></pre>
</div>
<script>
let chart;
function fmt(n, d=2) { return (n===null||n===undefined) ? '' : Number(n).toFixed(d); }
function cls(n) { return n>0 ? 'pos' : (n<0 ? 'neg' : 'mut'); }
async function refresh() {
  const d = document.getElementById('days').value;
  const [m, eq, tr, en, hlt, pos, ml, rg, at, cat, wd, cal] = await Promise.all([
    fetch('/api/metrics?days='+d).then(r=>r.json()),
    fetch('/api/equity?days='+d).then(r=>r.json()),
    fetch('/api/trades?days='+d+'&limit=200').then(r=>r.json()),
    fetch('/api/ensemble?days='+d+'&recent_limit=50').then(r=>r.json()).catch(()=>null),
    fetch('/api/health').then(r=>r.json()).catch(()=>null),
    fetch('/api/positions_open').then(r=>r.json()).catch(()=>null),
    fetch('/api/ml_recent?days='+Math.min(parseInt(d)||30,90)+'&limit=50').then(r=>r.json()).catch(()=>null),
    fetch('/api/regime_now').then(r=>r.json()).catch(()=>null),
    fetch('/api/attribution?days='+d).then(r=>r.json()).catch(()=>null),
    fetch('/api/catalysts_upcoming').then(r=>r.json()).catch(()=>null),
    fetch('/api/watchdog?limit=15').then(r=>r.json()).catch(()=>null),
    fetch('/api/calibration?days='+Math.min(parseInt(d)||30,90)).then(r=>r.json()).catch(()=>null),
  ]);
  renderHealth(hlt);
  renderOpenPositions(pos);
  renderRegime(rg, hlt);
  renderAttribution(at);
  renderMLRecent(ml);
  renderCatalysts(cat);
  renderLoopInsights(wd, cal);
  const mel = document.getElementById('metrics');
  mel.innerHTML = '';
  const cards = [
    ['trades', m.n_trades],
    ['win rate', (m.win_rate*100).toFixed(1)+'%'],
    ['avg win / avg loss', fmt(m.avg_win_pct*100,2)+'% / '+fmt(m.avg_loss_pct*100,2)+'%'],
    ['total pnl', '$'+fmt(m.total_pnl,2)],
  ];
  cards.forEach(([k,v]) => {
    const el = document.createElement('div'); el.className='card';
    el.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    mel.appendChild(el);
  });
  const labels = eq.points.map(p => p.ts);
  const data = eq.points.map(p => p.equity);
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('chart'), {
    type: 'line',
    data: { labels, datasets: [{ label: 'equity', data, borderColor: '#8ab4ff',
                                  backgroundColor: 'rgba(138,180,255,.08)',
                                  pointRadius: 0, tension: 0.2, fill: true }] },
    options: { responsive: true, scales: {
      x: { ticks: { color: '#9aa3c7', maxTicksLimit: 8 } },
      y: { ticks: { color: '#9aa3c7' } } },
      plugins: { legend: { display: false } } },
  });
  const tb = document.querySelector('#trades tbody');
  tb.innerHTML = '';
  tr.trades.slice().reverse().forEach(t => {
    const row = document.createElement('tr');
    row.innerHTML =
      `<td class="mut">${t.closed_at ? t.closed_at.replace('T',' ').slice(0,19) : ''}</td>`+
      `<td>${t.symbol}</td><td>${t.side}</td><td>${t.qty}</td>`+
      `<td>${fmt(t.entry_price)}</td><td>${fmt(t.exit_price)}</td>`+
      `<td class="${cls(t.pnl)}">${fmt(t.pnl)}</td>`+
      `<td class="${cls(t.pnl_pct)}">${t.pnl_pct!==null ? (t.pnl_pct*100).toFixed(2)+'%' : ''}</td>`+
      `<td class="mut">${t.entry_tag || ''}</td>`+
      `<td class="mut">${t.exit_reason || ''}</td>`;
    tb.appendChild(row);
  });
  renderEnsemble(en);
}

function renderEnsemble(en) {
  const sum = document.getElementById('ens-summary');
  const regTbody = document.querySelector('#ens-regimes tbody');
  const recTbody = document.querySelector('#ens-recent tbody');
  sum.innerHTML = '';
  regTbody.innerHTML = '';
  recTbody.innerHTML = '';
  if (!en) {
    sum.innerHTML = '<div class="card"><div class="k">ensemble</div>'+
      '<div class="v mut">n/a</div></div>';
    return;
  }
  const cards = [
    ['decisions', en.n_decisions],
    ['emitted', en.n_emitted],
    ['emit rate', (en.global_emit_rate*100).toFixed(1)+'%'],
    ['top contributors',
     (en.contributors_overall||[]).slice(0,4)
        .map(c=>c.source+':'+c.count).join(' · ') || '—'],
  ];
  cards.forEach(([k,v]) => {
    const el = document.createElement('div'); el.className='card';
    el.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    sum.appendChild(el);
  });
  // per-regime table
  const order = ['trend_lowvol','trend_highvol','range_lowvol','range_highvol','opening','closing'];
  const seen = new Set();
  const orderedRegimes = [];
  order.forEach(r => { if (en.by_regime[r]) { orderedRegimes.push(r); seen.add(r); } });
  Object.keys(en.by_regime).forEach(r => { if (!seen.has(r)) orderedRegimes.push(r); });
  orderedRegimes.forEach(r => {
    const d = en.by_regime[r];
    const wr = d.win_rate===null || d.win_rate===undefined ? '—' : (d.win_rate*100).toFixed(1)+'%';
    const mp = d.mean_pnl_pct===null || d.mean_pnl_pct===undefined ? '—' : (d.mean_pnl_pct*100).toFixed(2)+'%';
    const topC = (d.contributors||[]).slice(0,4)
      .map(c=>`<span class="chip">${c.source}:${c.count}</span>`).join('');
    const row = document.createElement('tr');
    row.innerHTML =
      `<td><span class="pill regime">${r}</span></td>`+
      `<td>${d.n}</td>`+
      `<td>${d.emits}</td>`+
      `<td>${(d.emit_rate*100).toFixed(1)}%</td>`+
      `<td class="mut">${d.matched_trades||0}</td>`+
      `<td class="${d.win_rate>0.5?'pos':(d.win_rate<0.45 && d.win_rate!==null?'neg':'mut')}">${wr}</td>`+
      `<td class="${d.mean_pnl_pct>0?'pos':(d.mean_pnl_pct<0?'neg':'mut')}">${mp}</td>`+
      `<td>${topC}</td>`;
    regTbody.appendChild(row);
  });
  // recent decisions
  (en.recent||[]).forEach(d => {
    const row = document.createElement('tr');
    const pill = d.emitted ?
      '<span class="pill emit">emit</span>' :
      '<span class="pill block">block</span>';
    row.innerHTML =
      `<td class="mut">${(d.ts||'').replace('T',' ').slice(0,19)}</td>`+
      `<td>${d.symbol}</td>`+
      `<td><span class="pill regime">${d.regime}</span></td>`+
      `<td>${pill}</td>`+
      `<td>${d.direction||'—'}</td>`+
      `<td>${d.score===null||d.score===undefined?'—':Number(d.score).toFixed(3)}</td>`+
      `<td class="mut">${d.opposing===null||d.opposing===undefined?'—':Number(d.opposing).toFixed(3)}</td>`+
      `<td class="mut">${d.n_inputs}</td>`+
      `<td class="mut">${d.reason}</td>`;
    recTbody.appendChild(row);
  });
}
function renderHealth(h) {
  const el = document.getElementById('health');
  el.innerHTML = '';
  if (!h) { el.innerHTML = '<div class="card"><div class="k">health</div><div class="v mut">n/a</div></div>'; return; }
  const ts = v => v ? v.replace('T',' ').slice(0,19) : '—';
  const pretty_bytes = n => n < 1024 ? n+'B' : n < 1.05e6 ? (n/1024).toFixed(1)+'K'
      : n < 1.05e9 ? (n/1048576).toFixed(1)+'M' : (n/1073741824).toFixed(2)+'G';
  const cards = [
    ['backend', h.backend],
    ['live trading', h.live_trading ? '🔴 LIVE' : 'paper'],
    ['last tick', ts(h.last_equity_snapshot)],
    ['last trade close', ts(h.last_trade_close)],
    ['open positions', (h.broker_open_positions == null ? 0 : h.broker_open_positions)],
    ['broker cash', h.broker_cash === null ? '—' : '$'+Number(h.broker_cash).toFixed(2)],
    ['day pnl', h.broker_day_pnl === null ? '—' :
        `<span class="${h.broker_day_pnl>=0?'pos':'neg'}">${h.broker_day_pnl>=0?'+':''}${Number(h.broker_day_pnl).toFixed(2)}</span>`],
    ['log size', pretty_bytes(h.log_size_bytes||0)],
  ];
  cards.forEach(([k,v]) => {
    const c = document.createElement('div'); c.className='card';
    c.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    el.appendChild(c);
  });
}

function renderOpenPositions(p) {
  const tb = document.querySelector('#open-pos tbody');
  tb.innerHTML = '';
  if (!p || !p.positions || p.positions.length === 0) {
    tb.innerHTML = '<tr><td colspan="10" class="mut">no open positions</td></tr>';
    return;
  }
  p.positions.forEach(pos => {
    const row = document.createElement('tr');
    row.innerHTML =
      `<td>${pos.symbol}</td>`+
      `<td class="${pos.qty>0?'pos':'neg'}">${pos.qty}</td>`+
      `<td>${fmt(pos.avg_price)}</td>`+
      `<td class="mut">${pos.is_option?(pos.right||'opt'):'stk'}</td>`+
      `<td class="mut">${pos.strike == null ? '—' : pos.strike}</td>`+
      `<td class="mut">${pos.expiry == null ? '—' : pos.expiry}</td>`+
      `<td>${fmt(pos.auto_profit_target)}</td>`+
      `<td>${fmt(pos.auto_stop_loss)}</td>`+
      `<td class="mut">${pos.consecutive_holds == null ? 0 : pos.consecutive_holds}</td>`+
      `<td><span class="chip">${pos.entry_tag||'—'}</span></td>`;
    tb.appendChild(row);
  });
}

function renderRegime(r, h) {
  const el = document.getElementById('regime-now');
  el.innerHTML = '';
  if (!r) { el.innerHTML = '<div class="card"><div class="k">regime</div><div class="v mut">n/a</div></div>'; return; }
  const cards = [
    ['current regime', r.regime ? `<span class="pill regime">${r.regime}</span>` : '—'],
    ['as of', r.as_of ? r.as_of.replace('T',' ').slice(0,19) : '—'],
    ['ensemble', h && h.ensemble_enabled ? '✓' : 'off'],
    ['lstm',     h && h.lstm_enabled     ? '✓' : 'off'],
  ];
  cards.forEach(([k,v]) => {
    const c = document.createElement('div'); c.className='card';
    c.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    el.appendChild(c);
  });
  if (r.recent_distribution && Object.keys(r.recent_distribution).length) {
    const mix = Object.entries(r.recent_distribution)
        .sort((a,b)=>b[1]-a[1])
        .map(([k,v])=>`<span class="chip">${k}: ${v}</span>`).join(' ');
    const c = document.createElement('div'); c.className='card';
    c.style.gridColumn = 'span 4';
    c.innerHTML = `<div class="k">last 200 decisions mix</div><div class="v" style="font-size:14px;">${mix}</div>`;
    el.appendChild(c);
  }
}

function renderAttribution(a) {
  const tb = document.querySelector('#attribution tbody');
  tb.innerHTML = '';
  if (!a || !a.by_entry_tag || a.by_entry_tag.length === 0) {
    tb.innerHTML = '<tr><td colspan="5" class="mut">no trades in window</td></tr>';
    return;
  }
  a.by_entry_tag.forEach(r => {
    const row = document.createElement('tr');
    row.innerHTML =
      `<td><span class="chip">${r.entry_tag}</span></td>`+
      `<td>${r.n}</td>`+
      `<td class="${r.win_rate>0.5?'pos':(r.win_rate<0.45?'neg':'mut')}">${(r.win_rate*100).toFixed(1)}%</td>`+
      `<td class="${r.mean_pnl_pct>0?'pos':'neg'}">${(r.mean_pnl_pct*100).toFixed(2)}%</td>`+
      `<td class="${r.total_pnl>0?'pos':'neg'}">$${Number(r.total_pnl).toFixed(2)}</td>`;
    tb.appendChild(row);
  });
}

function renderMLRecent(ml) {
  const sum = document.getElementById('ml-summary');
  const tb = document.querySelector('#ml-recent tbody');
  sum.innerHTML = '';
  tb.innerHTML = '';
  if (!ml) { sum.innerHTML = '<div class="card"><div class="k">lstm</div><div class="v mut">n/a</div></div>'; return; }
  const acc = ml.overall_accuracy;
  const accCls = acc > 0.4 ? 'pos' : (acc > 0.37 ? 'mut' : 'neg');
  const hitCls = ml.directional_hit_rate > 0.42 ? 'pos' : (ml.directional_hit_rate > 0.38 ? 'mut' : 'neg');
  const wrCls = ml.directional_wrong_side_rate < 0.30 ? 'pos' : (ml.directional_wrong_side_rate < 0.35 ? 'mut' : 'neg');
  const cards = [
    ['resolved', ml.n_resolved],
    ['unresolved', ml.n_unresolved],
    ['overall accuracy', `<span class="${accCls}">${(acc*100).toFixed(1)}%</span>`],
    ['directional hit', `<span class="${hitCls}">${(ml.directional_hit_rate*100).toFixed(1)}%</span>`],
    ['wrong side', `<span class="${wrCls}">${(ml.directional_wrong_side_rate*100).toFixed(1)}%</span>`],
  ];
  cards.forEach(([k,v]) => {
    const c = document.createElement('div'); c.className='card';
    c.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    sum.appendChild(c);
  });
  if (!ml.recent || ml.recent.length === 0) {
    tb.innerHTML = '<tr><td colspan="7" class="mut">no resolved predictions</td></tr>';
    return;
  }
  ml.recent.forEach(p => {
    const row = document.createElement('tr');
    const matchCls = p.pred === p.true ? 'pos' : 'neg';
    row.innerHTML =
      `<td class="mut">${(p.ts||'').replace('T',' ').slice(0,19)}</td>`+
      `<td>${p.symbol}</td>`+
      `<td>${p.pred}</td>`+
      `<td>${(p.confidence*100).toFixed(1)}%</td>`+
      `<td class="${matchCls}">${p.true||'—'}</td>`+
      `<td class="${p.fwd_return>0?'pos':'neg'}">${p.fwd_return===null?'—':(p.fwd_return*100).toFixed(2)+'%'}</td>`+
      `<td class="mut">${p.horizon_min}m</td>`;
    tb.appendChild(row);
  });
}

function renderCatalysts(c) {
  const tb = document.querySelector('#catalysts tbody');
  tb.innerHTML = '';
  if (!c || !c.events || c.events.length === 0) {
    tb.innerHTML = '<tr><td colspan="5" class="mut">no upcoming catalysts (run `tradebotctl.sh catalysts` to refresh)</td></tr>';
    return;
  }
  c.events.slice(0, 40).forEach(e => {
    const row = document.createElement('tr');
    row.innerHTML =
      `<td>${e.symbol}</td>`+
      `<td>${e.date}</td>`+
      `<td><span class="pill regime">${e.type}</span></td>`+
      `<td class="mut">${e.timing}</td>`+
      `<td class="mut">${e.details||''}</td>`;
    tb.appendChild(row);
  });
}

function renderLoopInsights(wd, cal) {
  // --- summary strip: heartbeat, watchdog event counts, calibration ratio ---
  const sum = document.getElementById('loop-summary');
  sum.innerHTML = '';
  const card = (k, v, color) => {
    const el = document.createElement('div'); el.className = 'card';
    const vstyle = color ? ('color:' + color) : '';
    el.innerHTML = `<div class="k">${k}</div><div class="v" style="${vstyle}">${v}</div>`;
    sum.appendChild(el);
  };
  // Heartbeat
  if (wd && wd.heartbeat_status) {
    const age = wd.heartbeat_age_sec;
    let label, color;
    if (wd.heartbeat_status === 'fresh') { label = age.toFixed(0)+'s ago'; color = '#6ee7a0'; }
    else if (wd.heartbeat_status === 'stale') { label = age.toFixed(0)+'s — STALE'; color = '#ff8686'; }
    else { label = 'never written'; color = '#f2c46b'; }
    card('heartbeat', label, color);
    const counts = wd.counts || {};
    card('watchdog events', `${counts.start||0} starts · ${counts.exit||0} exits · ${counts.heartbeat_stale||0} stale`);
  } else {
    card('heartbeat', '(watchdog not running)', '#f2c46b');
    card('watchdog events', '—');
  }
  // Calibration
  if (cal && cal.stats) {
    const s = cal.stats;
    const ratio = s.ratio;
    let color = '#cbd5ff';
    if (ratio >= 0.8 && ratio <= 1.2) color = '#6ee7a0';
    else if (ratio > 1.5 || (ratio > 0 && ratio < 0.5)) color = '#ff8686';
    else color = '#f2c46b';
    card('slippage ratio (obs/pred)',
      `${ratio.toFixed(2)}  ·  ${s.keep_or_tune}`, color);
    card('calibration n',
      `${s.n} fills (${s.mean_observed_bps.toFixed(1)} bps observed)`);
  } else {
    card('slippage ratio (obs/pred)', '(no calibration data yet)', '#f2c46b');
    card('calibration n', '—');
  }

  // --- recent watchdog events table ---
  const evTb = document.querySelector('#wd-events tbody');
  evTb.innerHTML = '';
  const events = (wd && wd.recent_events) || [];
  if (events.length === 0) {
    evTb.innerHTML = '<tr><td colspan="3" class="mut">no watchdog events yet — install via <code>tradebotctl watchdog-install</code></td></tr>';
  } else {
    events.slice(0, 15).forEach(e => {
      const detailParts = [];
      if (e.exit_code !== undefined) detailParts.push('rc=' + e.exit_code);
      if (e.duration_sec !== undefined) detailParts.push(e.duration_sec.toFixed(0) + 's');
      if (e.age_sec !== undefined) detailParts.push('age=' + e.age_sec);
      if (e.pid) detailParts.push('pid=' + e.pid);
      const tint = e.kind === 'heartbeat_stale' || e.kind === 'exit' && e.exit_code !== 0 ? '#ff8686'
                 : e.kind === 'clean_shutdown' ? '#cbd5ff' : '#6ee7a0';
      const row = document.createElement('tr');
      row.innerHTML =
        `<td class="mut">${(e.ts||'').replace('T',' ').slice(0,19)}</td>`+
        `<td style="color:${tint}">${e.kind}</td>`+
        `<td class="mut">${detailParts.join(' · ')}</td>`;
      evTb.appendChild(row);
    });
  }

  // --- calibration history table ---
  const calTb = document.querySelector('#cal-history tbody');
  calTb.innerHTML = '';
  const hist = (cal && cal.auto_history) || [];
  if (hist.length === 0) {
    calTb.innerHTML = '<tr><td colspan="4" class="mut">no auto-calibration cycles yet (needs 30+ fills)</td></tr>';
  } else {
    hist.slice(0, 15).forEach(h => {
      const changes = h.changes || {};
      const changeTxt = Object.keys(changes).length === 0
        ? '(no change — within tolerance)'
        : Object.entries(changes).map(([k,v]) => `${k}: ${v.old}→${v.new}`).join(', ');
      const ratio = h.ratio;
      let color = '#cbd5ff';
      if (ratio >= 0.8 && ratio <= 1.2) color = '#6ee7a0';
      else if (ratio > 1.5 || (ratio > 0 && ratio < 0.5)) color = '#ff8686';
      else color = '#f2c46b';
      const row = document.createElement('tr');
      row.innerHTML =
        `<td class="mut">${(h.ts||'').replace('T',' ').slice(0,19)}</td>`+
        `<td>${h.n_fills == null ? '' : h.n_fills}</td>`+
        `<td style="color:${color}">${ratio !== undefined ? ratio.toFixed(2) : ''}</td>`+
        `<td class="mut" style="max-width:340px;white-space:normal;">${changeTxt}</td>`;
      calTb.appendChild(row);
    });
  }
}

async function refreshLogs() {
  const q = document.getElementById('log-grep').value;
  const url = '/api/logs_tail?lines=200' + (q ? '&grep='+encodeURIComponent(q) : '');
  const r = await fetch(url).then(r=>r.json()).catch(()=>null);
  const view = document.getElementById('log-view');
  if (!r || r.missing) { view.textContent = '(no log file)'; return; }
  view.textContent = (r.lines || []).join('\\n') || '(empty)';
  view.scrollTop = view.scrollHeight;
}
document.getElementById('log-grep').addEventListener('keyup', e => {
  if (e.key === 'Enter') refreshLogs();
});
document.getElementById('log-refresh').addEventListener('click', refreshLogs);

document.getElementById('days').addEventListener('change', refresh);
refresh(); refreshLogs();
setInterval(refresh, 60000);
setInterval(refreshLogs, 15000);
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_INDEX_HTML)
