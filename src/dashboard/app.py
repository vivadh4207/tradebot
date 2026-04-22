"""FastAPI read-only dashboard: equity curve, trades, open positions.

Reads the local SQLite journal. No authentication — bind to localhost
and reach it through an SSH tunnel or a reverse-proxy with auth. Do
NOT expose to the public internet.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import os
from collections import Counter, defaultdict

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
import subprocess
import shlex

from ..core.config import load_settings
from ..storage.journal import build_journal
from ..storage.position_snapshot import load_snapshot


def _load_journal():
    root = Path(__file__).resolve().parents[2]
    s = load_settings(root / "config" / "settings.yaml")
    return build_journal(
        sqlite_path=s.get("storage.sqlite_path",
                           str(root / "logs" / "tradebot.sqlite")),
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
    """Currently open positions — TRADIER is source of truth, not
    local broker_state.json. Operator lost money last time because
    local showed clean while Tradier had 5 orphaned positions
    accumulating losses. We reverse that: Tradier is primary,
    local is an overlay with auto_pt / auto_sl / entry_tag.

    Returns:
      positions: list of Tradier positions enriched with local metadata
                 (auto_pt, auto_sl, entry_tag) where they match by symbol.
      sync: {status, local_only, tradier_only, in_both} — reconcile diff.
      source: 'tradier' | 'local_fallback'."""
    import re as _re
    from pathlib import Path as _P

    s, root = _settings()
    # Load local snapshot for metadata overlay
    local_snap = load_snapshot(s.get("broker.snapshot_path",
                                        str(root / "logs" / "broker_state.json")))
    local_by_sym: Dict[str, Any] = {}
    if local_snap is not None:
        for p in local_snap.positions:
            local_by_sym[p.symbol] = p

    # Query Tradier for truth
    tradier_positions: List[Any] = []
    tradier_error: Optional[str] = None
    try:
        from ..brokers.tradier_adapter import build_tradier_broker
        tb = build_tradier_broker()
        if tb is not None:
            tradier_positions = list(tb.positions())
    except Exception as e:                                  # noqa: BLE001
        tradier_error = str(e)[:200]

    # Get current marks so unrealized P&L is accurate
    marks: Dict[str, float] = {}
    try:
        from ..data.multi_provider import MultiProvider
        mp = MultiProvider.from_env()
        for p in tradier_positions:
            try:
                q = mp.latest_quote(p.symbol)
                if q is not None and q.mid and q.mid > 0:
                    marks[p.symbol] = float(q.mid)
            except Exception:
                pass
    except Exception:
        pass

    # Build Tradier-primary list, enriched with local metadata
    positions = []
    for p in tradier_positions:
        mark = marks.get(p.symbol)
        unrl = None
        if mark is not None and p.avg_price > 0:
            unrl = (mark - p.avg_price) * abs(p.qty) * p.multiplier * (
                1 if p.qty > 0 else -1
            )
        # Extract underlying from OCC if option
        underlying = p.symbol
        if p.is_option:
            m = _re.match(r"^([A-Z]{1,6})\d{6}[CP]\d{8}$", p.symbol)
            if m:
                underlying = m.group(1)
        # Overlay local metadata if symbol matches
        overlay = local_by_sym.get(p.symbol)
        auto_pt = overlay.auto_profit_target if overlay else None
        auto_sl = overlay.auto_stop_loss if overlay else None
        entry_tag = overlay.entry_tag if overlay else ""
        positions.append({
            "symbol": p.symbol,
            "qty": p.qty,
            "avg_price": p.avg_price,
            "mark": mark,
            "unrealized_pnl": round(unrl, 2) if unrl is not None else None,
            "is_option": p.is_option,
            "underlying": underlying,
            "multiplier": p.multiplier,
            "auto_profit_target": auto_pt,
            "auto_stop_loss": auto_sl,
            "entry_tag": entry_tag,
            "in_local": p.symbol in local_by_sym,
        })

    # Compute reconcile diff — the operator NEEDS to see this
    tradier_syms = {p.symbol for p in tradier_positions}
    local_syms = set(local_by_sym.keys())
    local_only = sorted(local_syms - tradier_syms)
    tradier_only = sorted(tradier_syms - local_syms)
    in_both = sorted(tradier_syms & local_syms)
    sync_status = "in_sync"
    if local_only or tradier_only:
        sync_status = "desynced"
    if tradier_error:
        sync_status = "tradier_unreachable"

    # Day P&L — prefer Tradier account if reachable, else local
    cash = None
    day_pnl = None
    try:
        if tb is not None:
            acct = tb.account()
            # Tradier AccountSummary dataclass has these fields (check
            # defensively since legacy code had a bug here):
            cash = float(getattr(acct, "cash", 0) or 0)
            day_pnl = float(getattr(acct, "day_pnl", 0) or 0)
    except Exception:
        pass
    if cash is None and local_snap is not None:
        cash = local_snap.cash
        day_pnl = local_snap.day_pnl

    return {
        "positions": positions,
        "cash": cash, "day_pnl": day_pnl,
        "sync": {
            "status": sync_status,
            "tradier_n": len(tradier_positions),
            "local_n": len(local_by_sym),
            "local_only": local_only,
            "tradier_only": tradier_only,
            "in_both": in_both,
            "tradier_error": tradier_error,
        },
        "source": ("tradier" if tradier_positions else
                     ("local_fallback" if local_snap else "empty")),
    }


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


@app.get("/api/signal_audit", response_class=JSONResponse)
def signal_audit_tail(limit: int = Query(100, ge=1, le=2000)):
    """Tail of the per-signal audit log. Requires TRADEBOT_SIGNAL_AUDIT=1
    at bot + dashboard startup; otherwise returns empty."""
    from src.core.signal_audit import read_tail
    return {"entries": read_tail(limit)}


@app.get("/api/llm_brain_tail", response_class=JSONResponse)
def llm_brain_tail(limit: int = Query(50, ge=1, le=500)):
    """Most recent LLM brain reviews from the signal_audit log. Filters
    down to source='llm_brain'."""
    from src.core.signal_audit import read_tail
    all_entries = read_tail(limit * 20)   # oversample, then filter
    brain = [e for e in all_entries if e.get("source") == "llm_brain"]
    return {"entries": brain[-limit:]}


@app.get("/api/strategy_audit", response_class=JSONResponse)
def strategy_audit_tail(limit: int = Query(20, ge=1, le=200)):
    """Most recent strategy-audit reports from the 70B auditor."""
    from src.intelligence.strategy_auditor import read_recent_audits
    return {"reports": read_recent_audits(limit)}


@app.post("/api/bot/strategy_audit", response_class=JSONResponse)
def run_strategy_audit():
    """On-demand strategy audit. Kicks off the 70B run synchronously;
    the endpoint can take 30-120s depending on GPU + context size."""
    if not _dashboard_controls_enabled():
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error":
                     "dashboard controls disabled — set TRADEBOT_DASHBOARD_CONTROLS=1 in .env"},
        )
    return _run_ctl("strategy-audit", timeout=300.0)


@app.get("/api/logs_tail", response_class=JSONResponse)
def logs_tail(lines: int = Query(80, ge=10, le=5000),
               grep: str = Query("", max_length=256),
               kinds: str = Query("all")):
    """Tail the bot's log and STRUCTURE each line into a readable event.
    Each returned item:
        ts, level, event, kv (dict of key=value pairs), human (string),
        icon (emoji tag), raw
    Filters via `grep` (substring) and `kinds` (comma-sep event categories):
      fills / exits / signals / warnings / errors / all
    """
    import re as _re
    from pathlib import Path
    s, root = _settings()
    log_path = root / "logs" / "tradebot.out"
    if not log_path.exists():
        return {"events": [], "path": str(log_path), "missing": True}
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = 128 * 1024
            buf = b""
            pos = size
            collected = 0
            while pos > 0 and collected <= lines * 4:
                read_size = min(chunk, pos)
                pos -= read_size
                f.seek(pos)
                buf = f.read(read_size) + buf
                collected = buf.count(b"\n")
            text = buf.decode("utf-8", errors="replace")
    except Exception as e:
        return {"events": [], "path": str(log_path), "error": str(e)}

    # Strip ANSI colour codes
    ansi_re = _re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    # structured-log format:  ISO_TS  [level ]  event   key=val key=val...
    line_re = _re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+"
        r"\[(?P<level>\w+)\s*\]\s+(?P<event>\S+)\s*(?P<rest>.*)$"
    )
    kv_re = _re.compile(r"(\w+)=(?:'([^']*)'|\"([^\"]*)\"|(\S+))")

    out: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        clean = ansi_re.sub("", raw_line).strip()
        if not clean:
            continue
        m = line_re.match(clean)
        if not m:
            continue
        ts = m.group("ts")
        level = m.group("level").lower()
        event = m.group("event")
        rest = m.group("rest") or ""
        kv: Dict[str, str] = {}
        for km in kv_re.finditer(rest):
            val = km.group(2) if km.group(2) is not None else (
                km.group(3) if km.group(3) is not None else km.group(4)
            )
            kv[km.group(1)] = val

        # Classify + build human description
        icon, kind, human = _humanize_event(event, level, kv)

        out.append({
            "ts": ts, "level": level, "event": event,
            "icon": icon, "kind": kind, "human": human,
            "kv": kv, "raw": clean[:400],
        })

    # Filter by kind
    kinds_set = set(k.strip() for k in kinds.lower().split(",") if k.strip())
    if kinds_set and "all" not in kinds_set:
        out = [e for e in out if e["kind"] in kinds_set]
    if grep:
        g = grep.lower()
        out = [e for e in out
                if g in e["event"].lower() or g in e["human"].lower()]

    # Most-recent last
    out = out[-lines:]
    return {"events": out, "path": str(log_path), "missing": False,
            "total": len(out)}


def _humanize_event(event: str, level: str, kv: Dict[str, str]
                    ) -> tuple:
    """Convert a structured log event into a human-readable one-liner
    + emoji icon + kind tag (fills / exits / signals / warnings)."""
    e = event.lower()
    sym = kv.get("symbol") or kv.get("sym") or ""
    price = kv.get("price") or ""

    # Entries / fills
    if e == "fill":
        right = (kv.get("right") or "").upper()
        strike = kv.get("strike", "?")
        dte = kv.get("dte", "")
        dte_txt = f" · {dte}d" if dte else ""
        return ("🟢", "fills",
                f"ENTRY · {sym} {right} ${strike}{dte_txt} @ ${price} "
                f"qty={kv.get('qty', '?')}")
    if e == "fast_exit":
        reason = (kv.get("reason") or "").strip("'\"")
        reason_head = reason.split(":")[0] if reason else "exit"
        pnl = kv.get("pnl_pct", "")
        usd = kv.get("realized_usd", "")
        usd_txt = f" ${usd}" if usd else ""
        pnl_txt = f" ({pnl}%)" if pnl else ""
        return ("🔴", "exits",
                f"EXIT · {sym} {reason_head}{pnl_txt}{usd_txt}")
    if e == "eod_force_close" or (e == "fast_exit" and "eod" in (kv.get("reason") or "")):
        return ("🌇", "exits", f"EOD close · {sym}")

    # Signals
    if e == "ensemble_emit":
        direction = kv.get("direction", "?")
        score = kv.get("score", "?")
        regime = kv.get("regime", "")
        icon = "🔼" if direction == "bullish" else "🔽" if direction == "bearish" else "•"
        return (icon, "signals",
                f"Signal · {sym} {direction} score {score}"
                + (f" [{regime}]" if regime else ""))
    if e == "ensemble_skip":
        reason = (kv.get("reason") or "").strip("'\"").split(":")[0]
        return ("⏭", "signals", f"Skip · {sym} {reason}")
    if e == "exec_chain_block":
        filt = kv.get("filter", "")
        reason = (kv.get("reason") or "").strip("'\"")
        return ("🛑", "signals",
                f"Block · {sym} {filt} — {reason[:80]}")
    if e == "exec_chain_pass":
        src = kv.get("signal") or kv.get("src") or ""
        return ("✅", "signals", f"Pass · {sym} [{src}]")

    # State
    if e == "market_state_snapshot":
        return ("📊", "state",
                f"State · regime={kv.get('regime', '?')} "
                f"vix={kv.get('vix', '?')} "
                f"breadth={kv.get('breadth_score', '?')}")
    if e == "data_adapter":
        return ("🔌", "state", f"Data adapter: {kv.get('kind', '?')}")
    if e == "strategy_mode":
        return ("🎯", "state", f"Mode: {kv.get('mode', '?')}")

    # Warnings / errors
    if level in ("warning", "warn", "error"):
        # Generic fallback with event + top kv
        top = " · ".join(f"{k}={v[:30]}" for k, v in list(kv.items())[:3])
        return ("⚠️" if level.startswith("warn") else "🛑",
                "warnings" if level.startswith("warn") else "errors",
                f"{event} · {top}")

    # Default passthrough
    top = " · ".join(f"{k}={v[:30]}" for k, v in list(kv.items())[:3])
    return ("·", "other", f"{event}" + (f" · {top}" if top else ""))


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


def _ctl_path() -> Path:
    """Path to tradebotctl.sh. Resolved from repo root, not CWD."""
    return Path(__file__).resolve().parents[2] / "scripts" / "tradebotctl.sh"


def _dashboard_controls_enabled() -> bool:
    """Dashboard start/stop buttons disabled unless explicitly opted in.

    Safety: the dashboard has no auth. Exposing process control on an
    open port would let anyone on the network stop the bot. Default is
    OFF; set TRADEBOT_DASHBOARD_CONTROLS=1 in .env to enable.
    """
    return os.getenv("TRADEBOT_DASHBOARD_CONTROLS", "").strip() in ("1", "true", "yes", "on")


def _run_ctl(action: str, timeout: float = 25.0,
              extra_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Shell out to tradebotctl.sh and return {ok, stdout, stderr, rc}.

    Supervisor-aware: tradebotctl internally detects whether the watchdog
    is running under launchd (Mac) or systemd --user (Linux) and routes
    start/stop accordingly instead of spawning a second bot alongside
    the supervised one.

    Whitelist prevents an attacker who bypasses routing from invoking
    arbitrary shell commands via the action string.
    """
    allowed = {
        "start", "stop", "restart", "status",
        "wipe-journal", "reset-paper",
        "walkforward", "putcall-oi", "doctor",
        "strategy-audit",
    }
    if action not in allowed:
        return {"ok": False, "error": f"action not allowed: {action}"}
    ctl = _ctl_path()
    if not ctl.exists():
        return {"ok": False, "error": f"tradebotctl.sh not found at {ctl}"}
    cmd = ["/bin/bash", str(ctl), action]
    if extra_args:
        # Extra args are appended as-is but each one is passed via list,
        # so shell-metacharacter injection is impossible.
        cmd.extend(str(a) for a in extra_args)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ctl.parent.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "action": action,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"{action} timed out after {timeout}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/bot/status", response_class=JSONResponse)
def bot_status():
    """Is the paper bot process currently running? Uses tradebotctl status
    which checks the PID file (portable on Mac + Linux)."""
    result = _run_ctl("status", timeout=5.0)
    # Parse "running (pid 12345)" vs "stopped"
    out = (result.get("stdout") or "").lower()
    state = "running" if "running" in out else "stopped"
    return {
        "state": state,
        "raw": result.get("stdout"),
        "controls_enabled": _dashboard_controls_enabled(),
    }


@app.post("/api/bot/start", response_class=JSONResponse)
def bot_start():
    """Start the paper bot. Requires TRADEBOT_DASHBOARD_CONTROLS=1."""
    if not _dashboard_controls_enabled():
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error":
                     "dashboard controls disabled — set TRADEBOT_DASHBOARD_CONTROLS=1 in .env"},
        )
    return _run_ctl("start")


@app.post("/api/bot/stop", response_class=JSONResponse)
def bot_stop():
    """Stop the paper bot. Cooperative shutdown: writes KILL file + SIGTERM."""
    if not _dashboard_controls_enabled():
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error":
                     "dashboard controls disabled — set TRADEBOT_DASHBOARD_CONTROLS=1 in .env"},
        )
    return _run_ctl("stop")


@app.post("/api/bot/restart", response_class=JSONResponse)
def bot_restart():
    """Stop + start. Useful after config changes."""
    if not _dashboard_controls_enabled():
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error":
                     "dashboard controls disabled — set TRADEBOT_DASHBOARD_CONTROLS=1 in .env"},
        )
    return _run_ctl("restart", timeout=45.0)


@app.post("/api/bot/reset_paper", response_class=JSONResponse)
def bot_reset_paper():
    """Nuclear reset: flatten Alpaca paper positions + wipe journal +
    restart the bot with a clean slate. DANGEROUS — wipes trade history.
    Requires TRADEBOT_DASHBOARD_CONTROLS=1."""
    if not _dashboard_controls_enabled():
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error":
                     "dashboard controls disabled — set TRADEBOT_DASHBOARD_CONTROLS=1 in .env"},
        )
    # reset_paper.py accepts --yes to skip its interactive confirm;
    # without it the script refuses to run. We pass it explicitly
    # because the UI button already required a confirm dialog.
    return _run_ctl("reset-paper", timeout=120.0, extra_args=["--yes"])


@app.post("/api/bot/walkforward", response_class=JSONResponse)
def bot_walkforward():
    """Run the nightly walk-forward edge report on demand. Posts the
    result to the #tradebot-reason Discord channel."""
    if not _dashboard_controls_enabled():
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error":
                     "dashboard controls disabled — set TRADEBOT_DASHBOARD_CONTROLS=1 in .env"},
        )
    return _run_ctl("walkforward", timeout=180.0)


@app.post("/webhook/tradingview", response_class=JSONResponse)
async def tradingview_webhook(req: Request):
    """Receive an alert from TradingView. Validated by shared secret,
    schema-checked, and appended to the bot's signal queue.

    TradingView alert 'Message' field should be JSON, e.g.
      {
        "secret": "your-long-shared-secret",
        "symbol": "SPY",
        "side": "buy",
        "reason": "rsi_oversold + vwap_reversion",
        "confidence": 0.7,
        "tf": "5m",
        "spot": {{close}},
        "ts": "{{time}}"
      }
    Configure the webhook URL as http://<host>:8000/webhook/tradingview
    (bound to localhost by default — use an SSH tunnel or a reverse-
    proxy with auth to expose it to TradingView's servers).
    """
    from src.signals.tradingview_webhook import ingest
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(status_code=400,
                             content={"ok": False, "error": "invalid JSON body"})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400,
                             content={"ok": False, "error": "body must be a JSON object"})
    result = ingest(body)
    return JSONResponse(
        status_code=result.status_code,
        content={"ok": result.ok, "error": result.error,
                 "alert_id": result.alert_id},
    )


@app.post("/api/bot/refresh_risk_switch", response_class=JSONResponse)
def bot_refresh_risk_switch():
    """Pull today's CBOE put/call OI and recompute the macro risk-off
    state. Normally driven by cron; this button lets the operator
    refresh it on demand."""
    if not _dashboard_controls_enabled():
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error":
                     "dashboard controls disabled — set TRADEBOT_DASHBOARD_CONTROLS=1 in .env"},
        )
    return _run_ctl("putcall-oi", timeout=30.0)


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
  .ctl-row { display: flex; gap: 10px; align-items: center; margin: 8px 0 18px;
             flex-wrap: wrap; padding: 12px 14px; background: #141a33;
             border: 1px solid #223; border-radius: 10px; }
  .ctl-row .label { color: #9aa3c7; font-size: 11px; text-transform: uppercase;
                    letter-spacing: .08em; margin-right: 4px; }
  .ctl-btn { background: #2b3360; color: #e6e9f5; border: 0; padding: 7px 18px;
             border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; }
  .ctl-btn:hover:not(:disabled) { background: #3a4680; }
  .ctl-btn.start { background: #1f4d37; }
  .ctl-btn.start:hover:not(:disabled) { background: #2a6b4c; }
  .ctl-btn.stop  { background: #5a2633; }
  .ctl-btn.stop:hover:not(:disabled)  { background: #7a3344; }
  .ctl-btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .ctl-state { font-family: ui-monospace, monospace; font-size: 12px;
               padding: 3px 9px; border-radius: 10px; }
  .ctl-state.running { background: #14291e; color: #6ee7b7; }
  .ctl-state.stopped { background: #2a1a1f; color: #fca5a5; }
  .ctl-state.unknown { background: #2b2f44; color: #9aa3c7; }
  .ctl-msg { color: #9aa3c7; font-size: 12px; font-family: ui-monospace, monospace; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head><body>
<h1>tradebot &nbsp;·&nbsp; <span class="mut">executive view</span></h1>

<div class="ctl-row" id="ctl-row">
  <span class="label">bot process</span>
  <span id="ctl-state" class="ctl-state unknown">checking…</span>
  <button id="ctl-start"   class="ctl-btn start" disabled>start</button>
  <button id="ctl-stop"    class="ctl-btn stop"  disabled>stop</button>
  <button id="ctl-restart" class="ctl-btn"       disabled>restart</button>
  <span style="width: 18px;"></span>
  <span class="label">flow</span>
  <button id="ctl-walkforward" class="ctl-btn" disabled>run walkforward</button>
  <button id="ctl-riskswitch"  class="ctl-btn" disabled>refresh risk switch</button>
  <button id="ctl-reset"       class="ctl-btn stop" disabled>reset paper</button>
  <span style="width: 18px;"></span>
  <span class="label">LLM</span>
  <button id="ctl-audit"       class="ctl-btn" disabled>run 70B audit</button>
  <span id="ctl-msg" class="ctl-msg"></span>
</div>

<h2>strategy audit &nbsp;·&nbsp; <span class="mut">70B LLM review of the whole setup</span></h2>
<div class="card" id="audit-latest">
  <div class="mut" style="font-size:12px;">No audits yet — click "run 70B audit" above, or wait for the nightly cron.</div>
</div>
<div class="card" style="margin-top:8px;">
  <div class="mut" style="margin-bottom:6px;font-size:12px;">recent audits</div>
  <table id="audit-history"><thead><tr>
    <th>time</th><th>health</th><th>issues</th><th>summary</th>
  </tr></thead><tbody></tbody></table>
</div>

<h2>LLM brain reviews &nbsp;·&nbsp; <span class="mut">per-trade 8B sanity check</span></h2>
<div class="card"><table id="brain-tail"><thead><tr>
  <th>time</th><th>symbol</th><th>action</th><th>mult</th><th>latency</th><th>cached</th><th>reason</th>
</tr></thead><tbody></tbody></table></div>

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

// ---- bot process controls + full-flow actions ----
const CTL_BTN_IDS = [
  'ctl-start', 'ctl-stop', 'ctl-restart',
  'ctl-walkforward', 'ctl-riskswitch', 'ctl-reset',
  'ctl-audit',
];
async function refreshBotStatus() {
  const stateEl = document.getElementById('ctl-state');
  const msgEl   = document.getElementById('ctl-msg');
  const start   = document.getElementById('ctl-start');
  const stop    = document.getElementById('ctl-stop');
  const restart = document.getElementById('ctl-restart');
  const walk    = document.getElementById('ctl-walkforward');
  const risk    = document.getElementById('ctl-riskswitch');
  const reset   = document.getElementById('ctl-reset');
  const audit   = document.getElementById('ctl-audit');
  try {
    const r = await fetch('/api/bot/status').then(r=>r.json());
    const running = r.state === 'running';
    stateEl.className = 'ctl-state ' + (running ? 'running' : 'stopped');
    stateEl.textContent = r.raw || (running ? 'running' : 'stopped');
    if (!r.controls_enabled) {
      CTL_BTN_IDS.forEach(id => document.getElementById(id).disabled = true);
      msgEl.textContent = 'controls disabled — set TRADEBOT_DASHBOARD_CONTROLS=1 in .env + restart dashboard';
      return;
    }
    start.disabled   = running;
    stop.disabled    = !running;
    restart.disabled = false;
    // flow actions are always available when controls are enabled —
    // they run regardless of bot state (walkforward reads the journal,
    // riskswitch pulls daily data, reset is its own thing)
    walk.disabled = false;
    risk.disabled = false;
    reset.disabled = false;
    audit.disabled = false;
    msgEl.textContent = '';
  } catch (e) {
    stateEl.className = 'ctl-state unknown';
    stateEl.textContent = 'error';
    msgEl.textContent = String(e);
  }
}
async function botAction(path, confirmMsg, timeoutHint) {
  if (confirmMsg && !window.confirm(confirmMsg)) return;
  const msgEl = document.getElementById('ctl-msg');
  CTL_BTN_IDS.forEach(id => document.getElementById(id).disabled = true);
  msgEl.textContent = (timeoutHint || (path + '…'));
  try {
    const r = await fetch('/api/bot/'+path, {method:'POST'}).then(r=>r.json());
    const txt = (r.stdout || r.error || '') + (r.stderr ? (' · ' + r.stderr) : '');
    const brief = (txt || (r.ok ? 'ok' : 'failed')).split('\\n').slice(0, 3).join(' · ');
    msgEl.textContent = brief;
  } catch (e) {
    msgEl.textContent = String(e);
  }
  // small delay so any state file catches up, then refresh
  setTimeout(refreshBotStatus, 1500);
}
document.getElementById('ctl-start').addEventListener('click',
  () => botAction('start'));
document.getElementById('ctl-stop').addEventListener('click',
  () => botAction('stop', 'Stop the paper bot process?\\nOpen positions stay open in the journal but stops will not be monitored until restart.'));
document.getElementById('ctl-restart').addEventListener('click',
  () => botAction('restart', 'Restart the paper bot?\\n(stop + start — loads the latest config)'));
document.getElementById('ctl-walkforward').addEventListener('click',
  () => botAction('walkforward', null, 'running walkforward (takes ~1 min)…'));
document.getElementById('ctl-riskswitch').addEventListener('click',
  () => botAction('refresh_risk_switch', null, 'pulling CBOE put/call OI…'));
document.getElementById('ctl-reset').addEventListener('click',
  () => botAction('reset_paper',
    'DANGEROUS: flattens all Alpaca paper positions AND truncates the trade journal. This is irreversible.\\n\\nContinue?',
    'resetting paper account (this takes ~30s)…'));
document.getElementById('ctl-audit').addEventListener('click',
  () => botAction('strategy_audit', null,
    'running 70B strategy audit (this takes 30-120s)…'));
refreshBotStatus();
setInterval(refreshBotStatus, 10000);

// ---- strategy audit panels ----
async function refreshAuditPanels() {
  try {
    const r = await fetch('/api/strategy_audit?limit=20').then(r=>r.json());
    const reports = (r && r.reports) || [];
    const latestCard = document.getElementById('audit-latest');
    const histTbody = document.querySelector('#audit-history tbody');
    if (reports.length === 0) {
      latestCard.innerHTML = '<div class="mut" style="font-size:12px;">No audits yet — click "run 70B audit" above, or wait for the nightly cron.</div>';
      histTbody.innerHTML = '';
      return;
    }
    const latest = reports[0];
    const cls = latest.overall_health >= 80 ? 'pos' :
                latest.overall_health >= 50 ? 'mut' : 'neg';
    let issuesHtml = '';
    (latest.issues || []).slice(0, 5).forEach(i => {
      const sevColor = i.severity === 'high' ? '#fca5a5' :
                        i.severity === 'medium' ? '#f2c46b' : '#9aa3c7';
      issuesHtml += `<div style="margin:6px 0;padding:6px 8px;background:#141a33;border-left:3px solid ${sevColor};border-radius:4px;">`
                  + `<div style="font-size:11px;color:${sevColor};text-transform:uppercase;">${i.severity} · ${i.area}</div>`
                  + `<div style="font-size:13px;margin-top:2px;">${(i.detail||'').replace(/</g,'&lt;')}</div>`
                  + (i.fix ? `<div style="font-size:12px;margin-top:3px;color:#6ee7b7;">Fix: ${(i.fix||'').replace(/</g,'&lt;')}</div>` : '')
                  + `</div>`;
    });
    let strengthsHtml = '';
    (latest.strengths || []).forEach(s => {
      strengthsHtml += `<span class="chip" style="margin:2px;">${(s||'').replace(/</g,'&lt;')}</span>`;
    });
    latestCard.innerHTML =
      `<div style="display:flex;align-items:baseline;gap:12px;">`+
        `<div class="v" style="font-size:32px;font-weight:600;" class="${cls}">${latest.overall_health}<span style="font-size:14px;color:#9aa3c7;">/100</span></div>`+
        `<div class="mut" style="font-size:12px;">${(latest.ts||'').replace('T',' ').slice(0,19)} · ${latest.model} · ${latest.latency_sec}s</div>`+
      `</div>`+
      `<div style="margin-top:6px;font-size:14px;">${(latest.summary||'').replace(/</g,'&lt;')}</div>`+
      (issuesHtml ? `<div style="margin-top:10px;"><div class="mut" style="font-size:11px;text-transform:uppercase;">issues</div>${issuesHtml}</div>` : '')+
      (strengthsHtml ? `<div style="margin-top:10px;"><div class="mut" style="font-size:11px;text-transform:uppercase;">strengths</div><div style="margin-top:4px;">${strengthsHtml}</div></div>` : '');
    histTbody.innerHTML = '';
    reports.slice(0, 20).forEach(rep => {
      const c = rep.overall_health >= 80 ? 'pos' :
                 rep.overall_health >= 50 ? 'mut' : 'neg';
      const nIssues = (rep.issues||[]).length;
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td class="mut">${(rep.ts||'').replace('T',' ').slice(0,19)}</td>`+
        `<td class="${c}">${rep.overall_health}</td>`+
        `<td>${nIssues}</td>`+
        `<td class="mut">${(rep.summary||'').substring(0,120)}</td>`;
      histTbody.appendChild(tr);
    });
  } catch (e) {
    // silent — no audits is a normal state
  }
}

// ---- LLM brain tail ----
async function refreshBrainTail() {
  try {
    const r = await fetch('/api/llm_brain_tail?limit=40').then(r=>r.json());
    const tbody = document.querySelector('#brain-tail tbody');
    tbody.innerHTML = '';
    const entries = (r && r.entries) || [];
    if (entries.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="mut">No LLM reviews yet. Enable with LLM_BRAIN_ENABLED=1 + TRADEBOT_SIGNAL_AUDIT=1 and restart the bot.</td></tr>';
      return;
    }
    entries.slice().reverse().forEach(e => {
      const meta = e.meta || {};
      const action = meta.action || '?';
      const cls = action === 'veto' ? 'neg' :
                   action === 'confirm' ? 'pos' : 'mut';
      const mult = e.confidence != null ? Number(e.confidence).toFixed(2)+'x' : '—';
      const cached = meta.from_cache ? '✓' : '';
      const latency = meta.latency_ms != null ? meta.latency_ms + 'ms' : '—';
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td class="mut">${(e.ts||'').replace('T',' ').slice(0,19)}</td>`+
        `<td>${e.symbol||'—'}</td>`+
        `<td class="${cls}">${action}</td>`+
        `<td>${mult}</td>`+
        `<td class="mut">${latency}</td>`+
        `<td class="mut">${cached}</td>`+
        `<td class="mut">${(e.rationale||'').replace(/</g,'&lt;')}</td>`;
      tbody.appendChild(tr);
    });
  } catch (e) {
    // silent
  }
}

refreshAuditPanels();
refreshBrainTail();
setInterval(refreshAuditPanels, 60000);
setInterval(refreshBrainTail, 15000);
</script>
</body></html>
"""


# ---------------------------------------------------------------
# Advanced endpoints — saves tracker, position advisories,
# strategy buckets, LLM chat, on-demand research / catalyst / scan,
# cleanup. Exposes the Discord flows via HTTP for the dashboard.
# ---------------------------------------------------------------


@app.get("/api/saves", response_class=JSONResponse)
def saves_api(hours: int = Query(24, ge=1, le=720)):
    """Saves tracker summary — defensive exits + $ saved vs. regret."""
    try:
        from ..intelligence.saves_tracker import summary
        return summary(since_hours=int(hours))
    except Exception as e:                                  # noqa: BLE001
        return {"error": str(e)[:200], "n_exits": 0}


@app.get("/api/advisories", response_class=JSONResponse)
def advisories_api():
    """Active position-fade advisories (last 6h)."""
    try:
        from ..intelligence.position_advisor import _store_path
        from pathlib import Path as _P
        import json as _json, time as _t
        p = _P(str(_store_path()))
        if not p.exists():
            return {"advisories": []}
        d = _json.loads(p.read_text() or "{}")
        cutoff = _t.time() - 6 * 3600
        out = []
        for aid, rec in d.items():
            if float(rec.get("ts", 0)) < cutoff:
                continue
            rec["id"] = aid
            out.append(rec)
        out.sort(key=lambda r: -float(r.get("ts", 0)))
        return {"advisories": out[:20]}
    except Exception as e:                                  # noqa: BLE001
        return {"error": str(e)[:200], "advisories": []}


@app.get("/api/strategy", response_class=JSONResponse)
def strategy_buckets_api(days: int = Query(7, ge=1, le=365)):
    """Per-bucket P&L (0DTE / short / swing)."""
    from datetime import datetime, timedelta, timezone
    import re as _re
    from collections import defaultdict as _dd
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        trades = j.closed_trades(since=since)
    finally:
        j.close()
    buckets = _dd(lambda: {"count": 0, "wins": 0, "losses": 0,
                             "total_pnl": 0.0, "pnl_pcts": []})
    for t in trades:
        tag = t.entry_tag or ""
        m = _re.search(r"\|strategy=([a-z0-9_]+)", tag)
        if m:
            bkt = m.group(1)
        else:
            dm = _re.search(r"\|dte=(\d+)", tag)
            if dm:
                dte = int(dm.group(1))
                bkt = ("0dte" if dte == 0 else
                         "swing" if dte >= 14 else "short")
            else:
                bkt = "unknown"
        b = buckets[bkt]
        b["count"] += 1
        pnl = t.pnl or 0
        b["total_pnl"] += pnl
        b["pnl_pcts"].append(t.pnl_pct or 0)
        if pnl > 0:
            b["wins"] += 1
        elif pnl < 0:
            b["losses"] += 1
    out = []
    for name, d in sorted(buckets.items(),
                            key=lambda kv: -kv[1]["total_pnl"]):
        n = d["count"]
        out.append({
            "bucket": name, "count": n,
            "wins": d["wins"], "losses": d["losses"],
            "win_rate": round(d["wins"] / n, 3) if n else None,
            "total_pnl": round(d["total_pnl"], 2),
            "avg_pnl_pct": (round(sum(d["pnl_pcts"]) / n, 4)
                              if n else None),
        })
    return {"buckets": out, "window_days": days}


@app.get("/api/exit_reasons", response_class=JSONResponse)
def exit_reasons_api(days: int = Query(7, ge=1, le=365)):
    """Exit reason breakdown for a donut/pie chart."""
    from datetime import datetime, timedelta, timezone
    from collections import Counter as _C
    j = _load_journal()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)
        trades = j.closed_trades(since=since)
    finally:
        j.close()
    c = _C()
    pnl_by_reason: Dict[str, float] = defaultdict(float)
    for t in trades:
        reason = (t.exit_reason or "unknown").split(":")[0] or "unknown"
        c[reason] += 1
        pnl_by_reason[reason] += float(t.pnl or 0)
    return {
        "reasons": [
            {"reason": r, "count": n,
             "pnl": round(pnl_by_reason[r], 2)}
            for r, n in c.most_common(15)
        ],
    }


@app.post("/api/chat", response_class=JSONResponse)
async def chat_api(request: Request):
    """LLM chat with INTENT-AWARE context. Classifies the question then
    pulls ONLY the relevant data so the LLM doesn't recycle the same
    4 data points for every question.

    Body: {message: str, model?: '70b'|'8b'}.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    msg = str(body.get("message", "")).strip()[:4000]
    model_pref = str(body.get("model", "70b")).lower()
    if not msg:
        return {"error": "empty message"}
    try:
        from ..intelligence.groq_client import build_llm_client_for
        role = "research" if model_pref == "70b" else "chat"
        client, model = build_llm_client_for(role)
        if client is None:
            return {"error": "no LLM configured (set GROQ_API_KEY)"}

        intent = _classify_chat_intent(msg)
        context, system_hint = _build_intent_context(intent, msg)
        prompt = (
            f"{system_hint}\n\n"
            f"CONTEXT ({intent}):\n{context}\n\n"
            f"OPERATOR: {msg}\n\nCOPILOT:"
        )
        resp = client.generate(
            model=model, prompt=prompt,
            temperature=0.35, max_tokens=500,
        )
        return {"response": (resp or "").strip(), "model": model,
                "intent": intent}
    except Exception as e:                                  # noqa: BLE001
        return {"error": str(e)[:200]}


def _classify_chat_intent(msg: str) -> str:
    """Classify the operator's question into a context bucket so we
    pull the RIGHT data for the LLM.

    Order matters — more specific intents checked first.
    """
    m = msg.lower()
    # Exit tuning (check BEFORE strategy because "0dte exit" contains "0dte")
    if any(w in m for w in ("tighten", "loosen", "threshold",
                              "exit rule", "exit tun", "exit thresh",
                              "give back", "give-back", "profit lock",
                              "trailing stop")):
        return "exit_tuning"
    # Filter blocks
    if any(w in m for w in ("not entering", "not trading", "no trade",
                              "blocked", "filter", "skip", "reject",
                              "blocking signal")):
        return "filter_blocks"
    # Trade review
    if any(w in m for w in ("last trade", "why did", "why exit",
                              "why close", "explain the", "review the",
                              "what went wrong", "what happened",
                              "recent trade")):
        return "trade_review"
    # Market regime
    if any(w in m for w in ("regime", "breadth", "vix",
                              "market state", "volatility")):
        return "market_regime"
    # Advisor
    if any(w in m for w in ("advisor", "fade advisory",
                              "should i hold", "should i close")):
        return "advisor"
    # Strategy buckets (check after exit_tuning)
    if any(w in m for w in ("bucket", "0dte vs", "swing vs",
                              "carrying", "per-strategy", "strategy perform",
                              "compare 0dte")):
        return "strategy"
    # P&L summary (counts as trade_review with slightly different prompt)
    if any(w in m for w in ("summarize", "summary", "today p&l",
                              "today pnl", "best trade", "worst trade")):
        return "trade_review"
    # Bot status
    if any(w in m for w in ("doing right now", "status", "health",
                              "is bot", "is the bot", "what's the bot",
                              "running", "alive")):
        return "bot_status"
    return "general"


def _build_intent_context(intent: str, msg: str) -> tuple:
    """Return (context_text, system_hint) tailored to the intent.
    Each intent pulls DIFFERENT data — that's why answers stop looking
    the same for every question."""
    import re as _re
    import json as _json
    from pathlib import Path as _P
    from datetime import datetime, timedelta, timezone

    # Always include minimal heartbeat so LLM knows bot is alive
    base: List[str] = []
    try:
        snap = _P(__file__).resolve().parents[2] / "logs" / "broker_state.json"
        if snap.exists():
            d = _json.loads(snap.read_text())
            base.append(
                f"bot_state: cash=${d.get('cash', 0):.2f}, "
                f"day_pnl=${d.get('day_pnl', 0):+.2f}, "
                f"local_positions={len(d.get('positions') or [])}"
            )
    except Exception:
        pass

    # --------- intent dispatch ---------
    if intent == "trade_review":
        parts = list(base)
        parts.append("### Recent closed trades (most recent last):")
        try:
            j = _load_journal()
            try:
                since = datetime.now(tz=timezone.utc) - timedelta(days=7)
                trades = j.closed_trades(since=since)[-8:]
            finally:
                j.close()
            for t in trades:
                closed = (t.closed_at.strftime("%H:%M")
                          if t.closed_at else "?")
                tag_sym = ""
                if t.entry_tag:
                    sm = _re.search(r"\|sym=([A-Z]{1,6})", t.entry_tag)
                    if sm:
                        tag_sym = sm.group(1)
                    strat_m = _re.search(r"\|strategy=(\w+)",
                                           t.entry_tag or "")
                    strat = strat_m.group(1) if strat_m else ""
                    dte_m = _re.search(r"\|dte=(\d+)", t.entry_tag or "")
                    dte = dte_m.group(1) if dte_m else "?"
                else:
                    strat, dte = "", "?"
                parts.append(
                    f"  - {closed} {tag_sym or t.symbol[:20]} "
                    f"dte={dte} strategy={strat} "
                    f"entry=${t.entry_price:.2f} exit=${t.exit_price or 0:.2f} "
                    f"pnl=${t.pnl or 0:+.2f} ({(t.pnl_pct or 0)*100:+.1f}%) "
                    f"reason={t.exit_reason or 'n/a'}"
                )
        except Exception as e:                          # noqa: BLE001
            parts.append(f"(journal read failed: {e})")
        hint = (
            "You are the operator's trading copilot. Answer SPECIFICALLY "
            "about the trades in the CONTEXT — cite real symbols, P&L, "
            "exit reasons, times. Explain the WHY (what reason fired, "
            "was it the right call, what would you tune). 3-5 sentences."
        )
        return "\n".join(parts), hint

    if intent == "strategy":
        parts = list(base)
        parts.append("### Per-bucket strategy performance (last 7d):")
        try:
            j = _load_journal()
            try:
                since = datetime.now(tz=timezone.utc) - timedelta(days=7)
                trades = j.closed_trades(since=since)
            finally:
                j.close()
            from collections import defaultdict as _dd
            b = _dd(lambda: {"n": 0, "w": 0, "l": 0,
                              "pnl": 0.0, "pnls": []})
            for t in trades:
                tag = t.entry_tag or ""
                m = _re.search(r"\|strategy=(\w+)", tag)
                if m:
                    name = m.group(1)
                else:
                    dm = _re.search(r"\|dte=(\d+)", tag)
                    dte = int(dm.group(1)) if dm else 7
                    name = ("0dte" if dte == 0 else
                             "swing" if dte >= 14 else "short")
                bs = b[name]
                bs["n"] += 1
                bs["pnl"] += t.pnl or 0
                bs["pnls"].append(t.pnl_pct or 0)
                if (t.pnl or 0) > 0:
                    bs["w"] += 1
                elif (t.pnl or 0) < 0:
                    bs["l"] += 1
            for name, bs in sorted(b.items(),
                                     key=lambda kv: -kv[1]["pnl"]):
                wr = bs["w"] / max(bs["n"], 1) * 100
                avg_p = sum(bs["pnls"]) / max(len(bs["pnls"]), 1) * 100
                parts.append(
                    f"  - {name}: {bs['n']} trades, {bs['w']}W/{bs['l']}L "
                    f"(winrate {wr:.0f}%), avg pnl {avg_p:+.1f}%, "
                    f"total ${bs['pnl']:+.2f}"
                )
        except Exception as e:                          # noqa: BLE001
            parts.append(f"(journal read failed: {e})")
        hint = (
            "You are the copilot. Compare the strategy buckets in "
            "CONTEXT — which is carrying P&L, which is losing. "
            "Recommend reallocation if one is clearly better. Cite "
            "real numbers. 4-6 sentences."
        )
        return "\n".join(parts), hint

    if intent == "filter_blocks":
        parts = list(base)
        parts.append("### Why signals are being blocked (last ~400 log lines):")
        try:
            log = _P(__file__).resolve().parents[2] / "logs" / "tradebot.out"
            if log.exists():
                size = log.stat().st_size
                with log.open("rb") as f:
                    f.seek(max(0, size - 300_000))
                    if size > 300_000:
                        f.readline()
                    text = f.read().decode("utf-8", errors="replace")
                text = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
                from collections import Counter as _C
                emits = _re.findall(r"ensemble_emit.*?direction=(\w+)"
                                      r".*?symbol=(\w+).*?score=([\d.]+)", text)
                skips = _re.findall(r"ensemble_skip.*?reason=(\S+)", text)
                blocks = _re.findall(r"exec_chain_block.*?filter=(\w+)"
                                       r".*?reason=['\"]?([^'\"]+?)['\"]?(?=\s|$)",
                                       text)
                parts.append(
                    f"  emits: {len(emits)} in window "
                    f"(last few: {emits[-3:]})"
                )
                skip_c = _C(s.split(':')[0] for s in skips)
                parts.append(
                    "  ensemble_skip reasons: " +
                    ", ".join(f"{k}×{v}" for k, v in skip_c.most_common(5))
                )
                block_c = _C((f, r.split(':')[0]) for f, r in blocks)
                parts.append(
                    "  exec_chain_block top filters: " +
                    ", ".join(f"{f}:{r}×{c}"
                              for (f, r), c in block_c.most_common(6))
                )
                regimes = _re.findall(r"regime=(\w+)", text)
                if regimes:
                    parts.append(f"  current regime: {regimes[-1]}")
        except Exception as e:                          # noqa: BLE001
            parts.append(f"(log read failed: {e})")
        hint = (
            "You are the copilot. Identify the TOP 2-3 filters causing "
            "the most blocks from CONTEXT. Name them specifically (f11, "
            "f12, f19, etc.) and explain what they check. Recommend a "
            "tunable that would unblock signals without breaking risk "
            "discipline. Cite real counts. 4-6 sentences."
        )
        return "\n".join(parts), hint

    if intent == "exit_tuning":
        parts = list(base)
        parts.append("### Saves tracker + exit reasons (last 48h):")
        try:
            from ..intelligence.saves_tracker import summary
            s = summary(since_hours=48)
            parts.append(
                f"  net saves: ${s.get('saved_usd_30m', 0):+.2f} · "
                f"{s.get('n_wins_30m', 0)} saves / "
                f"{s.get('n_regrets_30m', 0)} regrets · "
                f"{s.get('n_exits', 0)} defensive exits total"
            )
            by_r = s.get("by_reason", {})
            parts.append("  by exit reason:")
            for r, d in list(by_r.items())[:6]:
                parts.append(f"    - {r}: {d['count']}× · "
                             f"${d['saved_usd_30m']:+.2f}")
            by_d = s.get("by_dte_bucket", {})
            parts.append("  by DTE bucket:")
            for k, v in by_d.items():
                if v.get("count", 0) > 0:
                    parts.append(
                        f"    - {k.upper()}: {v['count']}× · "
                        f"${v['saved_usd_30m']:+.2f}"
                    )
        except Exception as e:                          # noqa: BLE001
            parts.append(f"(saves read failed: {e})")
        hint = (
            "You are the copilot. Use the saves data to recommend "
            "whether to TIGHTEN or LOOSEN exit thresholds. Rule: "
            "many regrets → thresholds too tight (loosen). Many saves → "
            "tight is working (keep). Analyze per DTE bucket. Be "
            "concrete — name the config knob. 4-6 sentences."
        )
        return "\n".join(parts), hint

    if intent == "market_regime":
        parts = list(base)
        try:
            log = _P(__file__).resolve().parents[2] / "logs" / "tradebot.out"
            if log.exists():
                size = log.stat().st_size
                with log.open("rb") as f:
                    f.seek(max(0, size - 120_000))
                    if size > 120_000:
                        f.readline()
                    text = f.read().decode("utf-8", errors="replace")
                text = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
                snapshots = _re.findall(
                    r"market_state_snapshot.*?regime=(\w+).*?vix=([\d.]+)"
                    r"(?:.*?breadth_score=(-?[\d.]+|None))?", text
                )
                if snapshots:
                    last = snapshots[-1]
                    parts.append(
                        f"  current: regime={last[0]} vix={last[1]} "
                        f"breadth_score={last[2] or 'None'}"
                    )
                    # Trend
                    if len(snapshots) >= 5:
                        vix_trend = [float(s[1]) for s in snapshots[-5:]]
                        parts.append(
                            f"  vix last 5 samples: "
                            f"{'→'.join(f'{v:.2f}' for v in vix_trend)}"
                        )
                emits = _re.findall(
                    r"ensemble_emit.*?direction=(\w+)"
                    r".*?symbol=(\w+).*?score=([\d.]+)",
                    text
                )[-5:]
                parts.append(f"  last 5 ensemble emits: {emits}")
        except Exception as e:                          # noqa: BLE001
            parts.append(f"(log read failed: {e})")
        hint = (
            "You are the copilot. Explain the current market regime in "
            "operator-friendly English. What does 'closing regime' mean? "
            "Is the VIX trending up or down? Is breadth risk-on or "
            "risk-off? How does this affect what the bot should do. "
            "Cite real numbers. 4-6 sentences."
        )
        return "\n".join(parts), hint

    if intent == "advisor":
        parts = list(base)
        try:
            from ..intelligence.position_advisor import _store_path
            import time as _t
            import json as _json2
            p = _P(str(_store_path()))
            if p.exists():
                d = _json2.loads(p.read_text() or "{}")
                cutoff = _t.time() - 6 * 3600
                active = [v for v in d.values()
                           if float(v.get("ts", 0)) > cutoff]
                parts.append(f"  active advisories: {len(active)}")
                for a in active[:4]:
                    parts.append(
                        f"    - {a.get('symbol')} "
                        f"peak={a.get('peak_pnl_pct', 0)*100:+.1f}% "
                        f"now={a.get('current_pnl_pct', 0)*100:+.1f}% "
                        f"rec={a.get('recommendation')} "
                        f"({a.get('rationale', '')[:100]})"
                    )
            else:
                parts.append("  no advisory file yet")
        except Exception as e:                          # noqa: BLE001
            parts.append(f"(advisor read failed: {e})")
        hint = (
            "You are the copilot. Walk through each active advisory, "
            "state whether you agree with the rec (close/trim/hold), "
            "and explain what a smart operator should do. If there are "
            "no advisories, say that plainly. 3-6 sentences."
        )
        return "\n".join(parts), hint

    if intent == "bot_status":
        parts = list(base)
        # Recent activity pulse
        try:
            log = _P(__file__).resolve().parents[2] / "logs" / "tradebot.out"
            if log.exists():
                size = log.stat().st_size
                with log.open("rb") as f:
                    f.seek(max(0, size - 60_000))
                    if size > 60_000:
                        f.readline()
                    text = f.read().decode("utf-8", errors="replace")
                text = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
                fills = len(_re.findall(r"\[info.*?\]\s*fill\s", text))
                exits = len(_re.findall(r"fast_exit", text))
                emits = len(_re.findall(r"ensemble_emit", text))
                skips = len(_re.findall(r"ensemble_skip", text))
                parts.append(
                    f"  recent activity (~60KB tail): "
                    f"{fills} fills, {exits} exits, "
                    f"{emits} emits, {skips} skips"
                )
                regimes = _re.findall(r"regime=(\w+)", text)
                if regimes:
                    parts.append(f"  regime: {regimes[-1]}")
        except Exception:
            pass
        hint = (
            "You are the copilot. Summarize in plain English what the "
            "bot is doing right now based on CONTEXT. Are signals "
            "firing, is it trading, are we in a good state. 3-5 "
            "sentences."
        )
        return "\n".join(parts), hint

    # --- general fallback: mixed snapshot ---
    parts = list(base)
    try:
        log = _P(__file__).resolve().parents[2] / "logs" / "tradebot.out"
        if log.exists():
            size = log.stat().st_size
            with log.open("rb") as f:
                f.seek(max(0, size - 80_000))
                if size > 80_000:
                    f.readline()
                text = f.read().decode("utf-8", errors="replace")
            text = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
            regimes = _re.findall(r"regime=(\w+)", text)
            vixs = _re.findall(r"\bvix=([0-9.]+)", text)
            if regimes:
                parts.append(f"regime: {regimes[-1]}")
            if vixs:
                parts.append(f"vix: {vixs[-1]}")
    except Exception:
        pass
    hint = (
        "You are the operator's trading copilot. Answer specifically. "
        "Cite numbers from CONTEXT when relevant. 3-5 sentences."
    )
    return "\n".join(parts), hint


@app.post("/api/action/research", response_class=JSONResponse)
async def action_research(request: Request):
    """Trigger options research agent."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    symbols = body.get("symbols") or ["SPY", "QQQ"]
    symbols = [str(s).upper()[:6] for s in symbols][:6]
    try:
        from ..data.multi_provider import MultiProvider
        from ..intelligence.options_research import OptionsResearchAgent
        mp = MultiProvider.from_env()
        agent = OptionsResearchAgent(mp)
        rep = agent.run(symbols)
        return {
            "ok": True,
            "markdown": agent.to_markdown(rep),
            "model": rep.model,
            "latency_sec": rep.latency_sec,
            "ideas": [
                {
                    "symbol": i.symbol, "direction": i.direction,
                    "strike": i.strike, "expiry": i.expiry,
                    "entry": i.entry, "profit_target": i.profit_target,
                    "stop_loss": i.stop_loss,
                    "confidence": i.confidence,
                    "rationale": i.rationale[:300],
                }
                for i in rep.ideas
            ],
        }
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "error": str(e)[:300]}


@app.post("/api/action/close", response_class=JSONResponse)
async def action_close(request: Request):
    """Queue a manual close intent. Body: {symbol, kind?: 'full_close'|'trim_half'}."""
    import json as _json, time as _t
    try:
        body = await request.json()
    except Exception:
        body = {}
    symbol = str(body.get("symbol", "")).upper().strip()
    kind = str(body.get("kind", "full_close"))
    if not symbol:
        return {"ok": False, "error": "symbol required"}
    try:
        from ..core.data_paths import data_path
        from pathlib import Path as _P
        intent_path = _P(data_path("manual_close_intents.json"))
        intent_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if intent_path.exists():
            try:
                existing = _json.loads(intent_path.read_text() or "[]")
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        existing.append({
            "symbol": symbol, "kind": kind,
            "source": "dashboard_manual", "ts": _t.time(),
        })
        intent_path.write_text(_json.dumps(existing, indent=2, default=str))
        return {"ok": True,
                "message": f"close intent queued for {symbol} ({kind})"}
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200]}


@app.get("/api/llm_health", response_class=JSONResponse)
def llm_health_api():
    """Which LLM will each role use? Diagnostic — shows if Groq key is
    loaded, if Ollama is reachable, and what each role routes to."""
    try:
        from ..intelligence.groq_client import llm_health
        return llm_health()
    except Exception as e:                                  # noqa: BLE001
        return {"error": str(e)[:300]}


@app.post("/api/action/bot_control", response_class=JSONResponse)
async def bot_control_with_verify(request: Request):
    """Start/stop/restart with BEFORE/AFTER status verification.
    Body: {action: 'start'|'stop'|'restart'}.
    Returns rich status so the dashboard can show what actually changed.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    action = str(body.get("action", "")).strip()
    if action not in ("start", "stop", "restart"):
        return {"ok": False, "error": f"bad action: {action}"}

    before = _run_ctl("status", timeout=5.0)
    before_running = (before.get("ok") and
                       "running" in (before.get("stdout") or "").lower()
                       and "not running" not in (before.get("stdout") or "").lower())

    t_to = 45.0 if action == "restart" else (25.0 if action == "start" else 15.0)
    result = _run_ctl(action, timeout=t_to)

    import time as _t
    _t.sleep(1.5)   # give launchd a moment to reflect
    after = _run_ctl("status", timeout=5.0)
    after_running = (after.get("ok") and
                      "running" in (after.get("stdout") or "").lower()
                      and "not running" not in (after.get("stdout") or "").lower())

    # Build human message
    changed = before_running != after_running
    msg_bits = [f"action: {action}"]
    msg_bits.append(
        f"before: {'running' if before_running else 'stopped'}"
    )
    msg_bits.append(
        f"after: {'running' if after_running else 'stopped'}"
    )
    if action == "restart" and before_running and after_running:
        msg_bits.append("bot restarted cleanly")
        changed = True
    elif action == "start" and after_running:
        msg_bits.append("bot started")
    elif action == "stop" and not after_running:
        msg_bits.append("bot stopped")
    elif action == "start" and before_running:
        msg_bits.append("(already was running — no change)")
    elif action == "stop" and not before_running:
        msg_bits.append("(already was stopped — no change)")
    else:
        msg_bits.append("⚠️ state didn't change as expected")

    return {
        "ok": bool(result.get("ok")) and (changed or action == "restart"),
        "message": " · ".join(msg_bits),
        "action": action,
        "before_running": before_running,
        "after_running": after_running,
        "stdout": result.get("stdout", "")[:800],
        "stderr": result.get("stderr", "")[:400],
        "rc": result.get("rc"),
    }


@app.get("/api/diagnostics", response_class=JSONResponse)
def diagnostics_api():
    """Full end-to-end diagnostic — tests each subsystem the dashboard
    depends on. Result: {checks: [{name, ok, detail}, ...]}.
    Answers 'does this actually work?' for every button on the dashboard."""
    import time as _t
    checks: List[Dict[str, Any]] = []

    # 1. Bot running?
    try:
        st = _run_ctl("status", timeout=5.0)
        is_up = "running" in (st.get("stdout") or "").lower() and \
                "not running" not in (st.get("stdout") or "").lower()
        checks.append({
            "name": "paper bot process",
            "ok": bool(is_up),
            "detail": (st.get("stdout") or "").strip()[:200],
        })
    except Exception as e:
        checks.append({"name": "paper bot process", "ok": False,
                        "detail": str(e)[:200]})

    # 2. Heartbeat
    try:
        from pathlib import Path as _P
        hb = _P(__file__).resolve().parents[2] / "logs" / "heartbeat.txt"
        if hb.exists():
            age = _t.time() - hb.stat().st_mtime
            checks.append({
                "name": "main_loop heartbeat",
                "ok": age < 300,
                "detail": f"{age:.0f}s ago (fresh if < 300s)",
            })
        else:
            checks.append({"name": "main_loop heartbeat", "ok": False,
                            "detail": "no heartbeat.txt"})
    except Exception as e:
        checks.append({"name": "main_loop heartbeat", "ok": False,
                        "detail": str(e)[:200]})

    # 3. Tradier
    try:
        from ..brokers.tradier_adapter import build_tradier_broker
        tb = build_tradier_broker()
        if tb is None:
            checks.append({"name": "Tradier broker", "ok": False,
                            "detail": "not configured (TRADIER_TOKEN missing)"})
        else:
            positions = list(tb.positions())
            checks.append({
                "name": "Tradier broker",
                "ok": True,
                "detail": f"{len(positions)} positions",
            })
    except Exception as e:
        checks.append({"name": "Tradier broker", "ok": False,
                        "detail": str(e)[:200]})

    # 4. Groq
    try:
        from ..intelligence.groq_client import build_groq_client
        g = build_groq_client()
        if g is None:
            checks.append({"name": "Groq 70B",
                            "ok": False,
                            "detail": "GROQ_API_KEY not loaded — "
                                      "restart bot after adding to .env"})
        else:
            reachable = g.ping()
            checks.append({"name": "Groq 70B",
                            "ok": bool(reachable),
                            "detail": "reachable" if reachable
                                      else "key set but ping failed"})
    except Exception as e:
        checks.append({"name": "Groq 70B", "ok": False,
                        "detail": str(e)[:200]})

    # 5. Ollama
    try:
        from ..intelligence.ollama_client import build_ollama_client
        oc = build_ollama_client()
        if oc is None:
            checks.append({"name": "Ollama 8B",
                            "ok": False,
                            "detail": "not configured"})
        else:
            checks.append({"name": "Ollama 8B",
                            "ok": bool(oc.ping()),
                            "detail": "reachable" if oc.ping()
                                      else "not running"})
    except Exception as e:
        checks.append({"name": "Ollama 8B", "ok": False,
                        "detail": str(e)[:200]})

    # 6. Data provider (Multi)
    try:
        from ..data.multi_provider import MultiProvider
        mp = MultiProvider.from_env()
        active = mp.active_providers()
        q = None
        try:
            q = mp.latest_quote("SPY")
        except Exception:
            pass
        ok = bool(active) and (q is not None and getattr(q, "mid", 0) > 0)
        checks.append({
            "name": "Data providers",
            "ok": ok,
            "detail": (f"active={active}; "
                       f"SPY quote mid="
                       f"{getattr(q, 'mid', None) if q else None}"),
        })
    except Exception as e:
        checks.append({"name": "Data providers", "ok": False,
                        "detail": str(e)[:200]})

    # 7. Journal
    try:
        j = _load_journal()
        try:
            n = len(j.closed_trades())
        finally:
            j.close()
        checks.append({"name": "Trade journal",
                        "ok": True,
                        "detail": f"{n} closed trades"})
    except Exception as e:
        checks.append({"name": "Trade journal", "ok": False,
                        "detail": str(e)[:200]})

    # 8. Advisory file
    try:
        from ..intelligence.position_advisor import _store_path
        from pathlib import Path as _P
        import json as _json
        p = _P(str(_store_path()))
        if p.exists():
            d = _json.loads(p.read_text() or "{}")
            checks.append({"name": "Position advisories",
                            "ok": True,
                            "detail": f"{len(d)} advisory records"})
        else:
            checks.append({"name": "Position advisories",
                            "ok": True,
                            "detail": "no file yet (no advisories fired)"})
    except Exception as e:
        checks.append({"name": "Position advisories", "ok": False,
                        "detail": str(e)[:200]})

    # 9. Saves tracker
    try:
        from ..intelligence.saves_tracker import summary
        s = summary(since_hours=24)
        checks.append({"name": "Saves tracker",
                        "ok": True,
                        "detail": f"{s.get('n_exits', 0)} exits tracked, "
                                  f"${s.get('saved_usd_30m', 0):+.2f} net"})
    except Exception as e:
        checks.append({"name": "Saves tracker", "ok": False,
                        "detail": str(e)[:200]})

    return {"checks": checks,
             "ok_count": sum(1 for c in checks if c["ok"]),
             "total": len(checks)}


@app.post("/api/action/cleanup", response_class=JSONResponse)
async def action_cleanup(request: Request):
    """List orphans + optionally close all. Body: {close?: bool}."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    do_close = bool(body.get("close", False))
    try:
        from ..brokers.tradier_adapter import build_tradier_broker
        from ..core.types import Order, Side
        import json as _json
        from pathlib import Path as _P
        tb = build_tradier_broker()
        if tb is None:
            return {"ok": False, "error": "tradier not configured"}
        # Local state
        snap = _P(__file__).resolve().parents[2] / "logs" / "broker_state.json"
        local_syms = set()
        if snap.exists():
            try:
                d = _json.loads(snap.read_text())
                local_syms = {p.get("symbol", "")
                              for p in (d.get("positions") or [])}
            except Exception:
                pass
        tradier_positions = list(tb.positions())
        tradier_syms = {p.symbol for p in tradier_positions}
        orphans_tradier = tradier_syms - local_syms
        orphans_local   = local_syms - tradier_syms
        actions: List[Dict[str, Any]] = []
        if do_close and orphans_tradier:
            for p in tradier_positions:
                if p.symbol not in orphans_tradier:
                    continue
                side = Side.SELL if p.qty > 0 else Side.BUY
                o = Order(symbol=p.symbol, side=side,
                          qty=abs(p.qty), is_option=p.is_option,
                          limit_price=max(0.01, p.avg_price * 0.5),
                          tif="DAY", tag="dashboard_cleanup")
                try:
                    tb.submit(o)
                    actions.append({"symbol": p.symbol, "status": "ok"})
                except Exception as e:
                    actions.append({"symbol": p.symbol,
                                     "status": "failed",
                                     "err": str(e)[:150]})
        return {
            "ok": True,
            "local_n": len(local_syms),
            "tradier_n": len(tradier_syms),
            "orphans_tradier": sorted(orphans_tradier),
            "orphans_local": sorted(orphans_local),
            "actions": actions,
            "tradier_positions": [
                {"symbol": p.symbol, "qty": p.qty,
                 "avg_price": p.avg_price, "is_option": p.is_option}
                for p in tradier_positions
            ],
        }
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200]}


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE_CACHE: Dict[str, str] = {}


def _render_template(name: str) -> str:
    """Load an HTML template from disk. Cached after first read unless
    TRADEBOT_DASHBOARD_DEBUG=1 is set (which reads every request for
    rapid redesign iteration)."""
    debug = os.getenv("TRADEBOT_DASHBOARD_DEBUG", "").strip() in ("1", "true", "yes")
    if not debug and name in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[name]
    path = _TEMPLATE_DIR / name
    try:
        html = path.read_text(encoding="utf-8")
    except Exception as e:
        return (f"<!doctype html><html><body><h1>template load failed</h1>"
                f"<pre>{name}: {e}</pre></body></html>")
    if not debug:
        _TEMPLATE_CACHE[name] = html
    return html


@app.get("/", response_class=HTMLResponse)
def index():
    """New data-dense financial dashboard. Source: templates/index.html.
    Design system derived from UI/UX Pro Max skill — dark financial
    palette, JetBrains Mono for numbers, Inter for labels, 12-col
    grid with 8px gap. See templates/index.html for styling."""
    return HTMLResponse(_render_template("index.html"))


@app.get("/legacy", response_class=HTMLResponse)
def index_legacy():
    """Previous dashboard, kept for reference. Bookmark this path if
    you need the old view while we validate the new design."""
    return HTMLResponse(_INDEX_HTML)
