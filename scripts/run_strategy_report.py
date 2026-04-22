"""Strategy-bucket P&L report — compares 0DTE vs short vs swing.

Reads closed trades from the journal, groups by strategy tag
(|strategy=<bucket> appended to entry tag), reports per-bucket:
  - count, win rate, avg pnl %, total $ pnl
  - best / worst trade
  - avg hold time
  - exit reason distribution (PT hit vs SL hit vs trailing vs timeout)

Runs on-demand (!strategy Discord command) or nightly via launchd.
Posts to Discord title='strategy_report' if configured, else default.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


_STRATEGY_RE = re.compile(r"\|strategy=([a-zA-Z0-9_]+)")
_DTE_RE = re.compile(r"\|dte=(\d+)")


def _strategy_from_tag(tag: Optional[str]) -> str:
    """Extract strategy bucket from entry tag. Falls back to DTE-based
    classification for pre-bucket trades so historical data still
    reports."""
    if not tag:
        return "unknown"
    m = _STRATEGY_RE.search(tag)
    if m:
        return m.group(1)
    d = _DTE_RE.search(tag)
    if d:
        try:
            dte = int(d.group(1))
            if dte <= 1:
                return "0dte"
            if dte <= 7:
                return "short"
            return "swing"
        except Exception:
            pass
    return "unknown"


def _load_trades(since_days: int = 30):
    """Pull closed trades from the journal."""
    try:
        from src.storage.journal import SqliteJournal
        from src.core.data_paths import data_path
    except Exception as e:
        print(f"[!] journal import failed: {e}")
        return []
    path = data_path("logs/tradebot.sqlite")
    if not Path(path).exists():
        print(f"[!] journal not found: {path}")
        return []
    since = datetime.now(tz=timezone.utc) - timedelta(days=since_days)
    j = SqliteJournal(str(path))
    return j.closed_trades(since=since)


def _aggregate(trades) -> Dict[str, Dict]:
    """Group trades by strategy bucket, compute stats."""
    buckets: Dict[str, Dict] = defaultdict(lambda: {
        "count": 0, "wins": 0, "losses": 0, "flat": 0,
        "total_pnl": 0.0, "pnl_pcts": [], "holds_sec": [],
        "best": None, "worst": None,
        "exit_reasons": defaultdict(int),
        "symbols": defaultdict(int),
    })
    for t in trades:
        b = _strategy_from_tag(t.entry_tag)
        d = buckets[b]
        d["count"] += 1
        pnl = t.pnl or 0.0
        pnl_pct = t.pnl_pct or 0.0
        d["total_pnl"] += pnl
        d["pnl_pcts"].append(pnl_pct)
        if pnl > 0:
            d["wins"] += 1
        elif pnl < 0:
            d["losses"] += 1
        else:
            d["flat"] += 1
        if t.opened_at and t.closed_at:
            d["holds_sec"].append((t.closed_at - t.opened_at).total_seconds())
        if d["best"] is None or pnl > (d["best"].pnl or 0):
            d["best"] = t
        if d["worst"] is None or pnl < (d["worst"].pnl or 0):
            d["worst"] = t
        reason = (t.exit_reason or "unknown").split(":")[0]
        d["exit_reasons"][reason] += 1
        d["symbols"][t.symbol] += 1
    return buckets


def _fmt_duration(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec/60:.0f}m"
    if sec < 86400:
        return f"{sec/3600:.1f}h"
    return f"{sec/86400:.1f}d"


def _format_report(buckets: Dict[str, Dict], since_days: int) -> str:
    if not buckets:
        return (
            f"**📈 Strategy Report (last {since_days}d)**\n"
            "No closed trades yet. Let the bot run longer."
        )
    lines: List[str] = []
    lines.append(f"**📈 Strategy Bucket Report · last {since_days}d**")
    total_pnl = sum(b["total_pnl"] for b in buckets.values())
    total_n = sum(b["count"] for b in buckets.values())
    lines.append(
        f"**Overall**: {total_n} trades · ${total_pnl:+,.2f} total P&L"
    )
    lines.append("")

    # Sort by total $ pnl descending — best bucket first
    ordered = sorted(
        buckets.items(),
        key=lambda kv: kv[1]["total_pnl"],
        reverse=True,
    )
    for name, d in ordered:
        n = d["count"]
        if n == 0:
            continue
        win_rate = d["wins"] / n * 100
        avg_pnl_pct = sum(d["pnl_pcts"]) / n * 100
        median_hold = (
            sorted(d["holds_sec"])[len(d["holds_sec"]) // 2]
            if d["holds_sec"] else 0
        )
        icon = {"0dte": "⚡", "short": "🎯", "swing": "🎢",
                 "unknown": "❓"}.get(name, "📊")
        lines.append(
            f"**{icon} {name.upper()}** · {n} trades · "
            f"win {win_rate:.0f}% · avg {avg_pnl_pct:+.1f}% · "
            f"**${d['total_pnl']:+,.2f}** total"
        )
        lines.append(
            f"   · median hold: {_fmt_duration(median_hold)} · "
            f"wins {d['wins']} / losses {d['losses']} / flat {d['flat']}"
        )
        if d["best"]:
            bsym = d["best"].symbol[:10]
            bpnl = d["best"].pnl or 0
            bpct = (d["best"].pnl_pct or 0) * 100
            lines.append(
                f"   · best: {bsym} ${bpnl:+,.2f} ({bpct:+.1f}%)"
            )
        if d["worst"] and (d["worst"].pnl or 0) < 0:
            wsym = d["worst"].symbol[:10]
            wpnl = d["worst"].pnl or 0
            wpct = (d["worst"].pnl_pct or 0) * 100
            lines.append(
                f"   · worst: {wsym} ${wpnl:+,.2f} ({wpct:+.1f}%)"
            )
        # Top exit reasons (helps tune the exit engine per bucket)
        top_reasons = sorted(
            d["exit_reasons"].items(), key=lambda kv: -kv[1]
        )[:3]
        if top_reasons:
            reason_str = ", ".join(f"{r}×{c}" for r, c in top_reasons)
            lines.append(f"   · exits: {reason_str}")
        lines.append("")

    lines.append(
        "_Operator tip: tilt allocation toward the bucket with highest "
        "**avg %** AND highest **win rate**. High total $ from one huge "
        "outlier is survivorship bias._"
    )
    out = "\n".join(lines)
    if len(out) > 1900:
        out = out[:1880].rstrip() + "\n… [truncated]"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-days", type=int, default=30,
                     help="Lookback window (default 30 days).")
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    trades = _load_trades(since_days=int(args.since_days))
    buckets = _aggregate(trades)
    body = _format_report(buckets, since_days=int(args.since_days))
    print(body)

    if not args.no_discord:
        try:
            from src.notify.base import build_notifier
            meta = {
                "Window":   f"{args.since_days}d",
                "Trades":   sum(b["count"] for b in buckets.values()),
                "Buckets":  len(buckets),
                "_footer":  "strategy_report",
            }
            build_notifier().notify(
                body, level="info", title="strategy_report", meta=meta,
            )
        except Exception as e:                              # noqa: BLE001
            print(f"[!] discord notify failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
