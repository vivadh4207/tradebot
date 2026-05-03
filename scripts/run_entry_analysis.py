"""Entry-quality analysis — which signal sources + regimes actually have edge.

Reads closed trades from the journal, groups by:
  - signal source (candle_patterns, orb, vwap_reversion, extreme_momentum, etc.)
  - regime at entry time (trend_lowvol, range_lowvol, etc.)
  - direction (bullish vs bearish)
  - DTE bucket (0dte / short / swing)

Output: matrix of win rate + avg pnl% + total $ per (signal × regime).

This is what you use to decide:
  - "candle_patterns loses money in range_lowvol → disable in that regime"
  - "orb wins 60% in opening → keep sized up"
  - "extreme_momentum + swing DTE = always wins" → increase weight

Post-hoc analysis only. Does NOT auto-adjust — the operator decides.

Run:
  .venv/bin/python scripts/run_entry_analysis.py --days 14
  .venv/bin/python scripts/run_entry_analysis.py --days 7 --min-trades 3

Post to Discord:
  .venv/bin/python scripts/run_entry_analysis.py --days 14 --discord
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _parse_entry_tag(tag: str) -> Dict[str, str]:
    """Parse the pipe-delimited entry tag into key/value parts."""
    out = {}
    if not tag:
        return out
    for part in tag.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _dte_bucket(dte_str: str) -> str:
    try:
        dte = int(dte_str)
    except Exception:
        return "unknown"
    if dte == 0:
        return "0dte"
    if dte <= 7:
        return "short"
    return "swing"


def _get_regime_for_trade(t: Any, log_path: Path) -> str:
    """Try to infer the regime at time-of-entry by log-grepping near
    the opened_at timestamp. Expensive — caller should cache."""
    if not t.opened_at:
        return "unknown"
    # Cheaper: read entire log once and index by timestamp
    # (caller should do this) — here we just do a small slice grep
    if not log_path.exists():
        return "unknown"
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            # Read up to 50KB centered on the trade's timestamp-prefix.
            # Since logs are chronological, we tail and search.
            f.seek(max(0, size - 2_000_000))
            text = f.read().decode("utf-8", errors="replace")
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
        # Find regime closest before opened_at
        open_ts = t.opened_at.replace(tzinfo=None).strftime(
            "%Y-%m-%dT%H:%M"
        )
        regime_snapshots = re.findall(
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}).*?regime=(\w+)", text
        )
        best_regime = "unknown"
        for ts, rg in regime_snapshots:
            if ts <= open_ts:
                best_regime = rg
        return best_regime
    except Exception:
        return "unknown"


def analyze(since_days: int = 14, min_trades: int = 2) -> Dict[str, Any]:
    from src.storage.journal import SqliteJournal
    from src.core.data_paths import data_path

    jpath = data_path("logs/tradebot.sqlite")
    j = SqliteJournal(str(jpath))
    since = datetime.now(tz=timezone.utc) - timedelta(days=since_days)
    try:
        trades = j.closed_trades(since=since)
    finally:
        j.close()

    # Cache regime snapshots from log ONCE
    log_path = ROOT / "logs" / "tradebot.out"
    regime_index: List[Tuple[str, str]] = []
    if log_path.exists():
        try:
            with log_path.open("rb") as f:
                size = f.tell()
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 5_000_000))
                text = f.read().decode("utf-8", errors="replace")
            text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
            regime_index = re.findall(
                r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}).*?regime=(\w+)",
                text,
            )
        except Exception:
            pass

    def _regime_at(ts: datetime) -> str:
        if not ts:
            return "unknown"
        key = ts.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M")
        last = "unknown"
        for k, rg in regime_index:
            if k <= key:
                last = rg
            else:
                break
        return last

    # Group by (src, regime, direction, dte_bucket)
    buckets: Dict[Tuple[str, str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "wins": 0, "losses": 0,
                  "total_pnl": 0.0, "pnl_pcts": []}
    )
    by_src: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "wins": 0, "total_pnl": 0.0, "pnl_pcts": []}
    )
    by_regime: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "wins": 0, "total_pnl": 0.0, "pnl_pcts": []}
    )

    for t in trades:
        kv = _parse_entry_tag(t.entry_tag or "")
        src = kv.get("src") or "unknown"
        direction = kv.get("right") or "unknown"   # call/put
        dte_bucket = _dte_bucket(kv.get("dte", ""))
        strategy_bucket = kv.get("strategy", "")
        regime = _regime_at(t.opened_at)
        pnl = t.pnl or 0
        pnl_pct = t.pnl_pct or 0

        key = (src, regime, direction, dte_bucket)
        b = buckets[key]
        b["n"] += 1
        b["total_pnl"] += pnl
        b["pnl_pcts"].append(pnl_pct)
        if pnl > 0:
            b["wins"] += 1
        elif pnl < 0:
            b["losses"] += 1

        # Marginal aggregations
        for agg, k in ((by_src, src), (by_regime, regime)):
            agg[k]["n"] += 1
            agg[k]["total_pnl"] += pnl
            agg[k]["pnl_pcts"].append(pnl_pct)
            if pnl > 0:
                agg[k]["wins"] += 1

    # Filter buckets with < min_trades (noise)
    signal_regime = [
        {
            "src": s, "regime": r, "direction": d, "dte": dte,
            "n": b["n"], "wins": b["wins"], "losses": b["losses"],
            "win_rate": round(b["wins"] / max(b["n"], 1), 3),
            "avg_pnl_pct": (round(sum(b["pnl_pcts"]) / max(b["n"], 1), 4)
                              if b["n"] else None),
            "total_pnl": round(b["total_pnl"], 2),
        }
        for (s, r, d, dte), b in buckets.items()
        if b["n"] >= min_trades
    ]
    signal_regime.sort(key=lambda x: -x["total_pnl"])

    src_summary = [
        {"src": s, "n": b["n"], "wins": b["wins"],
         "win_rate": round(b["wins"] / max(b["n"], 1), 3),
         "total_pnl": round(b["total_pnl"], 2),
         "avg_pnl_pct": (round(sum(b["pnl_pcts"]) / max(b["n"], 1), 4)
                           if b["n"] else None)}
        for s, b in by_src.items() if b["n"] >= min_trades
    ]
    src_summary.sort(key=lambda x: -x["total_pnl"])

    regime_summary = [
        {"regime": r, "n": b["n"], "wins": b["wins"],
         "win_rate": round(b["wins"] / max(b["n"], 1), 3),
         "total_pnl": round(b["total_pnl"], 2),
         "avg_pnl_pct": (round(sum(b["pnl_pcts"]) / max(b["n"], 1), 4)
                           if b["n"] else None)}
        for r, b in by_regime.items() if b["n"] >= min_trades
    ]
    regime_summary.sort(key=lambda x: -x["total_pnl"])

    return {
        "window_days": since_days,
        "total_trades": len(trades),
        "min_trades_filter": min_trades,
        "by_source_x_regime": signal_regime,
        "by_source": src_summary,
        "by_regime": regime_summary,
    }


def _format_discord(d: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"**🔬 Entry Analysis · last {d['window_days']}d**")
    lines.append(
        f"Total trades: **{d['total_trades']}** · "
        f"min n={d['min_trades_filter']} per bucket"
    )
    lines.append("")

    if d["by_source"]:
        lines.append("**Per signal source:**")
        for r in d["by_source"][:10]:
            icon = ("🟢" if r["total_pnl"] > 0
                      else "🔴" if r["total_pnl"] < 0 else "⚪")
            wr_pct = r["win_rate"] * 100
            avg_pct = (r["avg_pnl_pct"] or 0) * 100
            lines.append(
                f"  {icon} **{r['src']}**: {r['n']} trades · "
                f"win {wr_pct:.0f}% · "
                f"avg {avg_pct:+.1f}% · "
                f"**${r['total_pnl']:+.2f}**"
            )
        lines.append("")

    if d["by_regime"]:
        lines.append("**Per regime:**")
        for r in d["by_regime"][:10]:
            icon = "🟢" if r["total_pnl"] > 0 else "🔴" if r["total_pnl"] < 0 else "⚪"
            wr_pct = r["win_rate"] * 100
            lines.append(
                f"  {icon} **{r['regime']}**: {r['n']} trades · "
                f"win {wr_pct:.0f}% · "
                f"**${r['total_pnl']:+.2f}**"
            )
        lines.append("")

    lines.append("**Top 8 winning (signal × regime) combos:**")
    wins = [r for r in d["by_source_x_regime"] if r["total_pnl"] > 0][:8]
    if wins:
        for r in wins:
            lines.append(
                f"  🟢 {r['src']:<22} [{r['regime']}] {r['direction']} "
                f"{r['dte']}: {r['n']}× · win {r['win_rate']*100:.0f}% · "
                f"**${r['total_pnl']:+.2f}**"
            )
    else:
        lines.append("  _(none profitable)_")
    lines.append("")

    lines.append("**Top 8 losing (signal × regime) combos:**")
    losses = [r for r in d["by_source_x_regime"] if r["total_pnl"] < 0][:8]
    if losses:
        for r in losses:
            lines.append(
                f"  🔴 {r['src']:<22} [{r['regime']}] {r['direction']} "
                f"{r['dte']}: {r['n']}× · win {r['win_rate']*100:.0f}% · "
                f"**${r['total_pnl']:+.2f}**"
            )
    lines.append("")

    lines.append(
        "_Recommendation: disable or downweight any signal × regime "
        "combo with win rate <40% AND total pnl negative across "
        "≥5 trades. Keep (or upweight) combos with win rate >55% AND "
        "positive total pnl._"
    )

    out = "\n".join(lines)
    return out[:1900]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--min-trades", type=int, default=2)
    ap.add_argument("--discord", action="store_true",
                     help="Also post to Discord (title='entry_analysis').")
    args = ap.parse_args()

    d = analyze(since_days=args.days, min_trades=args.min_trades)
    body = _format_discord(d)
    print(body)

    if args.discord:
        try:
            from src.notify.base import build_notifier
            build_notifier().notify(
                body, level="info", title="entry_analysis",
                meta={"Window": f"{args.days}d",
                      "Total": d["total_trades"],
                      "_footer": "entry_analysis"},
            )
        except Exception as e:
            print(f"[!] discord notify failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
