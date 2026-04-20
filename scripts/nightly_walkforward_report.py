"""Nightly walk-forward edge-measurement report → Discord alerts.

Run as a cron job (added to deploy/cron/crontab.example at 20:00 ET):

  0 20 * * 1-5   $TRADEBOT_PY $TRADEBOT/scripts/nightly_walkforward_report.py

What it does:
  1. Runs a 365/63-day walk-forward over the journaled trade history
  2. Aggregates across windows: N windows, N tradable, median win rate,
     median EV per trade, median Sharpe-ish, max drawdown
  3. Posts a Discord embed to the `reason-to-trade` channel
     (DISCORD_WEBHOOK_URL_REASON) summarizing whether the strategy has
     measurable edge. If that webhook is unset, falls back to default.

Reading the report:
  - `tradable_ratio` = (# windows with positive EV + enough trades) / total.
    < 0.3 = no edge; stop trading.
    0.3-0.6 = marginal; pause, investigate signals.
    > 0.6 = edge exists; keep running.
  - `median_ev_pct` = expected per-trade P&L as % of equity. Must be
    POSITIVE to be worth trading at all (slippage eats EV < 0.5%).
  - `edge_trend` = direction of EV across chronological windows.
    "declining" = signal decay / overfitting; "stable" or "improving" = healthy.

This is the single best daily check for "is my strategy actually working?"
Run nightly, read the Discord, act on what the numbers say.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.core.config import load_settings
from src.storage.journal import build_journal
from src.backtest.walk_forward_runner import generate_windows, summarize
from src.notify.base import build_notifier
from src.notify.issue_reporter import alert_on_crash


def _edge_trend(windows: list) -> str:
    """Is EV improving, stable, or declining over time?"""
    evs = [w["ev"] for w in windows if w.get("tradable")]
    if len(evs) < 3:
        return "insufficient"
    # Simple linear regression slope sign
    n = len(evs)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(evs) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, evs))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = (num / den) if den > 0 else 0
    # Scale-aware threshold: 0.1% EV swing per window position
    if slope > 0.0005:
        return "improving"
    if slope < -0.0005:
        return "declining"
    return "stable"


def run(train_days: int, test_days: int, min_trades: int,
         max_windows: int) -> dict:
    s = load_settings(ROOT / "config" / "settings.yaml")
    j = build_journal(
        backend=s.get("storage.backend", "sqlite"),
        sqlite_path=s.get("storage.sqlite_path", str(ROOT / "logs" / "tradebot.sqlite")),
        dsn_env_var=s.get("storage.cockroach_dsn_env", "COCKROACH_DSN"),
        cockroach_schema=s.get("storage.cockroach_schema", "tradebot"),
    )
    try:
        windows = generate_windows(
            j, train_days=train_days, test_days=test_days,
            min_trades=min_trades, min_ev=0.0, max_windows=max_windows,
        )
    finally:
        j.close()
    report = summarize(windows)
    # Derived aggregates
    win_list = report.get("windows", [])
    total = len(win_list)
    tradable = sum(1 for w in win_list if w.get("tradable"))
    if win_list:
        wr = statistics.median(w["win_rate"] for w in win_list)
        ev = statistics.median(w["ev"] for w in win_list)
        n_trades = statistics.median(w["n"] for w in win_list)
    else:
        wr, ev, n_trades = 0.0, 0.0, 0
    report["aggregate"] = {
        "total_windows": total,
        "tradable_windows": tradable,
        "tradable_ratio": round(tradable / max(1, total), 3),
        "median_win_rate": round(wr, 3),
        "median_ev_pct": round(ev * 100, 3),
        "median_trades_per_window": int(n_trades),
        "edge_trend": _edge_trend(win_list),
        "train_days": train_days,
        "test_days": test_days,
    }
    return report


def verdict_for(agg: dict) -> tuple[str, str]:
    """Returns (verdict, color-tag) for Discord embed."""
    tr = agg.get("tradable_ratio", 0)
    ev = agg.get("median_ev_pct", 0)
    trend = agg.get("edge_trend", "insufficient")
    if agg.get("total_windows", 0) < 3:
        return ("insufficient_data — need ≥3 walk-forward windows to judge edge", "info")
    if ev <= 0 or tr < 0.3:
        return ("NO EDGE — strategy is not statistically profitable on historical bars. Stop live trading.", "error")
    if tr >= 0.6 and ev > 0.5:
        return (f"EDGE CONFIRMED — {tr:.0%} tradable windows, +{ev:.2f}% median EV per trade. {trend} trend.", "success")
    return (f"MARGINAL — {tr:.0%} tradable, +{ev:.2f}% EV. Edge weak; investigate.", "warn")


@alert_on_crash("nightly_walkforward", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--test-days", type=int, default=63)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--no-discord", action="store_true",
                     help="Print report to stdout only; skip Discord push.")
    args = ap.parse_args()

    report = run(args.train_days, args.test_days, args.min_trades, args.max_windows)
    agg = report["aggregate"]
    verdict, level = verdict_for(agg)

    # Structured meta for Discord embed
    meta = {
        "Verdict": verdict,
        "Tradable windows": f"{agg['tradable_windows']}/{agg['total_windows']} "
                               f"({agg['tradable_ratio']:.0%})",
        "Median win rate": f"{agg['median_win_rate']:.1%}",
        "Median EV / trade": f"{agg['median_ev_pct']:+.3f}%",
        "Median trades/window": str(agg["median_trades_per_window"]),
        "Edge trend": agg["edge_trend"],
        "Lookback": f"train={agg['train_days']}d, test={agg['test_days']}d",
        "_footer": f"nightly walkforward · {datetime.now(tz=timezone.utc).isoformat(timespec='minutes')}",
    }

    print(json.dumps(report, indent=2, default=str))

    if not args.no_discord:
        n = build_notifier()
        n.notify(
            f"**Walk-forward verdict**: {verdict}",
            title="backtest_report",
            level=level,
            meta=meta,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
