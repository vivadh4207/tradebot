"""Saves tracker report — proves (or disproves) the defensive exits.

Runs hourly:
  1. Fills in re-check prices for exits older than 30 min
  2. Once daily (by --daily flag or clock detection), posts summary
     to Discord

Outputs to Discord title='saves_report'. Example line:
  'Exits saved $347.20 in last 24h across 18 defensive closes.
   Top save: SPY 680C (green_to_red) exited $0.97, now $0.12 = saved $255.
   Top regret: QQQ 520C (profit_lock) exited $1.80, now $2.40 = left $60.'

Usage:
  # Just refresh re-check prices (cheap)
  python scripts/run_saves_report.py --recheck-only

  # Refresh + post Discord report
  python scripts/run_saves_report.py

  # Post report for a different window
  python scripts/run_saves_report.py --since-hours 168  # weekly
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.data.multi_provider import MultiProvider
from src.intelligence.saves_tracker import recheck_pending, summary


def _format_discord(s: dict) -> str:
    lines = []
    window = s["window_hours"]
    lines.append(f"**💰 Saves Tracker · last {window}h**")
    saved = s["saved_usd_30m"]
    icon = "🟢" if saved >= 0 else "🔴"
    lines.append(
        f"{icon} Exits saved **${saved:+,.2f}** "
        f"({s['n_wins_30m']} saves / {s['n_regrets_30m']} regrets) "
        f"across {s['n_exits']} defensive closes "
        f"({s['n_rechecked']} re-checked)."
    )
    if s.get("top_save"):
        t = s["top_save"]
        lines.append(
            f"🏆 Top save: **{t['underlying']}** "
            f"({t['exit_reason'].split(':')[0]}) — "
            f"exited ${t['exit_price']:.2f}, "
            f"now ${t['recheck_30m_price']:.2f} = "
            f"**saved ${t['saved_usd_30m']:+,.2f}**"
        )
    if s.get("top_regret"):
        t = s["top_regret"]
        lines.append(
            f"😬 Top regret: **{t['underlying']}** "
            f"({t['exit_reason'].split(':')[0]}) — "
            f"exited ${t['exit_price']:.2f}, "
            f"now ${t['recheck_30m_price']:.2f} = "
            f"**left ${abs(t['saved_usd_30m']):,.2f} on the table**"
        )
    lines.append("")
    lines.append("**By exit reason (top 5):**")
    for i, (reason, d) in enumerate(list(s["by_reason"].items())[:5], 1):
        mark = "🟢" if d["saved_usd_30m"] > 0 else "🔴"
        lines.append(
            f"  {mark} {reason}: {d['count']}× · "
            f"${d['saved_usd_30m']:+,.2f}"
        )
    lines.append("")
    lines.append("**By DTE bucket:**")
    for bucket, d in s["by_dte_bucket"].items():
        if d["count"] == 0:
            continue
        icon = {"0dte": "⚡", "short": "🎯", "swing": "🎢"}.get(bucket, "•")
        mark = "🟢" if d["saved_usd_30m"] > 0 else "🔴"
        lines.append(
            f"  {icon} {bucket.upper()}: {d['count']}× · "
            f"{mark} ${d['saved_usd_30m']:+,.2f}"
        )
    lines.append("")
    lines.append(
        "_Regrets = where holding would have done better. "
        "High regret count → loosen thresholds. "
        "High save count → keep tight._"
    )
    out = "\n".join(lines)
    return out[:1900]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=int, default=24)
    ap.add_argument("--recheck-only", action="store_true",
                     help="Only refresh prices, don't post report.")
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    mp = MultiProvider.from_env()
    n_updated = recheck_pending(mp)
    print(f"[saves] re-checked {n_updated} exits")
    if args.recheck_only:
        return 0

    s = summary(since_hours=int(args.since_hours))
    body = _format_discord(s)
    print(body)

    if not args.no_discord:
        try:
            from src.notify.base import build_notifier
            build_notifier().notify(
                body, level="info", title="saves_report",
                meta={
                    "Window": f"{args.since_hours}h",
                    "Net_USD": s["saved_usd_30m"],
                    "Saves":   s["n_wins_30m"],
                    "Regrets": s["n_regrets_30m"],
                    "_footer": "saves_tracker",
                },
            )
        except Exception as e:                              # noqa: BLE001
            print(f"[!] discord notify failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
