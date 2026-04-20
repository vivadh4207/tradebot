"""Compute win_rate / avg_win / avg_loss priors from the trade journal.

Usage:
  python scripts/compute_priors.py                       # SQLite, last 90 days
  python scripts/compute_priors.py --backend cockroach   # CockroachDB via COCKROACH_DSN
  python scripts/compute_priors.py --days 30 --symbol SPY

Output is both human-readable AND a YAML snippet you can paste back into
`config/settings.yaml` to update the sizer priors.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.storage.journal import build_journal
from src.notify.issue_reporter import alert_on_crash


@alert_on_crash("compute_priors", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser(description="Compute priors from the trade journal.")
    ap.add_argument("--backend", default="sqlite",
                    choices=["sqlite", "cockroach", "postgres"])
    ap.add_argument("--sqlite-path", default=str(ROOT / "logs" / "tradebot.sqlite"))
    ap.add_argument("--days", type=int, default=90,
                    help="Lookback window in days (default 90).")
    ap.add_argument("--symbol", default=None,
                    help="Filter to a single symbol (default: all).")
    ap.add_argument("--min-trades", type=int, default=30,
                    help="Refuse to emit priors with fewer closed trades (default 30).")
    args = ap.parse_args()

    j = build_journal(backend=args.backend, sqlite_path=args.sqlite_path)
    since = datetime.now(tz=timezone.utc) - timedelta(days=args.days)
    trades = j.closed_trades(since=since)
    if args.symbol:
        trades = [t for t in trades if t.symbol == args.symbol.upper()]

    n = len(trades)
    wins = [t for t in trades if (t.pnl or 0) > 0]
    losses = [t for t in trades if (t.pnl or 0) < 0]
    scratch = n - len(wins) - len(losses)
    win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0.0
    avg_win = (sum((t.pnl_pct or 0) for t in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(abs(t.pnl_pct or 0) for t in losses) / len(losses)) if losses else 0.0
    ev = win_rate * avg_win - (1 - win_rate) * avg_loss

    print(f"Lookback:    {args.days}d (since {since.date()})")
    print(f"Symbol:      {args.symbol or 'ALL'}")
    print(f"Trades:      n={n}  wins={len(wins)}  losses={len(losses)}  scratch={scratch}")
    print(f"win_rate:    {win_rate:.4f}")
    print(f"avg_win  %:  {avg_win:.4f}")
    print(f"avg_loss %:  {avg_loss:.4f}")
    print(f"EV per trade:{ev:+.4f}")

    if n < args.min_trades:
        print(f"\nInsufficient sample (< {args.min_trades}). Keep current priors.")
        return 1

    print("\nPaste into config/settings.yaml under the simulator priors (or")
    print("wire into SimConfig defaults):\n")
    print("simulator_priors:")
    print(f"  win_rate_prior: {round(win_rate, 4)}")
    print(f"  avg_win_prior:  {round(avg_win,  4)}")
    print(f"  avg_loss_prior: {round(avg_loss, 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
