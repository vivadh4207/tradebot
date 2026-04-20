"""Run a walk-forward analysis on the journaled trade history.

Prints per-window refitted priors + tradable verdict. Paste the most
recent tradable window into `config/settings.yaml` (or
`SimConfig` defaults) to deploy for the next forward period.
"""
from __future__ import annotations

import argparse
import json
import sys
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None,
                    help="sqlite|cockroach (default: from settings.yaml)")
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--test-days", type=int, default=63)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--min-ev", type=float, default=0.0)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    backend = args.backend or s.get("storage.backend", "sqlite")
    j = build_journal(
        backend=backend,
        sqlite_path=s.get("storage.sqlite_path", str(ROOT / "logs" / "tradebot.sqlite")),
        dsn_env_var=s.get("storage.cockroach_dsn_env", "COCKROACH_DSN"),
    )
    try:
        windows = generate_windows(
            j, train_days=args.train_days, test_days=args.test_days,
            min_trades=args.min_trades, min_ev=args.min_ev,
            max_windows=args.max_windows,
        )
    finally:
        j.close()

    report = summarize(windows)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    print(f"Walk-forward ({backend}): train={args.train_days}d  test={args.test_days}d")
    print(f"{'TRAIN':25s} {'TEST':25s} {'N':>4s} {'WR':>6s} "
          f"{'AW':>6s} {'AL':>6s} {'EV':>7s}  TRADABLE")
    for w in report["windows"]:
        print(f"{w['train']:25s} {w['test']:25s} "
              f"{w['n']:>4d} {w['win_rate']:>6.3f} "
              f"{w['avg_win']:>6.3f} {w['avg_loss']:>6.3f} "
              f"{w['ev']:>+7.4f}  {'yes' if w['tradable'] else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
