"""Run a backtest and print a performance report.

Data modes:
  --data synthetic   (default)  GBM-like bars, no network, good for pipeline tests
  --data historical            real bars via Alpaca (if keys) else yfinance
  --days N           (historical only) how many calendar days to backfill

Examples:
  python scripts/run_backtest.py
  python scripts/run_backtest.py --data historical --days 30
  python scripts/run_backtest.py --data historical --days 90 --total-bars 1500
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.core.config import load_settings
from src.core.clock import ET
from src.data.market_data import SyntheticDataAdapter
from src.data.historical_adapter import HistoricalMarketDataAdapter
from src.backtest.simulator import BacktestSimulator, SimConfig
from src.backtest.metrics import performance_report
from src.storage.journal import build_journal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "historical"], default="synthetic")
    ap.add_argument("--days", type=int, default=30,
                    help="(historical) calendar days to backfill")
    ap.add_argument("--total-bars", type=int, default=300)
    ap.add_argument("--timeframe-min", type=int, default=1)
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")

    if args.data == "historical":
        end = datetime.now(tz=ET)
        start = end - timedelta(days=args.days)
        data = HistoricalMarketDataAdapter(
            symbols=s.universe, start=start, end=end,
            timeframe_minutes=args.timeframe_min,
        )
        print(f"[data] historical: {args.days}d @ {args.timeframe_min}m "
              f"for {len(s.universe)} symbols")
    else:
        data = SyntheticDataAdapter(seed=42)
        print("[data] synthetic GBM")

    backend = s.get("storage.backend", "sqlite")
    sqlite_path = s.get("storage.sqlite_path", str(ROOT / "logs" / "tradebot.sqlite"))
    dsn_env = s.get("storage.cockroach_dsn_env", "COCKROACH_DSN")
    journal = build_journal(backend=backend, sqlite_path=sqlite_path, dsn_env_var=dsn_env)

    sim = BacktestSimulator(s.raw, data, SimConfig(
        starting_equity=s.paper_equity, verbose=False,
    ))
    sim.broker._journal = journal

    result = sim.run(s.universe, total_bars=args.total_bars)
    print("Final equity:", round(result["final_equity"], 2))
    print("Total pnl:   ", round(result["total_pnl"], 2))

    eq = result["equity_curve"] or [s.paper_equity]
    report = performance_report(eq, [], days_traded=len(eq))
    for k, v in report.to_dict().items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
