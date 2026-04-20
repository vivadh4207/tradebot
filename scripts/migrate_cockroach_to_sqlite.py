"""One-shot migration: copy every row from a Cockroach journal into a
local SQLite journal. Use this before flipping storage.backend to
sqlite if you want to keep your existing trade history, equity curve,
ensemble decisions, and ML predictions.

Usage:
  python scripts/migrate_cockroach_to_sqlite.py                   # default paths
  python scripts/migrate_cockroach_to_sqlite.py --dry-run         # count only
  python scripts/migrate_cockroach_to_sqlite.py \
         --sqlite-path logs/tradebot_from_cockroach.sqlite         # custom dest

Prerequisites:
  - COCKROACH_DSN (or the COCKROACH_* split fields) in .env
  - psycopg2 installed (already required for the Cockroach journal)

After running, flip storage.backend to sqlite in config/settings.yaml
and restart the bot. It will automatically pick up the migrated file.
"""
from __future__ import annotations

import argparse
import os
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

from src.core.config import load_settings
from src.core.logger import configure_logging
from src.notify.issue_reporter import alert_on_crash
from src.storage.journal import (
    CockroachJournal, SqliteJournal, resolve_cockroach_dsn,
)


def _all_history_fetch(journal, kind: str):
    """Pull every row the source has for one kind. Uses `since=None`
    where supported, or a very old date otherwise."""
    very_old = datetime(2000, 1, 1, tzinfo=timezone.utc)
    if kind == "closed_trades":
        return journal.closed_trades(since=very_old)
    if kind == "equity_series":
        return journal.equity_series(since=very_old, limit=10_000_000)
    if kind == "ensemble_decisions":
        return journal.ensemble_decisions(since=very_old, limit=10_000_000)
    if kind == "resolved_ml":
        return journal.resolved_ml_predictions(limit=10_000_000)
    return []


@alert_on_crash("migrate_cockroach_to_sqlite", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite-path", default=None,
                     help="Destination SQLite file (default: from settings.yaml)")
    ap.add_argument("--dry-run", action="store_true",
                     help="Count source rows; don't write anything.")
    ap.add_argument("--since-days", type=int, default=0,
                     help="If > 0, migrate only rows from the last N days. "
                          "Default 0 = everything.")
    args = ap.parse_args()

    configure_logging("INFO")
    settings = load_settings(ROOT / "config" / "settings.yaml")
    sqlite_path = args.sqlite_path or settings.get("storage.sqlite_path",
                                                     "logs/tradebot.sqlite")
    if not Path(sqlite_path).is_absolute():
        sqlite_path = str(ROOT / sqlite_path)

    # --- source: Cockroach ---
    dsn_env = settings.get("storage.cockroach_dsn_env", "COCKROACH_DSN")
    schema  = settings.get("storage.cockroach_schema", "tradebot")
    try:
        dsn = resolve_cockroach_dsn(dsn_env_var=dsn_env)
    except Exception as e:
        print(f"[!] cannot resolve Cockroach DSN: {e}")
        print("    Set COCKROACH_DSN in .env (or the COCKROACH_* fields)")
        return 2
    print(f"Source:  Cockroach schema='{schema}'")
    src = CockroachJournal(dsn, schema=schema)

    # --- destination: SQLite ---
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(sqlite_path).exists() and not args.dry_run:
        # Don't silently overwrite: make a backup on the side.
        backup = Path(str(sqlite_path) + ".pre-migrate.bak")
        print(f"  Existing SQLite file found; backup → {backup}")
        import shutil
        shutil.copy2(sqlite_path, backup)
    print(f"Dest:    SQLite {sqlite_path}")
    dst = SqliteJournal(sqlite_path)
    dst.init_schema()

    # --- pull + count ---
    print("\nReading from Cockroach...")
    trades = _all_history_fetch(src, "closed_trades")
    equity = _all_history_fetch(src, "equity_series")
    ensemble = _all_history_fetch(src, "ensemble_decisions")
    ml_resolved = _all_history_fetch(src, "resolved_ml")
    print(f"  closed_trades:      {len(trades):>8}")
    print(f"  equity_series:      {len(equity):>8}")
    print(f"  ensemble_decisions: {len(ensemble):>8}")
    print(f"  resolved_ml:        {len(ml_resolved):>8}")
    total = len(trades) + len(equity) + len(ensemble) + len(ml_resolved)
    print(f"  {'TOTAL':<19}: {total:>8}")

    if args.dry_run:
        print("\n[dry-run] not writing — rerun without --dry-run to migrate.")
        src.close()
        dst.close()
        return 0

    if args.since_days > 0:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=args.since_days)
        trades = [t for t in trades if t.closed_at and t.closed_at >= cutoff]
        equity = [(ts, eq, cash, pnl) for (ts, eq, cash, pnl) in equity
                    if ts >= cutoff]
        ensemble = [e for e in ensemble if e.ts >= cutoff]
        ml_resolved = [p for p in ml_resolved if p.ts >= cutoff]
        print(f"\n[since-days={args.since_days}] trimmed to: trades={len(trades)}, equity={len(equity)}, ensemble={len(ensemble)}, ml={len(ml_resolved)}")

    # --- insert into SQLite ---
    print("\nWriting to SQLite...")
    for t in trades:
        dst.record_trade(t)
    print(f"  closed_trades written")
    for ts, eq, cash, pnl in equity:
        dst.record_equity(ts, float(eq), float(cash), float(pnl))
    print(f"  equity_series written")
    for e in ensemble:
        dst.record_ensemble_decision(e)
    print(f"  ensemble_decisions written")
    for p in ml_resolved:
        pid = dst.record_ml_prediction(p)
        if p.true_class is not None and p.forward_return is not None:
            dst.resolve_ml_prediction(pid, p.true_class, float(p.forward_return))
    print(f"  ml_predictions written")

    src.close()
    dst.close()
    print(f"\ndone. verify with:")
    print(f"  sqlite3 {sqlite_path} 'SELECT COUNT(*) FROM trades'")
    print(f"\nNow flip config/settings.yaml storage.backend to 'sqlite'")
    print(f"(already default on fresh configs) and restart the bot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
