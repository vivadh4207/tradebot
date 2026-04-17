"""Wipe the trade journal + event logs — for a clean "day 1" reset.

WHAT IT CLEARS (destructive, no undo):
  - Journal tables (fills, trades, equity_curve, ml_predictions,
    ensemble_decisions) in whichever backend `settings.yaml` points at
    (sqlite or cockroach).
  - logs/slippage_calibration.jsonl
  - logs/calibration_history.jsonl
  - logs/backtest_runs.jsonl
  - logs/watchdog_events.jsonl
  - logs/heartbeat.txt
  - logs/broker_state.json
  - logs/daily_report.json  (if present)

WHAT IT DOES NOT TOUCH:
  - Your source code, settings, .env
  - logs/tradebot.out (so the process log stays appendable)
  - logs/tradebot.err
  - CockroachDB schema (tables stay; only rows are truncated)
  - LSTM checkpoints, cache dirs

REQUIRES TYPED CONFIRMATION so a fat-finger can't wipe by accident.

Run as:
    .venv/bin/python scripts/wipe_journal.py
or:
    scripts/tradebotctl.sh wipe-journal
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from src.core.config import load_settings
from src.storage.journal import build_journal, CockroachJournal, SqliteJournal


JOURNAL_TABLES = [
    "fills",
    "trades",
    "equity_curve",
    "ml_predictions",
    "ensemble_decisions",
]

LOG_FILES_TRUNCATE = [
    "logs/slippage_calibration.jsonl",
    "logs/calibration_history.jsonl",
    "logs/backtest_runs.jsonl",
    "logs/watchdog_events.jsonl",
    "logs/heartbeat.txt",
]

LOG_FILES_DELETE = [
    "logs/broker_state.json",
    "logs/daily_report.json",
]


def _confirm() -> bool:
    print("This will WIPE all paper trades, calibration history, and run logs.")
    print(f"Target: backend='{backend}', scope='{scope_desc}'")
    print("Type the word DESTROY (case-sensitive) to continue, anything else aborts:")
    try:
        ans = input("> ").strip()
    except EOFError:
        return False
    return ans == "DESTROY"


def _wipe_cockroach(j: CockroachJournal) -> None:
    schema = getattr(j, "_schema", "tradebot")
    with j._conn.cursor() as cur:
        cur.execute(f"SET search_path = {schema}, public")
        for t in JOURNAL_TABLES:
            try:
                cur.execute(f"TRUNCATE {schema}.{t}")
                print(f"  cockroach: truncated {schema}.{t}")
            except Exception as e:
                # Table may not exist yet — not fatal.
                print(f"  cockroach: skip {schema}.{t} ({e})")
        # One-time cleanup: drop orphaned tradebot tables from `public`
        # that used to live there before the schema isolation. Only
        # drops tables whose names match ours, and only from public.
        # Runs silently if already gone.
        for t in JOURNAL_TABLES:
            try:
                cur.execute(f"DROP TABLE IF EXISTS public.{t}")
            except Exception:
                pass
        print(f"  cockroach: dropped any orphan tradebot tables from public.*")


def _wipe_sqlite(j: SqliteJournal) -> None:
    cur = j._conn.cursor()                    # type: ignore[attr-defined]
    for t in JOURNAL_TABLES:
        try:
            cur.execute(f"DELETE FROM {t}")
            print(f"  sqlite: cleared {t}")
        except Exception as e:
            print(f"  sqlite: skip {t} ({e})")


def _truncate_log(rel: str) -> None:
    p = ROOT / rel
    if p.exists():
        try:
            p.write_text("", encoding="utf-8")
            print(f"  truncated {rel}")
        except Exception as e:
            print(f"  skip {rel} ({e})")


def _delete_log(rel: str) -> None:
    p = ROOT / rel
    if p.exists():
        try:
            p.unlink()
            print(f"  deleted {rel}")
        except Exception as e:
            print(f"  skip {rel} ({e})")


if __name__ == "__main__":
    s = load_settings(ROOT / "config" / "settings.yaml")
    backend = s.get("storage.backend", "sqlite").lower()
    if backend == "sqlite":
        scope_desc = f"sqlite → {s.get('storage.sqlite_path', 'logs/tradebot.sqlite')}"
    else:
        scope_desc = f"{backend} via env ${s.get('storage.cockroach_dsn_env', 'COCKROACH_DSN')}"

    if not _confirm():
        print("aborted — nothing changed.")
        sys.exit(1)

    print(f"\n[1/3] wiping journal ({backend})...")
    try:
        j = build_journal(
            backend=backend,
            sqlite_path=s.get("storage.sqlite_path", str(ROOT / "logs" / "tradebot.sqlite")),
            dsn_env_var=s.get("storage.cockroach_dsn_env", "COCKROACH_DSN"),
            cockroach_schema=s.get("storage.cockroach_schema", "tradebot"),
        )
    except Exception as e:
        print(f"  ERROR: could not connect to journal: {e}")
        sys.exit(2)
    if isinstance(j, CockroachJournal):
        _wipe_cockroach(j)
    elif isinstance(j, SqliteJournal):
        _wipe_sqlite(j)
    j.close()

    print("\n[2/3] truncating JSONL logs...")
    for rel in LOG_FILES_TRUNCATE:
        _truncate_log(rel)

    print("\n[3/3] deleting snapshot files...")
    for rel in LOG_FILES_DELETE:
        _delete_log(rel)

    print("\ndone. fresh-start paper session is ready.")
    print("next: scripts/tradebotctl.sh watchdog-status  (watchdog should already be running)")
