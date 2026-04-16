"""Verify the CockroachDB connection + schema.

Reads .env, assembles the DSN (from COCKROACH_DSN or the individual
COCKROACH_* fields), connects, runs `SELECT 1`, initializes the schema
(idempotent), and reports table counts.

Run:
  python scripts/test_db_connection.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.storage.journal import (
    CockroachJournal, resolve_cockroach_dsn,
)


def _redact(dsn: str) -> str:
    """Hide password so we can print the DSN safely."""
    if "://" not in dsn or "@" not in dsn:
        return dsn
    head, rest = dsn.split("://", 1)
    creds, tail = rest.split("@", 1)
    if ":" in creds:
        user, _ = creds.split(":", 1)
        creds = f"{user}:****"
    return f"{head}://{creds}@{tail}"


def main() -> int:
    try:
        dsn = resolve_cockroach_dsn()
    except RuntimeError as e:
        print(f"[config] {e}")
        return 2
    print(f"[config] DSN: {_redact(dsn)}")

    try:
        j = CockroachJournal(dsn)
    except ImportError as e:
        print(f"[deps] {e}")
        return 3
    except Exception as e:
        print("[connect] failed:")
        traceback.print_exc()
        return 4

    try:
        with j._conn.cursor() as cur:
            cur.execute("SELECT 1")
            _ = cur.fetchone()
        print("[connect] ok (SELECT 1 returned)")
        j.init_schema()
        print("[schema]  ok (tables created or already present)")
        with j._conn.cursor() as cur:
            for t in ("fills", "trades", "equity_curve"):
                cur.execute(f"SELECT count(*) FROM {t}")
                n = cur.fetchone()[0]
                print(f"[rows]    {t}: {n}")
    finally:
        j.close()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
