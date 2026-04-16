# Storage — Trade Journal

## What gets persisted

Three tables, defined in `src/storage/schema.sql`:

- **`fills`** — every raw broker fill (buy/sell, qty, price, fee, tag).
- **`trades`** — realized round-trips (entry + exit collapsed), with `pnl`,
  `pnl_pct`, `entry_tag`, `exit_reason`. This is what `compute_priors.py`
  reads.
- **`equity_curve`** — per-tick snapshots of equity, cash, and intraday P&L.

All timestamps are stored as UTC.

## Backends

Same `TradeJournal` interface; pick a backend by config.

### SQLite (default)

Zero setup. File-backed. Ideal for paper trading on a laptop.

```yaml
# config/settings.yaml
storage:
  backend: sqlite
  sqlite_path: logs/tradebot.sqlite
```

### CockroachDB (recommended for long-term / multi-machine / production)

Postgres-wire-compatible. Works identically with plain Postgres, Neon, or Supabase.

**You have two options for connecting. Pick one.**

#### Option A — full connection string (simplest, recommended)

In CockroachDB Cloud Console → **Connect → General connection string** → copy
the entire `postgresql://...` line. That single string contains
`host + port + user + password + database + sslmode` — nothing else needed.

Paste it into your `.env`:

```
COCKROACH_DSN=postgresql://user:password@host:26257/defaultdb?sslmode=verify-full
```

Leave every `COCKROACH_HOST` / `COCKROACH_USER` / `COCKROACH_PASSWORD` line
alone; they are ignored when `COCKROACH_DSN` is set.

#### Option B — individual fields

If you don't have the one-line DSN, fill the parts and the bot assembles the
DSN for you:

```
COCKROACH_DSN=
COCKROACH_HOST=your-cluster-host.cockroachlabs.cloud
COCKROACH_PORT=26257
COCKROACH_USER=your-sql-user
COCKROACH_PASSWORD=your-sql-password
COCKROACH_DATABASE=tradebot
COCKROACH_SSLMODE=verify-full
# Optional, only if your cluster requires a CA cert:
COCKROACH_SSLROOTCERT=
# Optional, multi-tenant Serverless routing:
COCKROACH_CLUSTER=
```

#### Common final steps

1. Switch the backend in `config/settings.yaml`:
   ```yaml
   storage:
     backend: cockroach
     cockroach_dsn_env: COCKROACH_DSN
   ```
2. Install the driver:
   ```bash
   pip install 'psycopg[binary]'
   ```
3. Verify before running the bot:
   ```bash
   python scripts/test_db_connection.py
   ```
   You should see `[connect] ok`, `[schema] ok`, and three table row counts
   (all zero on first run). The password is redacted in the printed DSN so
   logs stay safe.

Schema initialization happens automatically on first connection via
`journal.init_schema()` — tables are created idempotently (`IF NOT EXISTS`).

## Computing priors from the journal

```bash
# default: SQLite, last 90 days, all symbols
python scripts/compute_priors.py

# CockroachDB, last 30 days, SPY only, require 50+ trades
python scripts/compute_priors.py --backend cockroach --days 30 --symbol SPY --min-trades 50
```

Output is both a human-readable summary and a paste-ready YAML block. Keep
a running note of how priors evolve — large swings month-over-month mean
your sample is too small for Kelly to trust.

## Wiring the journal into the bot

```python
from src.storage.journal import build_journal
from src.brokers.paper import PaperBroker

journal = build_journal(
    backend="sqlite",                     # or "cockroach"
    sqlite_path="logs/tradebot.sqlite",
)
broker = PaperBroker(starting_equity=10_000, journal=journal)
```

The broker logs **every** fill and **every** realized round-trip
automatically. If the journal raises, the trade loop keeps running — we
never let logging kill the bot.

## Operational notes

- SQLite writes with `PRAGMA journal_mode=WAL;` so reads don't block
  writes. Safe for a single-process bot.
- Don't point two bot instances at the same SQLite file; use CockroachDB
  if you want to run multiple instances.
- Back up `logs/tradebot.sqlite` daily if you're using SQLite. For
  CockroachDB the cluster handles durability + backups.
- Never put the DSN in `.env.example` or any file that ships to git.
  Real values live only in `.env` which is `.gitignore`'d.
