# Storage — Trade Journal

## Backend

**Local SQLite file, zero setup.** Previous versions supported
CockroachDB as a second backend; it was removed because the bot's write
frequency (ensemble decisions + LSTM predictions + equity ticks + fills
per symbol per loop tick) outran Cockroach Serverless's free-tier
request-unit cap within about two weeks of live paper trading. SQLite
on local disk handles 10k+ writes/sec — well past anything the bot can
generate — and has no cloud-cost surface.

## What gets persisted

Defined in `src/storage/schema.sql`:

- **`fills`** — every raw broker fill (buy/sell, qty, price, fee, tag).
- **`trades`** — realized round-trips (entry + exit collapsed), with `pnl`,
  `pnl_pct`, `entry_tag`, `exit_reason`. This is what `compute_priors.py`
  reads.
- **`equity_curve`** — per-tick snapshots of equity, cash, intraday P&L.
- **`ml_predictions`** — every LSTM inference for post-hoc calibration.
- **`ensemble_decisions`** — every regime-weighted decision emit/block.

All timestamps stored as UTC.

## Config

```yaml
# config/settings.yaml
storage:
  sqlite_path: logs/tradebot.sqlite
```

On Jetson with SD-card data root set (`TRADEBOT_DATA_ROOT=/media/orin/tradebot-data`),
the SQLite file lives under the SD card instead of the repo `logs/`. See
`docs/jetson_plug_and_play.md` for the one-command setup.

## Computing priors from the journal

```bash
python scripts/compute_priors.py               # last 90 days, all symbols
python scripts/compute_priors.py --days 30 --symbol SPY --min-trades 50
```

Output is a human-readable summary plus a paste-ready YAML block. Keep a
running note of how priors evolve — large swings month-over-month mean
your sample is too small for Kelly to trust.

## Wiring the journal into the bot

```python
from src.storage.journal import build_journal
from src.brokers.paper import PaperBroker

journal = build_journal(sqlite_path="logs/tradebot.sqlite")
broker = PaperBroker(starting_equity=10_000, journal=journal)
```

The broker logs every fill and every realized round-trip automatically.
If the journal write fails, the trade loop keeps running — logging never
kills the bot, but failures DO alert via `issue_reporter.report_issue`
so silent data loss is impossible.

## Operational notes

- SQLite uses `PRAGMA journal_mode=WAL;` so reads don't block writes.
  Safe for a single-process bot.
- Don't point two bot instances at the same SQLite file.
- Back up the SQLite file daily. On Jetson: `cp logs/tradebot.sqlite
  logs/tradebot.$(date +%Y%m%d).sqlite.bak` in a nightly cron.
- Never put any secret in `.env.example`. Real values live in `.env`,
  which is gitignored.

## Wiping the journal

```bash
bash scripts/tradebotctl.sh wipe-journal
# requires typed DESTROY confirmation
```

Clears every table + slippage-calibration history + broker-state
snapshot. Preserves `tradebot.out` / `tradebot.err` log files so you
still have the process log.
