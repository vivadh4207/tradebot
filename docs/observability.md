# Observability — Notifier + Dashboard + Walk-Forward + Watchdog

## 1. Notifier (Discord / Slack webhook)

Push fills, halts, daily summaries, **watchdog crash alerts**,
**calibration-ratio excursions**, and errors to a phone-accessible
channel.

### Setup

Discord is recommended (fastest to set up, free, mobile push built in).

1. In Discord, pick a channel → **Edit Channel** → **Integrations** → **Webhooks** → **New Webhook** → copy URL.
2. Add to `.env`:
   ```
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   ```
   Slack alternative:
   ```
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   ```
   Set only ONE. If both are set, Discord wins.
3. Restart the bot. Fills, daily-loss halts, and EOD summaries will post automatically.

### What gets posted

| Event | Level | When |
|---|---|---|
| `[info] entry: SYMBOL buy x5 @ 123.45 src=momentum` | info | every filled entry |
| `[warn] HALT: Daily loss halt hit: day_pnl=-204.50 ...` | warn | daily loss exceeds `account.max_daily_loss_pct` |
| `[info] daily: EOD 2026-04-18: equity=9800.00 day_pnl=-200.00 ...` | info | once per session, after 15:45 ET |
| `[ERROR] tradebot: main_loop_error: <exception>` | error | any unhandled exception in main loop |
| `[warn] calibration: Slippage ratio=1.67 (n=47, pred=3.2bps, obs=5.3bps). Adjusted 2 constants.` | warn | auto-cal finds observed/predicted outside [0.5, 1.5] |
| `[ERROR] watchdog: tradebot CRASHED rc=1 after 842s. Last stderr: <line>` | error | child exits nonzero |
| `[info] watchdog: tradebot exited cleanly after 842s — restarting` | info | child exits 0 but we didn't ask it to |
| `[ERROR] watchdog: heartbeat stale 312s — killing child pid=60735 so launchd can restart` | error | main loop wedged; forced recycle |

### Guarantees

- Non-blocking — a slow webhook endpoint cannot stall the trade loop. All posts go through a daemon thread with a 64-message bounded queue; overflow drops the oldest message.
- Fail-soft — any network/JSON error is swallowed. Notifications are best-effort.
- No credentials, account numbers, or DSNs are ever logged or posted.

## 2. Dashboard

Read-only web UI that reads the journal and renders:
- 4 top metrics cards (trades, win rate, avg win/loss, total P&L)
- equity curve (Chart.js, 1-min refresh)
- closed-trades table with entry tag + exit reason

### Run it

```bash
pip install 'uvicorn[standard]' fastapi
python scripts/run_dashboard.py            # http://127.0.0.1:8000
```

On a remote VPS, SSH-tunnel it — never expose the port publicly (there's no auth layer):

```bash
ssh -L 8000:localhost:8000 user@your-vps
# then open http://localhost:8000 on your laptop
```

### API endpoints

| Path | Purpose |
|---|---|
| `GET /` | HTML page (self-contained, no templates to edit) |
| `GET /api/equity?days=30` | equity-curve points |
| `GET /api/trades?days=30&limit=500` | closed-trade rows |
| `GET /api/metrics?days=30` | aggregated stats + EV |
| `GET /api/calibration` | current auto-cal stats + recent history entries |
| `GET /api/daily_report` | today's EOD snapshot with `keep_or_tune` decision |
| `GET /api/var` | latest Monte Carlo VaR / CVaR report |
| `GET /api/backtest_runs` | reproducibility log of every backtest run |
| `GET /api/watchdog` | watchdog status: recent events + heartbeat age |

All reads come from whichever backend is configured in
`config/settings.yaml` (`sqlite` or `cockroach`). Watchdog +
calibration endpoints read from the JSONL files in `logs/`.

## 3. Walk-forward

Quarterly discipline: refit sizing priors from the most recent year of
closed trades and decide if the system is tradable next quarter.

### Run it

```bash
python scripts/run_walkforward.py                             # from settings backend
python scripts/run_walkforward.py --backend cockroach         # explicit
python scripts/run_walkforward.py --train-days 252 --test-days 63 --min-trades 50
```

Example output:

```
Walk-forward (cockroach): train=365d  test=63d
TRAIN                      TEST                      N     WR     AW     AL      EV  TRADABLE
2024-10-18 → 2025-10-17   2025-10-17 → 2025-12-19  184 0.581  0.028  0.019  +0.0083  yes
2024-07-19 → 2025-07-18   2025-07-18 → 2025-09-19  161 0.543  0.022  0.021  +0.0024  no
...
```

- `TRADABLE=yes` means the fit had `n >= min-trades` and `EV > min-ev`.
- Use the **most recent** tradable window's `WR / AW / AL` as the next quarter's priors (paste into `SimConfig` defaults or `settings.yaml`).
- If no recent window is tradable, the system doesn't have statistically supported edge right now — stay in paper.

### Historical bars (optional)

If you want to re-simulate the strategy on historical bars rather than
only measuring realized paper trades, `HistoricalDataProvider` pulls up
to 3 years of daily bars from Alpaca with disk caching in
`data_cache/<symbol>_daily_Ny.json`. Plug it into `BacktestSimulator`
in place of `SyntheticDataAdapter` for long-range walk-forward.

## 4. Watchdog (Mac launchd supervision)

On macOS the bot runs under `scripts/watchdog_run.py`, which is
supervised by launchd. The watchdog gives you three observability
surfaces the bot itself can't provide:

- **Crash alerts via the notifier.** Every abnormal child exit posts a
  Discord/Slack message with exit code and the last line of stderr.
- **Event log** at `logs/watchdog_events.jsonl`. Append-only JSONL with
  `start`, `exit`, `clean_shutdown`, and `heartbeat_stale` records.
- **Heartbeat freshness.** The bot writes `logs/heartbeat.txt` every
  main-loop tick. The watchdog reads it; `tradebotctl watchdog-status`
  prints the age.

Quick health check:

```bash
./scripts/tradebotctl.sh watchdog-status
```

Dashboard:

```
GET /api/watchdog
→ { "heartbeat_age_sec": 42, "last_event": {...}, "recent_events": [...] }
```

The dashboard HTML page surfaces these as the "Loop insights" panel
alongside the existing calibration panel — if the heartbeat age goes
red, the bot wedged and the watchdog is about to recycle it.

See `OPERATIONS.md § Watchdog supervision` for the layering of each
failure mode and which component handles it.
