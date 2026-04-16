# Catalyst calendar — earnings + FDA

## What it does

Blocks the bot from opening new positions on a symbol's full trading day
when a scheduled catalyst is coming. Handled by the same 14-filter chain
via `f05_economic_calendar` — `ctx.econ_blackout=True` when the bot's
pre-entry check finds the symbol on the blackout list.

Full-day blackout was chosen over a narrow time window because:
- Earnings announcements often leak minutes early or run late.
- Pre-announcement IV inflation, post-announcement gap risk, and
  skew-collapse all make the whole day dangerous, not just the headline
  moment.
- The bot can't micro-time announcements; staying out is the right move.

## Data sources (free, stacked by preference)

1. **`config/catalysts.yaml`** — manually maintained FDA PDUFA / adcom
   dates and any one-off overrides you add. Always consulted.
2. **Finnhub** — earnings calendar (covers all US equities). Activates
   automatically when `FINNHUB_KEY` is in `.env`.
3. **yfinance** — no API key required. Last-resort fallback for
   earnings. Slower (one HTTP call per symbol).

All three are consulted on each refresh; results are deduped by
`(symbol, date, event_type)`.

## Editing the manual overrides

`config/catalysts.yaml` (empty by default):

```yaml
fda:
  - symbol: MRNA
    date: 2026-05-15
    event: PDUFA decision
earnings:
  - symbol: AAPL
    date: 2026-05-02
    timing: amc
    event: Q2 earnings
```

`timing` is informational (`bmo`, `amc`, `unknown`). The bot blacks out
the whole day regardless.

## Refreshing the calendar

Call this daily — it writes nothing sensitive and is safe from cron.

```bash
# once, interactive
./scripts/tradebotctl.sh catalysts
./scripts/tradebotctl.sh catalysts --days 30 --json

# scheduled (already in deploy/cron/crontab.example)
0  7  * * 1-5   $CTL catalysts > $TRADEBOT/logs/catalysts.$(date +%Y%m%d).log 2>&1
```

The bot also refreshes on startup (`TradeBot.__init__` calls
`_refresh_catalysts()`), so the blackout list is always current when
launchd/systemd restarts the process.

## Observability

Upcoming catalysts are posted once per refresh to Discord/Slack:

```
[info] catalysts: 3 upcoming catalysts — AAPL:earnings:2026-05-02, NVDA:earnings:2026-05-21, MRNA:fda:2026-05-15
```

Blocked trades show up as filter decisions in `logs/tradebot.out` with
the `econ_blackout` reason. They do NOT generate a Discord notification
(would be too noisy) — check logs if you're debugging why a signal didn't
fire.

## Settings

`config/settings.yaml`:

```yaml
catalysts:
  enabled: true
  lookahead_days: 14                 # how far ahead to fetch per refresh
  static_yaml: config/catalysts.yaml # manual overrides
```

Disable everything by setting `enabled: false` — calendar stops refreshing
and the per-symbol blackout set stays empty.
