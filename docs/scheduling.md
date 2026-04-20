# Scheduling — cron / launchd / systemd

**TL;DR:**
- **Mac (local, personal)** → install the watchdog (`tradebotctl watchdog-install`) for supervision + `cron` for nightly reports. The watchdog wraps the bot under launchd with KeepAlive=true, Discord/Slack crash alerts, and a stale-heartbeat recycler.
- **Linux VPS (production)** → `deploy/systemd` for the bot + `cron` for nightly reports.
- **Don't mix modes** — if the watchdog is loaded, don't also run `tradebotctl start` / `stop` via cron. Cron should only run the periodic reports.

All control flows through one wrapper: `scripts/tradebotctl.sh`.

## The wrapper

```bash
# convenience — every subcommand uses the .venv python and loads .env
./scripts/tradebotctl.sh start           # spawn run_paper.py in background
./scripts/tradebotctl.sh stop            # graceful stop via KILL flag, then SIGTERM
./scripts/tradebotctl.sh restart
./scripts/tradebotctl.sh status          # running (pid N) | stopped
./scripts/tradebotctl.sh logs            # tail -f the log file
./scripts/tradebotctl.sh backtest        # one-shot synthetic backtest
./scripts/tradebotctl.sh priors --days 30
./scripts/tradebotctl.sh walkforward --max-windows 8
./scripts/tradebotctl.sh dashboard       # foreground (for development)
./scripts/tradebotctl.sh testdb          # verify CockroachDB connection
```

The wrapper is idempotent — `start` twice is a no-op; `stop` when nothing's
running is a no-op. Safe to call from cron.

## Mode 1 — cron (simplest, Mac or Linux)

Good for: running a long-lived dev machine, everything in one place.

Trade-off: cron only triggers at scheduled times; it won't auto-restart
the bot if it crashes mid-session. For that, use launchd/systemd.

### Install on Mac or Linux

1. Edit `deploy/cron/crontab.example` → replace `/ABSOLUTE/PATH/TO/tradebot` with your real path.
2. Install:
   ```bash
   crontab /absolute/path/to/tradebot/deploy/cron/crontab.example
   crontab -l                      # verify
   ```
3. **Mac only:** give cron full-disk access in
   `System Settings → Privacy & Security → Full Disk Access → + → /usr/sbin/cron`.
   Otherwise cron can't read files under `~/Documents`.

### What the default schedule does

| When (ET) | Action |
|---|---|
| 09:25, Mon–Fri | `tradebotctl start` (the bot then waits for 09:30 open internally) |
| 16:05, Mon–Fri | `tradebotctl stop` (EOD flat-close already ran at 15:45 inside the bot) |
| 18:00, Mon–Fri | `tradebotctl priors --days 30` → writes `logs/priors.YYYYMMDD.log` |
| 22:00, Mon–Fri | `tradebotctl walkforward` → writes `logs/walkforward.YYYYMMDD.log` |
| Sun 20:00 | Longer walk-forward (quarterly windows) |

All times use `TZ=America/New_York` in the crontab, so daylight-saving
transitions are handled correctly on both sides.

## Mode 2 — launchd + watchdog (Mac, recommended for anything past a toy run)

Good for: wanting the bot to auto-restart on crash, survive reboots,
start at login, **detect silent hangs** (main loop wedged while the
process is alive), and get a Discord/Slack alert every time it
restarts.

Architecture:

```
launchd (KeepAlive=true, ThrottleInterval=10, RunAtLoad=true)
  └── scripts/watchdog_run.py
        └── scripts/run_paper.py            (the actual bot)
```

The wrapper script `scripts/watchdog_run.py` does three things launchd
can't:
- posts a Discord/Slack message via `src.notify` on every abnormal exit
- records every start / exit / stale event to `logs/watchdog_events.jsonl`
- watches `logs/heartbeat.txt` (rewritten by the bot's main loop every
  tick) and kills + recycles the child if it goes stale

### Install (one-liner)

```bash
./scripts/tradebotctl.sh watchdog-install
```

This rewrites the paths in `deploy/launchd/com.tradebot.paper.plist` to
match your checkout, copies the plist to `~/Library/LaunchAgents/`, and
`launchctl load`s it. If a previous version is loaded it's unloaded
first so you pick up changes.

### Verify

```bash
./scripts/tradebotctl.sh watchdog-status
```

Sample output:

```
watchdog: running (pid=60735, last exit=0)
heartbeat: 42s ago
last event: {"kind":"start","pid":60735,"ts":"..."}
```

A healthy heartbeat in market hours is < 5 minutes old. "(never written
yet)" is normal for the first ~2 minutes after install while the child
imports and enters its main loop (that's the `WATCHDOG_STARTUP_GRACE_SEC`
= 120s).

### Tail logs

```bash
./scripts/tradebotctl.sh logs                    # bot stdout
tail -f logs/watchdog_events.jsonl               # watchdog's own log
tail -f logs/tradebot.err                         # child stderr
```

### Stop temporarily / permanently

```bash
launchctl stop com.tradebot.paper                # watchdog recycles it (brief pause)
./scripts/tradebotctl.sh watchdog-uninstall      # removes the LaunchAgent entirely
```

### Interaction with cron

With the watchdog managing the bot, **remove the `start`/`stop` lines
from cron** (keep the nightly reports: catalysts, priors,
walk-forward, calibration, etc.). The two modes shouldn't compete for
control of the bot.

### Tuning knobs (rarely needed)

The watchdog reads these env vars at startup:

| Var | Default | Meaning |
|---|---|---|
| `WATCHDOG_HEARTBEAT_STALE_SEC` | 300 | heartbeat age that trips "stale" → kill child |
| `WATCHDOG_CHECK_INTERVAL_SEC`  | 30  | how often we poll child + heartbeat |
| `WATCHDOG_STARTUP_GRACE_SEC`   | 120 | suppresses stale detection during bootstrap |

Set them in the plist's `<key>EnvironmentVariables</key>` dict if you
need to change them.

## Mode 3 — systemd (Linux VPS, production)

Good for: production deployment on a $5–10/mo VPS. Best supervision,
standard journald log management, auto-restart.

1. Edit `deploy/systemd/tradebot.service` — set `User`, `WorkingDirectory`, `EnvironmentFile`.
2. Install:
   ```bash
   sudo cp deploy/systemd/tradebot.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now tradebot
   sudo journalctl -u tradebot -f
   ```
3. Keep cron for nightly priors/walk-forward — systemd only supervises the
   main loop.

## Logs

| File | Written by | What it has |
|---|---|---|
| `logs/tradebot.out` | the bot itself | every filter decision, fill, exit, halt |
| `logs/tradebot.err` | the bot itself | unhandled exceptions (launchd/systemd only) |
| `logs/heartbeat.txt` | the bot's main loop | ISO timestamp, rewritten every tick; watchdog reads this |
| `logs/watchdog_events.jsonl` | the watchdog wrapper | start / exit / clean_shutdown / heartbeat_stale events |
| `logs/slippage_calibration.jsonl` | paper broker | per-fill predicted-vs-observed slippage |
| `logs/calibration_history.jsonl` | auto-cal model | every constant adjustment made by the auto-calibrator |
| `logs/cron.out` | cron wrapper | start/stop invocations, cron errors |
| `logs/priors.YYYYMMDD.log` | nightly cron | measured priors for the last N days |
| `logs/walkforward.YYYYMMDD.log` | nightly cron | sliding-window prior fits |

Rotation: add `logrotate` (Linux) or `newsyslog` (Mac) if logs grow. For
paper runs you'll be fine for months without it. The JSONL files
(`watchdog_events`, `slippage_calibration`, `calibration_history`) are
append-only and compress well if you ever want to archive them.

## Kill switch

```bash
# cooperative: main loop exits cleanly after finishing the current tick
# (works regardless of which supervisor is in use)
touch KILL

# launchd+watchdog mode: stop AND prevent auto-restart
./scripts/tradebotctl.sh watchdog-uninstall

# launchd+watchdog mode: brief pause only (watchdog respawns the child)
launchctl stop com.tradebot.paper

# bare `tradebotctl start` mode (no watchdog): hard stop
./scripts/tradebotctl.sh stop

# wipe everything (nuclear)
./scripts/tradebotctl.sh watchdog-uninstall 2>/dev/null || true
./scripts/tradebotctl.sh stop 2>/dev/null || true
crontab -r                    # removes ALL of YOUR cron jobs — careful
```

If you're on a VPS, `sudo systemctl stop tradebot`.
