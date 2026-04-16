# Scheduling — cron / launchd / systemd

**TL;DR:**
- **Mac (local, personal)** → use the `deploy/launchd` agent for the main bot + `cron` for nightly reports.
- **Linux VPS (production)** → use `deploy/systemd` for the bot + `cron` for nightly reports.
- **Don't mix modes** (don't run launchd AND cron-start for the same bot).

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

## Mode 2 — launchd (Mac, more robust)

Good for: wanting the bot to auto-restart on crash, survive reboots,
start at login. The bot itself handles market-open/close so it's fine to
let it run 24/7.

1. Edit `deploy/launchd/com.tradebot.paper.plist` — path to `.venv/bin/python` and repo are pre-filled for `/Users/vivekadhikari/Documents/Claude/Projects/tradebot`. Adjust if different.
2. Install:
   ```bash
   cp deploy/launchd/com.tradebot.paper.plist ~/Library/LaunchAgents/
   launchctl load  ~/Library/LaunchAgents/com.tradebot.paper.plist
   launchctl start com.tradebot.paper
   launchctl list | grep tradebot         # should show a numeric PID
   ```
3. Tail logs:
   ```bash
   tail -f logs/tradebot.out
   ```
4. Stop / unload:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.tradebot.paper.plist
   ```

With launchd managing the bot, **remove the `start`/`stop` lines from cron**
(keep the nightly priors/walk-forward lines). The two modes shouldn't
compete for control of the bot.

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
| `logs/cron.out` | cron wrapper | start/stop invocations, cron errors |
| `logs/priors.YYYYMMDD.log` | nightly cron | measured priors for the last N days |
| `logs/walkforward.YYYYMMDD.log` | nightly cron | sliding-window prior fits |

Rotation: add `logrotate` (Linux) or `newsyslog` (Mac) if logs grow. For
paper runs you'll be fine for months without it.

## Kill switch

```bash
# cooperative: main loop exits cleanly after finishing the current tick
touch KILL

# hard stop
./scripts/tradebotctl.sh stop

# wipe everything running under either mode
launchctl unload ~/Library/LaunchAgents/com.tradebot.paper.plist 2>/dev/null || true
./scripts/tradebotctl.sh stop
crontab -r                    # removes ALL of YOUR cron jobs — careful
```

If you're on a VPS, `sudo systemctl stop tradebot`.
