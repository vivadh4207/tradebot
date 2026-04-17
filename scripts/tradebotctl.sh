#!/usr/bin/env bash
# tradebotctl — control script for the tradebot paper loop + helpers.
#
# Usage:
#   tradebotctl.sh start          # start run_paper.py in background
#   tradebotctl.sh stop           # stop the running bot
#   tradebotctl.sh status         # running | stopped (with pid)
#   tradebotctl.sh restart        # stop + start
#   tradebotctl.sh logs           # tail the stdout/stderr log
#   tradebotctl.sh backtest       # one-shot synthetic backtest
#   tradebotctl.sh priors [...]   # compute_priors.py passthrough
#   tradebotctl.sh walkforward .. # run_walkforward.py passthrough
#   tradebotctl.sh dashboard [..] # run_dashboard.py passthrough (foreground)
#   tradebotctl.sh testdb         # test_db_connection.py
#
# Designed to be cron-safe: no tty assumptions, no prompts, nonzero exit on
# failure, idempotent start/stop.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TRADEBOT_PY:-$ROOT/.venv/bin/python}"
PID_FILE="$ROOT/logs/tradebot.pid"
LOG_FILE="$ROOT/logs/tradebot.out"
KILL_FILE="$ROOT/KILL"

# launchd (macOS) watchdog integration. Paths are derived, not assumed,
# so copying the repo elsewhere still works.
LAUNCHD_LABEL="com.tradebot.paper"
LAUNCHD_SRC_PLIST="$ROOT/deploy/launchd/${LAUNCHD_LABEL}.plist"
LAUNCHD_INSTALLED_PLIST="$HOME/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"

# launchd dashboard agent (separate from the bot watchdog).
DASHBOARD_LABEL="com.tradebot.dashboard"
DASHBOARD_SRC_PLIST="$ROOT/deploy/launchd/${DASHBOARD_LABEL}.plist"
DASHBOARD_INSTALLED_PLIST="$HOME/Library/LaunchAgents/${DASHBOARD_LABEL}.plist"

mkdir -p "$ROOT/logs"

# .env is intentionally NOT sourced here. Shell parsing is fragile with
# values that contain spaces, '&', '?', or URL-looking fragments (like a
# Postgres DSN). Every Python entry point calls python-dotenv's
# load_dotenv(...) itself, so the secrets make it into the process env
# exactly once, correctly.

_is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

cmd_start() {
  if _is_running; then
    echo "already running (pid $(cat "$PID_FILE"))"
    return 0
  fi
  rm -f "$KILL_FILE"
  echo "[$(date -Iseconds 2>/dev/null || date)] starting run_paper.py" >> "$LOG_FILE"
  nohup "$PY" "$ROOT/scripts/run_paper.py" >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 1
  if _is_running; then
    echo "started (pid $(cat "$PID_FILE"))"
  else
    echo "start failed — check $LOG_FILE"
    rm -f "$PID_FILE"
    return 1
  fi
}

cmd_stop() {
  if _is_running; then
    local pid; pid="$(cat "$PID_FILE")"
    touch "$KILL_FILE"                    # cooperative shutdown signal
    kill "$pid" 2>/dev/null || true
    # give it up to 10s to exit cleanly
    for _ in $(seq 1 10); do
      if ! kill -0 "$pid" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    echo "[$(date -Iseconds 2>/dev/null || date)] stopped" >> "$LOG_FILE"
    echo "stopped"
  else
    echo "not running"
    rm -f "$PID_FILE"
  fi
}

cmd_status() {
  if _is_running; then
    echo "running (pid $(cat "$PID_FILE"))"
  else
    echo "stopped"
  fi
}

cmd_logs() {
  touch "$LOG_FILE"
  tail -f "$LOG_FILE"
}

# ----- launchd watchdog (macOS only) -------------------------------------
# Install, uninstall, and inspect the com.tradebot.paper LaunchAgent. The
# agent runs scripts/watchdog_run.py, which supervises run_paper.py.
cmd_watchdog_install() {
  if [[ "$(uname)" != "Darwin" ]]; then
    echo "watchdog-install is macOS-only (uses launchd). On Linux use systemd."
    return 2
  fi
  if [[ ! -f "$LAUNCHD_SRC_PLIST" ]]; then
    echo "source plist missing: $LAUNCHD_SRC_PLIST"
    return 1
  fi
  mkdir -p "$(dirname "$LAUNCHD_INSTALLED_PLIST")"
  # Rewrite the hardcoded path in the shipped plist to match this checkout.
  # sed -i '' for BSD (macOS); temp file for portability.
  tmp_plist="$(mktemp)"
  awk -v new_root="$ROOT" -v new_py="$PY" '
    {
      gsub("/Users/vivekadhikari/Documents/Claude/Projects/tradebot/.venv/bin/python", new_py);
      gsub("/Users/vivekadhikari/Documents/Claude/Projects/tradebot", new_root);
      print;
    }
  ' "$LAUNCHD_SRC_PLIST" > "$tmp_plist"
  mv "$tmp_plist" "$LAUNCHD_INSTALLED_PLIST"
  # If a previous version is loaded, unload first so we pick up changes.
  launchctl unload "$LAUNCHD_INSTALLED_PLIST" 2>/dev/null || true
  if launchctl load "$LAUNCHD_INSTALLED_PLIST"; then
    echo "watchdog installed + loaded: $LAUNCHD_INSTALLED_PLIST"
    echo "check:    tradebotctl watchdog-status"
    echo "alerts:   set DISCORD_WEBHOOK_URL or SLACK_WEBHOOK_URL in .env"
  else
    echo "launchctl load failed — see console.app for launchd errors"
    return 1
  fi
}

cmd_watchdog_uninstall() {
  if [[ "$(uname)" != "Darwin" ]]; then
    echo "watchdog-uninstall is macOS-only."
    return 2
  fi
  if [[ -f "$LAUNCHD_INSTALLED_PLIST" ]]; then
    launchctl unload "$LAUNCHD_INSTALLED_PLIST" 2>/dev/null || true
    rm -f "$LAUNCHD_INSTALLED_PLIST"
    echo "watchdog unloaded + removed"
  else
    echo "not installed"
  fi
}

cmd_dashboard_install() {
  if [[ "$(uname)" != "Darwin" ]]; then
    echo "dashboard-install is macOS-only (uses launchd)."
    return 2
  fi
  if [[ ! -f "$DASHBOARD_SRC_PLIST" ]]; then
    echo "source plist missing: $DASHBOARD_SRC_PLIST"
    return 1
  fi
  mkdir -p "$(dirname "$DASHBOARD_INSTALLED_PLIST")"
  tmp_plist="$(mktemp)"
  awk -v new_root="$ROOT" -v new_py="$PY" '
    {
      gsub("/Users/vivekadhikari/Documents/Claude/Projects/tradebot/.venv/bin/python", new_py);
      gsub("/Users/vivekadhikari/Documents/Claude/Projects/tradebot", new_root);
      print;
    }
  ' "$DASHBOARD_SRC_PLIST" > "$tmp_plist"
  mv "$tmp_plist" "$DASHBOARD_INSTALLED_PLIST"
  launchctl unload "$DASHBOARD_INSTALLED_PLIST" 2>/dev/null || true
  if launchctl load "$DASHBOARD_INSTALLED_PLIST"; then
    echo "dashboard installed + loaded: $DASHBOARD_INSTALLED_PLIST"
    echo "access:   http://127.0.0.1:8000"
    echo "status:   tradebotctl dashboard-status"
    echo "logs:     tail -f $ROOT/logs/dashboard.out"
  else
    echo "launchctl load failed — see logs/dashboard.err"
    return 1
  fi
}

cmd_dashboard_uninstall() {
  if [[ "$(uname)" != "Darwin" ]]; then
    echo "dashboard-uninstall is macOS-only."
    return 2
  fi
  if [[ -f "$DASHBOARD_INSTALLED_PLIST" ]]; then
    launchctl unload "$DASHBOARD_INSTALLED_PLIST" 2>/dev/null || true
    rm -f "$DASHBOARD_INSTALLED_PLIST"
    echo "dashboard unloaded + removed"
  else
    echo "not installed"
  fi
}

cmd_dashboard_status() {
  if [[ "$(uname)" != "Darwin" ]]; then
    echo "dashboard-status is macOS-only."
    return 2
  fi
  local entry; entry="$(launchctl list 2>/dev/null | awk -v l="$DASHBOARD_LABEL" '$3 == l')"
  if [[ -z "$entry" ]]; then
    echo "dashboard: not loaded"
    echo "install:  tradebotctl dashboard-install"
    return 0
  fi
  local pid status
  pid="$(awk '{print $1}' <<< "$entry")"
  status="$(awk '{print $2}' <<< "$entry")"
  if [[ "$pid" == "-" ]]; then
    echo "dashboard: loaded but not running (last exit=$status) — check logs/dashboard.err"
  else
    echo "dashboard: running (pid=$pid, last exit=$status)"
    echo "open:      http://127.0.0.1:8000"
  fi
}

cmd_watchdog_status() {
  if [[ "$(uname)" != "Darwin" ]]; then
    echo "watchdog-status is macOS-only."
    return 2
  fi
  local entry; entry="$(launchctl list 2>/dev/null | awk -v l="$LAUNCHD_LABEL" '$3 == l')"
  if [[ -z "$entry" ]]; then
    echo "watchdog: not loaded"
    echo "install:  tradebotctl watchdog-install"
    return 0
  fi
  # columns: PID STATUS LABEL
  local pid status
  pid="$(awk '{print $1}' <<< "$entry")"
  status="$(awk '{print $2}' <<< "$entry")"
  if [[ "$pid" == "-" ]]; then
    echo "watchdog: loaded but not running (last exit=$status)"
  else
    echo "watchdog: running (pid=$pid, last exit=$status)"
  fi
  # Heartbeat freshness — tells you if the child is alive, not just the wrapper.
  if [[ -f "$ROOT/logs/heartbeat.txt" ]]; then
    # portable stat: macOS uses -f, Linux -c
    local hb_mtime now age
    hb_mtime="$(stat -f %m "$ROOT/logs/heartbeat.txt" 2>/dev/null || stat -c %Y "$ROOT/logs/heartbeat.txt" 2>/dev/null)"
    now="$(date +%s)"
    age=$(( now - hb_mtime ))
    echo "heartbeat: ${age}s ago"
  else
    echo "heartbeat: (never written yet)"
  fi
  # Last event from the watchdog's own log
  if [[ -f "$ROOT/logs/watchdog_events.jsonl" ]]; then
    echo "last event:"
    tail -n 1 "$ROOT/logs/watchdog_events.jsonl"
  fi
}

case "${1:-}" in
  start)        cmd_start ;;
  stop)         cmd_stop ;;
  restart)      cmd_stop || true; cmd_start ;;
  status)       cmd_status ;;
  logs)         cmd_logs ;;
  backtest)     shift; "$PY" "$ROOT/scripts/run_backtest.py" "$@" ;;
  priors)       shift; "$PY" "$ROOT/scripts/compute_priors.py" "$@" ;;
  walkforward)  shift; "$PY" "$ROOT/scripts/run_walkforward.py" "$@" ;;
  catalysts)    shift; "$PY" "$ROOT/scripts/refresh_catalysts.py" "$@" ;;
  train-lstm)   shift; "$PY" "$ROOT/scripts/train_lstm.py" "$@" ;;
  resolve-ml)   shift; "$PY" "$ROOT/scripts/resolve_ml_predictions.py" "$@" ;;
  analyze-ml)   shift; "$PY" "$ROOT/scripts/analyze_ml_predictions.py" "$@" ;;
  compare-lstm) shift; "$PY" "$ROOT/scripts/compare_lstm.py" "$@" ;;
  analyze-ensemble) shift; "$PY" "$ROOT/scripts/analyze_ensemble.py" "$@" ;;
  propose-weights)  shift; "$PY" "$ROOT/scripts/propose_weights.py" "$@" ;;
  var)              shift; "$PY" "$ROOT/scripts/daily_var.py" "$@" ;;
  drift)            shift; "$PY" "$ROOT/scripts/monitor_feature_drift.py" "$@" ;;
  orthogonalize)    shift; "$PY" "$ROOT/scripts/orthogonalize_signals.py" "$@" ;;
  calibrate)        shift; "$PY" "$ROOT/scripts/calibrate_slippage.py" "$@" ;;
  daily-report)     shift; "$PY" "$ROOT/scripts/daily_report.py" "$@" ;;
  dashboard)    shift; exec "$PY" "$ROOT/scripts/run_dashboard.py" "$@" ;;
  testdb)       shift; "$PY" "$ROOT/scripts/test_db_connection.py" "$@" ;;
  watchdog-install)   cmd_watchdog_install ;;
  watchdog-uninstall) cmd_watchdog_uninstall ;;
  watchdog-status)    cmd_watchdog_status ;;
  dashboard-install)   cmd_dashboard_install ;;
  dashboard-uninstall) cmd_dashboard_uninstall ;;
  dashboard-status)    cmd_dashboard_status ;;
  wipe-journal)       shift; "$PY" "$ROOT/scripts/wipe_journal.py" "$@" ;;
  reset-paper)        shift; "$PY" "$ROOT/scripts/reset_paper.py" "$@" ;;
  *)
    cat <<EOF
usage: $(basename "$0") {start|stop|restart|status|logs|backtest|priors|walkforward|dashboard|testdb|
                        watchdog-install|watchdog-uninstall|watchdog-status|
                        dashboard-install|dashboard-uninstall|dashboard-status|
                        wipe-journal}

The watchdog-* / dashboard-* subcommands are macOS-only; they manage
launchd agents. watchdog supervises the bot; dashboard supervises the
FastAPI read-only UI on http://127.0.0.1:8000.

Environment:
  TRADEBOT_PY   override python interpreter (default: \$ROOT/.venv/bin/python)
EOF
    exit 2
    ;;
esac
