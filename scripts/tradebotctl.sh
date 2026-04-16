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
  dashboard)    shift; exec "$PY" "$ROOT/scripts/run_dashboard.py" "$@" ;;
  testdb)       shift; "$PY" "$ROOT/scripts/test_db_connection.py" "$@" ;;
  *)
    cat <<EOF
usage: $(basename "$0") {start|stop|restart|status|logs|backtest|priors|walkforward|dashboard|testdb}

Environment:
  TRADEBOT_PY   override python interpreter (default: \$ROOT/.venv/bin/python)
EOF
    exit 2
    ;;
esac
