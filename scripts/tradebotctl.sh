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

# systemd user units (Linux / Jetson). Installed to ~/.config/systemd/user/
# so no sudo is required. Works for Jetson + any headless Linux.
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SYSTEMD_WATCHDOG_UNIT="tradebot-watchdog.service"
SYSTEMD_DASHBOARD_UNIT="tradebot-dashboard.service"
SYSTEMD_WATCHDOG_SRC="$ROOT/deploy/systemd/${SYSTEMD_WATCHDOG_UNIT}"
SYSTEMD_DASHBOARD_SRC="$ROOT/deploy/systemd/${SYSTEMD_DASHBOARD_UNIT}"

# Host OS — drives which supervisor we wire up.
HOST_OS="$(uname -s)"

_write_systemd_unit() {
  # $1 = source template path ; $2 = installed path
  # Rewrites __TRADEBOT_ROOT__ and __TRADEBOT_PY__ placeholders with the
  # current checkout's paths. Both macOS (BSD awk) and Linux (GNU awk)
  # accept this syntax.
  awk -v new_root="$ROOT" -v new_py="$PY" '
    {
      gsub("__TRADEBOT_ROOT__", new_root);
      gsub("__TRADEBOT_PY__", new_py);
      print;
    }
  ' "$1" > "$2"
}

mkdir -p "$ROOT/logs"

# .env is intentionally NOT sourced here. Shell parsing is fragile with
# values that contain spaces, '&', '?', or URL-looking fragments (like a
# Postgres DSN). Every Python entry point calls python-dotenv's
# load_dotenv(...) itself, so the secrets make it into the process env
# exactly once, correctly.

_is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# ----- supervisor detection -----------------------------------------------
# If the watchdog is installed under launchd (macOS) or systemd --user
# (Linux), start/stop/status MUST route through that supervisor rather
# than spawning a second PID-file-tracked process alongside it. Otherwise
# clicking "start" in the dashboard while the watchdog is running would
# end up with two bots sharing the journal + double-filling paper.
_active_supervisor() {
  # Prints one of: "launchd" | "systemd" | "pid" (fallback)
  if [[ "$HOST_OS" == "Darwin" && -f "$LAUNCHD_INSTALLED_PLIST" ]]; then
    if launchctl list 2>/dev/null | awk '{print $3}' | grep -qx "$LAUNCHD_LABEL"; then
      echo "launchd"; return
    fi
  fi
  if [[ "$HOST_OS" == "Linux" && -f "$SYSTEMD_USER_DIR/$SYSTEMD_WATCHDOG_UNIT" ]]; then
    if command -v systemctl >/dev/null 2>&1; then
      echo "systemd"; return
    fi
  fi
  echo "pid"
}

cmd_start() {
  local sup; sup="$(_active_supervisor)"
  case "$sup" in
    launchd)
      # launchd's KeepAlive wants the job loaded; `start` just kicks it
      launchctl start "$LAUNCHD_LABEL" 2>/dev/null
      echo "started via launchd ($LAUNCHD_LABEL)"
      return 0 ;;
    systemd)
      if systemctl --user start "$SYSTEMD_WATCHDOG_UNIT"; then
        echo "started via systemd ($SYSTEMD_WATCHDOG_UNIT)"
        return 0
      fi
      echo "systemctl start failed"; return 1 ;;
  esac
  # --- PID-file fallback (no supervisor installed) ---
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
  local sup; sup="$(_active_supervisor)"
  case "$sup" in
    launchd)
      launchctl stop "$LAUNCHD_LABEL" 2>/dev/null
      echo "stop sent to launchd ($LAUNCHD_LABEL)"
      return 0 ;;
    systemd)
      if systemctl --user stop "$SYSTEMD_WATCHDOG_UNIT"; then
        echo "stopped via systemd ($SYSTEMD_WATCHDOG_UNIT)"
        return 0
      fi
      echo "systemctl stop failed"; return 1 ;;
  esac
  # --- PID-file fallback ---
  if _is_running; then
    local pid; pid="$(cat "$PID_FILE")"
    touch "$KILL_FILE"                    # cooperative shutdown signal
    kill "$pid" 2>/dev/null || true
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
  local sup; sup="$(_active_supervisor)"
  case "$sup" in
    launchd)
      local entry; entry="$(launchctl list 2>/dev/null | awk -v l="$LAUNCHD_LABEL" '$3 == l')"
      local pid; pid="$(awk '{print $1}' <<< "$entry")"
      if [[ -n "$entry" && "$pid" != "-" ]]; then
        echo "running (launchd pid=$pid)"
      else
        echo "stopped (launchd loaded, not running)"
      fi
      return 0 ;;
    systemd)
      if systemctl --user is-active "$SYSTEMD_WATCHDOG_UNIT" >/dev/null 2>&1; then
        local pid; pid="$(systemctl --user show -p MainPID --value "$SYSTEMD_WATCHDOG_UNIT" 2>/dev/null)"
        echo "running (systemd pid=$pid)"
      else
        echo "stopped (systemd inactive)"
      fi
      return 0 ;;
  esac
  # --- PID-file fallback ---
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

# ----- supervisor install/uninstall/status ------------------------------
# macOS → launchd LaunchAgent.  Linux → systemd --user unit.  Both modes
# wrap scripts/watchdog_run.py, which supervises run_paper.py. This makes
# the same tradebotctl command work on a dev Mac and on a Jetson without
# host-specific tweaks.
cmd_watchdog_install() {
  case "$HOST_OS" in
    Darwin) _watchdog_install_launchd ;;
    Linux)  _watchdog_install_systemd ;;
    *) echo "unsupported OS for watchdog-install: $HOST_OS"; return 2 ;;
  esac
}

cmd_watchdog_uninstall() {
  case "$HOST_OS" in
    Darwin) _watchdog_uninstall_launchd ;;
    Linux)  _watchdog_uninstall_systemd ;;
    *) echo "unsupported OS for watchdog-uninstall: $HOST_OS"; return 2 ;;
  esac
}

cmd_watchdog_status() {
  case "$HOST_OS" in
    Darwin) _watchdog_status_launchd ;;
    Linux)  _watchdog_status_systemd ;;
    *) echo "unsupported OS for watchdog-status: $HOST_OS"; return 2 ;;
  esac
  _print_heartbeat_age
}

cmd_dashboard_install() {
  case "$HOST_OS" in
    Darwin) _dashboard_install_launchd ;;
    Linux)  _dashboard_install_systemd ;;
    *) echo "unsupported OS for dashboard-install: $HOST_OS"; return 2 ;;
  esac
}

cmd_dashboard_uninstall() {
  case "$HOST_OS" in
    Darwin) _dashboard_uninstall_launchd ;;
    Linux)  _dashboard_uninstall_systemd ;;
    *) echo "unsupported OS for dashboard-uninstall: $HOST_OS"; return 2 ;;
  esac
}

cmd_dashboard_status() {
  case "$HOST_OS" in
    Darwin) _dashboard_status_launchd ;;
    Linux)  _dashboard_status_systemd ;;
    *) echo "unsupported OS for dashboard-status: $HOST_OS"; return 2 ;;
  esac
}

# ----- macOS launchd helpers --------------------------------------------
_watchdog_install_launchd() {
  if [[ ! -f "$LAUNCHD_SRC_PLIST" ]]; then
    echo "source plist missing: $LAUNCHD_SRC_PLIST"; return 1
  fi
  mkdir -p "$(dirname "$LAUNCHD_INSTALLED_PLIST")"
  # Rewrite the baked-in dev path to this checkout's path. macOS only
  # needs this substitution; the template was authored on the original dev
  # box.
  local tmp_plist; tmp_plist="$(mktemp)"
  awk -v new_root="$ROOT" -v new_py="$PY" '
    {
      gsub("/Users/vivekadhikari/Documents/Claude/Projects/tradebot/.venv/bin/python", new_py);
      gsub("/Users/vivekadhikari/Documents/Claude/Projects/tradebot", new_root);
      print;
    }
  ' "$LAUNCHD_SRC_PLIST" > "$tmp_plist"
  mv "$tmp_plist" "$LAUNCHD_INSTALLED_PLIST"
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

_watchdog_uninstall_launchd() {
  if [[ -f "$LAUNCHD_INSTALLED_PLIST" ]]; then
    launchctl unload "$LAUNCHD_INSTALLED_PLIST" 2>/dev/null || true
    rm -f "$LAUNCHD_INSTALLED_PLIST"
    echo "watchdog unloaded + removed"
  else
    echo "not installed"
  fi
}

_watchdog_status_launchd() {
  local entry; entry="$(launchctl list 2>/dev/null | awk -v l="$LAUNCHD_LABEL" '$3 == l')"
  if [[ -z "$entry" ]]; then
    echo "watchdog: not loaded (launchd)"
    echo "install:  tradebotctl watchdog-install"
    return 0
  fi
  local pid status
  pid="$(awk '{print $1}' <<< "$entry")"
  status="$(awk '{print $2}' <<< "$entry")"
  if [[ "$pid" == "-" ]]; then
    echo "watchdog: loaded but not running (last exit=$status)"
  else
    echo "watchdog: running (pid=$pid, last exit=$status)"
  fi
}

_dashboard_install_launchd() {
  if [[ ! -f "$DASHBOARD_SRC_PLIST" ]]; then
    echo "source plist missing: $DASHBOARD_SRC_PLIST"; return 1
  fi
  mkdir -p "$(dirname "$DASHBOARD_INSTALLED_PLIST")"
  local tmp_plist; tmp_plist="$(mktemp)"
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
  else
    echo "launchctl load failed — see logs/dashboard.err"; return 1
  fi
}

_dashboard_uninstall_launchd() {
  if [[ -f "$DASHBOARD_INSTALLED_PLIST" ]]; then
    launchctl unload "$DASHBOARD_INSTALLED_PLIST" 2>/dev/null || true
    rm -f "$DASHBOARD_INSTALLED_PLIST"
    echo "dashboard unloaded + removed"
  else
    echo "not installed"
  fi
}

_dashboard_status_launchd() {
  local entry; entry="$(launchctl list 2>/dev/null | awk -v l="$DASHBOARD_LABEL" '$3 == l')"
  if [[ -z "$entry" ]]; then
    echo "dashboard: not loaded (launchd)"
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

# ----- Linux systemd --user helpers -------------------------------------
# No sudo required. Units live in ~/.config/systemd/user/. To survive
# logout (e.g. when SSH-ing into a Jetson), enable linger once per user:
#   sudo loginctl enable-linger $USER
_require_systemctl() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl not found — Linux install needs systemd."; return 1
  fi
}

_watchdog_install_systemd() {
  _require_systemctl || return 1
  if [[ ! -f "$SYSTEMD_WATCHDOG_SRC" ]]; then
    echo "source unit missing: $SYSTEMD_WATCHDOG_SRC"; return 1
  fi
  mkdir -p "$SYSTEMD_USER_DIR"
  _write_systemd_unit "$SYSTEMD_WATCHDOG_SRC" "$SYSTEMD_USER_DIR/$SYSTEMD_WATCHDOG_UNIT"
  systemctl --user daemon-reload
  if systemctl --user enable --now "$SYSTEMD_WATCHDOG_UNIT"; then
    echo "watchdog installed + started: $SYSTEMD_USER_DIR/$SYSTEMD_WATCHDOG_UNIT"
    echo "check:    tradebotctl watchdog-status"
    echo "logs:     journalctl --user -u $SYSTEMD_WATCHDOG_UNIT -f"
    echo "linger:   sudo loginctl enable-linger \$USER   # survive logout"
  else
    echo "systemctl --user enable failed"; return 1
  fi
}

_watchdog_uninstall_systemd() {
  _require_systemctl || return 1
  systemctl --user disable --now "$SYSTEMD_WATCHDOG_UNIT" 2>/dev/null || true
  rm -f "$SYSTEMD_USER_DIR/$SYSTEMD_WATCHDOG_UNIT"
  systemctl --user daemon-reload
  echo "watchdog unloaded + removed"
}

_watchdog_status_systemd() {
  _require_systemctl || return 1
  if [[ ! -f "$SYSTEMD_USER_DIR/$SYSTEMD_WATCHDOG_UNIT" ]]; then
    echo "watchdog: not installed (systemd user)"
    echo "install:  tradebotctl watchdog-install"
    return 0
  fi
  local active; active="$(systemctl --user is-active "$SYSTEMD_WATCHDOG_UNIT" 2>/dev/null || echo "inactive")"
  local pid; pid="$(systemctl --user show -p MainPID --value "$SYSTEMD_WATCHDOG_UNIT" 2>/dev/null)"
  echo "watchdog: $active (pid=${pid:-0})"
}

_dashboard_install_systemd() {
  _require_systemctl || return 1
  if [[ ! -f "$SYSTEMD_DASHBOARD_SRC" ]]; then
    echo "source unit missing: $SYSTEMD_DASHBOARD_SRC"; return 1
  fi
  mkdir -p "$SYSTEMD_USER_DIR"
  _write_systemd_unit "$SYSTEMD_DASHBOARD_SRC" "$SYSTEMD_USER_DIR/$SYSTEMD_DASHBOARD_UNIT"
  systemctl --user daemon-reload
  if systemctl --user enable --now "$SYSTEMD_DASHBOARD_UNIT"; then
    echo "dashboard installed + started: $SYSTEMD_USER_DIR/$SYSTEMD_DASHBOARD_UNIT"
    echo "access:   http://127.0.0.1:8000"
    echo "logs:     journalctl --user -u $SYSTEMD_DASHBOARD_UNIT -f"
  else
    echo "systemctl --user enable failed"; return 1
  fi
}

_dashboard_uninstall_systemd() {
  _require_systemctl || return 1
  systemctl --user disable --now "$SYSTEMD_DASHBOARD_UNIT" 2>/dev/null || true
  rm -f "$SYSTEMD_USER_DIR/$SYSTEMD_DASHBOARD_UNIT"
  systemctl --user daemon-reload
  echo "dashboard unloaded + removed"
}

_dashboard_status_systemd() {
  _require_systemctl || return 1
  if [[ ! -f "$SYSTEMD_USER_DIR/$SYSTEMD_DASHBOARD_UNIT" ]]; then
    echo "dashboard: not installed (systemd user)"
    echo "install:  tradebotctl dashboard-install"
    return 0
  fi
  local active; active="$(systemctl --user is-active "$SYSTEMD_DASHBOARD_UNIT" 2>/dev/null || echo "inactive")"
  local pid; pid="$(systemctl --user show -p MainPID --value "$SYSTEMD_DASHBOARD_UNIT" 2>/dev/null)"
  echo "dashboard: $active (pid=${pid:-0})"
  echo "open:      http://127.0.0.1:8000"
}

# Heartbeat freshness + last watchdog event. OS-agnostic — just reads
# logs/heartbeat.txt and logs/watchdog_events.jsonl.
_print_heartbeat_age() {
  if [[ -f "$ROOT/logs/heartbeat.txt" ]]; then
    local hb_mtime now age
    hb_mtime="$(stat -f %m "$ROOT/logs/heartbeat.txt" 2>/dev/null || stat -c %Y "$ROOT/logs/heartbeat.txt" 2>/dev/null)"
    now="$(date +%s)"
    age=$(( now - hb_mtime ))
    echo "heartbeat: ${age}s ago"
  else
    echo "heartbeat: (never written yet)"
  fi
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
  doctor)             exec /usr/bin/env bash "$ROOT/scripts/doctor.sh" ;;
  walkforward)        shift; "$PY" "$ROOT/scripts/nightly_walkforward_report.py" "$@" ;;
  putcall-oi)         shift; "$PY" "$ROOT/scripts/fetch_putcall_oi.py" "$@" ;;
  migrate-to-sqlite)  shift; "$PY" "$ROOT/scripts/migrate_cockroach_to_sqlite.py" "$@" ;;
  *)
    cat <<EOF
usage: $(basename "$0") {start|stop|restart|status|logs|backtest|priors|walkforward|dashboard|testdb|
                        watchdog-install|watchdog-uninstall|watchdog-status|
                        dashboard-install|dashboard-uninstall|dashboard-status|
                        wipe-journal|reset-paper}

The watchdog-* / dashboard-* subcommands auto-detect the OS:
  macOS  → install as launchd LaunchAgents (~/Library/LaunchAgents/)
  Linux  → install as systemd --user units (~/.config/systemd/user/)

watchdog supervises scripts/run_paper.py; dashboard supervises the
FastAPI read-only UI on http://127.0.0.1:8000.

On Linux/Jetson, survive logout with:
  sudo loginctl enable-linger \$USER

Environment:
  TRADEBOT_PY             override python interpreter (default: \$ROOT/.venv/bin/python)
  TRADEBOT_TORCH_DEVICE   force LSTM device: cpu|cuda|mps (default: auto-detect)
  TRADEBOT_DASHBOARD_CONTROLS  set to 1 to enable start/stop buttons in the dashboard UI
EOF
    exit 2
    ;;
esac
