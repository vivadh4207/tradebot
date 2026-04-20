#!/usr/bin/env bash
# tradebot doctor — one-shot readiness check. Runs on both macOS and
# Linux/Jetson and prints a pass/fail report for every dependency the
# bot needs before it can go live.
#
# Usage: bash scripts/doctor.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TRADEBOT_PY:-$ROOT/.venv/bin/python}"

PASS="ok"
FAIL="MISSING"
WARN="warn"

green() { printf "\033[32m%s\033[0m" "$1"; }
red()   { printf "\033[31m%s\033[0m" "$1"; }
yellow(){ printf "\033[33m%s\033[0m" "$1"; }

emit() {
  # $1 = label, $2 = state (ok|MISSING|warn), $3 = detail
  local state="$2" label="$1" detail="${3:-}"
  case "$state" in
    "$PASS") printf "  [%s] %-30s %s\n" "$(green ok)"   "$label" "$detail" ;;
    "$FAIL") printf "  [%s] %-30s %s\n" "$(red   XX)"   "$label" "$detail" ;;
    "$WARN") printf "  [%s] %-30s %s\n" "$(yellow '? ')" "$label" "$detail" ;;
  esac
}

hdr() { printf "\n== %s ==\n" "$1"; }

# ---- host ----
hdr "host"
emit "uname -s"       "$PASS" "$(uname -s)"
emit "uname -m"       "$PASS" "$(uname -m)"
if [[ "$(uname -s)" == "Darwin" ]]; then
  emit "supervisor"   "$PASS" "launchd (macOS)"
else
  if command -v systemctl >/dev/null 2>&1; then
    emit "supervisor" "$PASS" "systemd (Linux)"
  else
    emit "supervisor" "$FAIL" "systemd not found — service install will fail"
  fi
fi

# ---- python ----
hdr "python + venv"
if [[ -x "$PY" ]]; then
  emit "interpreter"  "$PASS" "$PY ($($PY --version 2>&1))"
else
  emit "interpreter"  "$FAIL" "not found at $PY (set TRADEBOT_PY or create .venv)"
fi
if [[ -f "$ROOT/requirements.txt" ]]; then
  emit "requirements"   "$PASS" "$ROOT/requirements.txt"
else
  emit "requirements"   "$WARN" "requirements.txt not found"
fi

# ---- core python modules ----
hdr "python modules"
# mod label ; python import name
for mod in "fastapi:fastapi" "alpaca-py:alpaca" "pydantic:pydantic" \
           "pyyaml:yaml" "python-dotenv:dotenv" "urllib3:urllib3"; do
  label="${mod%%:*}"
  import_name="${mod##*:}"
  if "$PY" -c "import $import_name" 2>/dev/null; then
    emit "$label"     "$PASS" ""
  else
    emit "$label"     "$FAIL" "pip install -r requirements.txt"
  fi
done
# torch is optional but needed for LSTM
if "$PY" -c "import torch" 2>/dev/null; then
  DEV="$("$PY" - <<'PY'
import torch
if torch.cuda.is_available(): print("cuda")
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): print("mps")
else: print("cpu")
PY
)"
  emit "torch"        "$PASS" "device=$DEV"
else
  emit "torch"        "$WARN" "not installed — LSTM signal disabled (OK if lstm_enabled: false)"
fi

# ---- files ----
hdr "repo layout"
for f in config/settings.yaml .env scripts/run_paper.py scripts/watchdog_run.py src/main.py; do
  if [[ -e "$ROOT/$f" ]]; then
    emit "$f"         "$PASS" ""
  else
    state="$FAIL"
    # .env is expected to be gitignored — if missing, prompt copy from .env.example
    [[ "$f" == ".env" ]] && state="$WARN"
    emit "$f"         "$state" "missing"
  fi
done
if [[ ! -d "$ROOT/logs" ]]; then
  emit "logs/"        "$WARN" "will be created on first run"
fi

# ---- env vars (presence only, never value) ----
hdr ".env keys present"
if [[ -f "$ROOT/.env" ]]; then
  for key in ALPACA_API_KEY_ID ALPACA_API_SECRET_KEY ALPACA_BASE_URL; do
    if grep -qE "^${key}=.+" "$ROOT/.env" 2>/dev/null; then
      emit "$key"     "$PASS" ""
    else
      emit "$key"     "$WARN" "unset or empty"
    fi
  done
  # Discord (at least one channel)
  if grep -qE "^DISCORD_WEBHOOK_URL(_[A-Z]+)?=https" "$ROOT/.env" 2>/dev/null; then
    emit "Discord webhook"  "$PASS" "at least one channel configured"
  else
    emit "Discord webhook"  "$WARN" "no Discord channel set — notifications go to log only"
  fi
else
  emit ".env file"    "$WARN" "copy .env.example to .env and fill in"
fi

# ---- supervisor install state ----
hdr "supervisor"
if [[ "$(uname -s)" == "Darwin" ]]; then
  PLIST="$HOME/Library/LaunchAgents/com.tradebot.paper.plist"
  if [[ -f "$PLIST" ]]; then
    emit "watchdog"   "$PASS" "installed (launchd)"
  else
    emit "watchdog"   "$WARN" "not installed → tradebotctl watchdog-install"
  fi
else
  UNIT="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/tradebot-watchdog.service"
  if [[ -f "$UNIT" ]]; then
    if systemctl --user is-active tradebot-watchdog.service >/dev/null 2>&1; then
      emit "watchdog" "$PASS" "running (systemd --user)"
    else
      emit "watchdog" "$WARN" "installed but inactive"
    fi
  else
    emit "watchdog"   "$WARN" "not installed → tradebotctl watchdog-install"
  fi
fi

# ---- live switch ----
hdr "safety"
if grep -qE "^LIVE_TRADING=true" "$ROOT/.env" 2>/dev/null; then
  emit "LIVE_TRADING"   "$WARN" "ENABLED — real money at risk"
else
  emit "LIVE_TRADING"   "$PASS" "false (paper only)"
fi

echo
echo "Done. Fix any XX (red) items before starting the bot."
