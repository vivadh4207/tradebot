#!/usr/bin/env bash
# Print a quick health snapshot for the Jetson AGX Orin dev kit + bot state.
set -euo pipefail

REPO="${1:-$(pwd)}"

header() { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }

header "System"
uname -a
head -1 /etc/nv_tegra_release 2>/dev/null || echo "(no nv_tegra_release)"

header "Power mode + clocks"
sudo nvpmodel -q 2>/dev/null | head -2 || true

header "Temperatures (thermal_zone)"
for z in /sys/class/thermal/thermal_zone*/temp; do
  t=$(cat "$z")
  zn=$(cat "$(dirname "$z")/type" 2>/dev/null || echo "?")
  printf '  %-10s %d°C\n' "$zn" "$((t / 1000))"
done 2>/dev/null | head -8

header "GPU load (last 1s)"
cat /sys/devices/gpu.0/load 2>/dev/null | awk '{printf "  gpu_load: %.1f%%\n", $1/10}' || \
  echo "  (gpu.0 sysfs not present — run with sudo)"

header "Memory"
free -h | head -2

header "CUDA"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -1
else
  echo "  (nvcc not on PATH)"
fi

header "Bot process"
if [[ -f "$REPO/logs/tradebot.pid" ]] && kill -0 "$(cat "$REPO/logs/tradebot.pid")" 2>/dev/null; then
  pid=$(cat "$REPO/logs/tradebot.pid")
  echo "  running (pid $pid)"
  ps -p "$pid" -o pid,%cpu,%mem,rss,etime,cmd | tail -n +1
else
  echo "  stopped"
fi

header "Systemd (if enabled)"
systemctl is-active tradebot 2>/dev/null || echo "  tradebot.service: inactive"
systemctl is-active tradebot-dashboard 2>/dev/null || echo "  tradebot-dashboard.service: inactive"

header "Last 10 log lines"
tail -n 10 "$REPO/logs/tradebot.out" 2>/dev/null || echo "  (no logs yet)"
