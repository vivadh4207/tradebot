#!/usr/bin/env bash
# Set the AGX Orin 64GB into MAXN power mode and pin max clocks.
# Needs sudo. Reversible via nvpmodel -m <mode>.
#
# MAXN on Orin 64GB: all 12 CPU cores + full GPU + memory controller unthrottled.
# Power draw: ~60W sustained. Plenty of cooling headroom with the dev-kit heatsink.
set -euo pipefail

echo "[*] Setting MAXN power mode (mode 0)…"
sudo nvpmodel -m 0

echo "[*] Pinning max clocks (CPU + GPU + EMC)…"
sudo jetson_clocks

echo "[*] Current state:"
sudo nvpmodel -q
echo
sudo jetson_clocks --show || true
