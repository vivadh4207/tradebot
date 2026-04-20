#!/usr/bin/env bash
# End-to-end Jetson AGX Orin bootstrap.
#
# Assumes:
#   - Fresh JetPack 6.x install
#   - User has sudo
#   - Internet access
#
# Run from the repo root:
#   bash deploy/jetson/setup.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

step() { printf '\n\033[1;32m[setup %d/%d]\033[0m %s\n' "$1" "$2" "$3"; }

TOTAL=9

step 1 $TOTAL "System prep"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev git build-essential cmake \
  curl tzdata
sudo timedatectl set-timezone America/New_York

step 2 $TOTAL "Set MAXN power mode + pin clocks"
bash "$REPO/deploy/jetson/scripts/set_power_mode.sh"

step 3 $TOTAL "Create tradebot venv"
[[ -d .venv ]] || python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel

step 4 $TOTAL "Install base requirements"
pip install -r requirements.txt

step 5 $TOTAL "Build llama-cpp-python with CUDA"
bash "$REPO/deploy/jetson/scripts/install_llama_cpp.sh" "$REPO"

step 6 $TOTAL "Install Jetson extras"
pip install -r "$REPO/deploy/jetson/requirements-jetson.txt" || true

step 6b $TOTAL "Install Jetson PyTorch (skip with SKIP_TORCH=1)"
if [[ "${SKIP_TORCH:-0}" != "1" ]]; then
  bash "$REPO/deploy/jetson/scripts/install_pytorch.sh" "$REPO" || \
    echo "[!] PyTorch install failed — LSTM training/inference disabled. Re-run manually."
else
  echo "  SKIP_TORCH=1 set — skipping."
fi

step 7 $TOTAL "Download a quantized 7B model for news classification"
bash "$REPO/deploy/jetson/scripts/download_model.sh" "$REPO"

step 8 $TOTAL "Merge LLM env into .env"
if [[ -f "$REPO/.env" ]]; then
  if ! grep -q "LLM_MODEL_PATH" "$REPO/.env"; then
    {
      echo ""
      echo "# --- Jetson local LLM (added by deploy/jetson/setup.sh) ---"
      cat "$REPO/deploy/jetson/config/llm.env.example"
    } >> "$REPO/.env"
    echo "  Appended LLM settings to .env."
  else
    echo "  LLM_MODEL_PATH already present in .env — leaving it alone."
  fi
else
  echo "  No .env found. Copy .env.example to .env and fill in your values."
fi

step 9 $TOTAL "Verify everything"
python -m pytest -q --no-header -p no:cacheprovider || true
bash "$REPO/deploy/jetson/scripts/jetson_health.sh" "$REPO" || true

cat <<'EOF'

============================================================
All set. Next steps:

  1. Edit .env if you haven't already (Alpaca keys + CockroachDB DSN).
  2. Benchmark the local LLM (should hit ~30-60 tok/s on GGUF Q4):
       bash deploy/jetson/scripts/benchmark_llm.sh .
  3. Install the systemd services:
       sudo cp deploy/jetson/services/tradebot.service /etc/systemd/system/
       sudo cp deploy/jetson/services/tradebot-dashboard.service /etc/systemd/system/
       sudo systemctl daemon-reload
       sudo systemctl enable --now tradebot tradebot-dashboard
       journalctl -u tradebot -f
  4. Health check any time:
       bash deploy/jetson/scripts/jetson_health.sh .
============================================================
EOF
