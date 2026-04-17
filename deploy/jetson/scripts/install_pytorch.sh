#!/usr/bin/env bash
# Install NVIDIA's Jetson-specific PyTorch wheel into the tradebot venv.
#
# PyPI's torch wheels are CPU-only on aarch64 (and cuDNN/cuBLAS don't ship).
# NVIDIA publishes Jetson wheels compiled against JetPack's CUDA 12.x /
# cuDNN 8.9 on the Jetson PyPI index.
#
# Docs: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
set -euo pipefail

REPO="${1:-$(pwd)}"
VENV_PY="$REPO/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[!] No venv at $VENV_PY — run setup.sh first."
  exit 1
fi

echo "[*] Installing Jetson PyTorch wheel (for JetPack 6 / CUDA 12.2)…"
# NVIDIA mirror index; versions here track JetPack. Adjust if NVIDIA moves the index.
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install \
  --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 \
  torch

"$VENV_PY" -c "
import torch
print('torch version :', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda device  :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(CPU only)')
"

echo "[OK] PyTorch installed. Next: bash deploy/jetson/scripts/train_lstm.sh"
