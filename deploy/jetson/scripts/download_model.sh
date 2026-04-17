#!/usr/bin/env bash
# Download a quantized GGUF model for local news sentiment.
#
# Recommended: Qwen2.5-7B-Instruct Q4_K_M (~4.7 GB). Good quality, ~50 tok/s
# on Jetson AGX Orin, fits easily in 64GB unified memory alongside the bot.
#
# Alternatives (edit $MODEL_URL below):
#   Llama-3.1-8B-Instruct Q4_K_M (~4.9 GB)  — Meta
#   Mistral-Nemo-12B-Instruct Q4_K_M (~7.1 GB) — larger, better reasoning
set -euo pipefail

REPO="${1:-$(pwd)}"
MODELS_DIR="$REPO/deploy/jetson/models"
mkdir -p "$MODELS_DIR"

MODEL_NAME="${MODEL_NAME:-qwen2.5-7b-instruct-q4_k_m.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf?download=true}"

DEST="$MODELS_DIR/$MODEL_NAME"

if [[ -f "$DEST" && $(stat -c %s "$DEST") -gt 1000000000 ]]; then
  echo "[*] Model already downloaded: $DEST ($(du -h "$DEST" | cut -f1))"
else
  echo "[*] Downloading $MODEL_NAME to $DEST…"
  # -L follow redirects, -C - resume partial downloads
  curl -L -C - -o "$DEST" "$MODEL_URL"
fi

echo
echo "Export this in your .env (or shell) so the bot picks it up:"
echo "  LLM_MODEL_PATH=$DEST"
echo "  LLM_N_GPU_LAYERS=-1"
echo "  LLM_N_CTX=4096"
