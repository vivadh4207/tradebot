#!/usr/bin/env bash
# Convenience wrapper: trains the LSTM signal on Jetson GPU.
# Uses the repo-configured universe by default.
set -euo pipefail

REPO="${1:-$(pwd)}"
shift || true
VENV_PY="$REPO/.venv/bin/python"

# Sensible Jetson defaults; override with args, e.g.
#   bash deploy/jetson/scripts/train_lstm.sh . --days 365 --epochs 60
"$VENV_PY" "$REPO/scripts/train_lstm.py" \
  --days "${DAYS:-180}" \
  --timeframe-min "${TIMEFRAME:-5}" \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH:-256}" \
  --out "$REPO/checkpoints/lstm_best.pt" \
  "$@"
