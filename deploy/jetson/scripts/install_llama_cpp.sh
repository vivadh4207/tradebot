#!/usr/bin/env bash
# Install llama-cpp-python with CUDA backend on Jetson.
#
# Requires:
#   - JetPack 6.x (CUDA 12.2+ already installed under /usr/local/cuda)
#   - Active tradebot .venv ($REPO/.venv/bin/python)
#
# Pass the repo root as arg 1, or run from the repo root.
set -euo pipefail

REPO="${1:-$(pwd)}"
VENV_PY="$REPO/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[!] No venv found at $VENV_PY"
  echo "    Create one with:  cd $REPO && python3 -m venv .venv"
  exit 1
fi

if [[ ! -d /usr/local/cuda ]]; then
  echo "[!] /usr/local/cuda not found. Install JetPack first."
  exit 1
fi

CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "[*] nvcc: $(nvcc --version | tail -1)"
echo "[*] Building llama-cpp-python with CUDA on-device (5-15 min)…"

# Build flags for CUDA on ARM64. -DGGML_CUDA=on enables GPU offload.
CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on -DCMAKE_CUDA_ARCHITECTURES=87" \
FORCE_CMAKE=1 \
"$VENV_PY" -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

echo "[*] Verifying llama-cpp-python sees CUDA…"
"$VENV_PY" -c "
from llama_cpp import llama_cpp
print('llama.cpp CUDA support:', llama_cpp.llama_supports_gpu_offload())
"
echo "[OK] llama-cpp-python installed with CUDA backend."
