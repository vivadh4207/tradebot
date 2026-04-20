# Jetson AGX Orin — Troubleshooting

## `install_llama_cpp.sh` shows `CUDA support: False`

The wheel was built without CUDA. Confirm:

```bash
ls /usr/local/cuda/bin/nvcc          # must exist
nvcc --version
echo $CUDA_HOME                       # should be /usr/local/cuda
```

Fix: set `CUDA_HOME` explicitly and rebuild from source:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87" FORCE_CMAKE=1 \
  .venv/bin/pip install --force-reinstall --no-cache-dir llama-cpp-python
```

`CMAKE_CUDA_ARCHITECTURES=87` is the compute capability for the Ampere
GPU in Orin. Using a different value (e.g. 89 for Ada) will silently
produce a wheel that won't launch kernels.

## `benchmark_llm.sh`: model loads but no GPU usage visible

Check what llama.cpp logged during load:

```bash
.venv/bin/python -c "
import os
from llama_cpp import Llama
m = Llama(model_path=os.environ['LLM_MODEL_PATH'],
          n_gpu_layers=-1, verbose=True, n_ctx=4096)
"
```

You should see `offloading N repeating layers to GPU` early in the log.
If it says `offloading 0` or similar, `LLM_N_GPU_LAYERS` is unset or the
wheel is CPU-only — go back to the previous section.

## `torch` is installed but pulls in CPU-only wheels

The Jetson requires the NVIDIA-published PyTorch wheels for aarch64. You
probably don't need torch for this bot at all — we use `llama-cpp-python`
directly, not via `transformers`. If something else installs torch:

```bash
pip uninstall -y torch torchvision
```

## Model download gets killed by the OS

Large GGUF files + NetworkManager can trigger OOM on the download itself
if you're under memory pressure. The download script uses `curl -C -`
which resumes partial downloads — just re-run it.

## `nvpmodel -m 0` says "permission denied"

Need sudo. The script uses `sudo`; if passwordless sudo is off, you'll
be prompted.

## `journalctl -u tradebot -f` shows `data_adapter kind=synthetic`

`ALPACA_API_KEY_ID`/`_SECRET` aren't reaching the service. Check that
`/home/orin/tradebot/.env` has the real values (not placeholders
wrapped in `<...>`) and that `EnvironmentFile=` in the systemd unit
points there. Then `sudo systemctl restart tradebot`.

## Bot logs show `news_classifier kind=KeywordClassifier` after install

Either `LLM_MODEL_PATH` isn't set in the `.env` the service loads, or
the file at that path doesn't exist. Inspect:

```bash
sudo systemctl cat tradebot | grep EnvironmentFile
grep LLM_MODEL_PATH /home/orin/tradebot/.env
ls -l $(grep LLM_MODEL_PATH /home/orin/tradebot/.env | cut -d= -f2)
```

## Clock drift / `session_closed` when the market should be open

Jetson's default TZ is UTC. `setup.sh` sets America/New_York, but if you
flashed a newer JetPack that reset it:

```bash
sudo timedatectl set-timezone America/New_York
timedatectl
```

## `docker compose up` fails with `could not select device driver ""`

NVIDIA container runtime isn't installed / enabled. Check
`/etc/docker/daemon.json` includes:

```json
{ "default-runtime": "nvidia",
  "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] } } }
```

Then `sudo systemctl restart docker`.

## `jetson-stats` complains about missing kernel module

`jtop` is optional — the bot runs fine without it. Suppress the warning:

```bash
pip uninstall -y jetson-stats
```

The telemetry thread silently no-ops when the package is absent.

## CockroachDB SSL error on first connect

CockroachDB Serverless uses certs signed by Let's Encrypt, and JetPack
6's default CA bundle sometimes lags:

```bash
sudo apt-get install -y ca-certificates
sudo update-ca-certificates
```

If that doesn't resolve it, add `&sslmode=require` (weaker: no hostname
verification) to the DSN while you sort out the CA bundle — but do NOT
leave that in place for long. Prefer `verify-full`.
