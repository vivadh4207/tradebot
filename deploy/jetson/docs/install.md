# Jetson AGX Orin 64GB — Install Guide

Tested on: JetPack 6.x (Jetson Linux 36.x, Ubuntu 22.04 base, CUDA 12.2).

## Prerequisites

- Jetson AGX Orin 64 GB dev kit
- JetPack 6.x flashed (via NVIDIA SDK Manager on a separate x86 host)
- User account on the Jetson (examples below use `orin`; change as needed)
- Internet access
- `sudo` on the Jetson

## Step-by-step

### 1. Clone the repo onto the Jetson

```bash
cd ~
git clone https://github.com/vivadh4207/tradebot.git
cd tradebot
```

### 2. Run the bootstrap

```bash
bash deploy/jetson/setup.sh
```

Nine steps, 10–20 minutes. Compiling `llama-cpp-python` with CUDA is the
slowest (5–15 min). The first `pip install` pulls ~300 MB of wheels.

What it does:

1. `apt-get install` build tools + `tzdata`, sets TZ to `America/New_York`.
2. `nvpmodel -m 0` (MAXN) + `jetson_clocks` — pins CPU/GPU/memory at max.
3. Creates `.venv` and upgrades pip/setuptools/wheel.
4. Installs `requirements.txt`.
5. Builds `llama-cpp-python` with `-DGGML_CUDA=on` and verifies GPU offload.
6. Installs `requirements-jetson.txt` (jetson-stats etc.).
7. Downloads `qwen2.5-7b-instruct-q4_k_m.gguf` (~4.7 GB) into `deploy/jetson/models/`.
8. Appends LLM env vars to `.env` (creates nothing if `.env` missing — do that first).
9. Runs `pytest` and prints a health snapshot.

### 3. Put your credentials in `.env`

If you haven't already:

```bash
cp .env.example .env
nano .env
```

Fill in:
- `ALPACA_API_KEY_ID` + `ALPACA_API_SECRET_KEY`
- `COCKROACH_DSN` (or the `COCKROACH_*` split fields)
- Optional: `DISCORD_WEBHOOK_URL` for fills + halt notifications

The `LLM_*` block was added by `setup.sh` automatically.

### 4. Verify the local LLM works

```bash
bash deploy/jetson/scripts/benchmark_llm.sh .
```

Expected: `[=] 3 calls in 4-8s` with `score=-0.8..-0.9` on the canned
negative-headline payload. If you see "llama.cpp CUDA support: False",
the `install_llama_cpp.sh` step didn't pick up CUDA — re-run it manually
with `CUDA_HOME=/usr/local/cuda` explicit.

### 5. Install the systemd services

```bash
sudo cp deploy/jetson/services/tradebot.service /etc/systemd/system/
sudo cp deploy/jetson/services/tradebot-dashboard.service /etc/systemd/system/

# Edit User=, WorkingDirectory= if your repo isn't at /home/orin/tradebot
sudo sed -i "s|/home/orin|$HOME|g; s|User=orin|User=$USER|g; s|Group=orin|Group=$USER|g" \
  /etc/systemd/system/tradebot.service /etc/systemd/system/tradebot-dashboard.service

sudo systemctl daemon-reload
sudo systemctl enable --now tradebot tradebot-dashboard
journalctl -u tradebot -f
```

The bot runs 24/7, auto-restarts on crash, survives reboots. The
dashboard binds to `127.0.0.1:8000`.

### 6. View the dashboard from your laptop

```bash
# on your laptop
ssh -L 8000:localhost:8000 orin@<jetson-ip>
# open http://localhost:8000
```

### 7. Optional: run via Docker instead

If you'd rather keep everything in containers:

```bash
cd deploy/jetson/docker
docker compose --env-file ../../../.env up -d --build
docker compose logs -f tradebot
```

The container uses `--runtime nvidia` and passes through the dev kit's
GPU to llama.cpp. The `models/` volume is read-only.

## Sanity check

```bash
bash deploy/jetson/scripts/jetson_health.sh .
```

You should see:
- MAXN mode
- GPU temp < 60°C idle
- Bot `running (pid N)`
- Recent log lines showing `data_adapter kind=alpaca`,
  `news_classifier kind=LocalLLMNewsClassifier`

If any of those show wrong, check [`troubleshooting.md`](troubleshooting.md).
