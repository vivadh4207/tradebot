# tradebot on NVIDIA Jetson AGX Orin 64 GB

Everything in this folder is specific to running on the Orin dev kit.
For the base bot, see the top-level [`README.md`](../../README.md).

## What you get vs. running on a Mac / VPS

| | Mac / VPS | Jetson AGX Orin 64GB |
|---|---|---|
| Bot itself | fine | fine (barely stresses the CPU) |
| News classifier | Anthropic API (~$/day) | **local Qwen/Llama on GPU** |
| Backtest | CPU | CPU (GPU pricer optional) |
| Uptime | depends on machine | 24/7 edge-class |
| Power draw | ~5W idle laptop | ~15-20W idle, ~45W under LLM load |
| Net benefit | baseline | **no API costs, no data leaves the box** |

The big practical win is replacing the Claude API call with a local LLM.
On the Orin 64GB, a 7B-parameter Q4 GGUF model runs at roughly 40-60 tok/s
through llama.cpp's CUDA backend — more than fast enough for the bot's
once-per-5-minutes-per-symbol news classification.

## Files

```
deploy/jetson/
├── setup.sh                     # one-shot bootstrap (9 steps, 10-20 min)
├── requirements-jetson.txt      # extras: jetson-stats, CuPy (optional)
├── README.md                    # you are here
├── scripts/
│   ├── set_power_mode.sh        # MAXN + jetson_clocks
│   ├── install_llama_cpp.sh     # build llama-cpp-python with CUDA
│   ├── download_model.sh        # fetch quantized GGUF (default: Qwen2.5-7B-Q4)
│   ├── benchmark_llm.sh         # verify GPU offload + measure tok/s
│   └── jetson_health.sh         # quick status snapshot
├── services/
│   ├── tradebot.service         # systemd, pinned to A78AE perf cores
│   └── tradebot-dashboard.service
├── config/
│   ├── jetson_settings.yaml     # tuned main-loop + cache intervals
│   └── llm.env.example          # env vars for the local LLM
├── python/
│   ├── gpu_pricer.py            # CuPy-vectorized BS + gamma
│   └── jetson_telemetry.py      # jtop → notifier bridge
├── docker/
│   ├── Dockerfile               # L4T PyTorch base + llama.cpp CUDA
│   └── docker-compose.yml       # bot + dashboard
├── models/                      # (gitignored) GGUF model files
└── docs/
    ├── install.md               # detailed step-by-step
    ├── performance.md           # benchmarks + tuning
    └── troubleshooting.md       # common gotchas
```

## 3-command quickstart

On a fresh JetPack 6 install, from the repo root:

```bash
bash deploy/jetson/setup.sh
bash deploy/jetson/scripts/benchmark_llm.sh .
bash deploy/jetson/scripts/jetson_health.sh .
```

`setup.sh` handles everything: timezone, MAXN mode, venv, Python deps,
llama-cpp-python CUDA build, model download, .env plumbing, and a test
run. See [`docs/install.md`](docs/install.md) for the long-form walk-through.

## Enabling local LLM news (the real reason to use the Jetson)

1. Run `bash deploy/jetson/scripts/download_model.sh .` (first time only,
   ~5 GB download).
2. Append the contents of `deploy/jetson/config/llm.env.example` to your
   `.env` (setup.sh does this automatically).
3. Restart the bot.

The existing `build_classifier()` factory detects `LLM_MODEL_PATH` and
swaps the classifier backend with zero code changes. If llama-cpp fails
to load the model for any reason, it falls back to Claude (if API key
set) or the keyword classifier — no silent failures.

## Safety reminder

Paper trading only by default. `LIVE_TRADING=false` is the default, and
flipping it requires TWO explicit settings (`LIVE_TRADING=true` AND
`broker.name != paper`). Don't route real money through any system you
haven't paper-tested for at least 30 days.
