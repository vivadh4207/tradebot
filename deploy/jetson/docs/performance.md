# Jetson AGX Orin 64 GB — Performance notes

Ballpark numbers measured on a stock dev kit in MAXN mode.
Your mileage will vary slightly by JetPack version and ambient temp.

## Bot resource use

The trading loop itself is tiny — a 3-minute orchestration cycle plus a
5-second fast-exit thread. Nothing that stresses the hardware.

| Component | CPU | RAM | GPU |
|---|---|---|---|
| `run_paper.py` main loop | <1% of 12 cores | ~200 MB | 0% |
| Fast-exit thread (5s) | <1% | (shared) | 0% |
| `run_dashboard.py` | <1% | ~80 MB | 0% |
| `LocalLLMNewsClassifier` loading | ~5% for 2s | +6 GB | — |
| `LocalLLMNewsClassifier` per call | — | steady | 60-80% spike |

## Local LLM throughput

Qwen2.5-7B-Instruct Q4_K_M GGUF, `n_gpu_layers=-1`, `n_ctx=4096`:

| Batch size | Time/call | Tokens/s (output) |
|---|---|---|
| 5 headlines  | ~1.2 s  | 50-65 tok/s |
| 10 headlines | ~1.5 s  | 55-70 tok/s |
| 20 headlines | ~2.0 s  | 55-70 tok/s |

With the bot's 5-minute cache TTL per symbol and a 10-symbol universe,
that's ~120 LLM calls per trading day — trivial load.

If you swap to a larger model (Llama-3.1-8B Q4 or Mistral-Nemo-12B Q4),
expect roughly -10% throughput and +1–2 GB RAM, but modestly better
classification quality on long or ambiguous headlines.

## GPU Black-Scholes (optional)

`deploy/jetson/python/gpu_pricer.py` uses CuPy. Single-contract pricing
is slower on GPU because of kernel launch overhead; batch pricing wins
starting around 1000+ contracts.

| Contracts | CPU (scipy) | GPU (CuPy) | Speedup |
|---|---|---|---|
| 1       | ~20 μs | ~500 μs | 0.04× (CPU wins) |
| 100     | ~1.5 ms | ~600 μs | 2.5× |
| 10,000  | ~120 ms | ~5 ms | 24× |
| 100,000 | ~1.2 s | ~40 ms | 30× |

Use GPU pricing for: fitting an SVI surface across the full chain,
re-pricing the entire book after a spot move, or a walk-forward backtest
that iterates thousands of contracts per tick. Stick to the CPU path for
the hot per-signal hot path.

## Power / thermal

At steady state (bot running + one LLM call every 5 min):

| State | Power | GPU temp | CPU temp |
|---|---|---|---|
| Idle | 11 W | 35°C | 38°C |
| Bot only | 15 W | 35°C | 42°C |
| Bot + LLM call burst | 45 W peak | 55°C | 48°C |

The dev-kit heatsink + fan handles it without sustained thermal
throttling. If you're going fanless / custom enclosure, budget extra
airflow for the +30°C spike during model loading.

## Tuning knobs (in order of impact)

1. **Power mode.** `sudo nvpmodel -m 0` + `sudo jetson_clocks` gives you
   ~40% more LLM throughput than the default. `deploy/jetson/setup.sh`
   does this for you.
2. **`LLM_N_GPU_LAYERS`.** `-1` offloads everything. Reducing (e.g. `20`)
   trades speed for lower GPU memory but there's no reason to on 64 GB.
3. **Model quantization.** Q4_K_M is the sweet spot. Q5_K_M is ~10%
   better but 20% slower and bigger. Q8_0 is production-quality-hungry
   on the GPU; skip it.
4. **Swap.** Orin 64GB does NOT need swap for this workload. Don't enable it.
5. **Dashboard refresh.** The auto-refresh is 60 s; drop to 15 s only if
   you're actively watching.

## Compared to the Mac / small VPS

| Scenario | Mac M-series | DigitalOcean $6/mo | Jetson Orin 64 GB |
|---|---|---|---|
| Bot loop | trivial | trivial | trivial |
| LLM news classification | API call | API call | **on-device** |
| Cost/month | $0 (electricity) | $6 | $0 (electricity) |
| Upfront | $0 | $0 | ~$2000 |
| Uptime | depends | 99.99% | depends on home internet |
| Latency to Alpaca | local | 10-30 ms | home internet + 20-50 ms |

The only material upside of the Jetson is **local LLM inference**. If
you don't care about that, a $6 VPS is strictly better for trading
infrastructure. If you already own a Jetson: great, the LLM savings are
real.
