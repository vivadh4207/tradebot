# Portability — Mac + Jetson/Linux

The repo runs unchanged on:

- macOS (Apple Silicon or Intel) — dev / paper testing
- Linux including NVIDIA Jetson (aarch64) — 24/7 paper or future live

No per-host code paths; all OS detection lives in `scripts/tradebotctl.sh`
and `src/signals/lstm_signal.py::_pick_device`.

## Same commands, both hosts

```
bash scripts/doctor.sh                 # readiness check (what's missing?)
bash scripts/tradebotctl.sh start      # PID-file + nohup (portable)
bash scripts/tradebotctl.sh status
bash scripts/tradebotctl.sh watchdog-install
bash scripts/tradebotctl.sh dashboard-install
```

Under the hood:

| Command | macOS | Linux/Jetson |
|---|---|---|
| `start` / `stop` / `restart` | PID file + POSIX signals | same |
| `watchdog-install` | launchd LaunchAgent | systemd `--user` unit |
| `watchdog-status` | `launchctl list` | `systemctl --user is-active` |
| `dashboard-install` | launchd LaunchAgent | systemd `--user` unit |
| LSTM device | auto: MPS → CPU | auto: CUDA → CPU |

## First-time setup

1. Copy `.env.example` to `.env` and fill in credentials.
2. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. `bash scripts/doctor.sh` — confirm no red items.
4. `bash scripts/tradebotctl.sh watchdog-install`
5. `bash scripts/tradebotctl.sh dashboard-install` (optional)

### Jetson extras

On Jetson AGX Orin, the dedicated `deploy/jetson/` pipeline handles
CUDA-accelerated local LLM inference (replacing Anthropic API with local
Qwen/Llama via llama-cpp CUDA). See `deploy/jetson/README.md`.

To keep the supervised bot running after you SSH out:

```
sudo loginctl enable-linger $USER
```

## Device detection

The LSTM signal picks a torch device in this priority order:

1. CUDA (Jetson, any Linux box with a CUDA GPU)
2. MPS (Apple Silicon Metal backend)
3. CPU (everything else)

Override with `TRADEBOT_TORCH_DEVICE=cpu|cuda|mps` in `.env` if you want
to pin the device explicitly (useful for debugging or benchmarking).

## Known host-specific differences

- **Timezone**: both supervisors export `TZ=America/New_York`. The
  bot's session-window logic depends on this; leave it alone unless you
  know the schema.
- **Heartbeat `stat` flags**: macOS is `stat -f %m`, Linux is `stat -c %Y`.
  `tradebotctl watchdog-status` tries both and uses whichever works.
- **Alpaca IEX feed**: set the same way on both hosts
  (`feed="iex"` in `AlpacaDataAdapter`). SIP requires a paid sub.
- **CockroachDB SSL**: same DSN works. On Jetson if you need a CA cert,
  drop it under `$HOME/.postgresql/root.crt` and set
  `COCKROACH_SSLROOTCERT` in `.env`.

## How to verify the port is clean

After moving the repo to a new host:

```
bash scripts/doctor.sh                    # must be all green except warnings
bash scripts/tradebotctl.sh status        # should say "stopped"
bash scripts/tradebotctl.sh watchdog-install
bash scripts/tradebotctl.sh watchdog-status
# wait 30 seconds
tail -f logs/tradebot.out                 # confirm the loop is ticking
```

If any of these fail on one host but pass on the other, file it — a
platform-specific branch somewhere leaked in.
