# Jetson AGX Orin — first-time setup, step by step

Zero prior knowledge assumed. If you just pulled the Orin out of the
box, this walks you from "I have a Jetson" to "the bot is running 24/7
with the local LLM doing news classification on-GPU".

Budget about **2–3 hours end to end**. Most of it is waiting for
downloads. The hands-on typing is probably 20 minutes total.

---

## 0. What you should have in hand before you start

From the Orin dev-kit box:

- Jetson AGX Orin 64GB dev kit
- USB-C power brick + cable
- microSD card (Orin boots from eMMC; SD is optional backup)

Buy separately if you don't already have them:

- **USB keyboard + mouse** (any wired pair; needed for first boot)
- **HDMI or DisplayPort monitor** + matching cable (first boot only — after SSH works you can unplug)
- **Ethernet cable** OR your **Wi-Fi password** (Ethernet is easier)
- **USB-C to USB-A cable** (for flashing if the pre-installed JetPack is too old — probably unnecessary)

Keep a second machine (your Mac) nearby. Most of the interesting typing happens there via SSH once the Jetson has network.

---

## 1. Physical hookup

1. Plug monitor into the DisplayPort or HDMI port on the Orin.
2. Plug USB keyboard + mouse into any USB-A port.
3. Plug in Ethernet if you have it.
4. Plug in the USB-C power brick last. The dev kit powers on immediately — no power switch.

The green LED should come on. First boot shows NVIDIA/Ubuntu splash; this takes 30–90 seconds.

---

## 2. First-boot wizard (Ubuntu + NVIDIA EULA)

You'll see NVIDIA's out-of-box setup. Click through:

1. Accept the NVIDIA SDK license.
2. Accept the Ubuntu license.
3. Pick language / keyboard layout / timezone. Set **timezone to America/New_York** — the bot's trading-session math assumes it.
4. Create a user. I'd use `orin` as the username to match the shipped systemd units; any other name works too but you'll need to edit the units. Pick a strong password.
5. Join Wi-Fi (or confirm Ethernet worked).
6. Let the wizard reboot once when it finishes.

After the reboot you're sitting at an Ubuntu desktop. The Orin's installed JetPack is already set up with CUDA, cuDNN, TensorRT — you don't need to flash anything.

---

## 3. Get SSH working (you want to stop using that keyboard ASAP)

Open the Jetson's terminal (Ctrl+Alt+T from the desktop):

```bash
sudo apt-get update
sudo apt-get install -y openssh-server
sudo systemctl enable --now ssh
ip -4 addr show | grep -E 'inet.*(eth|wlan)' | awk '{print $2}'
```

Write down the IP address it prints (for example `192.168.1.42`).

From your **Mac terminal**:

```bash
ssh orin@192.168.1.42        # use whatever username + IP you set
```

Accept the fingerprint once. From here, everything is done over SSH.
You can unplug the monitor/keyboard from the Jetson if you want — it'll run headless.

**Optional but nice:** on your Mac, `ssh-copy-id orin@192.168.1.42` so you don't have to re-type the password every time.

---

## 4. System prep (2 minutes, done over SSH)

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git python3-venv python3-dev build-essential cmake curl tzdata
sudo timedatectl set-timezone America/New_York
```

If `timedatectl` complains, it's usually because NTP isn't synced yet. Wait a minute and retry, or run `sudo systemctl restart systemd-timesyncd`.

---

## 5. Clone the tradebot repo

```bash
cd ~
git clone <your-git-remote-here> tradebot
cd tradebot
```

Replace `<your-git-remote-here>` with whatever your GitHub / GitLab URL is. If you're pushing from your Mac, you might want to use the HTTPS URL and paste a Personal Access Token when prompted, or `ssh-keygen` on the Jetson and add that key to your git host.

---

## 6. Run the Jetson bootstrap (one command, ~30–45 minutes)

This is the big one — it does nearly everything:

```bash
bash deploy/jetson/setup.sh
```

What it does, in order (you'll see `[setup N/9]` messages):

1. `apt install` the build toolchain
2. Set the Orin to **MAXN power mode** (highest performance)
3. Create `.venv` and install base Python requirements
4. Build **llama-cpp-python with CUDA** (this is the slow step — compiling against JetPack's CUDA headers takes 15–20 minutes)
5. Install Jetson-extra Python packages (`jetson-stats`, optional CuPy)
6. Install the **Jetson PyTorch wheel** — this is NVIDIA's aarch64 CUDA build, NOT plain `pip install torch` (plain pip gives you a CPU-only build that won't use your GPU)
7. Download a quantized 7B LLM (~5 GB — default is Qwen2.5-7B-Instruct Q4)
8. Append the LLM env vars to `.env`
9. Run `pytest` and a health check

If step 4 (llama-cpp) fails, re-run **just** that step: `bash deploy/jetson/scripts/install_llama_cpp.sh .`

If step 6 (PyTorch) fails: `bash deploy/jetson/scripts/install_pytorch.sh .`

---

## 7. Fill in the `.env` file

If you didn't already carry a `.env` from the Mac, copy the template and fill it in:

```bash
cp .env.example .env
nano .env                # or vim — whichever you're comfortable with
```

Minimum things to set:

- `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` (your paper-trading keys)
- `ALPACA_BASE_URL=https://paper-api.alpaca.markets`
- `COCKROACH_DSN=postgresql://...` (same DSN you used on the Mac — the cluster can be shared)
- At least one Discord webhook (pick any of `DISCORD_WEBHOOK_URL`, `_TRADES`, `_ALERTS`, `_REASON`)
- `LIVE_TRADING=false` (leave it false until you've run paper for at least 30 days)

The setup script already appended the LLM block (`LLM_MODEL_PATH=...`) for you.

**If you used the same git repo from your Mac**, you can also just `scp` your `.env` across from your Mac instead of re-entering everything:

```bash
# From your Mac:
scp ~/Documents/Claude/Projects/tradebot/.env orin@192.168.1.42:~/tradebot/.env
```

---

## 8. Run the doctor (sanity check)

```bash
bash scripts/doctor.sh
```

Every row should be green (`ok`) or yellow (`?`). If anything is red (`XX`), fix it before moving on. Typical fixes:

- Red on `interpreter`: you're outside the venv. `source .venv/bin/activate`.
- Red on `fastapi`/`pydantic`/etc.: `pip install -r requirements.txt`.
- Yellow on `.env` keys: re-check step 7.
- Yellow on `watchdog`: expected — next step installs it.

You want the `torch` row to say **`device=cuda`**. If it says `cpu`, the NVIDIA PyTorch wheel didn't install correctly; re-run `bash deploy/jetson/scripts/install_pytorch.sh .` and check its output.

---

## 9. Benchmark the local LLM (optional but satisfying)

```bash
bash deploy/jetson/scripts/benchmark_llm.sh .
```

Expected output: **~30–60 tokens/second** on the default Qwen2.5-7B Q4. If you see less than 15 tok/s, the model isn't running on the GPU — the script prints a hint about what to check.

This is the moment that made the Jetson worth buying: you now run the news classifier locally for free, instead of paying Anthropic per call.

---

## 10. Install the watchdog + dashboard as services

The same commands that work on your Mac work here — `tradebotctl` auto-detects Linux and installs systemd user units:

```bash
# Keep services running even after you SSH out
sudo loginctl enable-linger $USER

bash scripts/tradebotctl.sh watchdog-install
bash scripts/tradebotctl.sh dashboard-install

# Verify:
bash scripts/tradebotctl.sh watchdog-status
bash scripts/tradebotctl.sh dashboard-status
```

After ~30 seconds:

- Watchdog status should say `active (pid=N)`.
- The dashboard should be reachable from your Mac at `http://<jetson-ip>:8000` — but **only** if you change the bind address. By default the dashboard binds to 127.0.0.1 (local only). To access it from your laptop on the same LAN, either:

```bash
# Option A (recommended): SSH tunnel from your Mac — no network exposure
ssh -L 8000:127.0.0.1:8000 orin@192.168.1.42
# then open http://127.0.0.1:8000 on your Mac

# Option B: bind to the LAN (less safe — anyone on your network can see it)
# Edit scripts/run_dashboard.py and change host="127.0.0.1" to host="0.0.0.0"
```

Stick with Option A unless you're sure your home network is trustworthy.

---

## 11. Verify end-to-end

```bash
# watch the bot's stdout
tail -f logs/tradebot.out

# or use journalctl for the supervisor's view
journalctl --user -u tradebot-watchdog.service -f
```

During market hours you should see:

- a `startup` line
- every few seconds, a `main_loop_tick` line
- when a signal fires, an `ensemble_emit` or `entry_filtered` line

If it's outside market hours (9:30am–4:00pm ET Mon–Fri), the bot idles — that's normal, not broken.

Sanity check the dashboard. Through your SSH tunnel:

```
http://127.0.0.1:8000
```

You should see the same dashboard you had on the Mac, running off the shared Cockroach journal.

---

## 12. Day-to-day cheat sheet

```bash
# Status + quick health
bash scripts/tradebotctl.sh watchdog-status
bash scripts/tradebotctl.sh dashboard-status
bash deploy/jetson/scripts/jetson_health.sh .

# Tail logs
tail -f logs/tradebot.out
tail -f logs/watchdog.out

# Restart after config change
systemctl --user restart tradebot-watchdog.service

# Stop everything
systemctl --user stop tradebot-watchdog.service
systemctl --user stop tradebot-dashboard.service

# Turn it off for the weekend, turn it back on Monday
systemctl --user stop tradebot-watchdog.service
# ...then Monday morning:
systemctl --user start tradebot-watchdog.service

# Pull the latest code from your Mac push
cd ~/tradebot && git pull && systemctl --user restart tradebot-watchdog.service

# Run the nightly walk-forward report manually
.venv/bin/python scripts/nightly_walkforward_report.py
```

If you want the nightly report to actually run nightly, add this to your user crontab (`crontab -e`):

```
0 20 * * 1-5  cd /home/orin/tradebot && .venv/bin/python scripts/nightly_walkforward_report.py >> logs/nightly.out 2>&1
```

---

## 13. Common gotchas

**"GPU is not being used"** — In a Python shell, run:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
If `False` or `ImportError`, the Jetson PyTorch wheel didn't install. Re-run `bash deploy/jetson/scripts/install_pytorch.sh .`. Plain `pip install torch` will NOT give you GPU support on aarch64 — you must use NVIDIA's wheel.

**"llama-cpp says GGML_ASSERT or crashes when loading the model"** — Usually an out-of-memory issue or a corrupted download. Retry:
```bash
rm -f deploy/jetson/models/*.gguf
bash deploy/jetson/scripts/download_model.sh .
bash deploy/jetson/scripts/benchmark_llm.sh .
```

**"Watchdog keeps restarting the bot"** — Check `logs/watchdog_events.jsonl` for exit codes and `logs/tradebot.err` for stack traces. Most common cause: `.env` missing a required key — fix the env var, then `systemctl --user restart tradebot-watchdog.service`.

**"Running hot / fan loud"** — MAXN mode runs the Orin flat-out. If you want quieter, `sudo nvpmodel -m 2` drops you to 30W mode (slower LLM, same bot perf). Check current mode with `sudo nvpmodel -q`.

**"SSH keeps timing out"** — Your router probably assigned a new IP. On the Jetson console, `ip -4 addr` to see the new one. Consider setting a DHCP reservation in your router so the IP stays fixed.

**"Dashboard shows no data"** — The Cockroach DSN in `.env` is either missing or points at a different database than your Mac uses. If the Mac and Jetson should share history, both `.env` files must have the exact same `COCKROACH_DSN`.

---

## 14. When to graduate to live trading

**Not yet.** Keep `LIVE_TRADING=false` for at least 30 market days of Jetson-only paper operation. Read the nightly Discord `#tradebot-reason` channel every morning; if you see `EDGE CONFIRMED` three weeks in a row with positive median EV and a non-declining trend, you're allowed to *consider* flipping the switch.

See `docs/advanced_quant_review.md` for why patience is the correct default here.

---

## Summary of the commands you'll run

```bash
# 1. First boot (physical)
# 2. SSH prep (on Jetson):
sudo apt-get install -y openssh-server && sudo systemctl enable --now ssh
# 3. From Mac:
ssh orin@<jetson-ip>
# 4. On Jetson, one-time:
sudo apt-get update && sudo apt-get install -y git python3-venv python3-dev build-essential cmake
git clone <your-repo> tradebot && cd tradebot
bash deploy/jetson/setup.sh
cp .env.example .env  # edit it with your keys
bash scripts/doctor.sh
sudo loginctl enable-linger $USER
bash scripts/tradebotctl.sh watchdog-install
bash scripts/tradebotctl.sh dashboard-install
# 5. From Mac (for dashboard access):
ssh -L 8000:127.0.0.1:8000 orin@<jetson-ip>
# open http://127.0.0.1:8000 on your Mac
```

That's the whole thing.
