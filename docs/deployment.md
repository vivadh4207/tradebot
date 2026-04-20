# Deployment

**Short answer:** Do NOT deploy this on Vercel. Trading bots need long-running
stateful processes; Vercel is built for short request/response serverless
functions and is a bad fit technically and operationally.

## Why not Vercel (or Netlify / Cloudflare Workers)

- Serverless functions time out after ~10–60 seconds. The bot needs a loop
  that runs continuously for the 6.5-hour market session.
- No persistent in-memory state between invocations. The bot holds
  positions, spread history, rolling volume averages, consecutive-holds
  counters — all lost between cold starts.
- No WebSocket / streaming market data in the free tier of most serverless
  platforms. Even where supported, each invocation pays cold-start latency
  which is fatal for a 5-second fast-exit thread.
- Vercel cannot run background threads (like `fast_loop`).
- Vercel's runtime upper-bounds Python install size. `scipy`, `statsmodels`,
  `alpaca-py` push you to the limit quickly.
- No outbound API key rotation hooks, no kill switch, no systemd.

## What to use instead

### Recommended: small Linux VPS with systemd (simple, reliable)

| Provider | Tier | Monthly | Notes |
|---|---|---|---|
| Hetzner | CX22 (2 vCPU, 4 GB) | ~€4 | Lowest cost; EU datacenter |
| DigitalOcean | Basic (1 vCPU, 1 GB) | $6 | Easy, NY-metro availability |
| AWS Lightsail | 2GB | $7 | AWS credits friendly |
| Oracle Cloud | Always-Free VM.Standard.A1 | $0 | Generous free tier, ARM64 |

Network latency to the Alpaca API from a US VPS (NY/NJ region) is typically
10–30 ms — fine for this bot; 5-second fast loop tolerates this easily.

### Setup sketch (Ubuntu 22.04 / 24.04)

```bash
# 1. System deps
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv git

# 2. Clone + install
git clone <repo-url> /opt/tradebot && cd /opt/tradebot
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Secrets
cp .env.example .env && nano .env  # fill Alpaca paper keys

# 4. Systemd service
sudo tee /etc/systemd/system/tradebot.service > /dev/null <<'EOF'
[Unit]
Description=tradebot (paper)
After=network.target

[Service]
Type=simple
User=tradebot
WorkingDirectory=/opt/tradebot
EnvironmentFile=/opt/tradebot/.env
ExecStart=/opt/tradebot/.venv/bin/python scripts/run_paper.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo useradd -r -s /usr/sbin/nologin tradebot
sudo chown -R tradebot:tradebot /opt/tradebot
sudo systemctl daemon-reload
sudo systemctl enable --now tradebot
sudo journalctl -u tradebot -f      # tail live logs
```

Kill switch:
```bash
sudo touch /opt/tradebot/KILL          # handled at next loop tick
sudo systemctl stop tradebot           # hard stop
```

### Acceptable alternatives

- **Local Mac with the included launchd watchdog.** Good for paper and
  small live books, provided you disable sleep (`sudo pmset -c
  disablesleep 1` or `caffeinate -is`) and use the included watchdog
  (`tradebotctl watchdog-install`). The watchdog restarts on crash,
  alerts via Discord/Slack, and recycles on silent hangs — see
  `docs/scheduling.md § Mode 2`. A home Mac is still less reliable than
  a VPS (ISP outages, power blips), but the robustness gap has narrowed
  meaningfully. Lid-closed on AC is safe with `disablesleep 1`.
- **Fly.io / Railway / Render.** These support long-running processes;
  workable but generally more expensive than a $5 VPS for this workload.
- **Docker on an always-on server.** Good. Add a `Dockerfile` and `docker-compose.yml`
  when you reach that stage — see `docs/docker.md` (TBD).

### Bad ideas

- Vercel, Netlify, Cloudflare Workers, AWS Lambda (cron) — serverless.
- Phone-hosted or battery-powered devices.
- Shared-tenancy "always-free" containers that get evicted.

## Operational checklist for live (after 30+ days of paper)

1. `LIVE_TRADING=true` in `.env` **and** `broker.name: alpaca` in `settings.yaml`.
2. API key scope: read + trade; explicitly deny transfers/withdrawals.
3. Daily loss halt tested end-to-end in paper.
4. `tradebot.service` enabled with `Restart=on-failure` (systemd).
5. Monitoring: tail `journalctl` into your phone via an SSH shortcut, or
   wire a Discord/Slack webhook that posts fills and daily P&L.
6. Manual reconcile before market open each day for the first 2 weeks of
   live: log into the broker UI, verify positions match `broker.positions()`.
