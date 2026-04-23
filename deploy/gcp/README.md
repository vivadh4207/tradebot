# GCP migration — low-latency order execution

Move the bot from your Mac to a Google Compute Engine VM in
**us-east4 (Ashburn, Virginia)** — same AZ as most broker order-entry
endpoints. Expected latency improvement: **80-120ms → 5-15ms** per
order to Tradier/Alpaca.

## Why us-east4 (Ashburn)

| Region | Location | Latency to NYSE/NASDAQ/OPRA |
|---|---|---|
| **us-east4** | Ashburn VA | **<1 ms** (same data center campus as AWS us-east-1) |
| us-east1 | Moncks Corner SC | ~6-10 ms |
| us-central1 | Council Bluffs IA | ~25-30 ms |
| us-west1 | Oregon | ~65-75 ms |

Tradier's API endpoints + Alpaca Markets both live in AWS us-east-1
(also Ashburn). GCP us-east4 is literally cross-datacenter from there
— order round-trips are faster than from your Mac by 100-200ms.

**For 0DTE options especially**, this matters: a 100ms advantage at
entry compounds over a session of 20 trades + exits. You capture
prices before the quote walks away.

## Instance sizing

| Machine type | vCPU/RAM | Monthly cost | When to use |
|---|---|---|---|
| **e2-small** | 2 / 2 GB | **~$14** | Light paper bot (recommended to start) |
| e2-medium | 2 / 4 GB | ~$27 | Bot + local Ollama 8B |
| e2-standard-2 | 2 / 8 GB | ~$55 | + Groq-heavy research + dashboard |
| n2-standard-2 | 2 / 8 GB | ~$85 | Lowest jitter, live-trading grade |

**Start with e2-small** — the bot's hot path is ~50-100MB RAM. Upgrade
only if you see swap pressure or OOM.

**Free tier** — GCP has a permanent free tier with **1 e2-micro** in
us-west1 per billing account, but that's not in us-east4 (kills the
latency benefit). For a paper bot, `e2-small` in us-east4 at $14/mo
is worth the edge.

## Step-by-step migration (~45 min, first time)

### 1. Create the VM

```bash
# Install gcloud CLI if you don't have it:
#   https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create a VM in us-east4 (Ashburn)
gcloud compute instances create tradebot \
  --zone=us-east4-a \
  --machine-type=e2-small \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB \
  --boot-disk-type=pd-balanced \
  --tags=tradebot \
  --metadata=enable-oslogin=TRUE
```

Open outbound firewall only (the VM doesn't need inbound except SSH):

```bash
# SSH only (default allowed via identity-aware proxy — no firewall change needed)
# Dashboard port 8765 — DO NOT open to internet.
# Access via cloudflared tunnel only (step 5).
```

### 2. SSH + bootstrap

```bash
gcloud compute ssh tradebot --zone=us-east4-a

# On the VM:
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev \
  git build-essential sqlite3
```

### 3. Clone + deps

```bash
git clone https://github.com/vivadh4207/tradebot.git
cd tradebot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt   # or install whatever your requirements file is
```

### 4. Copy `.env` securely

Never commit `.env`. Use `gcloud` to upload:

```bash
# From your Mac:
gcloud compute scp ~/tradebot/.env tradebot:~/tradebot/.env --zone=us-east4-a
```

Or create it fresh on the VM with the same keys (Tradier, Groq, etc.).

### 5. Install the Cloudflare Tunnel on the VM

Same flow as the Mac dashboard exposure — but on the VM you ALSO
need a tunnel for the dashboard (since you won't be SSH-port-forwarding
all the time).

```bash
# On the VM:
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
cloudflared tunnel login
cloudflared tunnel create tradebot-gcp
cloudflared tunnel route dns tradebot-gcp tradebot.yourdomain.com
```

### 6. Run the bot under systemd (equivalent of launchd)

Copy the systemd unit file (already in `deploy/systemd/`):

```bash
# On the VM:
sudo cp deploy/systemd/tradebot-paper.service /etc/systemd/system/
# Edit paths if necessary:
sudo nano /etc/systemd/system/tradebot-paper.service
# Set User=YOUR_USERNAME, WorkingDirectory=/home/YOUR_USERNAME/tradebot

sudo systemctl daemon-reload
sudo systemctl enable tradebot-paper
sudo systemctl start tradebot-paper
sudo systemctl status tradebot-paper     # verify running
```

For the dashboard + discord bot, similar service files — add them as
needed. Or run the bot standalone:

```bash
cd ~/tradebot
.venv/bin/python -m src.main
```

### 7. Verify latency improvement

From the VM, ping Tradier's API:

```bash
ping -c 5 api.tradier.com
# Expected from us-east4: ~1-3ms
# Expected from your Mac: ~40-80ms (depending on your ISP)
```

### 8. Monitor

```bash
# Logs
journalctl -u tradebot-paper -f

# Heartbeat from the Mac:
curl https://tradebot.yourdomain.com/api/health \
  -H "Authorization: Bearer $DASHBOARD_REMOTE_TOKEN"
```

## Data continuity

Copy your trade journal (SQLite) across so strategy history persists:

```bash
# Stop both bots first, then:
gcloud compute scp tradebot:~/tradebot/logs/tradebot.sqlite \
  ~/tradebot/logs/tradebot.sqlite --zone=us-east4-a
```

## Cost controls

- Stop the VM when NOT trading (weekends, holidays) to cut cost ~70%:
  ```bash
  gcloud compute instances stop tradebot --zone=us-east4-a
  gcloud compute instances start tradebot --zone=us-east4-a
  ```
- Set a budget alert at $25/mo in GCP Console → Billing → Budgets
- e2-small at $14/mo × ~22 trading days × ~7 hrs market hours =
  if you STOP the VM off-hours, you pay ~$5/mo.

## Security hardening (when you go live)

1. **Firewall** — only allow SSH via IAP (default). Never expose port 8765.
2. **Service account** — give the VM a minimal IAM role (no project-wide perms).
3. **VPC** — keep the VM in its own VPC with no external IP; use Cloud NAT for outbound.
4. **Secrets** — instead of `.env` on disk, use **Secret Manager**:
   ```bash
   gcloud secrets create tradier-token --data-file=- <<< "$TRADIER_TOKEN"
   # Then fetch at bot startup
   ```
5. **Automated backups** — nightly snapshot of the boot disk + journal export to GCS.

## Hybrid option (keep Mac as fallback)

If you're not ready to migrate fully, run the **order hot path** on GCP
and keep everything else (analytics, dashboard, journal) on the Mac:

1. Bot on GCP handles: signal → filter chain → order submit → broker reconcile
2. Mac handles: research, advisor, saves tracker, dashboard
3. They share state via the journal sync'd over SFTP every 5 min

More complex; only worth it if you've got data/tooling you don't want to move.

## Rollback

If GCP doesn't work out, just stop the systemd service and run on Mac
again. Copy the journal back:

```bash
gcloud compute scp tradebot:~/tradebot/logs/tradebot.sqlite \
  ~/tradebot/logs/tradebot.sqlite --zone=us-east4-a
```

The `.env` and codebase stay identical — you only moved where Python runs.
