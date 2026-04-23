# Remote dashboard — Vercel + Cloudflare Tunnel

Access your local bot dashboard from anywhere (phone, laptop at work, etc.)
via a public Vercel URL. Authentication stays local — Vercel never sees
your bot data.

## Architecture

```
  [Your Mac]  ——  Cloudflare Tunnel  ——  Internet  ——  Vercel (static site)
  bot:8765          tradebot.X.com                       tradebot.vercel.app
                                                         (your phone browser)
                         ↑                                     ↑
                  bearer token                          same bearer token
```

The dashboard HTML is served by Vercel (static). It calls your bot's
FastAPI via a Cloudflare Tunnel with a bearer token. Bot data never
goes through Vercel's servers.

## One-time setup (~10 minutes)

### 1. Install Cloudflare Tunnel on your Mac

```bash
brew install cloudflared
cloudflared tunnel login    # opens browser → pick your Cloudflare account
cloudflared tunnel create tradebot
```

If you don't have a custom domain, use a free `trycloudflare.com` URL:

```bash
cloudflared tunnel --url http://localhost:8765
# → gives you something like https://dashed-words.trycloudflare.com
```

For a stable URL (recommended — otherwise it changes on every restart),
use a domain you own:

```bash
cloudflared tunnel route dns tradebot tradebot.yourdomain.com
```

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: tradebot
credentials-file: /Users/YOU/.cloudflared/<TUNNEL_UUID>.json
ingress:
  - hostname: tradebot.yourdomain.com
    service: http://localhost:8765
  - service: http_status:404
```

Run it persistently via launchd:

```bash
sudo cloudflared service install
# Or manually:
cloudflared tunnel run tradebot
```

### 2. Set the bot's bearer token

Generate a strong random token and add it to your bot's `.env`:

```bash
cd ~/tradebot
TOKEN=$(openssl rand -hex 32)
echo "DASHBOARD_REMOTE_TOKEN=$TOKEN" >> .env
echo "DASHBOARD_CORS_ORIGINS=https://your-vercel-url.vercel.app" >> .env
echo "Your token: $TOKEN"
```

Restart the dashboard so `.env` reloads:

```bash
launchctl kickstart -k gui/$(id -u)/com.tradebot.dashboard
```

### 3. Build + deploy to Vercel

```bash
cd ~/tradebot/deploy/vercel
bash build.sh              # generates dashboard.html
```

Then either:

**Option A — drag-and-drop deploy:**
1. Go to https://vercel.com/new
2. Drag the `deploy/vercel` folder to the page
3. Click Deploy — done in 10 seconds

**Option B — Vercel CLI:**
```bash
npm i -g vercel
vercel --prod            # follow the prompts; use defaults
```

Vercel gives you a URL like `https://tradebot-xyz.vercel.app`.

### 4. Connect

Open `https://tradebot-xyz.vercel.app` on any device. The login screen asks for:
- **API base URL** → `https://tradebot.yourdomain.com` (or your trycloudflare URL)
- **Bearer token** → the token you generated in step 2

Both stay in your browser's localStorage. Vercel never sees them.

## Rebuilding after dashboard changes

Whenever you change `src/dashboard/templates/index.html`:

```bash
bash deploy/vercel/build.sh
cd deploy/vercel
vercel --prod
```

## Security notes

- **The bearer token is required** — bot rejects any `/api/*` call without it (once `DASHBOARD_REMOTE_TOKEN` is set in `.env`).
- **CORS** is restricted to origins you whitelist in `DASHBOARD_CORS_ORIGINS`.
- **No credentials are transmitted to Vercel** — they're client-side only.
- **Rotate the token** if you ever share a screenshot — update `.env` and the login page's stored token.
- **HTTPS everywhere** — Cloudflare Tunnel and Vercel both serve over TLS by default.

## Troubleshooting

**"Could not reach API" on login:**
- Is the tunnel running? `cloudflared tunnel info tradebot`
- Is the bot's dashboard running? `launchctl list | grep dashboard`
- Try `curl -H "Authorization: Bearer $TOKEN" https://YOUR-TUNNEL/api/health` from your Mac

**"unauthorized" errors:**
- Double-check the token in `.env` matches what you pasted into the login
- Restart the dashboard after changing `.env` (it only loads at startup)

**Stale data:**
- Dashboard polls every 5-30s. Hard-refresh (Cmd+Shift+R) for immediate pull.

**CORS errors in browser console:**
- Check `DASHBOARD_CORS_ORIGINS` in `.env` includes your exact Vercel URL (including `https://`).
