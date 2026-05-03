"""Intraday Discord recap — runs every 20 min during market hours.

Posts a compact market-context summary for SPX / SPY / QQQ / IWM / DIA
plus a snapshot of the bot's recent activity (latest signals, fills,
filter blocks). Designed to be kept open in a separate terminal so the
operator can see at a glance what the bot is doing and where the
market is moving.

Source of truth:
  * Quotes: Tradier /v1/markets/quotes (works for SPX cash-index too)
  * Signals/fills/blocks: tail of logs/tradebot.out

Loop:
  while True:
    if market_open:  post()
    else:            log_skip()
    sleep until next 20-minute boundary

Run:
    python scripts/intraday_recap.py

Or as launchd / systemd user service: see deploy/.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

LOG_FILE = ROOT / "logs" / "tradebot.out"
RECAP_INTERVAL_SEC = int(os.getenv("RECAP_INTERVAL_SEC", "1200"))  # 20 min
SYMBOLS = os.getenv("RECAP_SYMBOLS", "SPX,SPY,QQQ,IWM,DIA").split(",")

# Branding — override via env so you can rebrand without code changes.
BRAND_NAME = os.getenv("DISCORD_BRAND_NAME", "Market Pulse").strip()
BRAND_FOOTER = os.getenv(
    "DISCORD_BRAND_FOOTER",
    "Market Pulse · Educational only · Not financial advice",
).strip()
DISCLAIMER = os.getenv(
    "DISCORD_DISCLAIMER",
    "Information shown is generated from public market data and "
    "automated analysis. It is provided for educational and "
    "informational purposes only and does not constitute financial, "
    "investment, or trading advice. Trading options involves "
    "substantial risk of loss; you can lose more than your initial "
    "investment. Past performance is not indicative of future "
    "results. Consult a licensed financial advisor before making "
    "any investment decisions.",
).strip()
# Embed accent color (hex int). Operator can override; defaults to a
# muted blue that prints well on both Discord light + dark themes.
EMBED_COLOR_NEUTRAL = int(os.getenv("DISCORD_EMBED_COLOR", "0x5865F2"), 16)
EMBED_COLOR_BULL = 0x2E7D32     # green
EMBED_COLOR_BEAR = 0xC62828     # red


def _market_open_et(now_utc: datetime) -> bool:
    """True if we're in the regular US session 09:30-16:00 ET, weekdays."""
    et = now_utc - timedelta(hours=4)        # crude EDT; close enough
    if et.weekday() >= 5:
        return False
    minutes = et.hour * 60 + et.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def _tradier_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Pull quotes for `symbols` from Tradier sandbox. Cash indexes
    like SPX are returned with the same /v1/markets/quotes endpoint."""
    token = os.getenv("TRADIER_TOKEN", "").strip()
    if not token:
        return {}
    url = (
        "https://sandbox.tradier.com/v1/markets/quotes"
        "?symbols=" + ",".join(s.strip() for s in symbols if s.strip())
    )
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"[recap] tradier_quote_err {e}")
        return {}
    rows = (data.get("quotes") or {}).get("quote") or []
    if isinstance(rows, dict):                 # one symbol → dict not list
        rows = [rows]
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sym = (r.get("symbol") or "").upper()
        if not sym:
            continue
        out[sym] = r
    return out


def _format_quote_block(quotes: Dict[str, Dict[str, Any]],
                          symbols: List[str]) -> str:
    """Pretty-print a one-line-per-symbol summary."""
    lines: List[str] = []
    for s in symbols:
        s = s.strip().upper()
        q = quotes.get(s)
        if not q:
            lines.append(f"`{s:5s}` no quote")
            continue
        last = float(q.get("last") or 0.0)
        chg = float(q.get("change") or 0.0)
        chg_pct = float(q.get("change_percentage") or 0.0)
        hi = float(q.get("high") or 0.0)
        lo = float(q.get("low") or 0.0)
        vol = int(q.get("volume") or 0)
        arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "·")
        lines.append(
            f"`{s:5s}` {arrow} ${last:>8,.2f} "
            f"({chg:+.2f} / {chg_pct:+.2f}%)  "
            f"H ${hi:,.2f}  L ${lo:,.2f}  vol {vol:,}"
        )
    return "\n".join(lines)


def _tail_log(lines: int = 800) -> List[str]:
    if not LOG_FILE.exists():
        return []
    try:
        with LOG_FILE.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = min(size, 256 * 1024)
            f.seek(size - block)
            chunk = f.read().decode("utf-8", errors="replace")
        return chunk.splitlines()[-lines:]
    except Exception:
        return []


def _bot_activity(window_min: int = 25) -> Dict[str, Any]:
    """Count emit / pass / block / fill in the last `window_min` minutes."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=window_min)
    emit = passes = blocks = fills = 0
    last_emits: List[str] = []
    last_blocks: List[str] = []
    last_fills: List[str] = []
    for line in _tail_log():
        # Each structlog line starts with ISO ts. Cheap parse.
        if len(line) < 20 or line[10] != "T":
            continue
        try:
            ts = datetime.fromisoformat(line[:26].rstrip("Z").replace("Z", ""))
            ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        if "ensemble_emit" in line:
            emit += 1
            if len(last_emits) < 5:
                last_emits.append(line)
        elif "exec_chain_pass" in line:
            passes += 1
        elif "exec_chain_block" in line:
            blocks += 1
            if len(last_blocks) < 4:
                last_blocks.append(line)
        elif "auto_pt=" in line:
            fills += 1
            if len(last_fills) < 4:
                last_fills.append(line)
    return {
        "emit": emit, "pass": passes, "block": blocks, "fill": fills,
        "last_emits": last_emits, "last_blocks": last_blocks,
        "last_fills": last_fills,
    }


def _trim(line: str, n: int = 130) -> str:
    """Strip ANSI color codes + trim to n chars for Discord."""
    import re as _re
    line = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
    if len(line) > n:
        line = line[:n] + "…"
    return line


def _post_discord(payload_obj: Dict[str, Any]) -> bool:
    """Post a Discord webhook payload. Returns True on success.
    Prefers DISCORD_WEBHOOK_URL_NEWS so the market-context channel gets
    both news AND recap; falls back to DISCORD_WEBHOOK_URL otherwise."""
    url = (os.getenv("DISCORD_WEBHOOK_URL_NEWS")
            or os.getenv("DISCORD_WEBHOOK_URL")
            or os.getenv("DISCORD_WEBHOOK")
            or "").strip()
    if not url:
        print("[recap] no DISCORD_WEBHOOK_URL_NEWS set — printing instead")
        print(json.dumps(payload_obj, indent=2))
        return False
    payload = json.dumps(payload_obj).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        print(f"[recap] discord_post_err {e}")
        return False


def _market_color(quotes: Dict[str, Dict[str, Any]]) -> int:
    """Embed color based on SPY's % change — bull green / bear red /
    neutral blue. Picks the first ETF in SYMBOLS that has a quote."""
    for sym in ("SPY", "QQQ", "IWM"):
        q = quotes.get(sym)
        if q is None:
            continue
        try:
            pct = float(q.get("change_percentage") or 0.0)
        except Exception:
            return EMBED_COLOR_NEUTRAL
        if pct > 0.20:   return EMBED_COLOR_BULL
        if pct < -0.20:  return EMBED_COLOR_BEAR
        return EMBED_COLOR_NEUTRAL
    return EMBED_COLOR_NEUTRAL


def build_recap_payload() -> Dict[str, Any]:
    """Build a Discord webhook payload with a polished embed.

    Layout:
      title:       "📊 Market Snapshot"
      description: ETF/index price grid (compact, fixed-width)
      fields:      Market activity stats + recent fills/blocks/signals
      footer:      Brand + disclaimer
      timestamp:   ISO time
    """
    now = datetime.now(tz=timezone.utc)
    quotes = _tradier_quotes(SYMBOLS)
    qblock = _format_quote_block(quotes, SYMBOLS)
    a = _bot_activity()

    fields: List[Dict[str, Any]] = []

    # Bot activity row
    fields.append({
        "name": "Engine activity (last 25 min)",
        "value": (f"Signals **{a['emit']}**  ·  "
                   f"Passed **{a['pass']}**  ·  "
                   f"Filtered **{a['block']}**  ·  "
                   f"Fills **{a['fill']}**"),
        "inline": False,
    })

    # Recent fills
    if a["last_fills"]:
        body = "\n".join(_trim(ln, 200) for ln in a["last_fills"][:3])
        fields.append({
            "name": "Recent executions",
            "value": f"```\n{body}\n```",
            "inline": False,
        })

    # Recent blocks (why entries didn't fire)
    if a["last_blocks"]:
        body = "\n".join(_trim(ln, 200) for ln in a["last_blocks"][:3])
        fields.append({
            "name": "Recently filtered",
            "value": f"```\n{body}\n```",
            "inline": False,
        })

    # Recent signals
    if a["last_emits"]:
        body = "\n".join(_trim(ln, 200) for ln in a["last_emits"][:4])
        fields.append({
            "name": "Latest signals",
            "value": f"```\n{body}\n```",
            "inline": False,
        })

    embed = {
        "title": "📊 Market Snapshot",
        "description": f"```\n{qblock}\n```",
        "color": _market_color(quotes),
        "fields": fields,
        "footer": {"text": BRAND_FOOTER},
        "timestamp": now.isoformat(),
    }

    return {
        "username": BRAND_NAME,
        "embeds": [embed],
    }


def build_disclaimer_payload() -> Dict[str, Any]:
    """A standalone disclaimer post — sent once per session start so a
    fresh viewer of the channel always sees the legal notice without
    cluttering each recap with the full text."""
    return {
        "username": BRAND_NAME,
        "embeds": [{
            "title": "Disclaimer",
            "description": DISCLAIMER,
            "color": EMBED_COLOR_NEUTRAL,
            "footer": {"text": BRAND_FOOTER},
        }],
    }


def main() -> int:
    print(f"[recap] starting — brand='{BRAND_NAME}' "
           f"symbols={len(SYMBOLS)} interval={RECAP_INTERVAL_SEC}s")
    disclaimer_posted = False
    while True:
        now = datetime.now(tz=timezone.utc)
        if not _market_open_et(now):
            # Reset the daily disclaimer flag overnight so the next
            # session opens with one legal notice posted.
            disclaimer_posted = False
            time.sleep(300)
            continue
        # Post the disclaimer once per session
        if not disclaimer_posted:
            _post_discord(build_disclaimer_payload())
            disclaimer_posted = True
        payload = build_recap_payload()
        ok = _post_discord(payload)
        print(f"[recap] {now.isoformat()} posted={ok}")
        # Sleep until the next 20-min boundary so posts are aligned
        cycle_sec = RECAP_INTERVAL_SEC
        next_boundary = (
            (int(now.timestamp()) // cycle_sec + 1) * cycle_sec
            - int(now.timestamp())
        )
        time.sleep(max(60.0, next_boundary))


if __name__ == "__main__":
    raise SystemExit(main())
