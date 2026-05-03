"""Market news collector — pulls headlines from multiple free sources
and posts them to a dedicated Discord channel.

Run continuously:
    python3 scripts/news_collector.py

It self-throttles by source (each source has its own min-interval) and
dedupes by URL hash so the same headline never gets posted twice.
Persists seen-hashes to logs/news_seen.txt so a restart doesn't repost
yesterday's news.

Sources wired in (all free):
  * Finnhub /news?category=general              (market-wide)
  * Finnhub /company-news per universe ticker   (per-symbol)
  * Yahoo Finance via yfinance                  (per-symbol, scrapes)
  * Alpaca News (REST)                           (existing creds)
  * SEC EDGAR 8-K RSS                           (filings)
  * Federal Reserve press releases RSS          (FOMC, monetary policy)
  * Treasury press releases RSS                  (debt issuance, sanctions)

Optional (add the API key to .env to enable):
  * MarketAux         — MARKETAUX_KEY  (100 req/day free)
  * AlphaVantage      — ALPHAVANTAGE_KEY (25 req/day, has sentiment)
  * NewsAPI.org       — NEWSAPI_KEY    (100 req/day free, generic)

Posts to env DISCORD_WEBHOOK_URL_NEWS (separate channel) — falls back
to DISCORD_WEBHOOK_URL if that's not set so it still works out-of-box.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

SEEN_FILE = ROOT / "logs" / "news_seen.txt"
SEEN_TTL_SEC = 36 * 3600                 # forget hashes older than 36h

# Branding (override via env)
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
EMBED_COLOR = int(os.getenv("DISCORD_EMBED_COLOR", "0x5865F2"), 16)
DISCLAIMER_INTERVAL_SEC = 6 * 3600       # repost disclaimer every 6h

# Per-source min interval to avoid burning rate-limit. Map name → secs.
SOURCE_INTERVAL = {
    "finnhub_general":   600,    # 10 min
    "finnhub_company":   900,    # 15 min (loops over universe — slower)
    "yahoo":             900,    # 15 min
    "alpaca":            300,    # 5  min
    "sec_edgar":         600,    # 10 min
    "fed":              1800,    # 30 min
    "treasury":         1800,    # 30 min
    "marketaux":        3600,    # 60 min (low quota)
    "alphavantage":     3600,    # 60 min
    "newsapi":          3600,    # 60 min
}
LOOP_TICK_SEC = 60
DISCORD_BATCH_LIMIT = 12         # cap posts per cycle so we don't spam

# Universe pulled from settings.yaml at startup.
UNIVERSE: List[str] = []


# ---------- helpers ----------

def _load_universe() -> List[str]:
    """Read trading + monitor universe from settings.yaml — cheap parse."""
    syms: List[str] = []
    yml = ROOT / "config" / "settings.yaml"
    if not yml.exists():
        return ["SPY", "QQQ", "IWM", "DIA"]
    try:
        text = yml.read_text(encoding="utf-8")
        in_block = None
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("universe:") or s.startswith("monitor_universe:"):
                in_block = s.split(":", 1)[0]
                continue
            if in_block is None:
                continue
            # Hyphen-list under the block. Stop on next top-level key.
            if line.startswith(" ") and s.startswith("- "):
                tok = s[2:].split("#", 1)[0].strip().strip('"').strip("'")
                if tok and tok.isupper() and 1 <= len(tok) <= 5:
                    if tok not in syms:
                        syms.append(tok)
            elif s and not line.startswith(" "):
                in_block = None
    except Exception:
        pass
    return syms or ["SPY", "QQQ", "IWM", "DIA"]


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="replace"))
        h.update(b"|")
    return h.hexdigest()[:16]


def _load_seen() -> Dict[str, float]:
    if not SEEN_FILE.exists():
        return {}
    out: Dict[str, float] = {}
    cutoff = time.time() - SEEN_TTL_SEC
    try:
        for line in SEEN_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or " " not in line:
                continue
            ts_s, h = line.split(" ", 1)
            try:
                ts = float(ts_s)
            except Exception:
                continue
            if ts >= cutoff:
                out[h] = ts
    except Exception:
        pass
    return out


def _save_seen(seen: Dict[str, float]) -> None:
    try:
        SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(seen.items(), key=lambda kv: kv[1])
        SEEN_FILE.write_text(
            "\n".join(f"{ts:.0f} {h}" for h, ts in rows) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _http_get(url: str, headers: Optional[Dict[str, str]] = None,
                timeout: float = 10.0) -> Optional[bytes]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        print(f"[news] http_err {url[:80]} {e}")
        return None


def _strip(s: str, n: int = 220) -> str:
    s = html.unescape(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > n:
        s = s[:n].rstrip() + "…"
    return s


# ---------- source: Finnhub general ----------

def fetch_finnhub_general() -> List[Dict[str, Any]]:
    key = os.getenv("FINNHUB_KEY", "").strip()
    if not key:
        return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={key}"
    data = _http_get(url)
    if not data:
        return []
    try:
        rows = json.loads(data)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for r in rows[:30]:
        url_ = r.get("url") or ""
        head = r.get("headline") or ""
        if not head or not url_:
            continue
        out.append({
            "src": "finnhub", "ticker": "MARKET",
            "headline": _strip(head),
            "summary": _strip(r.get("summary") or "", 280),
            "url": url_,
            "ts": float(r.get("datetime") or time.time()),
        })
    return out


# ---------- source: Finnhub company-news ----------

def fetch_finnhub_company(symbol: str) -> List[Dict[str, Any]]:
    key = os.getenv("FINNHUB_KEY", "").strip()
    if not key:
        return []
    end = _now().date()
    start = end - timedelta(days=2)
    url = (f"https://finnhub.io/api/v1/company-news?symbol={symbol}"
            f"&from={start.isoformat()}&to={end.isoformat()}&token={key}")
    data = _http_get(url)
    if not data:
        return []
    try:
        rows = json.loads(data)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for r in rows[:10]:
        head = r.get("headline") or ""
        url_ = r.get("url") or ""
        if not head or not url_:
            continue
        out.append({
            "src": "finnhub", "ticker": symbol,
            "headline": _strip(head),
            "summary": _strip(r.get("summary") or "", 240),
            "url": url_,
            "ts": float(r.get("datetime") or time.time()),
        })
    return out


# ---------- source: Yahoo Finance per ticker ----------

def fetch_yahoo(symbol: str) -> List[Dict[str, Any]]:
    try:
        import yfinance as yf
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    try:
        items = yf.Ticker(symbol).news or []
    except Exception:
        return []
    for it in items[:10]:
        head = it.get("title") or ""
        url_ = it.get("link") or it.get("url") or ""
        ts = it.get("providerPublishTime") or time.time()
        if not head or not url_:
            continue
        out.append({
            "src": "yahoo", "ticker": symbol,
            "headline": _strip(head),
            "summary": _strip(it.get("publisher") or "", 80),
            "url": url_, "ts": float(ts),
        })
    return out


# ---------- source: Alpaca news ----------

def fetch_alpaca() -> List[Dict[str, Any]]:
    k = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or ""
    s = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY") or ""
    if not k or not s:
        return []
    syms = ",".join(UNIVERSE[:25])
    since = (_now() - timedelta(hours=4)).isoformat()
    url = (f"https://data.alpaca.markets/v1beta1/news"
            f"?symbols={syms}&start={since}&limit=30")
    data = _http_get(url, headers={
        "APCA-API-KEY-ID": k.strip(),
        "APCA-API-SECRET-KEY": s.strip(),
    })
    if not data:
        return []
    try:
        rows = json.loads(data).get("news") or []
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        head = r.get("headline") or ""
        url_ = r.get("url") or ""
        if not head or not url_:
            continue
        ticks = r.get("symbols") or []
        out.append({
            "src": "alpaca",
            "ticker": (ticks[0] if ticks else "MARKET"),
            "headline": _strip(head),
            "summary": _strip(r.get("summary") or "", 240),
            "url": url_,
            "ts": time.time(),
        })
    return out


# ---------- source: RSS feeds (no key) ----------

def _parse_rss(content: bytes, src_name: str,
                  ticker_hint: str = "MACRO") -> List[Dict[str, Any]]:
    """Tolerant RSS/Atom parser — only pulls title + link + pubDate."""
    out: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(content)
    except Exception:
        return out
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"
    items = root.iter(f"{ns}item") if ns else root.iter("item")
    items = list(items)
    if not items:
        items = list(root.iter("{http://www.w3.org/2005/Atom}entry"))
    for it in items[:20]:
        title = ""
        link = ""
        for child in it:
            tag = child.tag.split("}", 1)[-1]
            if tag == "title" and child.text:
                title = child.text
            elif tag == "link":
                link = (child.attrib.get("href")
                          or (child.text or "")).strip()
            elif tag == "summary" and not title:
                title = child.text or ""
        if title and link:
            out.append({
                "src": src_name, "ticker": ticker_hint,
                "headline": _strip(title),
                "summary": "",
                "url": link, "ts": time.time(),
            })
    return out


def fetch_sec_edgar() -> List[Dict[str, Any]]:
    """SEC EDGAR 8-K (current report — material events) RSS."""
    url = ("https://www.sec.gov/cgi-bin/browse-edgar"
           "?action=getcurrent&type=8-K&company=&dateb=&owner=include"
           "&count=40&output=atom")
    data = _http_get(url, headers={"User-Agent": "tradebot-news/1.0"})
    if not data:
        return []
    return _parse_rss(data, "sec_8k", ticker_hint="SEC")


def fetch_fed() -> List[Dict[str, Any]]:
    url = "https://www.federalreserve.gov/feeds/press_all.xml"
    data = _http_get(url, headers={"User-Agent": "tradebot-news/1.0"})
    if not data:
        return []
    return _parse_rss(data, "fed", ticker_hint="MACRO")


def fetch_treasury() -> List[Dict[str, Any]]:
    url = "https://home.treasury.gov/news/press-releases.rss"
    data = _http_get(url, headers={"User-Agent": "tradebot-news/1.0"})
    if not data:
        return []
    return _parse_rss(data, "treasury", ticker_hint="MACRO")


# ---------- source: optional API-key sources (stubbed if no key) ----------

def fetch_marketaux() -> List[Dict[str, Any]]:
    # Accept either name — user-facing convention varies.
    key = (os.getenv("MARKETAUX_KEY")
            or os.getenv("MARKET_AUX")
            or os.getenv("MARKETAUX")
            or "").strip()
    if not key:
        return []
    syms = ",".join(UNIVERSE[:20])
    url = (f"https://api.marketaux.com/v1/news/all?symbols={syms}"
            f"&filter_entities=true&language=en&api_token={key}")
    data = _http_get(url)
    if not data:
        return []
    try:
        rows = json.loads(data).get("data") or []
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for r in rows[:25]:
        head = r.get("title") or ""
        url_ = r.get("url") or ""
        if not head or not url_:
            continue
        ents = r.get("entities") or []
        tk = (ents[0].get("symbol") if ents else None) or "MARKET"
        out.append({
            "src": "marketaux", "ticker": tk,
            "headline": _strip(head),
            "summary": _strip(r.get("description") or "", 240),
            "url": url_, "ts": time.time(),
        })
    return out


def fetch_alphavantage() -> List[Dict[str, Any]]:
    key = (os.getenv("ALPHAVANTAGE_KEY")
            or os.getenv("ALPHA_VANTAGE_KEY")
            or os.getenv("ALPHAVANTAGE")
            or "").strip()
    if not key:
        return []
    syms = ",".join(UNIVERSE[:25])
    url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
            f"&tickers={syms}&limit=30&apikey={key}")
    data = _http_get(url)
    if not data:
        return []
    try:
        rows = json.loads(data).get("feed") or []
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for r in rows[:25]:
        head = r.get("title") or ""
        url_ = r.get("url") or ""
        if not head or not url_:
            continue
        sent = r.get("overall_sentiment_label") or "neutral"
        ts = r.get("ticker_sentiment") or [{}]
        tk = (ts[0].get("ticker") if ts else None) or "MARKET"
        out.append({
            "src": f"alphav[{sent}]", "ticker": tk,
            "headline": _strip(head),
            "summary": _strip(r.get("summary") or "", 240),
            "url": url_, "ts": time.time(),
        })
    return out


def fetch_newsapi() -> List[Dict[str, Any]]:
    key = os.getenv("NEWSAPI_KEY", "").strip()
    if not key:
        return []
    q = "stock+market+OR+Fed+OR+FOMC+OR+earnings"
    url = (f"https://newsapi.org/v2/everything?q={q}"
            f"&language=en&pageSize=25&sortBy=publishedAt&apiKey={key}")
    data = _http_get(url)
    if not data:
        return []
    try:
        rows = json.loads(data).get("articles") or []
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for r in rows[:20]:
        head = r.get("title") or ""
        url_ = r.get("url") or ""
        if not head or not url_:
            continue
        out.append({
            "src": "newsapi", "ticker": "MARKET",
            "headline": _strip(head),
            "summary": _strip(r.get("description") or "", 240),
            "url": url_, "ts": time.time(),
        })
    return out


# ---------- discord post ----------

def _emoji_for(src: str) -> str:
    s = src.lower()
    if "finnhub" in s:   return ":newspaper:"
    if "yahoo" in s:     return ":bar_chart:"
    if "alpaca" in s:    return ":zap:"
    if "sec" in s:       return ":bank:"
    if "fed" in s:       return ":bank:"
    if "treasury" in s:  return ":dollar:"
    if "alphav" in s:
        if "negative" in s: return ":warning:"
        if "positive" in s: return ":chart_with_upwards_trend:"
        return ":newspaper:"
    if "marketaux" in s: return ":satellite:"
    return ":newspaper:"


def _post_payload(url: str, payload: Dict[str, Any]) -> bool:
    """POST one Discord webhook payload."""
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if 200 <= resp.status < 300:
                return True
            print(f"[news] discord_status={resp.status}")
            return False
    except Exception as e:
        print(f"[news] discord_err {e}")
        return False


def _post_discord(items: List[Dict[str, Any]]) -> bool:
    """Post news items to the configured Discord webhook as a single
    branded embed digest (capped at 25 fields per Discord's limit)."""
    url = (os.getenv("DISCORD_WEBHOOK_URL_NEWS")
            or os.getenv("DISCORD_WEBHOOK_URL")
            or os.getenv("DISCORD_WEBHOOK")
            or "").strip()
    if not url:
        for it in items:
            print(f"[news] {it['src']:14s} {it['ticker']:6s} "
                   f"{it['headline'][:120]}")
        return False
    if not items:
        return True

    # Group up to 10 items per embed for clean display.
    BATCH = 10
    success = True
    for batch_start in range(0, len(items), BATCH):
        batch = items[batch_start:batch_start + BATCH]
        lines: List[str] = []
        for it in batch:
            emo = _emoji_for(it["src"])
            ts = datetime.fromtimestamp(
                it["ts"], tz=timezone.utc
            ).strftime("%H:%M")
            head = it["headline"]
            url_ = it["url"]
            line = (f"{emo} `{ts}` `{it['ticker']:>5s}` "
                     f"[{head}]({url_})")
            lines.append(line)
        embed = {
            "title": "📰 Market Headlines",
            "description": "\n".join(lines)[:4000],
            "color": EMBED_COLOR,
            "footer": {"text": BRAND_FOOTER},
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        payload = {"username": BRAND_NAME, "embeds": [embed]}
        if not _post_payload(url, payload):
            success = False
        time.sleep(1)
    return success


def _post_disclaimer() -> bool:
    """One-shot disclaimer post for legal cover. Called once per
    DISCLAIMER_INTERVAL_SEC."""
    url = (os.getenv("DISCORD_WEBHOOK_URL_NEWS")
            or os.getenv("DISCORD_WEBHOOK_URL")
            or os.getenv("DISCORD_WEBHOOK")
            or "").strip()
    if not url:
        return False
    payload = {
        "username": BRAND_NAME,
        "embeds": [{
            "title": "Disclaimer",
            "description": DISCLAIMER,
            "color": EMBED_COLOR,
            "footer": {"text": BRAND_FOOTER},
        }],
    }
    return _post_payload(url, payload)


# ---------- main loop ----------

def main() -> int:
    global UNIVERSE
    UNIVERSE = _load_universe()
    print(f"[news] universe ({len(UNIVERSE)}): {UNIVERSE}")
    seen = _load_seen()
    print(f"[news] loaded {len(seen)} seen-hashes")

    last_run: Dict[str, float] = {k: 0.0 for k in SOURCE_INTERVAL}
    last_universe_idx = 0
    last_disclaimer_ts = 0.0

    while True:
        now = time.time()
        batch: List[Dict[str, Any]] = []

        # General-market sources (whole-feed each call)
        broadcasters = {
            "finnhub_general": fetch_finnhub_general,
            "alpaca": fetch_alpaca,
            "sec_edgar": fetch_sec_edgar,
            "fed": fetch_fed,
            "treasury": fetch_treasury,
            "marketaux": fetch_marketaux,
            "alphavantage": fetch_alphavantage,
            "newsapi": fetch_newsapi,
        }
        for name, fn in broadcasters.items():
            if now - last_run[name] < SOURCE_INTERVAL[name]:
                continue
            last_run[name] = now
            try:
                rows = fn() or []
            except Exception as e:
                print(f"[news] {name}_err {e}")
                rows = []
            print(f"[news] {name} pulled {len(rows)}")
            batch.extend(rows)

        # Per-symbol sources (rotate one symbol per cycle to keep API quota low)
        for name, fn in (("finnhub_company", fetch_finnhub_company),
                          ("yahoo", fetch_yahoo)):
            if now - last_run[name] < SOURCE_INTERVAL[name]:
                continue
            if not UNIVERSE:
                continue
            sym = UNIVERSE[last_universe_idx % len(UNIVERSE)]
            last_universe_idx = (last_universe_idx + 1) % max(1, len(UNIVERSE))
            last_run[name] = now
            try:
                rows = fn(sym) or []
            except Exception as e:
                print(f"[news] {name}_err {e}")
                rows = []
            print(f"[news] {name}({sym}) pulled {len(rows)}")
            batch.extend(rows)

        # Dedup
        fresh: List[Dict[str, Any]] = []
        for it in batch:
            h = _hash(it.get("url", ""), it.get("headline", ""))
            if h in seen:
                continue
            seen[h] = now
            fresh.append(it)

        # Trim seen to TTL window
        cutoff = now - SEEN_TTL_SEC
        seen = {h: t for h, t in seen.items() if t >= cutoff}

        # Post a fresh disclaimer every DISCLAIMER_INTERVAL_SEC so any
        # new viewer of the channel sees the legal notice.
        if now - last_disclaimer_ts >= DISCLAIMER_INTERVAL_SEC:
            if _post_disclaimer():
                last_disclaimer_ts = now

        # Post the freshest first, capped per cycle
        fresh.sort(key=lambda x: x.get("ts", 0), reverse=True)
        if fresh:
            print(f"[news] posting {min(len(fresh), DISCORD_BATCH_LIMIT)} of "
                   f"{len(fresh)} fresh items")
            _post_discord(fresh[:DISCORD_BATCH_LIMIT])
            _save_seen(seen)

        time.sleep(LOOP_TICK_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
