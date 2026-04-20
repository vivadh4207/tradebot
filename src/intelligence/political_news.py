"""Political news aggregator for the 70B strategy auditor.

Three sources, all optional, all fail-graceful:

  1. **Alpaca News** (already authenticated for the bot's existing
     news sentiment pipeline). Filtered by a political keyword list.
     Covers Reuters, Bloomberg, CNBC, WSJ. Most reliable source.

  2. **X / Twitter via Nitter RSS** — Nitter is a community-run X
     mirror that exposes user timelines as RSS. No API key. Brittle
     (instances go up and down; rate-limited). We treat it as a
     best-effort signal. Configurable instance + account list.

  3. **Arbitrary RSS feeds** — the cleanest way to follow Truth Social,
     White House press, Fed speeches, Congress releases. Each user
     configures which feeds to pull. Truth Social doesn't have a stable
     official API, but most prominent accounts are mirrored via
     trumpstwitter.com-style archives that do publish RSS.

The output is a ranked list of headlines, each tagged with:
  - source (alpaca | x:@handle | rss:url)
  - headline text
  - timestamp
  - an optional "political_score" 0..1 when the provider can infer it

Feeds the `StrategyAuditor` snapshot under key `political_news`. The
70B model sees it and incorporates political risk into its audit —
e.g. "Rate-decision speech tomorrow; selling premium into FOMC is a
red flag against this snapshot's credit_spreads.enabled=true."

## Security + ToS notes
- Nitter's ToS permits read-only RSS consumption but individual
  instances may ban aggressive polling. We cache results and limit
  to one pull per run.
- Scraping X directly is against X's ToS — don't do it. Use their
  official API (paid) OR use Nitter RSS, which is scraped by
  volunteer instances, not by us.
- Truth Social has no official API. RSS is what mirror sites publish.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


_log = logging.getLogger(__name__)


# ------------------------------------------------------------ config


# Curated political + geopolitical + economic-panic keywords. Broad
# enough to catch market-moving events, narrow enough to exclude
# celebrity / sports noise. Used for Alpaca News filtering and for
# X/RSS relevance scoring. Lowercased compare.
_DEFAULT_POLITICAL_KEYWORDS = [
    # --- Monetary / Fed ---
    "fed", "federal reserve", "powell", "fomc", "rate cut", "rate hike",
    "interest rate", "inflation", "cpi", "ppi", "jobs report", "nfp",
    "unemployment", "stagflation",
    # --- Executive / Legislative ---
    "white house", "president", "congress", "senate", "house of representatives",
    "bipartisan", "gridlock", "shutdown", "debt ceiling", "budget",
    # --- Trade / sanctions ---
    "tariff", "trade war", "sanction", "sanctions", "export ban",
    "import ban", "decoupling",
    # --- Geopolitical actors (nation-state) ---
    "china", "russia", "taiwan", "iran", "north korea", "middle east",
    "saudi arabia", "saudi", "uae", "qatar", "israel", "gaza", "lebanon",
    "syria", "yemen", "iraq", "afghanistan", "ukraine", "europe",
    # --- Non-state actors / groups ---
    "houthi", "houthis", "hezbollah", "hamas", "isis", "taliban",
    # --- Military action ---
    "military action", "military strike", "military exercise",
    "military deployment", "military operation",
    "air strike", "airstrike", "drone strike", "missile strike",
    "ballistic missile", "cruise missile",
    "naval", "warship", "destroyer", "aircraft carrier", "submarine",
    "airspace", "no-fly zone", "troops", "deployment", "ground forces",
    # --- Conflict / escalation ---
    "war", "ceasefire", "escalation", "retaliation", "reprisal",
    "invasion", "occupation", "blockade", "siege", "hostage",
    "conflict", "clash", "skirmish", "proxy war", "regional conflict",
    "geopolitical", "geopolitical risk", "geopolitical tension",
    # --- Strait of Hormuz / shipping chokepoints ---
    "strait of hormuz", "hormuz", "bab el-mandeb", "red sea",
    "suez canal", "panama canal", "shipping lane", "tanker",
    # --- Oil + energy ---
    "oil", "oil supply", "oil shock", "oil embargo", "oil spike",
    "oil price", "crude", "crude oil", "wti", "brent",
    "barrel", "refinery", "pipeline", "aramco",
    "opec", "opec+", "energy crisis", "gas supply", "natural gas",
    # --- Regulatory ---
    "sec", "ftc", "doj", "antitrust", "regulation", "executive order",
    # --- Elections ---
    "election", "primary", "poll", "campaign", "ballot",
    # --- Economic crisis / panic / fury language ---
    "crisis", "panic", "crash", "plunge", "plummet", "collapse",
    "meltdown", "bloodbath", "rout", "tumble", "freefall",
    "selloff", "sell-off", "turmoil", "volatility spike", "vix spike",
    "circuit breaker", "halt trading",
    "recession", "depression", "downturn", "slump", "contraction",
    "debt crisis", "financial crisis", "banking crisis", "credit crisis",
    "sovereign debt", "default", "bailout", "bank run",
    # --- Political fury / outrage language ---
    "fury", "furor", "outrage", "outraged", "backlash", "uproar",
    "denounce", "denounced", "condemn", "condemned", "retaliate",
    "retaliation", "anger", "worldwide anger", "protest", "riot",
    "unrest", "demonstration",
]


# Reasonable defaults — operator customizes via settings.yaml.
_DEFAULT_X_HANDLES = [
    "WhiteHouse",
    "FederalReserve",
    "POTUS",
    "USTradeRep",
    "SecYellen",
    "SpeakerJohnson",
    "SenSchumer",
    "realDonaldTrump",      # via Nitter — may be unreliable
]


_DEFAULT_NITTER_INSTANCE = "https://nitter.privacydev.net"


# ------------------------------------------------------------ data types


@dataclass
class PoliticalHeadline:
    source: str              # "alpaca" | "x:@handle" | "rss:<url>"
    headline: str
    timestamp: str           # ISO8601 UTC
    url: str = ""
    political_score: float = 0.0   # 0..1 — heuristic keyword density


@dataclass
class PoliticalNewsConfig:
    enabled: bool = False
    alpaca_enabled: bool = True
    x_enabled: bool = False
    rss_enabled: bool = False
    nitter_instance: str = _DEFAULT_NITTER_INSTANCE
    x_handles: List[str] = field(default_factory=lambda: list(_DEFAULT_X_HANDLES))
    rss_feeds: List[str] = field(default_factory=list)
    political_keywords: List[str] = field(
        default_factory=lambda: list(_DEFAULT_POLITICAL_KEYWORDS))
    lookback_hours: int = 24
    max_headlines: int = 40
    cache_seconds: float = 900.0        # 15 min
    http_timeout_sec: float = 8.0


# ------------------------------------------------------------ scoring


def _political_score(text: str, keywords: List[str]) -> float:
    """Crude keyword-density score. Returns 0..1.

    Used for ranking headlines — Alpaca's general feed doesn't
    pre-filter by topic. A higher score means more political keywords
    hit per word, so the top-N slice is disproportionately political.
    """
    t = (text or "").lower()
    if not t:
        return 0.0
    total_words = max(1, len(t.split()))
    matches = sum(1 for kw in keywords if kw in t)
    # Normalize to 0..1 roughly — 3+ keyword hits in a short headline = ~1.0
    return min(1.0, matches / max(3.0, total_words / 20))


# ------------------------------------------------------------ providers


class PoliticalNewsProvider:
    """Aggregates political headlines from multiple sources with a shared
    cache so the 70B auditor can see them without slowing entries."""

    def __init__(self, cfg: Optional[PoliticalNewsConfig] = None,
                 *, alpaca_news_client=None):
        """
        Args:
          cfg: configuration. If None, defaults are used.
          alpaca_news_client: existing AlpacaNewsProvider or equivalent.
                               The provider's `.recent(symbols, hours)`
                               is called to reuse the already-auth feed.
        """
        self.cfg = cfg or PoliticalNewsConfig()
        self._alpaca = alpaca_news_client
        self._cache: Optional[List[PoliticalHeadline]] = None
        self._cache_ts: float = 0.0

    # ---------- public API ----------

    def headlines(self, *, force_refresh: bool = False) -> List[PoliticalHeadline]:
        """Ranked list of political headlines. Returns [] on total
        outage — the auditor handles that case."""
        if not self.cfg.enabled:
            return []
        now = time.time()
        if (not force_refresh
                and self._cache is not None
                and (now - self._cache_ts) < self.cfg.cache_seconds):
            return self._cache

        out: List[PoliticalHeadline] = []
        if self.cfg.alpaca_enabled:
            try:
                out.extend(self._fetch_alpaca())
            except Exception as e:
                _log.warning("political_news_alpaca_failed err=%s", e)

        if self.cfg.x_enabled:
            for handle in self.cfg.x_handles:
                try:
                    out.extend(self._fetch_nitter_rss(handle))
                except Exception as e:
                    _log.info("political_news_x_skip handle=%s err=%s",
                               handle, e)

        if self.cfg.rss_enabled:
            for url in self.cfg.rss_feeds:
                try:
                    out.extend(self._fetch_generic_rss(url))
                except Exception as e:
                    _log.info("political_news_rss_skip url=%s err=%s",
                               url, e)

        # Dedupe (coarse — first-140-chars match) and rank by political_score
        seen = set()
        uniq: List[PoliticalHeadline] = []
        for h in out:
            key = (h.headline or "")[:140].lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(h)
        uniq.sort(key=lambda h: h.political_score, reverse=True)
        uniq = uniq[:self.cfg.max_headlines]

        self._cache = uniq
        self._cache_ts = now
        return uniq

    def snapshot_for_auditor(self) -> Dict[str, Any]:
        """Compact structure the StrategyAuditor embeds in its prompt.
        Limited headlines to keep the 70B context window manageable."""
        hs = self.headlines()
        return {
            "n_headlines": len(hs),
            "lookback_hours": self.cfg.lookback_hours,
            "top": [{
                "source": h.source,
                "headline": h.headline[:220],
                "timestamp": h.timestamp,
                "score": round(h.political_score, 3),
            } for h in hs[:12]],
        }

    # ---------- sources ----------

    def _fetch_alpaca(self) -> List[PoliticalHeadline]:
        """Uses the already-authenticated Alpaca News client. Filters
        by keyword density so we only keep political-looking items."""
        if self._alpaca is None:
            return []
        raw = []
        # Try the most common call shapes gracefully
        for call in (
            lambda: self._alpaca.recent(hours=self.cfg.lookback_hours),
            lambda: self._alpaca.recent_news(hours=self.cfg.lookback_hours),
            lambda: self._alpaca.news(hours=self.cfg.lookback_hours),
        ):
            try:
                raw = call() or []
                if raw:
                    break
            except Exception:
                continue
        out: List[PoliticalHeadline] = []
        for item in raw:
            headline = (getattr(item, "headline", None)
                         or (item.get("headline") if isinstance(item, dict) else "")
                         or "")
            ts = (getattr(item, "created_at", None)
                   or (item.get("created_at") if isinstance(item, dict) else None)
                   or datetime.now(tz=timezone.utc).isoformat())
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            url = (getattr(item, "url", "")
                    or (item.get("url", "") if isinstance(item, dict) else "")
                    or "")
            score = _political_score(headline, self.cfg.political_keywords)
            if score <= 0:
                continue
            out.append(PoliticalHeadline(
                source="alpaca",
                headline=str(headline)[:300],
                timestamp=str(ts),
                url=str(url),
                political_score=score,
            ))
        return out

    def _fetch_nitter_rss(self, handle: str) -> List[PoliticalHeadline]:
        """Nitter RSS for one X handle. `handle` without leading @."""
        if not handle:
            return []
        base = self.cfg.nitter_instance.rstrip("/")
        url = f"{base}/{handle}/rss"
        items = _fetch_rss(url, timeout=self.cfg.http_timeout_sec)
        out: List[PoliticalHeadline] = []
        for it in items:
            text = it.get("title", "") + " " + it.get("description", "")
            text = re.sub(r"<[^>]+>", "", text).strip()
            score = _political_score(text, self.cfg.political_keywords)
            if score <= 0:
                continue
            out.append(PoliticalHeadline(
                source=f"x:@{handle}",
                headline=text[:300],
                timestamp=it.get("pubDate", datetime.now(tz=timezone.utc).isoformat()),
                url=it.get("link", ""),
                political_score=score,
            ))
        return out

    def _fetch_generic_rss(self, url: str) -> List[PoliticalHeadline]:
        items = _fetch_rss(url, timeout=self.cfg.http_timeout_sec)
        out: List[PoliticalHeadline] = []
        for it in items:
            headline = it.get("title", "")
            if not headline:
                continue
            score = _political_score(headline, self.cfg.political_keywords)
            if score <= 0:
                continue
            out.append(PoliticalHeadline(
                source=f"rss:{url}",
                headline=headline[:300],
                timestamp=it.get("pubDate",
                                   datetime.now(tz=timezone.utc).isoformat()),
                url=it.get("link", ""),
                political_score=score,
            ))
        return out


# ------------------------------------------------------------ RSS parser
# Tiny stdlib-only RSS reader — avoids adding feedparser as a dep.

def _fetch_rss(url: str, *, timeout: float = 8.0) -> List[Dict[str, str]]:
    """Return list of {title, description, pubDate, link} dicts.
    Parses RSS 2.0 and a common subset of Atom. Network / parse failures
    return []. NEVER raises."""
    import urllib.request
    import urllib.error
    from xml.etree import ElementTree as ET

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "tradebot-political-news/1.0 (read-only)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception:
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []

    out: List[Dict[str, str]] = []
    # RSS 2.0: root/channel/item
    items = root.findall(".//item")
    if items:
        for it in items:
            out.append({
                "title": (it.findtext("title") or "").strip(),
                "description": (it.findtext("description") or "").strip(),
                "pubDate": (it.findtext("pubDate") or "").strip(),
                "link": (it.findtext("link") or "").strip(),
            })
        return out

    # Atom: root/{ns}entry
    ns = "{http://www.w3.org/2005/Atom}"
    entries = root.findall(f"{ns}entry")
    for e in entries:
        title = e.findtext(f"{ns}title") or ""
        summary = e.findtext(f"{ns}summary") or ""
        updated = e.findtext(f"{ns}updated") or ""
        link_el = e.find(f"{ns}link")
        link = link_el.get("href", "") if link_el is not None else ""
        out.append({
            "title": title.strip(),
            "description": summary.strip(),
            "pubDate": updated.strip(),
            "link": link,
        })
    return out


# ------------------------------------------------------------ factory


def build_political_news_provider(settings,
                                    *, alpaca_news_client=None) -> Optional[PoliticalNewsProvider]:
    """Construct from settings.raw.political_news. Returns None if
    the whole feature is disabled."""
    cfg_d = (settings.raw.get("political_news", {}) or {})
    if not cfg_d.get("enabled", False):
        return None
    cfg = PoliticalNewsConfig(
        enabled=True,
        alpaca_enabled=bool(cfg_d.get("alpaca_enabled", True)),
        x_enabled=bool(cfg_d.get("x_enabled", False)),
        rss_enabled=bool(cfg_d.get("rss_enabled", False)),
        nitter_instance=str(cfg_d.get("nitter_instance",
                                         _DEFAULT_NITTER_INSTANCE)),
        x_handles=list(cfg_d.get("x_handles", _DEFAULT_X_HANDLES)),
        rss_feeds=list(cfg_d.get("rss_feeds", [])),
        political_keywords=list(cfg_d.get("political_keywords",
                                            _DEFAULT_POLITICAL_KEYWORDS)),
        lookback_hours=int(cfg_d.get("lookback_hours", 24)),
        max_headlines=int(cfg_d.get("max_headlines", 40)),
        cache_seconds=float(cfg_d.get("cache_seconds", 900.0)),
        http_timeout_sec=float(cfg_d.get("http_timeout_sec", 8.0)),
    )
    return PoliticalNewsProvider(cfg, alpaca_news_client=alpaca_news_client)
