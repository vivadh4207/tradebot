"""Catalyst deep-dive — aggregates news, social, earnings, FOMC/CPI
events, sentiment, and political-news feeds; asks the LLM to produce
a structured "what's driving markets right now" report.

This is a wider lens than run_options_research.py (which focuses on
specific underlyings + strikes). The catalyst dive answers: WHY is the
tape doing what it's doing?

Sources pulled in parallel (all optional; missing keys skipped):
  - Finnhub company news with sentiment scores
  - Polygon news per ticker (if key set)
  - Yahoo Finance news
  - Political news (Alpaca + Nitter/X + curated RSS) via
    src.intelligence.political_news — Fed speeches, White House,
    Treasury, CNBC, Reuters, Bloomberg, WSB, Truth Social
  - Earnings calendar for universe via yfinance (ETFs skipped)
  - Per-ticker sentiment aggregate

LLM prompt demands structured output with:
  - One-line tape read
  - Top 3 catalysts driving markets RIGHT NOW
  - Upcoming catalysts (next 48h)
  - Social-media pulse (what retail is watching)
  - Risk flags

Posts to Discord #catalyst channel (or catch-all).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.notify.base import build_notifier
from src.notify.issue_reporter import alert_on_crash
from src.data.multi_provider import MultiProvider
from src.intelligence.ollama_client import build_ollama_client


_PROMPT_TEMPLATE = """You are the overnight catalyst analyst for a
retail options trading bot. Review the SNAPSHOT below and produce a
structured deep-dive on what's driving the tape.

RULES (anti-hallucination):
  1. Every named entity (ticker, person, event) MUST appear in the
     SNAPSHOT. Do not cite sources or headlines that aren't listed.
  2. Sentiment values must come from the sentiment_by_symbol or
     aggregate_sentiment block — don't invent them.
  3. If you're unsure, say so — don't fill with speculation.

OUTPUT STRICT JSON with these fields:
  {{
    "tape_read": "1-sentence overall take",
    "top_catalysts_now": [
      {{"headline": "...", "source": "...",
        "tickers": ["SPY","..."], "sentiment": -0.3,
        "why_it_matters": "...", "timeframe": "intraday|short-term|medium"}}
    ],
    "upcoming_48h": [
      {{"event": "FOMC minutes release",
        "when": "2026-04-23 14:00 ET",
        "impact": "high|medium|low",
        "expected_direction": "hawkish|dovish|uncertain"}}
    ],
    "social_pulse": "what WSB/Reddit/X are discussing — 1-2 sentences with specific ticker mentions from snapshot",
    "risk_flags": ["flag 1", "flag 2"],
    "positioning_bias": "bullish|bearish|neutral, with brief rationale"
  }}

SNAPSHOT:
{snapshot}

YOUR JSON:
"""


def _gather_snapshot(mp: MultiProvider, universe: List[str]
                       ) -> Dict[str, Any]:
    """Fan out to every provider for news + sentiment. De-dupes by
    headline hash. Adds political_news aggregator output if the module
    is configured. Returns a compact dict ready for the LLM prompt."""
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    # 1. News pooled across providers, tagged by symbol
    all_news: List[Dict[str, Any]] = []
    seen: set = set()
    for sym in universe:
        for item in (mp.news(sym, limit=15) or []):
            key = (item.headline or "").lower()[:120]
            if key in seen or not key:
                continue
            seen.add(key)
            all_news.append({
                "ts": item.ts,
                "symbol": sym,
                "headline": item.headline,
                "summary": (item.summary or "")[:180],
                "source": item.source,
                "sentiment": item.sentiment_score,
                "tickers": list(item.tickers or [])[:6],
            })
    # 2. Market-wide headlines (no symbol filter) — captures macro
    for item in (mp.news(None, limit=15) or []):
        key = (item.headline or "").lower()[:120]
        if key in seen or not key:
            continue
        seen.add(key)
        all_news.append({
            "ts": item.ts,
            "symbol": None,
            "headline": item.headline,
            "summary": (item.summary or "")[:180],
            "source": item.source,
        })

    # 3. Aggregate sentiment per symbol
    sentiment_by_symbol = {}
    for sym in universe:
        s = mp.news_sentiment(sym)
        if s is not None:
            sentiment_by_symbol[sym] = round(s, 3)

    # 4. Political-news provider (Fed / WH / Treasury / Truth Social / WSB)
    political: List[Dict[str, Any]] = []
    try:
        from src.core.config import load_settings
        from src.intelligence.political_news import build_political_news_provider
        s = load_settings(ROOT / "config" / "settings.yaml")
        pol = build_political_news_provider(s)
        if pol is not None:
            snap = pol.snapshot_for_auditor()
            if isinstance(snap, dict):
                political = list(snap.get("headlines", []))[:30]
            elif isinstance(snap, list):
                political = snap[:30]
    except Exception:
        pass

    # 5. Earnings calendar per stock (ETFs skipped)
    earnings: List[Dict[str, Any]] = []
    try:
        import yfinance as yf
        for sym in universe:
            if sym in ("SPY", "QQQ", "IWM", "DIA"):
                continue        # ETFs don't have earnings
            try:
                t = yf.Ticker(sym)
                cal = getattr(t, "calendar", None)
                if cal is not None and not cal.empty:
                    earnings.append({
                        "symbol": sym,
                        "earnings_date": str(cal.iloc[0, 0]),
                    })
            except Exception:
                continue
    except Exception:
        pass

    # 6. Known macro events (next 48h). This is a hardcoded minimum;
    # production would pull from an econ-calendar provider.
    macro_events = _static_macro_events_next_48h()

    # 7. Live VIX — if any provider supports it (Yahoo does)
    vix = None
    for p in getattr(mp, "_providers", []):
        fn = getattr(p, "latest_vix", None)
        if fn:
            try:
                vix = fn()
                if vix:
                    break
            except Exception:
                pass

    return {
        "now_utc": now_iso,
        "universe": universe,
        "vix": (round(float(vix), 2) if vix is not None else None),
        "aggregate_sentiment": sentiment_by_symbol,
        "news": all_news[:60],
        "political_news": political[:20],
        "earnings": earnings[:20],
        "macro_events_48h": macro_events,
        "providers_active": mp.active_providers(),
    }


def _static_macro_events_next_48h() -> List[Dict[str, Any]]:
    """Placeholder scheduler for common US macro events. Replace with a
    real econ-calendar provider when you get one (ForexFactory free RSS
    works, or Finnhub's /calendar/economic on paid tier)."""
    today = datetime.now(tz=timezone.utc)
    dow = today.weekday()                           # Mon=0 ... Sun=6
    events = []
    # CPI / FOMC schedules are monthly and irregular; leaving the list
    # empty here rather than misleading the LLM with stale dates.
    # Operator should add key dates to .env as JSON if needed.
    import os as _os
    raw = _os.getenv("MACRO_EVENTS_JSON", "").strip()
    if raw:
        try:
            events = json.loads(raw)
        except Exception:
            pass
    return events


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.find("{")
    e = raw.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(raw[s:e + 1])
    except Exception:
        return None


def _sanitize_mentions(s: str) -> str:
    import re as _re
    return _re.sub(r"@everyone|@here|<@[!&]?\d+>|<@&\d+>",
                    "[mention-stripped]", s or "")


def _format_discord(report: Dict[str, Any], *, model: str,
                     latency: float) -> str:
    lines: List[str] = []
    lines.append(f"**📰 Catalyst Deep Dive · {model} · {latency:.0f}s**")
    tape = report.get("tape_read")
    if tape:
        lines.append(f"**Tape:** {tape}")
    pb = report.get("positioning_bias")
    if pb:
        lines.append(f"**Bias:** {pb}")
    tops = report.get("top_catalysts_now") or []
    if tops:
        lines.append("\n**📌 Top catalysts right now**")
        for i, c in enumerate(tops[:5], 1):
            head = c.get("headline", "?")
            src = c.get("source", "?")
            tickers = ",".join(c.get("tickers", [])[:4])
            s_val = c.get("sentiment")
            s_tag = f" · sent {s_val:+.2f}" if isinstance(s_val, (int, float)) else ""
            lines.append(f"{i}. *{head[:140]}* — [{src}]{s_tag}")
            if tickers:
                lines.append(f"   tickers: {tickers}")
            if c.get("why_it_matters"):
                lines.append(f"   💡 {c['why_it_matters'][:200]}")
    upc = report.get("upcoming_48h") or []
    if upc:
        lines.append("\n**⏰ Next 48h**")
        for e in upc[:5]:
            lines.append(
                f"  · {e.get('event','?')} — {e.get('when','?')} "
                f"({e.get('impact','?')} / {e.get('expected_direction','?')})"
            )
    soc = report.get("social_pulse")
    if soc:
        lines.append(f"\n**🐦 Social pulse:** {soc[:300]}")
    risks = report.get("risk_flags") or []
    if risks:
        lines.append("\n**⚠️ Risk flags**")
        for r in risks[:4]:
            lines.append(f"  · {r[:200]}")
    out = _sanitize_mentions("\n".join(lines))
    if len(out) > 1900:
        out = out[:1880].rstrip() + "\n… [truncated]"
    return out


def _call_llm(prompt: str, *, model_name: Optional[str] = None,
               timeout_sec: float = 300.0, max_tokens: int = 800
               ) -> tuple:
    client = build_ollama_client()
    client.cfg.timeout_sec = float(timeout_sec)
    if not client.ping():
        return "", ""
    import os as _os
    candidates: List[str] = []
    if model_name:
        candidates.append(model_name)
    else:
        candidates = [
            (_os.getenv("LLM_AUDITOR_MODEL", "").strip() or "llama3.1:70b"),
            (_os.getenv("LLM_BRAIN_MODEL", "").strip() or "llama3.1:8b"),
        ]
    for m in candidates:
        try:
            raw = client.generate(
                model=m, prompt=prompt,
                temperature=0.2, max_tokens=max_tokens, num_ctx=8192,
                stop=["\n\nSNAPSHOT:", "\n\nYOUR JSON:"],
            )
            if raw and raw.strip():
                return raw, m
        except Exception:
            continue
    return "", ""


@alert_on_crash("catalyst_dive", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL",
                     help="Catalyst-scan universe (default: big tech + indexes)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--timeout-sec", type=float, default=300.0)
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()
    universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    mp = MultiProvider.from_env()

    snap = _gather_snapshot(mp, universe)
    prompt = _PROMPT_TEMPLATE.format(
        snapshot=json.dumps(snap, indent=2, default=str)[:14000],
    )
    import time as _time
    started = _time.time()
    raw, used_model = _call_llm(
        prompt, model_name=args.model,
        timeout_sec=float(args.timeout_sec),
        max_tokens=int(args.max_tokens),
    )
    latency = _time.time() - started

    parsed = _parse_llm_json(raw)
    if not parsed:
        body = (f"**📰 Catalyst Deep Dive · {used_model or 'n/a'} · "
                f"{latency:.0f}s**\n_no structured LLM output — "
                f"see logs/catalyst_dive.err_")
    else:
        body = _format_discord(parsed, model=used_model or "n/a",
                                 latency=latency)
    print(body)

    if not args.no_discord:
        meta = {
            "Model":       used_model or "(none)",
            "Latency":     f"{latency:.0f}s",
            "News":        len(snap.get("news", [])),
            "Political":   len(snap.get("political_news", [])),
            "Earnings":    len(snap.get("earnings", [])),
            "Symbols":     ",".join(universe[:6]),
            "Providers":   ",".join(snap.get("providers_active", [])),
            "_footer":     "catalyst_dive",
        }
        build_notifier().notify(
            body, level="info", title="catalysts", meta=meta,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
