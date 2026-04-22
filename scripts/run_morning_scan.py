"""Morning ticker scanner — runs pre-market each weekday, screens the
market for highest-opportunity setups, posts a ranked watchlist to
Discord.

Screens across (all optional, fail-soft):
  1. News-mentioned tickers (Finnhub / Yahoo / Polygon) — counts
     how many times each ticker appears in the last 24h of market news
  2. Highest IV-rank candidates from the default + mentioned universe
     (Yahoo / Tradier / Polygon chain)
  3. Biggest pre-market movers (via Yahoo intraday 1-day data)
  4. Highest relative-volume names
  5. Unusual options activity proxy (volume/OI ratio)

Output: a ranked watchlist with per-ticker rationale, posted to
  Discord title='llm_ideas' (or map DISCORD_WEBHOOK_URL_SCAN for a
  dedicated channel).

Runs daily via launchd (see deploy/launchd/com.tradebot.morning_scan.plist)
at 09:15 ET weekdays. Can also be invoked from Discord `!scan`.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
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
from src.intelligence.symbol_scanner import SymbolScanner


# Seed universe — big-liquid names the scanner always evaluates plus
# dynamic mentions from news.
_SEED = [
    "SPY", "QQQ", "IWM", "DIA",             # ETFs
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",    # mega-tech
    "AMD", "AVGO", "TSM", "ORCL", "CRM",    # other tech
    "JPM", "GS", "BAC", "WFC",               # banks
    "XOM", "CVX",                            # energy
    "JNJ", "UNH", "LLY", "PFE",              # healthcare
    "WMT", "HD", "COST",                     # retail
]


@dataclass
class TickerScore:
    symbol: str
    spot: Optional[float] = None
    news_mentions: int = 0
    day_change_pct: Optional[float] = None
    relative_volume: Optional[float] = None
    atm_iv: Optional[float] = None
    pc_oi_ratio: Optional[float] = None
    unusual_options: Optional[float] = None      # today_vol / open_interest
    composite_score: float = 0.0
    rationale: List[str] = field(default_factory=list)


def _screen_ticker(mp: MultiProvider, symbol: str,
                    news_counter: Dict[str, int]) -> Optional[TickerScore]:
    # Defense-in-depth: reject obviously-bad symbols before hitting the
    # network. Must be 2-5 uppercase letters, not in the blocklist.
    from src.intelligence.symbol_scanner import (
        _VALID_SYMBOL_RE, _BLOCKED_SYMBOLS,
    )
    if (not symbol or len(symbol) < 2 or len(symbol) > 5
            or not _VALID_SYMBOL_RE.match(symbol)
            or symbol in _BLOCKED_SYMBOLS):
        return None
    ts = TickerScore(symbol=symbol)
    # 1. Live quote + day change
    q = mp.latest_quote(symbol)
    if q is None:
        return None
    if q.mid is None or q.mid <= 0:
        return None
    ts.spot = q.mid
    # yfinance has fast day-change. Silence its "possibly delisted"
    # stderr spam for symbols that aren't tickers — the ticker got
    # through the blocklist somehow; we just want to drop it quietly.
    try:
        import logging as _lg
        import yfinance as yf
        _lg.getLogger("yfinance").setLevel(_lg.ERROR)
        t = yf.Ticker(symbol)
        fi = getattr(t, "fast_info", None)
        if fi is not None:
            prev = getattr(fi, "previous_close", None)
            last = getattr(fi, "last_price", None) or q.mid
            if prev and prev > 0 and last:
                ts.day_change_pct = round((last - prev) / prev, 4)
            # Relative volume: today's volume / 10-day avg
            try:
                hist = t.history(period="15d", interval="1d")
                if len(hist) >= 2:
                    today_vol = float(hist["Volume"].iloc[-1])
                    avg_10 = float(hist["Volume"].iloc[-11:-1].mean())
                    if avg_10 > 0:
                        ts.relative_volume = round(today_vol / avg_10, 2)
            except Exception:
                pass
    except Exception:
        pass
    # 2. News mentions for this ticker
    ts.news_mentions = news_counter.get(symbol, 0)
    # 3. Options chain → ATM IV, P/C OI, unusual activity
    try:
        chain = mp.option_chain(symbol)
        if chain and ts.spot:
            calls = [c for c in chain if (c.right or "").lower() == "call"]
            puts = [c for c in chain if (c.right or "").lower() == "put"]
            if calls and puts:
                atm_c = min(calls, key=lambda c: abs(c.strike - ts.spot))
                atm_p = min(puts, key=lambda c: abs(c.strike - ts.spot))
                ivs = [v for v in (atm_c.implied_vol, atm_p.implied_vol)
                       if v is not None]
                if ivs:
                    ts.atm_iv = round(sum(ivs) / len(ivs), 4)
                oi_c = sum(c.open_interest or 0 for c in calls)
                oi_p = sum(c.open_interest or 0 for c in puts)
                if oi_c > 0:
                    ts.pc_oi_ratio = round(oi_p / oi_c, 3)
                vol_total = sum(c.volume or 0 for c in chain)
                oi_total = oi_c + oi_p
                if oi_total > 0:
                    ts.unusual_options = round(vol_total / oi_total, 3)
    except Exception:
        pass

    # 4. Composite score (weighted sum; tune over time)
    score = 0.0
    if ts.news_mentions:
        score += min(3.0, ts.news_mentions / 3.0)
        ts.rationale.append(f"{ts.news_mentions}× news mentions")
    if ts.day_change_pct is not None:
        abs_chg = abs(ts.day_change_pct)
        score += min(2.5, abs_chg * 100)
        ts.rationale.append(f"{ts.day_change_pct*100:+.2f}% day")
    if ts.relative_volume is not None and ts.relative_volume > 1.0:
        score += min(2.0, (ts.relative_volume - 1.0))
        ts.rationale.append(f"rel-vol {ts.relative_volume:.2f}×")
    if ts.atm_iv is not None and ts.atm_iv > 0.30:
        score += min(2.0, (ts.atm_iv - 0.30) * 10)
        ts.rationale.append(f"IV {ts.atm_iv*100:.1f}%")
    if ts.unusual_options is not None and ts.unusual_options > 0.1:
        score += min(1.5, (ts.unusual_options - 0.1) * 10)
        ts.rationale.append(f"vol/OI {ts.unusual_options:.2f}")
    if ts.pc_oi_ratio is not None:
        if ts.pc_oi_ratio > 1.3:
            ts.rationale.append(f"P/C {ts.pc_oi_ratio:.2f} (bearish tilt)")
        elif ts.pc_oi_ratio < 0.7:
            ts.rationale.append(f"P/C {ts.pc_oi_ratio:.2f} (bullish tilt)")
    ts.composite_score = round(score, 2)
    return ts


def _count_news_mentions(mp: MultiProvider) -> Dict[str, int]:
    """Pool market-wide news across providers, extract ticker mentions.

    Uses the canonical extractor in symbol_scanner._extract_tickers_from_text
    which respects _BLOCKED_SYMBOLS — otherwise headlines like
    "US markets fell on EU concerns" produce fake tickers "US" and "EU".
    """
    from src.intelligence.symbol_scanner import _extract_tickers_from_text
    counter: Dict[str, int] = {}
    items = mp.news(None, limit=60) or []
    for item in items:
        text = f"{item.headline} {item.summary}"
        for sym in _extract_tickers_from_text(text):
            counter[sym] = counter.get(sym, 0) + 1
        # Provider-native tickers (Finnhub, Polygon emit these) — more
        # trustworthy than regex-extracted ones, so weight them 2x.
        for t in (item.tickers or []):
            t = t.upper().strip()
            if not t or len(t) < 2 or not t.isalpha():
                continue
            counter[t] = counter.get(t, 0) + 2
    return counter


def _detect_session() -> str:
    """Auto-detect pre|post|intraday|offhours from current ET clock.
    Pre: 04:00-09:30 ET.  Post: 16:00-20:00 ET.  Intraday: 09:30-16:00.
    Returns 'pre', 'post', 'intraday', or 'offhours'. Weekdays only.
    """
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now = datetime.now()
    wd = now.weekday()
    if wd >= 5:
        return "offhours"
    mins = now.hour * 60 + now.minute
    if 4 * 60 <= mins < 9 * 60 + 30:
        return "pre"
    if 9 * 60 + 30 <= mins < 16 * 60:
        return "intraday"
    if 16 * 60 <= mins < 20 * 60:
        return "post"
    return "offhours"


def _llm_commentary(scores: List[TickerScore], session: str,
                      news_counter: Dict[str, int]) -> str:
    """Ask the 70B (Groq preferred) for a session-aware narrative over
    the top setups. Returns '' on any failure (caller falls back to
    the quantitative-only message)."""
    if not scores:
        return ""
    try:
        from src.intelligence.groq_client import build_llm_client_for
        client, model = build_llm_client_for("research")
        if client is None:
            return ""
        top = scores[:8]
        top_names = [s.symbol for s in top]
        news_mentions = {
            k: v for k, v in sorted(
                news_counter.items(), key=lambda kv: -kv[1],
            )[:15]
            if k in top_names or v >= 3
        }
        snap = {
            "session": session,
            "top_candidates": [
                {
                    "symbol": s.symbol, "spot": s.spot,
                    "day_change_pct": s.day_change_pct,
                    "relative_volume": s.relative_volume,
                    "atm_iv": s.atm_iv,
                    "pc_oi_ratio": s.pc_oi_ratio,
                    "unusual_options": s.unusual_options,
                    "news_mentions": s.news_mentions,
                    "composite_score": s.composite_score,
                    "rationale_tags": s.rationale,
                } for s in top
            ],
            "news_mention_counts": news_mentions,
        }
        if session == "pre":
            framing = (
                "PRE-MARKET BRIEF (before 09:30 ET). Your job: frame the "
                "session setup, flag the 2-3 tickers with the highest "
                "confluence, call out key levels / catalysts, and name "
                "the bear case explicitly. Do NOT propose specific "
                "strikes — just the directional read and why."
            )
        elif session == "post":
            framing = (
                "POST-MARKET RECAP (after 16:00 ET). Your job: summarize "
                "what moved today, flag names with lingering momentum "
                "into tomorrow, note IV compression/expansion, and "
                "call out what BOTH bulls and bears should watch at "
                "the next open."
            )
        else:
            framing = (
                "INTRADAY SCAN. Your job: highlight the 2-3 highest-"
                "confluence names RIGHT NOW, describe what's driving "
                "them, and name the primary risk to each thesis."
            )
        prompt = (
            f"You are a disciplined options desk analyst. {framing}\n\n"
            "Cite ONLY numbers present in the SNAPSHOT below. Do NOT "
            "invent prices, levels, or headlines. Keep the response "
            "under 12 lines of plain text. Use bull/bear parity — every "
            "directional call must include what would invalidate it.\n\n"
            f"SNAPSHOT:\n{json.dumps(snap, indent=2, default=str)[:6000]}\n\n"
            "YOUR ANALYSIS:\n"
        )
        raw = client.generate(
            model=model, prompt=prompt,
            temperature=0.2, max_tokens=450,
            num_ctx=4096,
        )
        return (raw or "").strip()[:1100]
    except Exception:
        return ""


_SESSION_STYLE = {
    "pre":      ("🌅 Pre-Market Brief",
                 "Setups to watch at the open. Confluence-ranked."),
    "post":     ("🌆 Post-Market Recap",
                 "What moved today — and what to watch tomorrow."),
    "intraday": ("🔎 Intraday Scan",
                 "Highest-confluence names right now."),
    "offhours": ("🌙 Off-Hours Scan",
                 "Overnight read — ranked by composite score."),
}


def _format_discord(scores: List[TickerScore], *,
                      limit: int = 12,
                      session: str = "intraday",
                      commentary: str = "") -> str:
    title, subtitle = _SESSION_STYLE.get(session, _SESSION_STYLE["intraday"])
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        clock = now_et.strftime('%Y-%m-%d %H:%M ET')
    except Exception:
        clock = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines: List[str] = []
    lines.append(f"**{title} · {clock}**")
    lines.append(f"_{subtitle}_")
    lines.append(
        f"Scanned {len(scores)} symbols. Ranked by composite score "
        "(news + momentum + IV + vol + unusual)."
    )
    if commentary:
        lines.append("")
        lines.append("**📝 Desk note**")
        lines.append(commentary)
    lines.append("")
    lines.append("**📊 Top candidates**")
    for i, s in enumerate(scores[:limit], 1):
        spot = f"${s.spot:.2f}" if s.spot else "—"
        chg = (f"{s.day_change_pct*100:+.2f}%"
                if s.day_change_pct is not None else "—")
        lines.append(f"**{i}. {s.symbol}** · {spot} · {chg} · score {s.composite_score}")
        if s.rationale:
            lines.append(f"   · {' · '.join(s.rationale[:4])}")
    if len(scores) > limit:
        lines.append(f"\n… +{len(scores) - limit} more below threshold")
    out = "\n".join(lines)
    if len(out) > 1900:
        out = out[:1880].rstrip() + "\n… [truncated]"
    return out


# Each session routes to a Discord channel via notifier title. The
# notifier maps titles → webhooks in src/notify/base.py (see
# DISCORD_WEBHOOK_URL_<TITLE> env vars). If a dedicated webhook isn't
# mapped the fallback catch-all webhook is used — same channel, just
# labeled distinctly in the message header.
_SESSION_TITLE = {
    "pre":      "scan_premarket",
    "post":     "scan_postmarket",
    "intraday": "llm_ideas",
    "offhours": "llm_ideas",
}


@alert_on_crash("morning_scan", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-only", action="store_true",
                     help="Screen only seed universe, skip dynamic news scan.")
    ap.add_argument("--limit", type=int, default=12,
                     help="Top N tickers to report (default 12).")
    ap.add_argument("--no-discord", action="store_true")
    ap.add_argument("--session",
                     choices=["pre", "post", "intraday", "offhours", "auto"],
                     default="auto",
                     help="Which session framing to use. 'auto' picks based "
                          "on current ET clock.")
    ap.add_argument("--no-llm", action="store_true",
                     help="Skip LLM commentary (quantitative-only output).")
    args = ap.parse_args()

    session = args.session if args.session != "auto" else _detect_session()
    print(f"[scan] session={session}")

    mp = MultiProvider.from_env()
    if not mp.active_providers():
        print("[!] no data providers configured. Check .env.")
        return 2

    # Build universe: seed + news-mentioned (unless --seed-only)
    news_counter = _count_news_mentions(mp)
    universe = list(_SEED)
    if not args.seed_only:
        mentioned = sorted(
            (s for s in news_counter if s not in universe),
            key=lambda s: -news_counter[s],
        )[:15]
        universe.extend(mentioned)

    print(f"Scanning {len(universe)} tickers: {universe}")
    results: List[TickerScore] = []
    for sym in universe:
        try:
            r = _screen_ticker(mp, sym, news_counter)
            if r is not None:
                results.append(r)
        except Exception as e:                          # noqa: BLE001
            print(f"  skip {sym}: {e}")

    # Rank by composite score
    results.sort(key=lambda r: r.composite_score, reverse=True)

    commentary = ""
    if not args.no_llm:
        commentary = _llm_commentary(results, session, news_counter)

    body = _format_discord(
        results, limit=int(args.limit),
        session=session, commentary=commentary,
    )
    print(body)

    if not args.no_discord:
        meta = {
            "Session":  session,
            "Scanned":  len(results),
            "Seed":     len(_SEED),
            "Dynamic":  len(universe) - len(_SEED),
            "Top":      min(int(args.limit), len(results)),
            "LLM":      "on" if commentary else "off",
            "_footer":  f"morning_scan/{session}",
        }
        build_notifier().notify(
            body, level="info",
            title=_SESSION_TITLE.get(session, "llm_ideas"),
            meta=meta,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
