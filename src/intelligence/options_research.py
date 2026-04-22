"""OptionsResearchAgent — pulls live option chain + quote + news across
all configured data providers, feeds to 70B (or 8B fallback), returns a
ranked list of actionable setups.

Outputs are ADVISORY ONLY. The rules engine / ensemble makes trades.
This agent posts ideas to Discord so the operator can:
  1. See what the LLM thinks is interesting
  2. Decide whether to toggle specific signal weights
  3. Cross-check against the bot's own decisions

Invocation paths:
  - `!research SPY` in Discord → one-shot for that symbol
  - Systemd/launchd timer → every 30 min during session, posts to
    DISCORD_WEBHOOK_URL_IDEAS (or catch-all) with title='llm_ideas'
  - Nightly macro sweep uses this agent's chain data too

Not intended to be fast: a full run can take 30-120s on 70B.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


_log = logging.getLogger(__name__)


@dataclass
class TradeIdea:
    """One actionable idea the LLM surfaced. Kept loose so different
    prompts can return different shapes and still serialize cleanly."""
    symbol: str                           # SPY, QQQ, or underlying
    direction: str                        # "call" | "put" | "spread"
    strike: Optional[float] = None
    expiry: Optional[str] = None          # ISO date
    rationale: str = ""
    entry: Optional[float] = None
    profit_target: Optional[float] = None
    stop_loss: Optional[float] = None
    time_horizon: str = ""                # "intraday" | "1-3d" | "1-2wk"
    confidence: str = "medium"            # low | medium | high


@dataclass
class ResearchReport:
    ts: str
    underlyings: List[str]
    spot_by_symbol: Dict[str, float]
    quote_sources: Dict[str, List[str]]   # symbol → [provider names]
    n_headlines: int
    sentiment_by_symbol: Dict[str, Optional[float]]
    ideas: List[TradeIdea] = field(default_factory=list)
    raw_llm: str = ""
    model: str = ""
    latency_sec: float = 0.0
    notes: str = ""


_PROMPT_TEMPLATE = """You are an options strategist reviewing live data
for a retail paper-trading bot. Propose 2-3 actionable setups for the
next 1-3 trading days. Return STRICT JSON only — no prose, no
markdown outside the JSON.

Rules for each idea:
  - Direction is "call" (bullish), "put" (bearish), or "spread" (defined risk).
  - Strike must be a real listed strike from the supplied chain (integer
    dollars for SPY/QQQ). Do not invent fractional strikes.
  - Expiry must be one of the supplied expirations.
  - profit_target and stop_loss are PREMIUM levels (dollars), not
    percentages. Set profit_target at roughly +40-80% of entry and
    stop_loss at -30-50% of entry.
  - time_horizon: "intraday" | "1-3d" | "1-2wk".
  - confidence: "low" | "medium" | "high". Use "high" ONLY when
    multiple independent signals agree.

Rationale must reference:
  - At least one specific number from the SNAPSHOT (spot, IV, OI, delta)
  - A specific news headline OR a specific chart read from the chain
  - Why the chosen strike + expiry over nearby alternatives

Output JSON shape:
  {{"ideas": [
      {{"symbol":"SPY","direction":"put","strike":705,"expiry":"2026-05-02",
        "entry":3.40,"profit_target":5.00,"stop_loss":2.00,
        "time_horizon":"1-3d","confidence":"medium",
        "rationale":"..."}},
      ...
    ],
    "notes": "one-sentence overall read"}}

SNAPSHOT:
{snapshot}

YOUR JSON:
"""


def _best_contracts_near_atm(chain: List, spot: float,
                               n_each_side: int = 4) -> List[Dict[str, Any]]:
    """Return the n_each_side nearest-ATM calls AND puts, compact rep,
    preserving bid/ask/OI/IV/delta. Keeps prompt size bounded."""
    calls = [c for c in chain if (c.right or "").lower() == "call"]
    puts = [c for c in chain if (c.right or "").lower() == "put"]
    calls.sort(key=lambda c: abs(c.strike - spot))
    puts.sort(key=lambda c: abs(c.strike - spot))
    picked = list(calls[:n_each_side]) + list(puts[:n_each_side])
    out: List[Dict[str, Any]] = []
    for c in picked:
        out.append({
            "symbol": c.symbol,
            "strike": c.strike,
            "expiry": c.expiry,
            "right": c.right,
            "bid": c.bid, "ask": c.ask, "last": c.last,
            "volume": c.volume,
            "open_interest": c.open_interest,
            "iv": c.implied_vol,
            "delta": c.delta, "gamma": c.gamma,
            "theta": c.theta, "vega": c.vega,
            "source": c.source,
        })
    return out


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


class OptionsResearchAgent:
    """Pulls live cross-provider data, hands to an LLM, returns
    structured trade ideas."""

    def __init__(
        self,
        multi_provider,                       # src.data.multi_provider.MultiProvider
        *,
        model_name: Optional[str] = None,
        max_tokens: int = 700,
        timeout_sec: float = 240.0,
        n_strikes_each_side: int = 4,
    ):
        self._mp = multi_provider
        self._model = model_name
        self._max_tokens = int(max_tokens)
        self._timeout = float(timeout_sec)
        self._n_each_side = int(n_strikes_each_side)

    # ------------------------------------------------ public

    def run(self, underlyings: List[str],
             extra_context: Optional[Dict[str, Any]] = None,
             ) -> ResearchReport:
        """Build snapshot → call LLM → parse → return ResearchReport."""
        import time as _time
        started = _time.time()
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        spot_by_symbol: Dict[str, float] = {}
        quote_sources: Dict[str, List[str]] = {}
        chains: Dict[str, List[Dict[str, Any]]] = {}
        sentiment_by_symbol: Dict[str, Optional[float]] = {}
        headlines: List[Dict[str, Any]] = []

        for sym in underlyings:
            # Multi-source quote (median when sources disagree).
            all_q = self._mp.all_quotes(sym)
            if all_q:
                spot_by_symbol[sym] = statistics.median(
                    [q.mid for q in all_q if q.mid > 0]
                ) if all_q else 0.0
                quote_sources[sym] = [q.source for q in all_q]
            else:
                quote_sources[sym] = []
            # Chain — first non-empty wins inside multi_provider.
            chain = self._mp.option_chain(sym)
            if chain and sym in spot_by_symbol:
                chains[sym] = _best_contracts_near_atm(
                    chain, spot_by_symbol[sym], self._n_each_side,
                )
            # Sentiment + news
            sentiment_by_symbol[sym] = self._mp.news_sentiment(sym)
            for item in (self._mp.news(sym, limit=8) or []):
                headlines.append({
                    "ts": item.ts,
                    "symbol": sym,
                    "headline": item.headline,
                    "summary": item.summary[:180],
                    "source": item.source,
                })

        snapshot: Dict[str, Any] = {
            "now_utc": now_iso,
            "underlyings": underlyings,
            "spot": {k: round(v, 2) for k, v in spot_by_symbol.items()},
            "quote_sources": quote_sources,
            "sentiment": sentiment_by_symbol,
            "chains": chains,
            "headlines": headlines[:40],    # cap
        }
        if extra_context:
            snapshot["extra"] = extra_context

        prompt = _PROMPT_TEMPLATE.format(
            snapshot=json.dumps(snapshot, indent=2, default=str)[:14000],
        )

        raw, used_model = self._call_llm(prompt)
        ideas: List[TradeIdea] = []
        notes = ""
        parsed = _parse_llm_json(raw) if raw else None
        if parsed:
            for d in (parsed.get("ideas") or [])[:5]:
                try:
                    ideas.append(TradeIdea(
                        symbol=str(d.get("symbol", "")).upper(),
                        direction=str(d.get("direction", "")).lower(),
                        strike=(float(d["strike"])
                                 if d.get("strike") is not None else None),
                        expiry=str(d.get("expiry") or "") or None,
                        rationale=str(d.get("rationale", ""))[:500],
                        entry=(float(d["entry"])
                                if d.get("entry") is not None else None),
                        profit_target=(float(d["profit_target"])
                                        if d.get("profit_target") is not None else None),
                        stop_loss=(float(d["stop_loss"])
                                    if d.get("stop_loss") is not None else None),
                        time_horizon=str(d.get("time_horizon", "") or ""),
                        confidence=str(d.get("confidence", "medium"))[:10],
                    ))
                except Exception:
                    continue
            notes = str(parsed.get("notes", ""))[:240]

        return ResearchReport(
            ts=now_iso,
            underlyings=underlyings,
            spot_by_symbol=spot_by_symbol,
            quote_sources=quote_sources,
            n_headlines=len(headlines),
            sentiment_by_symbol=sentiment_by_symbol,
            ideas=ideas,
            raw_llm=(raw or "")[:1500],
            model=used_model,
            latency_sec=_time.time() - started,
            notes=notes,
        )

    def to_markdown(self, rep: ResearchReport) -> str:
        """Render a Discord-sized summary."""
        lines: List[str] = []
        lines.append(
            f"**Options Research · {', '.join(rep.underlyings)} · "
            f"model={rep.model} · {rep.latency_sec:.0f}s**"
        )
        if rep.spot_by_symbol:
            spot_bits = [f"{k}={v:.2f}" for k, v in rep.spot_by_symbol.items()]
            lines.append("· spot: " + " / ".join(spot_bits))
        if any(rep.sentiment_by_symbol.values()):
            snt = [f"{k}={v:+.2f}"
                   for k, v in rep.sentiment_by_symbol.items() if v is not None]
            if snt:
                lines.append("· sentiment: " + " / ".join(snt))
        if rep.notes:
            lines.append(f"· read: {rep.notes}")
        if not rep.ideas:
            lines.append("· no actionable ideas surfaced.")
            if rep.raw_llm and len(rep.raw_llm) < 300:
                lines.append(f"  raw: {rep.raw_llm}")
            return "\n".join(lines)
        for i, idea in enumerate(rep.ideas, 1):
            bits = [f"**{i}. {idea.symbol} {idea.direction.upper()}**"]
            if idea.strike is not None:
                bits.append(f"${idea.strike:g}")
            if idea.expiry:
                bits.append(idea.expiry)
            if idea.confidence:
                bits.append(f"({idea.confidence})")
            lines.append(" ".join(bits))
            pt = f"PT={idea.profit_target}" if idea.profit_target else ""
            sl = f"SL={idea.stop_loss}" if idea.stop_loss else ""
            entry = f"entry={idea.entry}" if idea.entry else ""
            tag = f"horizon={idea.time_horizon}" if idea.time_horizon else ""
            tail = "  ".join(x for x in [entry, pt, sl, tag] if x)
            if tail:
                lines.append(f"   {tail}")
            if idea.rationale:
                lines.append(f"   _{idea.rationale[:240]}_")
        out = "\n".join(lines)
        if len(out) > 1700:
            out = out[:1690].rstrip() + "\n… [truncated]"
        return out

    # ------------------------------------------------ llm

    def _call_llm(self, prompt: str) -> tuple:
        """Try 70B first (more thoughtful), fall back to 8B if 70B not
        configured or fails. Returns (raw_text, model_used)."""
        import os as _os
        model_candidates: List[str] = []
        if self._model:
            model_candidates.append(self._model)
        else:
            audit = (_os.getenv("LLM_AUDITOR_MODEL", "").strip()
                     or "llama3.1:70b")
            brain = (_os.getenv("LLM_BRAIN_MODEL", "").strip()
                     or "llama3.1:8b")
            model_candidates = [audit, brain]
        try:
            from .ollama_client import build_ollama_client
            client = build_ollama_client()
            client.cfg.timeout_sec = self._timeout
            if not client.ping():
                _log.warning("options_research_ollama_unreachable")
                return "", ""
            for model in model_candidates:
                try:
                    raw = client.generate(
                        model=model,
                        prompt=prompt,
                        temperature=0.2,
                        max_tokens=self._max_tokens,
                        num_ctx=6144,
                        stop=["\n\nSNAPSHOT:", "\n\nYOUR JSON:"],
                    )
                    if raw and raw.strip():
                        return raw, model
                except Exception as e:                  # noqa: BLE001
                    _log.info("options_research_model_failed model=%s err=%s",
                              model, e)
                    continue
        except Exception as e:                          # noqa: BLE001
            _log.warning("options_research_llm_err err=%s", e)
        return "", ""
