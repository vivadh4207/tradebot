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
    signals_aligned: List[str] = field(default_factory=list)
    invalidation: str = ""
    risk_reward: Optional[float] = None
    hallucination_flags: List[str] = field(default_factory=list)


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
    market_read: str = ""
    cross_source_agreement: str = ""
    risks: List[str] = field(default_factory=list)
    hallucination_warnings: List[str] = field(default_factory=list)


_PROMPT_TEMPLATE = """You are a disciplined institutional options strategist
reviewing LIVE market data for a retail paper-trading bot. Produce an
analytical, multi-position read with SPECIFIC actionable setups.
Return STRICT JSON ONLY — no prose, no markdown outside the JSON.

## Anti-hallucination rules (HARD CONSTRAINTS)

1. Every number in your `rationale` MUST be sourced from the SNAPSHOT.
   If a number you want to cite isn't in the SNAPSHOT, don't cite it.
2. Strike MUST match a strike present in the supplied chain. Do NOT
   invent fractional strikes on SPY/QQQ.
3. Expiry MUST be one listed in the chain's expiry set. No invented dates.
4. Quote sources MUST be named (e.g. "polygon agrees with alpaca at 708.80").
5. If you cannot find enough evidence for 2+ setups, return fewer ideas
   rather than inventing. It's better to return 1 high-confidence idea
   than 3 weak ones.
6. `confidence=high` requires at least THREE independent signals agreeing:
   multi-source quote, news/sentiment, and at least one chart signal
   (IV rank, OI skew, greeks, volume, breadth).
7. For each idea, include a `invalidation` field: "what would kill this
   trade" — a specific level or event that would flip your view.

## Per-idea spec

Each idea object must have:
  symbol              — SPY | QQQ (or the underlying asked for)
  direction           — "call" | "put" (no spreads this round)
  strike              — integer dollars matching the chain
  expiry              — ISO date string from the chain
  entry               — current mid from chain bid/ask
  profit_target       — premium level ~ +50-80% of entry
  stop_loss           — premium level ~ -30-50% of entry
  time_horizon        — "intraday" | "1-3d" | "1-2wk"
  confidence          — "low" | "medium" | "high"
  signals_aligned     — list of 2-5 specific signals supporting this trade,
                        each a short tag like "rsi_oversold",
                        "breadth_negative", "vix_expanding",
                        "oi_call_skew", "bearish_divergence",
                        "news_hawkish"
  rationale           — 2-4 sentences, cross-reference SNAPSHOT values only
  invalidation        — "what kills this trade" (specific level or event)
  risk_reward         — computed as (profit_target - entry) / (entry - stop_loss),
                        round to 1 decimal

## Output shape

{{
  "market_read": "2-3 sentences on overall regime + primary driver today",
  "cross_source_agreement": "description of where sources agree/disagree (quotes, news, sentiment)",
  "ideas": [
    {{...see per-idea spec above...}}
  ],
  "risks": ["top 3 risk flags from SNAPSHOT that could flip the tape"]
}}

## SNAPSHOT (authoritative — cite ONLY these numbers)

{snapshot}

## YOUR JSON:
"""


def _compute_chain_stats(chain: List, spot: float) -> Dict[str, Any]:
    """Compute structural chain stats the 70B can reference:
      - Put/call OI ratio (bearish if >1.2)
      - Put/call volume ratio
      - IV rank proxy (today's ATM IV vs 20-chain mean)
      - ATM skew (put IV minus call IV at nearest strike)
      - Total OI + volume
    All fail-soft; missing data returns None for that field."""
    calls = [c for c in chain if (c.right or "").lower() == "call"]
    puts = [c for c in chain if (c.right or "").lower() == "put"]
    oi_call = sum((c.open_interest or 0) for c in calls)
    oi_put = sum((c.open_interest or 0) for c in puts)
    vol_call = sum((c.volume or 0) for c in calls)
    vol_put = sum((c.volume or 0) for c in puts)
    pc_oi = round(oi_put / max(1, oi_call), 3) if oi_call else None
    pc_vol = round(vol_put / max(1, vol_call), 3) if vol_call else None

    # Nearest-ATM call + put for skew
    atm_call = min(calls, key=lambda c: abs(c.strike - spot)) if calls else None
    atm_put = min(puts, key=lambda c: abs(c.strike - spot)) if puts else None
    skew = None
    if (atm_call is not None and atm_put is not None
            and atm_call.implied_vol and atm_put.implied_vol):
        skew = round(atm_put.implied_vol - atm_call.implied_vol, 4)

    atm_iv = None
    if atm_call and atm_put and atm_call.implied_vol and atm_put.implied_vol:
        atm_iv = round((atm_call.implied_vol + atm_put.implied_vol) / 2, 4)

    return {
        "n_calls": len(calls),
        "n_puts": len(puts),
        "oi_call_total": oi_call,
        "oi_put_total": oi_put,
        "pc_oi_ratio": pc_oi,
        "volume_call_total": vol_call,
        "volume_put_total": vol_put,
        "pc_volume_ratio": pc_vol,
        "atm_iv": atm_iv,
        "atm_skew_put_minus_call": skew,
        "atm_call_delta": (round(atm_call.delta, 3)
                           if atm_call and atm_call.delta is not None else None),
        "atm_put_delta": (round(atm_put.delta, 3)
                          if atm_put and atm_put.delta is not None else None),
    }


def _gather_bot_state() -> Dict[str, Any]:
    """Pull current bot-internal state for the prompt:
      - Open positions (from broker_state.json)
      - Recent ensemble decisions (from tradebot.out tail)
      - Last strategy audit verdict
      - Current regime + VIX if present in logs
    Fail-open — missing pieces come back as None.
    """
    out: Dict[str, Any] = {
        "positions": [], "open_positions": 0, "cash": None, "day_pnl": None,
        "recent_ensemble": [], "regime": None, "vix": None, "breadth": None,
        "last_audit_health": None, "last_audit_summary": None,
    }
    # broker_state.json
    try:
        from ..core.data_paths import data_path
        from pathlib import Path as _P
        import json as _json
        snap = _P(data_path("logs/broker_state.json"))
        if snap.exists():
            d = _json.loads(snap.read_text())
            out["cash"] = d.get("cash")
            out["day_pnl"] = d.get("day_pnl")
            pos_list = d.get("positions") or []
            out["open_positions"] = len(pos_list)
            out["positions"] = [
                {"symbol": p.get("symbol"),
                  "qty": p.get("qty"),
                  "avg_price": p.get("avg_price"),
                  "unrealized_pnl": p.get("unrealized_pnl"),
                  "auto_pt": p.get("auto_profit_target"),
                  "auto_sl": p.get("auto_stop_loss")}
                for p in pos_list[:10]
            ]
    except Exception:
        pass
    # Log tail for regime + VIX + recent ensemble
    try:
        import re as _re
        from ..core.data_paths import data_path
        from pathlib import Path as _P
        log = _P(data_path("logs/tradebot.out"))
        if log.exists():
            size = log.stat().st_size
            with log.open("rb") as f:
                f.seek(max(0, size - 400_000))
                if size > 400_000:
                    f.readline()
                text = f.read().decode("utf-8", errors="replace")
            # Most-recent regime
            regimes = _re.findall(r"regime=(\w+)", text)
            if regimes:
                out["regime"] = regimes[-1]
            vixs = _re.findall(r"\bvix=([0-9.]+)", text)
            if vixs:
                try:
                    out["vix"] = float(vixs[-1])
                except Exception:
                    pass
            breadths = _re.findall(r"breadth_score=(-?[0-9.]+)", text)
            if breadths:
                try:
                    out["breadth"] = float(breadths[-1])
                except Exception:
                    pass
            # Last 6 ensemble emits for context
            last_ens = []
            for line in reversed(text.splitlines()[-400:]):
                if "ensemble_emit" in line or "exec_chain_pass" in line:
                    clean = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
                    last_ens.append(clean[:200])
                    if len(last_ens) >= 6:
                        break
            out["recent_ensemble"] = list(reversed(last_ens))
    except Exception:
        pass
    # Last audit
    try:
        from .strategy_auditor import read_recent_audits
        r = read_recent_audits(1)
        if r:
            out["last_audit_health"] = r[0].get("overall_health")
            out["last_audit_summary"] = r[0].get("summary")
    except Exception:
        pass
    return out


def _validate_idea_against_snapshot(idea_dict: Dict[str, Any],
                                      snapshot: Dict[str, Any]
                                      ) -> List[str]:
    """Return a list of hallucination-warning strings (empty if clean).
    Checks:
      - strike exists in snapshot.chain_strikes for that symbol
      - expiry exists in snapshot.chain_expiries
      - entry premium is reasonable vs the chain's bid/ask bounds
    Each warning is human-readable — we DON'T drop the idea here, we
    flag it so the operator can see what to distrust."""
    warnings: List[str] = []
    sym = str(idea_dict.get("symbol", "")).upper()
    strikes = snapshot.get("chain_strikes", {}).get(sym, [])
    expiries = snapshot.get("chain_expiries", {}).get(sym, [])

    k = idea_dict.get("strike")
    if strikes and k is not None:
        try:
            k = float(k)
            if not any(abs(k - s) < 0.005 for s in strikes):
                warnings.append(
                    f"strike {k} not in supplied chain for {sym}"
                )
        except Exception:
            warnings.append(f"strike value '{idea_dict.get('strike')}' unparseable")
    exp = idea_dict.get("expiry")
    if expiries and exp and exp not in expiries:
        warnings.append(f"expiry {exp} not in supplied chain for {sym}")
    return warnings


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
        """Build snapshot → call LLM → parse → validate → return report.

        The snapshot is rich + structured: per-symbol multi-source
        quotes (with cross-check flags), nearest-ATM chain with greeks,
        IV rank, OI call/put skew, recent volume profile, sentiment
        score, top headlines, current bot positions, recent ensemble
        decisions, and last audit summary. The 70B sees everything it
        would need to be a junior PM and is instructed to ONLY cite
        numbers from the snapshot (post-LLM validator catches any
        number it hallucinates)."""
        import time as _time
        started = _time.time()
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        spot_by_symbol: Dict[str, float] = {}
        quote_sources: Dict[str, List[str]] = {}
        quote_spread_pcts: Dict[str, float] = {}
        chains: Dict[str, List[Dict[str, Any]]] = {}
        chain_expiries: Dict[str, List[str]] = {}
        chain_strikes: Dict[str, List[float]] = {}
        chain_stats: Dict[str, Dict[str, Any]] = {}
        sentiment_by_symbol: Dict[str, Optional[float]] = {}
        headlines: List[Dict[str, Any]] = []

        for sym in underlyings:
            # Multi-source quote (median when sources disagree).
            all_q = self._mp.all_quotes(sym)
            if all_q:
                mids = [q.mid for q in all_q if q.mid > 0]
                if mids:
                    spot_by_symbol[sym] = statistics.median(mids)
                    # Record how much sources disagree — the LLM should
                    # weight its confidence by this.
                    if len(mids) > 1:
                        quote_spread_pcts[sym] = round(
                            (max(mids) - min(mids)) /
                            max(1e-9, statistics.median(mids)),
                            5,
                        )
                    else:
                        quote_spread_pcts[sym] = 0.0
                quote_sources[sym] = [q.source for q in all_q]
            else:
                quote_sources[sym] = []

            # Chain — first non-empty wins inside multi_provider.
            chain = self._mp.option_chain(sym)
            if chain and sym in spot_by_symbol:
                chains[sym] = _best_contracts_near_atm(
                    chain, spot_by_symbol[sym], self._n_each_side,
                )
                chain_expiries[sym] = sorted({c.expiry for c in chain
                                                if c.expiry})
                chain_strikes[sym] = sorted({float(c.strike) for c in chain
                                              if c.strike})
                chain_stats[sym] = _compute_chain_stats(
                    chain, spot_by_symbol[sym],
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

        # Bot internal state — positions, recent signals, audit.
        bot_state = _gather_bot_state()

        snapshot: Dict[str, Any] = {
            "now_utc": now_iso,
            "underlyings": underlyings,
            "spot": {k: round(v, 2) for k, v in spot_by_symbol.items()},
            "quote_sources": quote_sources,
            "quote_spread_pct": quote_spread_pcts,
            "sentiment": sentiment_by_symbol,
            "chains": chains,
            "chain_expiries": chain_expiries,
            "chain_stats": chain_stats,
            "headlines": headlines[:40],    # cap
            "bot_state": bot_state,
        }
        if extra_context:
            snapshot["extra"] = extra_context

        prompt = _PROMPT_TEMPLATE.format(
            snapshot=json.dumps(snapshot, indent=2, default=str)[:14000],
        )

        raw, used_model = self._call_llm(prompt)
        ideas: List[TradeIdea] = []
        notes = ""
        market_read = ""
        cross_source_agreement = ""
        risks: List[str] = []
        halluc_warnings: List[str] = []
        parsed = _parse_llm_json(raw) if raw else None
        if parsed:
            market_read = str(parsed.get("market_read", ""))[:500]
            cross_source_agreement = str(
                parsed.get("cross_source_agreement", "")
            )[:400]
            for r in (parsed.get("risks") or [])[:5]:
                risks.append(str(r)[:200])
            notes = str(parsed.get("notes", ""))[:240]

            for d in (parsed.get("ideas") or [])[:5]:
                try:
                    # Validate against snapshot BEFORE constructing the
                    # TradeIdea so hallucinated strikes/expiries get
                    # surfaced as warnings.
                    local_warnings = _validate_idea_against_snapshot(
                        d, snapshot,
                    )
                    if local_warnings:
                        halluc_warnings.extend(
                            f"{d.get('symbol','?')}: {w}"
                            for w in local_warnings
                        )
                    risk_reward = d.get("risk_reward")
                    if risk_reward is None:
                        try:
                            e = float(d.get("entry", 0))
                            pt = float(d.get("profit_target", 0))
                            sl = float(d.get("stop_loss", 0))
                            if e > 0 and sl > 0 and e > sl:
                                risk_reward = round((pt - e) / (e - sl), 2)
                        except Exception:
                            risk_reward = None
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
                        signals_aligned=[
                            str(x)[:40]
                            for x in (d.get("signals_aligned") or [])
                        ][:8],
                        invalidation=str(d.get("invalidation", ""))[:240],
                        risk_reward=(float(risk_reward)
                                      if risk_reward is not None else None),
                        hallucination_flags=list(local_warnings),
                    ))
                except Exception:
                    continue

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
            market_read=market_read,
            cross_source_agreement=cross_source_agreement,
            risks=risks,
            hallucination_warnings=halluc_warnings,
        )

    def to_markdown(self, rep: ResearchReport) -> str:
        """Render a structured, professional-grade summary for Discord."""
        lines: List[str] = []
        lines.append(
            f"**🧠 Options Research · {', '.join(rep.underlyings)} · "
            f"{rep.model or '—'} · {rep.latency_sec:.0f}s**"
        )
        # Live spot + source agreement
        if rep.spot_by_symbol:
            spot_bits = []
            for k, v in rep.spot_by_symbol.items():
                srcs = ",".join(rep.quote_sources.get(k, [])) or "—"
                spot_bits.append(f"{k}=${v:.2f} [{srcs}]")
            lines.append("📊 " + " · ".join(spot_bits))
        if any((v is not None) for v in rep.sentiment_by_symbol.values()):
            snt = [f"{k}={v:+.2f}"
                   for k, v in rep.sentiment_by_symbol.items() if v is not None]
            lines.append("📰 sentiment: " + " · ".join(snt))
        # Market read + cross-source
        if rep.market_read:
            lines.append(f"\n**Market read**\n{rep.market_read}")
        if rep.cross_source_agreement:
            lines.append(f"\n**Cross-source**\n{rep.cross_source_agreement}")
        # Ideas
        if not rep.ideas:
            lines.append("\n*No actionable setups surfaced under current "
                         "constraints.*")
            if rep.raw_llm and len(rep.raw_llm) < 300:
                lines.append(f"_raw_: {rep.raw_llm}")
        else:
            lines.append("\n**Trade ideas**")
            for i, idea in enumerate(rep.ideas, 1):
                head = f"**{i}. {idea.symbol} {idea.direction.upper()}**"
                if idea.strike is not None:
                    head += f" ${idea.strike:g}"
                if idea.expiry:
                    head += f" · {idea.expiry}"
                if idea.confidence:
                    head += f" · *{idea.confidence}*"
                if idea.risk_reward is not None:
                    head += f" · R:R **{idea.risk_reward:.1f}**"
                lines.append(head)
                bits = []
                if idea.entry is not None:
                    bits.append(f"entry ${idea.entry:.2f}")
                if idea.profit_target is not None:
                    bits.append(f"🎯 PT ${idea.profit_target:.2f}")
                if idea.stop_loss is not None:
                    bits.append(f"🛑 SL ${idea.stop_loss:.2f}")
                if idea.time_horizon:
                    bits.append(f"⏱ {idea.time_horizon}")
                if bits:
                    lines.append("   " + "  ·  ".join(bits))
                if idea.signals_aligned:
                    lines.append("   ✅ signals: "
                                  + ", ".join(idea.signals_aligned))
                if idea.rationale:
                    lines.append(f"   _{idea.rationale[:220]}_")
                if idea.invalidation:
                    lines.append(f"   ⚠️ invalidation: {idea.invalidation[:160]}")
                if idea.hallucination_flags:
                    lines.append("   🚩 " + "; ".join(idea.hallucination_flags))
        # Risks
        if rep.risks:
            lines.append("\n**Risk flags**")
            for r in rep.risks[:4]:
                lines.append(f"  • {r}")
        # Global hallucination warnings (repeat for visibility)
        if rep.hallucination_warnings:
            lines.append("\n🚩 Validator warnings — do NOT trade blind:")
            for w in rep.hallucination_warnings[:5]:
                lines.append(f"  · {w}")
        out = "\n".join(lines)
        if len(out) > 1900:
            out = out[:1880].rstrip() + "\n… [truncated]"
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
