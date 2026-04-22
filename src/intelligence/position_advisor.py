"""Position-fade advisor — LLM + chart review for open positions.

When a position's unrealized pnl fades from peak (or hits critical
moments like "was +5% now flat"), this module:

  1. Gathers the position snapshot, recent bars, option quote, news
  2. Asks 70B (Groq) for a fast recommendation: close / hold / trim
  3. Posts to Discord with action buttons so operator can execute
     the LLM's advice in one click — OR override

The automatic fast_exit layers still run in parallel. This advisor is
a SECOND opinion + a manual override path. Operator requested:
  "automated should also be there but manual is a plus."

Fires when:
  - Peak pnl >= +3% AND current pnl has retraced >= 40% from peak
  - OR position has been open > 15 minutes and is flat/negative
  - Rate-limited per-position (one advisory per 10 min)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional


_log = logging.getLogger(__name__)
_lock = RLock()


@dataclass
class FadeAdvisory:
    symbol: str
    direction: str                 # 'call' | 'put'
    strike: Optional[float]
    expiry: Optional[str]
    qty: int
    peak_pnl_pct: float
    current_pnl_pct: float
    entry_price: float
    current_price: float
    recommendation: str = ""        # 'close' | 'hold' | 'trim'
    confidence: str = "medium"
    rationale: str = ""
    key_levels: List[str] = field(default_factory=list)
    urgency: str = "normal"         # 'urgent' | 'normal' | 'low'
    bars_summary: str = ""          # one-line chart read
    model: str = ""
    ts: float = 0.0
    # ---- Rich LLM insights (new) ----
    chart_signals: List[str] = field(default_factory=list)
    # e.g. "lower-high pattern on last 3 bars", "volume spike against us",
    #      "broke session VWAP at 679.80", "RSI(14) rolled over from 68→52"
    risk_reward_remaining: str = ""
    # e.g. "Upside capped at ~$1.55 resistance ($0.08 more). Downside
    #       open to $0.40 support ($0.57 loss)."
    alternative_actions: List[str] = field(default_factory=list)
    # e.g. ["Trim 50% and set stop at entry ($1.45)",
    #        "Roll to further-dated contract for breathing room"]
    fade_trigger: str = ""
    # e.g. "peak_retrace_30pct" | "green_to_red" | "stale_flat"
    time_context: str = ""
    # e.g. "Position held 42 min, 3h 17m to expiry — theta accelerating"


class PositionAdvisor:
    """Per-position cadence-throttled fade advisor."""

    def __init__(self, *, cooldown_sec: int = 600):
        self._cooldown = int(cooldown_sec)
        self._last_fire: Dict[str, float] = {}
        # Peak pnl tracking for decision logic — the fast_exit engine
        # stores peak_pnl_pct on the Position itself; we read that.

    def _peak(self, pos) -> Optional[float]:
        return getattr(pos, "peak_pnl_pct", None)

    def _should_fire(self, pos, current_pnl_pct: float) -> bool:
        """Decide whether this position qualifies for an LLM advisory.
        Operator: 'position went green then red — we waited.' Lowered
        threshold so we fire EARLIER, while still in profit."""
        peak = self._peak(pos) or 0.0
        # A: peak >= +3% and retraced >= 25% from peak (was 40%)
        if peak >= 0.03:
            give_back = (peak - current_pnl_pct) / max(peak, 1e-9)
            if give_back >= 0.25:
                return True
        # B: green-to-red — was positive, now negative. ALWAYS fire
        # regardless of peak size; this is exactly the scenario the
        # operator called out.
        if peak > 0 and current_pnl_pct < 0:
            return True
        # C: open > 10 min (was 15) and flat/negative (stuck trade)
        try:
            held_sec = time.time() - float(pos.entry_ts)
        except Exception:
            held_sec = 0
        if held_sec > 600 and current_pnl_pct <= 0.005:
            return True
        return False

    def _rate_limited(self, key: str) -> bool:
        now = time.time()
        last = self._last_fire.get(key, 0.0)
        if now - last < self._cooldown:
            return True
        self._last_fire[key] = now
        return False

    def maybe_advise(self, pos, current_price: float,
                      current_pnl_pct: float, bars=None
                      ) -> Optional[FadeAdvisory]:
        """If conditions fire, return an advisory (posts to Discord
        is caller's job). Returns None when gated or LLM fails."""
        if not self._should_fire(pos, current_pnl_pct):
            return None
        key = f"{pos.symbol}:{id(pos)}"
        if self._rate_limited(key):
            return None
        return self._build(pos, current_price, current_pnl_pct, bars)

    def _bars_summary(self, bars) -> str:
        if not bars or len(bars) < 5:
            return "no bars available"
        recent = bars[-10:]
        highs = [b.high for b in recent]
        lows = [b.low for b in recent]
        closes = [b.close for b in recent]
        vols = [b.volume or 0 for b in recent]
        trend = "up" if closes[-1] > closes[0] else (
            "down" if closes[-1] < closes[0] else "flat"
        )
        ret = (closes[-1] - closes[0]) / max(closes[0], 1e-9) * 100
        avg_v = sum(vols[:-3]) / max(1, len(vols[:-3]))
        recent_v = sum(vols[-3:]) / max(1, 3)
        vol_trend = ("rising" if recent_v > 1.2 * avg_v else
                       "falling" if recent_v < 0.8 * avg_v else "flat")
        return (
            f"10-bar {trend} ({ret:+.2f}%), "
            f"high={max(highs):.2f} low={min(lows):.2f}, "
            f"vol {vol_trend}"
        )

    def _build(self, pos, current_price: float,
                current_pnl_pct: float, bars) -> Optional[FadeAdvisory]:
        peak = self._peak(pos) or current_pnl_pct
        adv = FadeAdvisory(
            symbol=pos.symbol,
            direction=(pos.right.value if pos.right else "long"),
            strike=getattr(pos, "strike", None),
            expiry=(str(pos.expiry) if getattr(pos, "expiry", None)
                      else None),
            qty=int(abs(pos.qty)),
            peak_pnl_pct=float(peak),
            current_pnl_pct=float(current_pnl_pct),
            entry_price=float(pos.avg_price or 0),
            current_price=float(current_price),
            bars_summary=self._bars_summary(bars),
            ts=time.time(),
        )

        # Fallback rule-based recommendation — this is what gets
        # ACTED ON if the LLM times out. Tuned aggressive per operator
        # feedback ("we waited when it went green to red").
        give_back = (peak - current_pnl_pct) / max(peak, 1e-9) if peak > 0 else 0
        if peak > 0 and current_pnl_pct < 0:
            # GREEN-TO-RED: immediate close, no waiting.
            adv.recommendation = "close"
            adv.urgency = "urgent"
            adv.rationale = (
                f"Position was +{peak*100:.1f}% peak, NOW NEGATIVE at "
                f"{current_pnl_pct*100:+.1f}%. Cut to preserve capital."
            )
        elif give_back >= 0.50:
            # Gave back half or more — close urgently.
            adv.recommendation = "close"
            adv.urgency = "urgent"
            adv.rationale = (
                f"Gave back {give_back*100:.0f}% of +{peak*100:.1f}% peak. "
                f"Take the remaining {current_pnl_pct*100:+.1f}% NOW."
            )
        elif give_back >= 0.25:
            # Moderate fade — trim half, trail the rest.
            adv.recommendation = "trim"
            adv.urgency = "urgent"   # still urgent so it auto-executes
            adv.rationale = (
                f"Gave back {give_back*100:.0f}% from +{peak*100:.1f}% peak. "
                "Trim half, trail the rest with a tight stop."
            )
        else:
            adv.recommendation = "hold"
            adv.rationale = "Still in decent profit; watch next 5 bars."
        adv.confidence = "medium"

        # LLM overlay — if Groq/Ollama available, get a richer opinion.
        try:
            from .groq_client import build_llm_client_for
            client, model = build_llm_client_for("research")
            if client is None:
                return adv
            prompt = _build_prompt(adv, bars)
            raw = client.generate(
                model=model, prompt=prompt,
                temperature=0.1, max_tokens=260,
                num_ctx=3072,
            )
            if raw and raw.strip():
                parsed = _parse_llm(raw)
                if parsed:
                    rec = str(parsed.get("recommendation", "")).lower().strip()
                    if rec in ("close", "hold", "trim"):
                        adv.recommendation = rec
                    if parsed.get("urgency") in ("urgent", "normal", "low"):
                        adv.urgency = parsed["urgency"]
                    if parsed.get("confidence"):
                        adv.confidence = str(parsed["confidence"])[:10]
                    if parsed.get("rationale"):
                        adv.rationale = str(parsed["rationale"])[:500]
                    if parsed.get("key_levels"):
                        adv.key_levels = [str(x)[:80]
                                             for x in parsed["key_levels"][:4]]
                    if parsed.get("chart_signals"):
                        adv.chart_signals = [
                            str(x)[:100]
                            for x in parsed["chart_signals"][:4]
                        ]
                    if parsed.get("risk_reward_remaining"):
                        adv.risk_reward_remaining = str(
                            parsed["risk_reward_remaining"]
                        )[:300]
                    if parsed.get("alternative_actions"):
                        adv.alternative_actions = [
                            str(x)[:120]
                            for x in parsed["alternative_actions"][:3]
                        ]
                    if parsed.get("time_context"):
                        adv.time_context = str(parsed["time_context"])[:200]
                    adv.model = model
        except Exception as e:                              # noqa: BLE001
            _log.info("position_advisor_llm_err err=%s", e)
        return adv


# ---- stateless helpers -------------------------------------------


def _build_prompt(adv: FadeAdvisory, bars) -> str:
    # Rich bar snapshot with computed signals the LLM can cite.
    recent_bars: List[Dict[str, Any]] = []
    if bars:
        for b in bars[-15:]:
            recent_bars.append({
                "o": round(b.open, 2), "h": round(b.high, 2),
                "l": round(b.low, 2), "c": round(b.close, 2),
                "v": int(b.volume or 0),
            })
    # Pre-compute chart metrics so LLM can cite them by name
    computed = _compute_chart_metrics(bars) if bars else {}

    # Pull session-wide context: recent ensemble emits, regime, VIX.
    # Fail-open — missing pieces are omitted from the snapshot.
    session_ctx = _gather_session_context()

    snap = {
        "symbol": adv.symbol,
        "direction": adv.direction,
        "strike": adv.strike, "expiry": adv.expiry,
        "qty": adv.qty,
        "entry_price": adv.entry_price,
        "current_price": adv.current_price,
        "peak_pnl_pct": round(adv.peak_pnl_pct * 100, 2),
        "current_pnl_pct": round(adv.current_pnl_pct * 100, 2),
        "gave_back_pct": round(
            ((adv.peak_pnl_pct - adv.current_pnl_pct) /
              max(adv.peak_pnl_pct, 1e-9)) * 100, 1
        ) if adv.peak_pnl_pct > 0 else None,
        "bars_summary": adv.bars_summary,
        "recent_underlying_bars": recent_bars,
        "chart_metrics": computed,
        "session_context": session_ctx,
    }
    return (
        "You are a senior options-desk risk manager reviewing a FADING "
        "position for a long-options paper bot. The position was in "
        "profit and is retracing. Your job: recommend close / hold / "
        "trim with SPECIFIC, EVIDENCE-BASED reasoning the operator can "
        "review in Discord.\n\n"
        "## Decision rules\n\n"
        "1. If the position was in profit and is now NEGATIVE: always "
        "recommend 'close', urgency='urgent'.\n"
        "2. If give-back >= 50% and chart shows reversal (lower highs, "
        "VWAP loss, volume against us): 'close', urgency='urgent'.\n"
        "3. If give-back 25-50% and reversal signs are partial: 'trim' "
        "50% at urgency='urgent' — lock half, let the rest trail.\n"
        "4. If the fade is noise INSIDE a still-valid trend (higher "
        "lows on the underlying, VWAP holding, volume flat): 'hold'.\n"
        "5. For 0DTE positions: be EXTRA aggressive — theta accelerates "
        "in final hours, no recovery window.\n\n"
        "## What the rationale MUST include\n\n"
        "- What SPECIFIC chart feature triggered your call (name it: "
        "  'lower-high pattern on last 3 bars', 'VWAP break at X', "
        "  'volume surge against position', etc.)\n"
        "- The key price level to watch next\n"
        "- Expected outcome if held vs closed\n\n"
        "## Output schema — STRICT JSON, no other text\n\n"
        "{\n"
        '  "recommendation": "close|hold|trim",\n'
        '  "urgency": "urgent|normal|low",\n'
        '  "confidence": "low|medium|high",\n'
        '  "rationale": "3-4 sentences citing SNAPSHOT values BY NAME",\n'
        '  "chart_signals": ["2-4 named chart features driving the call"],\n'
        '  "key_levels": ["price + what it means", "..."],\n'
        '  "risk_reward_remaining": "one sentence: what more can we gain vs lose if we hold",\n'
        '  "alternative_actions": ["1-2 alternatives besides the primary rec"],\n'
        '  "time_context": "DTE + minutes held + theta note"\n'
        "}\n\n"
        f"SNAPSHOT:\n{json.dumps(snap, indent=2, default=str)[:6000]}\n\n"
        "YOUR JSON:"
    )


def _compute_chart_metrics(bars) -> Dict[str, Any]:
    """Pre-compute signals the LLM can cite by name — saves tokens and
    grounds the output in real numbers from the snapshot."""
    if not bars or len(bars) < 5:
        return {}
    recent = bars[-15:]
    highs = [b.high for b in recent]
    lows = [b.low for b in recent]
    closes = [b.close for b in recent]
    volumes = [b.volume or 0 for b in recent]

    # Lower-high / higher-low pattern detection
    last3_highs = highs[-3:]
    last3_lows = lows[-3:]
    lower_highs = len(last3_highs) == 3 and last3_highs[0] > last3_highs[1] > last3_highs[2]
    higher_lows = len(last3_lows) == 3 and last3_lows[0] < last3_lows[1] < last3_lows[2]

    # VWAP (volume-weighted average) across the window
    typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    tpv = sum(t * v for t, v in zip(typical, volumes))
    vsum = sum(volumes) or 1
    vwap = tpv / vsum
    last_close = closes[-1]
    vwap_pos = ("above" if last_close > vwap * 1.001
                else "below" if last_close < vwap * 0.999
                else "at")

    # Volume trend (recent 3 vs prior 10)
    if len(volumes) >= 13:
        recent_v = sum(volumes[-3:]) / 3
        baseline_v = sum(volumes[-13:-3]) / 10
        vol_ratio = recent_v / max(baseline_v, 1)
    else:
        vol_ratio = 1.0

    # 15-bar range
    rng_high = max(highs)
    rng_low = min(lows)
    pct_of_range = (
        (last_close - rng_low) / max(rng_high - rng_low, 1e-9)
    ) if rng_high > rng_low else 0.5

    return {
        "last_close": round(last_close, 2),
        "vwap": round(vwap, 2),
        "price_vs_vwap": vwap_pos,
        "vwap_dist_pct": round((last_close - vwap) / max(vwap, 1e-9) * 100, 2),
        "last_3_highs": [round(h, 2) for h in last3_highs],
        "last_3_lows": [round(l, 2) for l in last3_lows],
        "lower_high_pattern": lower_highs,
        "higher_low_pattern": higher_lows,
        "volume_ratio_recent_vs_baseline": round(vol_ratio, 2),
        "15bar_range_high": round(rng_high, 2),
        "15bar_range_low": round(rng_low, 2),
        "pct_of_range": round(pct_of_range * 100, 1),
        "net_15bar_pct": round(
            (closes[-1] - closes[0]) / max(closes[0], 1e-9) * 100, 2
        ),
    }


def _gather_session_context() -> Dict[str, Any]:
    """Pull current regime + VIX + recent ensemble activity from the log
    tail. Fail-open — missing pieces returned as None."""
    out: Dict[str, Any] = {}
    try:
        from ..core.data_paths import data_path
        from pathlib import Path as _P
        log_path = _P(data_path("logs/tradebot.out"))
        if not log_path.exists():
            return out
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            f.seek(max(0, size - 80_000))
            if size > 80_000:
                f.readline()
            text = f.read().decode("utf-8", errors="replace")
        import re as _re
        # Strip ANSI before matching
        text = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
        regimes = _re.findall(r"regime=(\w+)", text)
        if regimes:
            out["regime"] = regimes[-1]
        vixs = _re.findall(r"\bvix=([0-9.]+)", text)
        if vixs:
            try:
                out["vix"] = float(vixs[-1])
            except Exception:
                pass
        # Last 3 ensemble emits for trend context
        emits = []
        for line in reversed(text.splitlines()[-400:]):
            if "ensemble_emit" in line:
                emits.append(line[:180])
                if len(emits) >= 3:
                    break
        if emits:
            out["recent_ensemble_emits"] = list(reversed(emits))
    except Exception:
        pass
    return out


def _parse_llm(raw: str) -> Optional[Dict[str, Any]]:
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


# ---- persistence for Discord buttons -----------------------------
# Advisories are written to a file so the Discord button handler can
# resolve an advisory_id back to its position + recommendation.

def _store_path():
    try:
        from ..core.data_paths import data_path
        return Path(data_path("position_advisories.json"))
    except Exception:
        return Path("data/position_advisories.json")


def save_advisory(adv: FadeAdvisory) -> str:
    """Persist an advisory. Returns a short id for Discord button
    custom_id (truncated to 80 chars)."""
    import hashlib
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        try:
            d = json.loads(path.read_text() or "{}")
        except Exception:
            d = {}
        aid = hashlib.sha1(
            f"{adv.symbol}:{adv.ts}:{adv.entry_price}".encode()
        ).hexdigest()[:16]
        # Expire entries older than 6h (prevents unbounded growth)
        cutoff = time.time() - 6 * 3600
        d = {k: v for k, v in d.items() if v.get("ts", 0) > cutoff}
        d[aid] = {
            "symbol": adv.symbol, "direction": adv.direction,
            "strike": adv.strike, "expiry": adv.expiry,
            "qty": adv.qty,
            "entry_price": adv.entry_price,
            "current_price": adv.current_price,
            "peak_pnl_pct": adv.peak_pnl_pct,
            "current_pnl_pct": adv.current_pnl_pct,
            "recommendation": adv.recommendation,
            "rationale": adv.rationale,
            "ts": adv.ts,
        }
        path.write_text(json.dumps(d, indent=2, default=str))
    return aid


def load_advisory(aid: str) -> Optional[Dict[str, Any]]:
    path = _store_path()
    if not path.exists():
        return None
    try:
        with _lock:
            d = json.loads(path.read_text() or "{}")
        return d.get(aid)
    except Exception:
        return None
