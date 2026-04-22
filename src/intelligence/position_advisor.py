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
        """Decide whether this position qualifies for an LLM advisory."""
        peak = self._peak(pos) or 0.0
        # A: peak >= +3% and retraced >= 40% from peak
        if peak >= 0.03:
            give_back = (peak - current_pnl_pct) / max(peak, 1e-9)
            if give_back >= 0.40:
                return True
        # B: open > 15 min and flat/negative (stuck trade)
        try:
            held_sec = time.time() - float(pos.entry_ts)
        except Exception:
            held_sec = 0
        if held_sec > 900 and current_pnl_pct <= 0.005:
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

        # Fallback rule-based recommendation first — ensures SOMETHING
        # gets posted even if LLM times out.
        give_back = (peak - current_pnl_pct) / max(peak, 1e-9) if peak > 0 else 0
        if peak > 0 and current_pnl_pct < 0:
            adv.recommendation = "close"
            adv.urgency = "urgent"
            adv.rationale = (
                f"Position was +{peak*100:.1f}% peak, now NEGATIVE at "
                f"{current_pnl_pct*100:+.1f}%. Cut and preserve capital."
            )
        elif give_back >= 0.60:
            adv.recommendation = "close"
            adv.urgency = "urgent"
            adv.rationale = (
                f"Gave back {give_back*100:.0f}% of +{peak*100:.1f}% peak. "
                f"Take the remaining {current_pnl_pct*100:+.1f}% before "
                "it goes negative."
            )
        elif give_back >= 0.40:
            adv.recommendation = "trim"
            adv.urgency = "normal"
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
                        adv.rationale = str(parsed["rationale"])[:300]
                    if parsed.get("key_levels"):
                        adv.key_levels = [str(x)[:60]
                                             for x in parsed["key_levels"][:3]]
                    adv.model = model
        except Exception as e:                              # noqa: BLE001
            _log.info("position_advisor_llm_err err=%s", e)
        return adv


# ---- stateless helpers -------------------------------------------


def _build_prompt(adv: FadeAdvisory, bars) -> str:
    recent_bars: List[Dict[str, Any]] = []
    if bars:
        for b in bars[-12:]:
            recent_bars.append({
                "o": round(b.open, 2), "h": round(b.high, 2),
                "l": round(b.low, 2), "c": round(b.close, 2),
                "v": int(b.volume or 0),
            })
    snap = {
        "symbol": adv.symbol,
        "direction": adv.direction,
        "strike": adv.strike, "expiry": adv.expiry,
        "qty": adv.qty,
        "entry_price": adv.entry_price,
        "current_price": adv.current_price,
        "peak_pnl_pct": round(adv.peak_pnl_pct * 100, 2),
        "current_pnl_pct": round(adv.current_pnl_pct * 100, 2),
        "bars_summary": adv.bars_summary,
        "recent_underlying_bars": recent_bars,
    }
    return (
        "You are a risk-management copilot for a long-options paper bot. "
        "A position has faded from its peak profit — you must recommend "
        "'close', 'hold', or 'trim' and justify in 2 sentences max. "
        "Rules: if the position was in profit and is now negative, ALWAYS "
        "recommend 'close'. If the chart shows reversal (lower highs, "
        "VWAP loss), recommend 'close' or 'trim'. If the fade is noise "
        "inside a still-valid trend, 'hold' is acceptable.\n\n"
        "Respond with STRICT JSON only:\n"
        "{\n"
        '  "recommendation": "close|hold|trim",\n'
        '  "urgency": "urgent|normal|low",\n'
        '  "confidence": "low|medium|high",\n'
        '  "rationale": "2-sentence explanation citing SNAPSHOT values",\n'
        '  "key_levels": ["short level or trigger", "..."]\n'
        "}\n\n"
        f"SNAPSHOT:\n{json.dumps(snap, indent=2)}\n\n"
        "YOUR JSON:"
    )


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
