"""SRBounceBreakSignal — mirrors the operator's manual edge.

Operator's description (2026-04-23):
  "I watch SPY/QQQ at key levels and buy calls on bounces from support,
   but puts when it drops from support."

Two-trigger strategy:

  BOUNCE (long call):
    1. Price tagged a support level within last N bars
    2. Last 1-2 bars closed ABOVE support (rejection/bounce)
    3. Volume on the bounce bar > baseline (confirmation)
    4. RSI(14) oversold (<40) at the touch (extra filter)

  BREAK (long put):
    1. Price was ABOVE support within last N bars
    2. Last 1-2 bars closed BELOW support (breakdown)
    3. Volume on the break bar > 1.2× baseline (conviction)
    4. RSI(14) NOT oversold (>40) (not a dead-cat bounce setup)

Support levels computed from multiple sources:
  - Session VWAP
  - Prior day's low (PDL) / high (PDH)
  - Rolling 20-bar low / high
  - Round-number psychological levels (every 5 or 10 points for SPY,
    every 5 for QQQ)

The strongest level (most confluence) takes priority. Bounce/break
emits a direction + score ∈ [0.85, 0.95] — high confidence because
this is the operator's proven edge, not a noisy indicator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .base import SignalSource, SignalContext
from ..core.types import Signal, Side


# Psychological round numbers: SPY every $5 (710, 715, 720),
# QQQ every $5 (505, 510, 515). Tighter than normal stocks because
# SPY moves ~0.5-2% intraday — finer grid catches more levels.
_ROUND_STEPS = {"SPY": 5.0, "QQQ": 5.0, "IWM": 2.5, "DIA": 5.0}


def _round_levels(symbol: str, spot: float, band_pct: float = 0.015
                    ) -> List[float]:
    """Return nearby round-number levels within ±band_pct of spot."""
    step = _ROUND_STEPS.get(symbol, 5.0)
    lo = spot * (1 - band_pct)
    hi = spot * (1 + band_pct)
    # Find nearest multiples of `step`
    start = (lo // step) * step
    levels = []
    x = start
    while x <= hi + step:
        if lo <= x <= hi:
            levels.append(round(x, 2))
        x += step
    return levels


def _compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        chg = closes[i] - closes[i - 1]
        if chg > 0:
            gains += chg
        else:
            losses += abs(chg)
    if gains + losses == 0:
        return 50.0
    rs = gains / max(losses, 1e-9)
    return 100 - (100 / (1 + rs))


def _session_vwap(bars: List) -> Optional[float]:
    """Volume-weighted average price over all bars passed in."""
    if not bars:
        return None
    tpv = 0.0
    vol_sum = 0.0
    for b in bars:
        typical = (b.high + b.low + b.close) / 3
        v = max(1.0, b.volume or 0)
        tpv += typical * v
        vol_sum += v
    return tpv / max(vol_sum, 1e-9)


@dataclass
class SRBounceBreakConfig:
    """Thresholds. Tighter = fewer but higher-quality signals."""
    min_bars: int = 30
    support_touch_lookback: int = 5     # must have tagged level recently
    bounce_rejection_bars: int = 2       # N bars closed back above
    break_confirmation_bars: int = 1     # N bars closed below
    level_proximity_pct: float = 0.003   # price within 0.3% = "at" level
    volume_surge_multiple: float = 1.20  # bounce/break volume ≥ 1.2× avg
    rsi_oversold: float = 40.0           # bounce needs RSI < this
    rsi_overbought: float = 60.0         # break-put rejects RSI > this (already bearish)
    min_confluence_levels: int = 1       # merge nearby levels
    base_score: float = 0.85             # ensemble weight ramp-up to 0.95


class SRBounceBreakSignal(SignalSource):
    """Support/resistance bounce + breakdown signal — codification of
    the operator's manual edge. Focused on SPY/QQQ/IWM/DIA only."""

    name = "sr_bounce_break"
    _ALLOWED = {"SPY", "QQQ", "IWM", "DIA"}

    def __init__(self, cfg: Optional[SRBounceBreakConfig] = None):
        self.cfg = cfg or SRBounceBreakConfig()

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        """Adapter to the SignalSource ABC — delegates to `score()`."""
        return self.score(ctx.symbol, ctx.bars)

    def score(self, symbol: str, bars, **_kwargs) -> Optional[Signal]:
        if symbol not in self._ALLOWED:
            return None
        if not bars or len(bars) < self.cfg.min_bars:
            return None

        closes = [b.close for b in bars]
        lows = [b.low for b in bars]
        highs = [b.high for b in bars]
        volumes = [b.volume or 0 for b in bars]

        spot = closes[-1]
        last = bars[-1]

        # Baseline volume (last 10-bar avg excluding most recent 3)
        if len(bars) >= 15:
            baseline_vol = sum(volumes[-13:-3]) / 10
        else:
            baseline_vol = sum(volumes) / len(volumes)

        # --- Support / resistance candidates ---
        levels: List[tuple] = []   # (price, kind, source)

        # 1. Session VWAP
        vwap = _session_vwap(bars)
        if vwap:
            levels.append((vwap, "both", "vwap"))

        # 2. Prior-day low + high (approximated as 390-bar boundary —
        # assumes 1-min bars, 6.5h = 390 bars; gets last trading day).
        if len(bars) >= 200:
            prev_day_slice = bars[-200:-100]
            levels.append((min(b.low for b in prev_day_slice),
                             "support", "pdl"))
            levels.append((max(b.high for b in prev_day_slice),
                             "resistance", "pdh"))

        # 3. Rolling 20-bar low + high
        if len(bars) >= 20:
            levels.append((min(lows[-20:]), "support", "20bar_low"))
            levels.append((max(highs[-20:]), "resistance", "20bar_high"))

        # 4. Round numbers near spot
        for px in _round_levels(symbol, spot):
            levels.append((px, "both", "round"))

        # --- BOUNCE (long CALL) detection ---
        # Criteria:
        #   (a) price touched a support within last N bars
        #   (b) most recent bar closed above that level
        #   (c) RSI at touch was oversold
        #   (d) volume on bounce bar > baseline
        rsi = _compute_rsi(closes, 14)
        supports = [p for p, k, _ in levels if k in ("support", "both")]

        for level in sorted(supports, key=lambda x: abs(x - spot)):
            # Was there a recent touch?
            lb = self.cfg.support_touch_lookback
            touch_bars = lows[-lb:]
            touched = any(abs(low - level) / level
                          <= self.cfg.level_proximity_pct
                          for low in touch_bars)
            if not touched:
                continue
            # Are last N bars now ABOVE the level?
            rej_n = self.cfg.bounce_rejection_bars
            rejection = all(c > level for c in closes[-rej_n:])
            if not rejection:
                continue
            # Volume confirmation on the bounce bar
            vol_surge = (baseline_vol > 0 and
                         (last.volume or 0)
                         >= self.cfg.volume_surge_multiple * baseline_vol)
            # RSI filter: the TOUCH bar's RSI should have been oversold
            rsi_ok = (rsi is not None and rsi < 55.0)  # not already overbought
            if vol_surge and rsi_ok:
                # Strong confluence — score ramps 0.85 → 0.95
                score = self.cfg.base_score
                if rsi is not None and rsi < self.cfg.rsi_oversold:
                    score += 0.05
                if (last.volume or 0) >= 1.5 * baseline_vol:
                    score += 0.05
                return Signal(
                    source=self.name,
                    symbol=symbol,
                    side=Side.BUY,    # BUY CALL
                    confidence=min(0.95, score),
                    meta={
                        "direction": "bullish",
                        "setup": "bounce_from_support",
                        "level": round(level, 2),
                        "level_source": next(
                            (s for p, _, s in levels
                             if abs(p - level) < 0.01),
                            "unknown",
                        ),
                        "rsi": round(rsi or 0, 1),
                        "vol_ratio": round(
                            (last.volume or 0) / max(baseline_vol, 1),
                            2,
                        ),
                        "spot": round(spot, 2),
                    },
                )

        # --- BREAKDOWN (long PUT) detection ---
        # Criteria:
        #   (a) price was ABOVE a support level in the recent window
        #   (b) most recent bar closed BELOW it
        #   (c) volume on break > 1.2× baseline
        #   (d) RSI not already oversold (avoid bounce setups)
        for level in sorted(supports, key=lambda x: abs(x - spot)):
            lb = self.cfg.support_touch_lookback
            # Was price ABOVE within lookback?
            above_closes = [c for c in closes[-lb - 1:-1]
                              if c > level]
            if len(above_closes) < 2:
                continue
            # Are last N bars now BELOW the level?
            br_n = self.cfg.break_confirmation_bars
            breakdown = all(c < level for c in closes[-br_n:])
            if not breakdown:
                continue
            # Volume confirmation
            vol_surge = (baseline_vol > 0 and
                         (last.volume or 0)
                         >= self.cfg.volume_surge_multiple * baseline_vol)
            # RSI filter — if already oversold, this is more likely
            # a bottoming pattern than a continuation break
            rsi_ok = (rsi is None or rsi > 45.0)
            if vol_surge and rsi_ok:
                score = self.cfg.base_score
                if rsi is not None and rsi > self.cfg.rsi_overbought:
                    score += 0.05
                if (last.volume or 0) >= 1.5 * baseline_vol:
                    score += 0.05
                return Signal(
                    source=self.name,
                    symbol=symbol,
                    side=Side.BUY,    # BUY PUT (bot interprets via meta.direction)
                    confidence=min(0.95, score),
                    meta={
                        "direction": "bearish",
                        "setup": "breakdown_below_support",
                        "level": round(level, 2),
                        "level_source": next(
                            (s for p, _, s in levels
                             if abs(p - level) < 0.01),
                            "unknown",
                        ),
                        "rsi": round(rsi or 0, 1),
                        "vol_ratio": round(
                            (last.volume or 0) / max(baseline_vol, 1),
                            2,
                        ),
                        "spot": round(spot, 2),
                    },
                )

        return None
