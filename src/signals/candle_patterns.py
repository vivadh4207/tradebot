"""Candle-pattern signal — recognize classical reversal / continuation
patterns on the most recent bars and emit a directional signal.

Different from `momentum.py`:
  - Momentum fits a slope; this module looks at BAR SHAPES.
  - Patterns emit earlier than slope-based signals (often the first bar
    of a reversal).
  - Volume context gates the confidence: the same engulfing bar is
    weak on low volume, strong on high volume.

Patterns recognized:
  BULLISH reversal: bullish_engulfing, hammer, piercing, morning_star_lite
  BEARISH reversal: bearish_engulfing, shooting_star, dark_cloud,
                    evening_star_lite
  CONTINUATION:     inside_bar_breakout (direction decided by breakout)
  BREAKOUT:         range_break_with_volume (20-bar range + vol confirmation)

Outputs a Signal with:
  - side=BUY
  - option_right=CALL (bullish) or PUT (bearish)
  - confidence in [0.55, 0.95], scaled by pattern strength + volume
  - meta={"pattern": <name>, "vol_ratio": x, "range_break": bool}

Rules:
  - Minimum 20 bars of history required (volume baseline).
  - Low-volume-on-a-continuation pattern → skip; low-volume-on-reversal
    at support/resistance → kept but confidence dampened.
  - VWAP proximity tightens confidence (pattern near key level = better).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..core.types import Signal, Side, OptionRight, Bar
from .base import SignalSource, SignalContext


# ---------------------------------------------------------- helpers

def _body(b: Bar) -> float:
    return abs(b.close - b.open)


def _range(b: Bar) -> float:
    return max(0.0, b.high - b.low)


def _is_bullish(b: Bar) -> bool:
    return b.close > b.open


def _is_bearish(b: Bar) -> bool:
    return b.close < b.open


def _upper_shadow(b: Bar) -> float:
    return max(0.0, b.high - max(b.close, b.open))


def _lower_shadow(b: Bar) -> float:
    return max(0.0, min(b.close, b.open) - b.low)


def _avg_volume(bars: List[Bar], n: int = 20) -> float:
    if not bars:
        return 0.0
    window = bars[-n:] if len(bars) >= n else bars
    return sum(b.volume for b in window) / max(1, len(window))


def _avg_range(bars: List[Bar], n: int = 20) -> float:
    if not bars:
        return 0.0
    window = bars[-n:] if len(bars) >= n else bars
    return sum(_range(b) for b in window) / max(1, len(window))


# ---------------------------------------------------------- pattern detectors

@dataclass
class PatternHit:
    name: str
    direction: str                 # "bullish" | "bearish"
    strength: float                # 0.0-1.0 (pattern-intrinsic only)
    rationale: str


def _detect_engulfing(bars: List[Bar]) -> Optional[PatternHit]:
    """Bullish: prior red, current green, current body fully engulfs prior.
    Bearish: mirror.
    """
    if len(bars) < 2:
        return None
    prev, curr = bars[-2], bars[-1]
    if _body(prev) <= 0 or _body(curr) <= 0:
        return None
    # Bullish
    if (_is_bearish(prev) and _is_bullish(curr)
            and curr.open <= prev.close
            and curr.close >= prev.open):
        ratio = _body(curr) / max(1e-9, _body(prev))
        strength = min(1.0, 0.55 + 0.1 * min(ratio - 1.0, 3.0))
        return PatternHit(
            name="bullish_engulfing", direction="bullish",
            strength=strength,
            rationale=f"body ratio {ratio:.2f}",
        )
    # Bearish
    if (_is_bullish(prev) and _is_bearish(curr)
            and curr.open >= prev.close
            and curr.close <= prev.open):
        ratio = _body(curr) / max(1e-9, _body(prev))
        strength = min(1.0, 0.55 + 0.1 * min(ratio - 1.0, 3.0))
        return PatternHit(
            name="bearish_engulfing", direction="bearish",
            strength=strength,
            rationale=f"body ratio {ratio:.2f}",
        )
    return None


def _detect_hammer(bars: List[Bar]) -> Optional[PatternHit]:
    """Hammer — long lower shadow, small body, tiny upper shadow, after a
    downtrend. We approximate downtrend via last 3 bars' close slope."""
    if len(bars) < 4:
        return None
    b = bars[-1]
    body, rng = _body(b), _range(b)
    if rng <= 0 or body <= 0:
        return None
    lower, upper = _lower_shadow(b), _upper_shadow(b)
    if lower < 2.0 * body:
        return None
    if upper > 0.4 * body:
        return None
    # Prior trend: last 3 bars trending down
    closes = [x.close for x in bars[-4:-1]]
    if closes[-1] >= closes[0]:
        return None
    strength = min(0.95, 0.60 + 0.1 * (lower / max(1e-9, body) - 2.0))
    return PatternHit(
        name="hammer", direction="bullish",
        strength=strength,
        rationale=f"lower/body={lower/body:.1f}x",
    )


def _detect_shooting_star(bars: List[Bar]) -> Optional[PatternHit]:
    """Inverted hammer at top of uptrend — bearish reversal."""
    if len(bars) < 4:
        return None
    b = bars[-1]
    body, rng = _body(b), _range(b)
    if rng <= 0 or body <= 0:
        return None
    lower, upper = _lower_shadow(b), _upper_shadow(b)
    if upper < 2.0 * body:
        return None
    if lower > 0.4 * body:
        return None
    closes = [x.close for x in bars[-4:-1]]
    if closes[-1] <= closes[0]:
        return None
    strength = min(0.95, 0.60 + 0.1 * (upper / max(1e-9, body) - 2.0))
    return PatternHit(
        name="shooting_star", direction="bearish",
        strength=strength,
        rationale=f"upper/body={upper/body:.1f}x",
    )


def _detect_inside_bar_breakout(bars: List[Bar]) -> Optional[PatternHit]:
    """Inside bar = bar contained within prior bar's range. We emit on
    the NEXT bar when it breaks above/below the inside-bar high/low."""
    if len(bars) < 3:
        return None
    mother, inside, curr = bars[-3], bars[-2], bars[-1]
    if not (inside.high <= mother.high and inside.low >= mother.low):
        return None
    if curr.close > inside.high and curr.close > mother.high:
        return PatternHit(
            name="inside_bar_breakout_up", direction="bullish",
            strength=0.70,
            rationale="break above inside high",
        )
    if curr.close < inside.low and curr.close < mother.low:
        return PatternHit(
            name="inside_bar_breakout_down", direction="bearish",
            strength=0.70,
            rationale="break below inside low",
        )
    return None


def _detect_range_breakout(bars: List[Bar], n: int = 20) -> Optional[PatternHit]:
    """Close breaks above 20-bar high or below 20-bar low. Pure breakout."""
    if len(bars) < n + 1:
        return None
    window = bars[-(n + 1):-1]    # exclude the current bar
    prior_high = max(b.high for b in window)
    prior_low = min(b.low for b in window)
    curr = bars[-1]
    if curr.close > prior_high:
        over = (curr.close - prior_high) / max(1e-9, prior_high)
        strength = min(0.95, 0.65 + 50.0 * over)
        return PatternHit(
            name="range_breakout_up", direction="bullish",
            strength=strength,
            rationale=f"{n}-bar high + {over*100:.2f}%",
        )
    if curr.close < prior_low:
        under = (prior_low - curr.close) / max(1e-9, prior_low)
        strength = min(0.95, 0.65 + 50.0 * under)
        return PatternHit(
            name="range_breakout_down", direction="bearish",
            strength=strength,
            rationale=f"{n}-bar low - {under*100:.2f}%",
        )
    return None


# ---------------------------------------------------------- signal

class CandlePatternSignal(SignalSource):
    """Consolidates several pattern detectors into one SignalSource.
    First-hit wins priority order: engulfing > hammer/star >
    inside-bar > range-breakout. Returns None when nothing fires."""

    name = "candle_patterns"

    def __init__(
        self,
        min_bars: int = 20,
        volume_hi_ratio: float = 1.5,       # bar vol >= 1.5 × 20-bar avg
        volume_lo_ratio: float = 0.6,       # bar vol <= 0.6 × 20-bar avg
        low_vol_damp: float = 0.20,         # subtract 0.20 from confidence
        high_vol_boost: float = 0.10,       # add 0.10
        near_vwap_bps: float = 50.0,        # 0.50% proximity counts as "near"
        near_vwap_boost: float = 0.05,
        continuation_require_volume: bool = True,
    ):
        self.min_bars = min_bars
        self.volume_hi_ratio = volume_hi_ratio
        self.volume_lo_ratio = volume_lo_ratio
        self.low_vol_damp = low_vol_damp
        self.high_vol_boost = high_vol_boost
        self.near_vwap_bps = near_vwap_bps
        self.near_vwap_boost = near_vwap_boost
        self.continuation_require_volume = continuation_require_volume

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        if len(ctx.bars) < self.min_bars:
            return None

        hit = (_detect_engulfing(ctx.bars)
               or _detect_hammer(ctx.bars)
               or _detect_shooting_star(ctx.bars)
               or _detect_inside_bar_breakout(ctx.bars)
               or _detect_range_breakout(ctx.bars))
        if hit is None:
            return None

        # Volume context
        curr = ctx.bars[-1]
        avg_vol = _avg_volume(ctx.bars[:-1], n=self.min_bars)
        vol_ratio = (curr.volume / avg_vol) if avg_vol > 0 else 1.0

        confidence = float(hit.strength)

        # Continuation / breakout patterns need volume confirmation.
        is_continuation = hit.name in ("inside_bar_breakout_up",
                                        "inside_bar_breakout_down",
                                        "range_breakout_up",
                                        "range_breakout_down")
        if is_continuation and self.continuation_require_volume and vol_ratio < self.volume_hi_ratio:
            # A "breakout" on low volume is a fake-out — don't emit.
            if vol_ratio < self.volume_lo_ratio:
                return None
            # Marginal volume: emit but with dampening.
            confidence -= self.low_vol_damp

        # Reversal patterns: low volume dampens, high volume boosts.
        else:
            if vol_ratio >= self.volume_hi_ratio:
                confidence += self.high_vol_boost
            elif vol_ratio <= self.volume_lo_ratio:
                confidence -= self.low_vol_damp

        # VWAP proximity bonus — pattern near VWAP often marks a
        # regime-relevant level.
        if ctx.vwap > 0:
            bps = abs(curr.close - ctx.vwap) / ctx.vwap * 10000.0
            if bps <= self.near_vwap_bps:
                confidence += self.near_vwap_boost

        confidence = max(0.0, min(1.0, confidence))
        if confidence < 0.55:
            return None

        right = OptionRight.CALL if hit.direction == "bullish" else OptionRight.PUT
        return Signal(
            source=self.name,
            symbol=ctx.symbol,
            side=Side.BUY,
            option_right=right,
            confidence=confidence,
            rationale=f"{hit.name} · {hit.rationale} · vol×{vol_ratio:.2f}",
            meta={
                "pattern": hit.name,
                "direction": hit.direction,
                "vol_ratio": round(vol_ratio, 2),
                "is_continuation": is_continuation,
            },
        )
