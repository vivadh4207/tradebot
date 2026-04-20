"""Technical-analysis signal — consolidates classical chart signals
used by discretionary traders that the bot was previously missing.

Detectors (all stdlib-only, numpy for speed):

  1. RSI divergence
     - Bearish: price makes new high, RSI makes lower high → reversal
     - Bullish: price makes new low, RSI makes higher low → reversal

  2. Double top / double bottom
     - Two peaks within 0.5% of each other separated by a trough
     - Inverse for double bottom

  3. Median-line break (50-SMA)
     - Close crosses below 50-SMA with slope turning down → bearish
     - Close crosses above 50-SMA with slope turning up → bullish
     - Requires volume confirmation

  4. Bollinger reclaim
     - Bar was outside BB (above upper / below lower) and current bar
       closes back inside → momentum exhaustion

  5. Multi-timeframe RSI confluence
     - Aggregates 5-min bars to 15-min bars and checks both agree
       on oversold (<30) or overbought (>70) before emitting

Confidence is shaped by:
  - Pattern strength (depth of divergence, width of peaks, etc.)
  - Volume context (same volume gate as candle_patterns)
  - Timeframe confluence bonus

Emits one `Signal` per tick — best pattern wins if multiple fire.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..core.types import Signal, Side, OptionRight, Bar
from .base import SignalSource, SignalContext


# ---------------------------------------------------------- indicators

def _closes(bars: List[Bar]) -> np.ndarray:
    return np.asarray([b.close for b in bars], dtype=float)


def _volumes(bars: List[Bar]) -> np.ndarray:
    return np.asarray([b.volume for b in bars], dtype=float)


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Standard Wilder's RSI. Returns one value per input close; first
    `period` entries are NaN.
    """
    if len(closes) < period + 1:
        return np.full(len(closes), np.nan)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Wilder's smoothing
    avg_gain = np.zeros_like(closes)
    avg_loss = np.zeros_like(closes)
    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    out = np.full(len(closes), np.nan)
    for i in range(period, len(closes)):
        al = avg_loss[i]
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + avg_gain[i] / al)
    return out


def bollinger_bands(closes: np.ndarray, period: int = 20,
                     n_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (upper, mid, lower). Uses simple rolling mean + std."""
    n = len(closes)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if n < period:
        return upper, mid, lower
    for i in range(period - 1, n):
        window = closes[i - period + 1: i + 1]
        m = window.mean()
        s = window.std(ddof=0)
        mid[i] = m
        upper[i] = m + n_std * s
        lower[i] = m - n_std * s
    return upper, mid, lower


def sma(closes: np.ndarray, period: int) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period:
        return out
    for i in range(period - 1, n):
        out[i] = closes[i - period + 1: i + 1].mean()
    return out


def aggregate_to_timeframe(bars: List[Bar], group_size: int) -> List[Bar]:
    """Collapse every `group_size` consecutive bars into one bar.
    Useful for computing 15-min from 1-min bars (group_size=15),
    or 15-min from 5-min (group_size=3)."""
    if group_size <= 1 or len(bars) < group_size:
        return list(bars)
    out: List[Bar] = []
    for i in range(0, len(bars) - (len(bars) % group_size), group_size):
        chunk = bars[i: i + group_size]
        agg = Bar(
            symbol=chunk[0].symbol,
            ts=chunk[-1].ts,
            open=chunk[0].open,
            high=max(b.high for b in chunk),
            low=min(b.low for b in chunk),
            close=chunk[-1].close,
            volume=sum(b.volume for b in chunk),
        )
        out.append(agg)
    return out


# ---------------------------------------------------------- detectors


@dataclass
class TechHit:
    name: str
    direction: str       # "bullish" | "bearish"
    strength: float      # 0.0-1.0
    rationale: str


def _local_extrema(values: np.ndarray, window: int = 3) -> Tuple[List[int], List[int]]:
    """Return (peak_indices, trough_indices) — index i is a peak if
    values[i] is the maximum of values[i-window : i+window+1]."""
    peaks: List[int] = []
    troughs: List[int] = []
    n = len(values)
    for i in range(window, n - window):
        w = values[i - window: i + window + 1]
        if np.isnan(w).any():
            continue
        if values[i] == w.max() and values[i] > values[i - 1]:
            peaks.append(i)
        if values[i] == w.min() and values[i] < values[i - 1]:
            troughs.append(i)
    return peaks, troughs


def detect_rsi_divergence(closes: np.ndarray, rsi_values: np.ndarray,
                            lookback: int = 30) -> Optional[TechHit]:
    """Scan last `lookback` bars for classic price-vs-RSI divergence.

    Bearish: the two most recent peaks — price higher but RSI lower.
    Bullish: the two most recent troughs — price lower but RSI higher.
    """
    if len(closes) < lookback + 5:
        return None
    tail_closes = closes[-lookback:]
    tail_rsi = rsi_values[-lookback:]
    peaks, troughs = _local_extrema(tail_closes, window=2)
    # Bearish divergence: price higher, RSI lower
    if len(peaks) >= 2:
        a, b = peaks[-2], peaks[-1]
        if (tail_closes[b] > tail_closes[a]
                and tail_rsi[b] < tail_rsi[a]
                and tail_rsi[a] > 60):   # only meaningful near overbought
            spread = float(tail_rsi[a] - tail_rsi[b])
            strength = min(0.90, 0.60 + spread / 50.0)
            return TechHit(
                name="bearish_rsi_divergence", direction="bearish",
                strength=strength,
                rationale=f"price↑ rsi↓ by {spread:.1f}",
            )
    # Bullish divergence: price lower, RSI higher
    if len(troughs) >= 2:
        a, b = troughs[-2], troughs[-1]
        if (tail_closes[b] < tail_closes[a]
                and tail_rsi[b] > tail_rsi[a]
                and tail_rsi[a] < 40):
            spread = float(tail_rsi[b] - tail_rsi[a])
            strength = min(0.90, 0.60 + spread / 50.0)
            return TechHit(
                name="bullish_rsi_divergence", direction="bullish",
                strength=strength,
                rationale=f"price↓ rsi↑ by {spread:.1f}",
            )
    return None


def detect_double_top_bottom(closes: np.ndarray,
                                lookback: int = 40,
                                tol_pct: float = 0.005) -> Optional[TechHit]:
    """Two peaks within `tol_pct` of each other, separated by a trough
    that's at least 1% below the peaks → double top (bearish).
    Inverse → double bottom (bullish).
    """
    if len(closes) < lookback:
        return None
    tail = closes[-lookback:]
    peaks, troughs = _local_extrema(tail, window=2)
    # Double top — need 2 peaks with trough between
    if len(peaks) >= 2:
        a, b = peaks[-2], peaks[-1]
        p_a, p_b = tail[a], tail[b]
        if abs(p_a - p_b) / max(p_a, p_b) <= tol_pct:
            # trough between them must be at least 1% below
            between = tail[a + 1: b]
            if len(between) > 0:
                tr = between.min()
                if (max(p_a, p_b) - tr) / max(p_a, p_b) >= 0.01:
                    # bearish only if current close is below both peaks
                    if tail[-1] < min(p_a, p_b):
                        return TechHit(
                            name="double_top", direction="bearish",
                            strength=0.75,
                            rationale=f"peaks ~{p_a:.2f},{p_b:.2f}",
                        )
    # Double bottom
    if len(troughs) >= 2:
        a, b = troughs[-2], troughs[-1]
        t_a, t_b = tail[a], tail[b]
        if abs(t_a - t_b) / max(t_a, t_b) <= tol_pct:
            between = tail[a + 1: b]
            if len(between) > 0:
                pk = between.max()
                if (pk - min(t_a, t_b)) / pk >= 0.01:
                    if tail[-1] > max(t_a, t_b):
                        return TechHit(
                            name="double_bottom", direction="bullish",
                            strength=0.75,
                            rationale=f"troughs ~{t_a:.2f},{t_b:.2f}",
                        )
    return None


def detect_median_break(closes: np.ndarray, volumes: np.ndarray,
                          period: int = 50,
                          vol_confirm_ratio: float = 1.3) -> Optional[TechHit]:
    """50-SMA cross with slope + volume confirmation."""
    if len(closes) < period + 5:
        return None
    sma50 = sma(closes, period)
    if np.isnan(sma50[-2]) or np.isnan(sma50[-1]):
        return None
    # Cross up: prev close below SMA, current close above
    crossed_up = closes[-2] < sma50[-2] and closes[-1] > sma50[-1]
    crossed_dn = closes[-2] > sma50[-2] and closes[-1] < sma50[-1]
    if not (crossed_up or crossed_dn):
        return None
    # Slope of SMA over last 5 bars
    slope = (sma50[-1] - sma50[-5]) / max(1e-9, sma50[-5])
    # Volume confirmation
    vol_avg = volumes[-period:].mean()
    vol_ratio = volumes[-1] / max(1e-9, vol_avg)
    if vol_ratio < vol_confirm_ratio:
        return None
    # Clean cross on volume is meaningful even if SMA slope is still
    # slightly in the opposite direction (typical at turning points).
    # Only suppress when slope is strongly against the cross.
    slope_tol = 0.003          # 0.3% allowed mismatch
    if crossed_up and slope >= -slope_tol:
        return TechHit(
            name="median_break_up", direction="bullish",
            strength=min(0.90, 0.65 + 100.0 * max(0.0, slope)),
            rationale=f"close>{period}SMA slope={slope*100:.2f}% vol×{vol_ratio:.1f}",
        )
    if crossed_dn and slope <= slope_tol:
        return TechHit(
            name="median_break_down", direction="bearish",
            strength=min(0.90, 0.65 - 100.0 * min(0.0, slope)),
            rationale=f"close<{period}SMA slope={slope*100:.2f}% vol×{vol_ratio:.1f}",
        )
    return None


def detect_bollinger_reclaim(closes: np.ndarray,
                                upper: np.ndarray,
                                lower: np.ndarray) -> Optional[TechHit]:
    """Bar was outside the band; current bar closes back inside →
    momentum exhaustion."""
    if len(closes) < 3 or np.isnan(upper[-1]) or np.isnan(upper[-2]):
        return None
    # Prev bar above upper, current back inside → bearish reclaim
    if closes[-2] > upper[-2] and closes[-1] < upper[-1]:
        return TechHit(
            name="bb_upper_reclaim", direction="bearish",
            strength=0.70,
            rationale="price back inside upper BB",
        )
    if closes[-2] < lower[-2] and closes[-1] > lower[-1]:
        return TechHit(
            name="bb_lower_reclaim", direction="bullish",
            strength=0.70,
            rationale="price back inside lower BB",
        )
    return None


def detect_multi_tf_rsi(bars_short: List[Bar],
                          bars_long: List[Bar]) -> Optional[TechHit]:
    """Confluence across two timeframes. If RSI(short) AND RSI(long)
    agree on overbought/oversold, emit.
    """
    if len(bars_short) < 20 or len(bars_long) < 20:
        return None
    rsi_s = rsi(_closes(bars_short))
    rsi_l = rsi(_closes(bars_long))
    if np.isnan(rsi_s[-1]) or np.isnan(rsi_l[-1]):
        return None
    s, l = float(rsi_s[-1]), float(rsi_l[-1])
    # Both overbought → bearish
    if s >= 70 and l >= 70:
        strength = min(0.90, 0.65 + (min(s, l) - 70) / 100.0)
        return TechHit(
            name="multi_tf_rsi_overbought", direction="bearish",
            strength=strength,
            rationale=f"rsi_s={s:.1f} rsi_l={l:.1f}",
        )
    # Both oversold → bullish
    if s <= 30 and l <= 30:
        strength = min(0.90, 0.65 + (30 - max(s, l)) / 100.0)
        return TechHit(
            name="multi_tf_rsi_oversold", direction="bullish",
            strength=strength,
            rationale=f"rsi_s={s:.1f} rsi_l={l:.1f}",
        )
    return None


# ---------------------------------------------------------- signal

class TechnicalAnalysisSignal(SignalSource):
    """Consolidates divergence, double-top/bottom, median-break,
    Bollinger-reclaim, and multi-timeframe RSI into one signal source.

    Priority: divergence > double-top/bottom > median-break > BB reclaim
    > multi-tf RSI confluence. First hit wins.
    """

    name = "technical_analysis"

    def __init__(
        self,
        min_bars: int = 60,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        sma_period: int = 50,
        short_tf_group: int = 5,     # 5-bar group for "5-min"-equivalent
        long_tf_group: int = 15,    # 15-bar group for "15-min"-equivalent
        low_vol_damp: float = 0.15,
    ):
        self.min_bars = min_bars
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sma_period = sma_period
        self.short_tf_group = short_tf_group
        self.long_tf_group = long_tf_group
        self.low_vol_damp = low_vol_damp

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        if len(ctx.bars) < self.min_bars:
            return None

        closes = _closes(ctx.bars)
        volumes = _volumes(ctx.bars)
        rsi_vals = rsi(closes, self.rsi_period)
        upper, _mid, lower = bollinger_bands(closes, self.bb_period, self.bb_std)

        # Priority-ordered first-hit.
        hit: Optional[TechHit] = None
        hit = (detect_rsi_divergence(closes, rsi_vals)
                or detect_double_top_bottom(closes)
                or detect_median_break(closes, volumes, self.sma_period)
                or detect_bollinger_reclaim(closes, upper, lower))
        if hit is None:
            # Multi-TF RSI uses downsampled bars
            short_bars = aggregate_to_timeframe(ctx.bars, self.short_tf_group)
            long_bars = aggregate_to_timeframe(ctx.bars, self.long_tf_group)
            hit = detect_multi_tf_rsi(short_bars, long_bars)
        if hit is None:
            return None

        confidence = float(hit.strength)

        # Low-volume dampening — consistent with candle_patterns.
        vol_avg = float(volumes[-self.bb_period:].mean()) if len(volumes) >= self.bb_period else 0.0
        if vol_avg > 0:
            vol_ratio = float(volumes[-1]) / vol_avg
            if vol_ratio < 0.6:
                confidence -= self.low_vol_damp

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
            rationale=f"{hit.name} · {hit.rationale}",
            meta={
                "pattern": hit.name,
                "direction": hit.direction,
            },
        )
