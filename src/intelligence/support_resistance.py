"""Volume-weighted support / resistance detector.

Scans a bar series, finds local swing highs and swing lows, and scores
each level by the volume that traded within a narrow band of that
price across the full lookback window. The highest-scoring levels are
the most likely to act as S/R zones.

Why volume-weighted: a pivot point matters in proportion to how much
conviction (volume) was transacted there. A local low on a light-volume
day isn't a real support; a local low on a high-volume day, retested
twice more on moderate volume, is.

Used by the 0DTE credit-spread runner to pick a short-strike location
that sits inside a real institutional zone rather than an arbitrary
20-bar minimum. Replaces the crude `_pivot_low()` function.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional

from ..core.types import Bar


_log = logging.getLogger(__name__)


@dataclass
class SRLevel:
    """One S/R level + its strength score."""
    price: float
    kind: Literal["support", "resistance"]
    touches: int             # number of swing points that hit this level
    volume: float            # cumulative volume traded inside the band
    first_ts: float          # earliest bar timestamp that touched
    last_ts: float           # most recent bar timestamp that touched

    @property
    def score(self) -> float:
        """Composite strength. Higher = stronger level.
        volume * touches — tested multiple times on heavy vol is
        structurally stronger than one heavy-vol touch."""
        return self.volume * max(1, self.touches)


@dataclass
class SRConfig:
    pivot_window: int = 3         # bar must be max/min over ±pivot_window
    band_pct: float = 0.002        # ±0.2% band around each candidate
    min_touches: int = 1
    top_k: int = 5


def find_levels(
    bars: List[Bar],
    kind: Literal["support", "resistance", "both"] = "both",
    cfg: Optional[SRConfig] = None,
) -> List[SRLevel]:
    """Return the top-K S/R levels from a bar series, sorted by score
    (strongest first).

    Algorithm:
      1. Scan for pivot highs (local max over ±window) and lows.
      2. Cluster pivots within `band_pct` of each other — e.g. pivots
         at 571.20 and 571.35 with band=0.2% cluster into one level
         at the volume-weighted mean.
      3. For each cluster, sum the volume of every bar whose high-low
         overlapped the ±band zone (not just the pivot bar — any bar
         that transacted inside the zone adds to the score).
      4. Filter by min_touches, return top_k by score.
    """
    cfg = cfg or SRConfig()
    if len(bars) < 2 * cfg.pivot_window + 1:
        return []

    pivots_hi: List[int] = []
    pivots_lo: List[int] = []
    w = cfg.pivot_window
    for i in range(w, len(bars) - w):
        center_hi = bars[i].high
        center_lo = bars[i].low
        # Strict inequality vs. at least one neighbor in each direction —
        # otherwise a flat plateau registers every bar as a pivot.
        hi_left  = max(b.high for b in bars[i - w: i])
        hi_right = max(b.high for b in bars[i + 1: i + w + 1])
        lo_left  = min(b.low  for b in bars[i - w: i])
        lo_right = min(b.low  for b in bars[i + 1: i + w + 1])
        if center_hi > hi_left and center_hi > hi_right:
            pivots_hi.append(i)
        if center_lo < lo_left and center_lo < lo_right:
            pivots_lo.append(i)

    def _cluster(indices: List[int], price_of) -> List[SRLevel]:
        """Cluster pivots whose prices are within band_pct of each
        other. Returns SRLevels."""
        if not indices:
            return []
        clusters: List[List[int]] = []
        by_price = sorted(indices, key=lambda i: price_of(bars[i]))
        for i in by_price:
            px = price_of(bars[i])
            placed = False
            for cluster in clusters:
                cluster_px = sum(price_of(bars[j]) for j in cluster) / len(cluster)
                if abs(px - cluster_px) / max(cluster_px, 1e-9) <= cfg.band_pct:
                    cluster.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
        out: List[SRLevel] = []
        for cluster in clusters:
            if len(cluster) < cfg.min_touches:
                continue
            # Volume-weighted mean of the pivot prices
            total_pv_v = sum(price_of(bars[j]) * bars[j].volume for j in cluster)
            total_v = sum(bars[j].volume for j in cluster)
            if total_v <= 0:
                mean_px = sum(price_of(bars[j]) for j in cluster) / len(cluster)
            else:
                mean_px = total_pv_v / total_v
            # Sum volume of ALL bars (not just the pivot bars) that
            # transacted inside the ±band around this level
            lo = mean_px * (1 - cfg.band_pct)
            hi = mean_px * (1 + cfg.band_pct)
            in_band_vol = sum(b.volume for b in bars
                              if b.low <= hi and b.high >= lo)
            touch_timestamps = [bars[j].ts.timestamp()
                                 if hasattr(bars[j].ts, "timestamp")
                                 else bars[j].ts for j in cluster]
            first_ts = min(touch_timestamps)
            last_ts = max(touch_timestamps)
            kind_val = "resistance" if price_of is _high else "support"
            out.append(SRLevel(
                price=round(mean_px, 4),
                kind=kind_val,            # type: ignore
                touches=len(cluster),
                volume=float(in_band_vol),
                first_ts=float(first_ts),
                last_ts=float(last_ts),
            ))
        return out

    levels: List[SRLevel] = []
    if kind in ("support", "both"):
        levels += _cluster(pivots_lo, _low)
    if kind in ("resistance", "both"):
        levels += _cluster(pivots_hi, _high)
    levels.sort(key=lambda lv: lv.score, reverse=True)
    return levels[:cfg.top_k]


def nearest_support(bars: List[Bar], spot: float,
                     cfg: Optional[SRConfig] = None,
                     max_distance_pct: float = 0.02) -> Optional[SRLevel]:
    """Find the strongest support at or below `spot` within
    max_distance_pct. Returns None if none found."""
    levels = find_levels(bars, kind="support", cfg=cfg)
    below = [lv for lv in levels
             if lv.price <= spot
             and (spot - lv.price) / max(spot, 1e-9) <= max_distance_pct]
    if not below:
        return None
    # Strongest (highest score) is the best pick — not just the
    # closest. A weak nearby support is worse than a strong one 0.5% away.
    below.sort(key=lambda lv: lv.score, reverse=True)
    return below[0]


def nearest_resistance(bars: List[Bar], spot: float,
                         cfg: Optional[SRConfig] = None,
                         max_distance_pct: float = 0.02) -> Optional[SRLevel]:
    levels = find_levels(bars, kind="resistance", cfg=cfg)
    above = [lv for lv in levels
             if lv.price >= spot
             and (lv.price - spot) / max(spot, 1e-9) <= max_distance_pct]
    if not above:
        return None
    above.sort(key=lambda lv: lv.score, reverse=True)
    return above[0]


def _high(b: Bar) -> float: return b.high
def _low(b: Bar) -> float:  return b.low
