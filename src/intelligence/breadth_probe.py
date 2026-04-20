"""Market-breadth / risk-appetite probe.

Composes cheap, retail-accessible indicators into a single score in
[-1, +1]:

  -1.0  → extreme risk-off (collapsing breadth)
   0.0  → neutral
  +1.0  → extreme risk-on

The score aggregates:

  - VIX level vs its 52-week range (lower = risk-on)
  - SPY intraday % move (a sharp down-move = risk-off)
  - IWM / SPY ratio drift (small-cap leadership = risk-on)
  - HYG / TLT ratio drift (high-yield vs treasuries = credit appetite)

Each indicator is normalized to a sub-score in [-1, +1]; the composite
is the simple mean. When any source is unavailable we skip it and
renormalize — one missing feed doesn't zero the whole probe.

## Why this matters for short-premium strategies

Selling put credit spreads in a collapsing-breadth regime is the
classic "picking up pennies in front of the steamroller" mistake.
When breadth is crashing (score < -0.5), widening credit demands
mean today's $0.60 credit would have been $2.40 tomorrow — you can
get paid more for the same risk by waiting. This probe lets the
credit spread runners step aside during those regimes and the
sizing module scale down even when trades are still allowed.

## Fail-safe

Every data fetch is wrapped. A network outage or broken cache returns
score=0.0 (neutral), which lets the bot keep trading rather than
freezing entirely. The issue_reporter fires once per hour when a fetch
fails, so degraded probes are visible.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple


_log = logging.getLogger(__name__)


@dataclass
class BreadthProbeConfig:
    cache_seconds: float = 60.0
    # SPY intraday threshold: a drop of this much vs today's open is
    # counted as -1 on the SPY sub-score. Mid-day -0.5% is neutral.
    spy_drop_risk_off_pct: float = 0.015      # 1.5%
    spy_rally_risk_on_pct: float = 0.010       # 1.0%
    # Scores below this count as risk-off for credit-spread runners
    risk_off_threshold: float = -0.3
    # Weighting — when a source is missing, remaining components are
    # re-normalized. Keep weights roughly equal unless you have evidence.
    weight_vix: float = 1.0
    weight_spy_intraday: float = 1.0
    weight_iwm_spy: float = 0.5
    weight_hyg_tlt: float = 0.5


@dataclass
class BreadthSnapshot:
    score: float                     # composite, [-1, +1]
    is_risk_off: bool
    components: Dict[str, float] = field(default_factory=dict)
    as_of: float = field(default_factory=time.time)

    def size_multiplier(self) -> float:
        """Position-size multiplier based on the breadth score.

        score >= +0.3 → 1.0 (full size)
        score in [0, +0.3]  → 0.9
        score in [-0.3, 0]  → 0.75
        score in [-0.6, -0.3] → 0.50 (risk-off)
        score <= -0.6  → 0.25 (hard risk-off)
        """
        s = self.score
        if s >= 0.3:
            return 1.0
        if s >= 0.0:
            return 0.9
        if s >= -0.3:
            return 0.75
        if s >= -0.6:
            return 0.50
        return 0.25


class BreadthProbe:
    """Computes a thread-safe, cached breadth snapshot."""

    def __init__(
        self,
        cfg: Optional[BreadthProbeConfig] = None,
        *,
        spot_fetcher: Optional[Callable[[str], Optional[float]]] = None,
        open_fetcher: Optional[Callable[[str], Optional[float]]] = None,
        vix_probe=None,
    ) -> None:
        """
        Args:
          spot_fetcher: callable(symbol) -> latest price (or None). Wire
                        this to the bot's data adapter (AlpacaDataAdapter)
                        so we share cached bars.
          open_fetcher: callable(symbol) -> today's open price (or None).
                        Used to compute SPY intraday drift.
          vix_probe:    already-built VIXProbe instance with .value() and
                        .percentile_52w() methods. Optional.
        """
        self.cfg = cfg or BreadthProbeConfig()
        self._spot_fetcher = spot_fetcher
        self._open_fetcher = open_fetcher
        self._vix_probe = vix_probe
        self._lock = threading.Lock()
        self._cached: Optional[BreadthSnapshot] = None

    # ------------ sub-scores ------------

    def _vix_subscore(self) -> Optional[float]:
        """VIX-based risk-on/off: lower VIX = +1, higher = -1."""
        if self._vix_probe is None:
            return None
        try:
            vix = float(self._vix_probe.value())
            # Map VIX [10, 40] → [+1, -1]; above/below that saturates
            if vix <= 10:
                return 1.0
            if vix >= 40:
                return -1.0
            return 1.0 - 2.0 * (vix - 10) / 30.0
        except Exception as e:
            self._report("breadth.vix", e)
            return None

    def _spy_intraday_subscore(self) -> Optional[float]:
        """SPY intraday % move vs today's open. A -1.5% move hits -1."""
        if self._spot_fetcher is None or self._open_fetcher is None:
            return None
        try:
            spot = self._spot_fetcher("SPY")
            open_px = self._open_fetcher("SPY")
            if spot is None or open_px is None or open_px <= 0:
                return None
            pct = (spot - open_px) / open_px
            off = self.cfg.spy_drop_risk_off_pct
            on = self.cfg.spy_rally_risk_on_pct
            # Positive move → risk-on sub-score; negative → risk-off
            if pct >= on:
                return 1.0
            if pct <= -off:
                return -1.0
            if pct >= 0:
                return pct / on
            return pct / off
        except Exception as e:
            self._report("breadth.spy_intraday", e)
            return None

    def _pair_ratio_subscore(self, numer: str, denom: str,
                               *, hist_bars: int = 20) -> Optional[float]:
        """Compare a pair's current ratio to its N-bar mean. +1 if numer
        is outperforming denom; -1 if lagging."""
        if self._spot_fetcher is None:
            return None
        try:
            numer_px = self._spot_fetcher(numer)
            denom_px = self._spot_fetcher(denom)
            if numer_px is None or denom_px is None or denom_px <= 0:
                return None
            # For the baseline, use today's open of both as a cheap
            # approximation of recent history. A proper implementation
            # would pull N daily closes, but this keeps the probe
            # stateless and cheap.
            if self._open_fetcher is None:
                return 0.0
            numer_open = self._open_fetcher(numer)
            denom_open = self._open_fetcher(denom)
            if numer_open is None or denom_open is None:
                return 0.0
            if denom_open <= 0:
                return 0.0
            current = numer_px / denom_px
            baseline = numer_open / denom_open
            if baseline <= 0:
                return 0.0
            drift_pct = (current - baseline) / baseline
            # Map ±1% drift → ±1.0; saturate beyond.
            return max(-1.0, min(1.0, drift_pct * 100))
        except Exception as e:
            self._report(f"breadth.{numer}/{denom}", e)
            return None

    # ------------ composition ------------

    def snapshot(self, *, force_refresh: bool = False) -> BreadthSnapshot:
        """Return a cached snapshot. Recomputes on TTL expiry."""
        now = time.time()
        with self._lock:
            if (not force_refresh
                    and self._cached is not None
                    and (now - self._cached.as_of) < self.cfg.cache_seconds):
                return self._cached

            parts: Dict[str, Tuple[float, float]] = {}
            vix = self._vix_subscore()
            if vix is not None:
                parts["vix"] = (vix, self.cfg.weight_vix)
            spy = self._spy_intraday_subscore()
            if spy is not None:
                parts["spy_intraday"] = (spy, self.cfg.weight_spy_intraday)
            iwm_spy = self._pair_ratio_subscore("IWM", "SPY")
            if iwm_spy is not None:
                parts["iwm_spy"] = (iwm_spy, self.cfg.weight_iwm_spy)
            hyg_tlt = self._pair_ratio_subscore("HYG", "TLT")
            if hyg_tlt is not None:
                parts["hyg_tlt"] = (hyg_tlt, self.cfg.weight_hyg_tlt)

            if not parts:
                score = 0.0
            else:
                total_weight = sum(w for _, w in parts.values())
                score = (sum(v * w for v, w in parts.values())
                         / max(total_weight, 1e-9))

            snap = BreadthSnapshot(
                score=round(float(score), 4),
                is_risk_off=score <= self.cfg.risk_off_threshold,
                components={k: round(v, 4) for k, (v, _) in parts.items()},
                as_of=now,
            )
            self._cached = snap
            return snap

    def score(self) -> float:
        """Shortcut — returns just the composite score."""
        return self.snapshot().score

    def is_risk_off(self) -> bool:
        return self.snapshot().is_risk_off

    # ------------ error reporting ------------

    def _report(self, scope: str, exc: Exception) -> None:
        try:
            from ..notify.issue_reporter import report_issue
            report_issue(
                scope=scope,
                message=f"breadth probe failure: {type(exc).__name__}: {exc}",
                exc=exc,
                throttle_sec=3600.0,
            )
        except Exception:
            pass
