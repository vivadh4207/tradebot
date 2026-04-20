"""Extreme-momentum shock scanner.

Detects sudden, high-conviction moves that the standard momentum signal
(small-slope continuous drift) misses. Pattern:

  - Underlying has moved `min_move_pct` or more over the last
    `lookback_bars` bars (default: 3% over 5 bars)
  - Cumulative volume over that window is `min_volume_multiple`× the
    prior 20-bar average (default: 3×)
  - Optional: catalyst calendar shows a relevant event in the last
    `catalyst_within_minutes` (default: disabled)

Emits a high-confidence directional signal (CALL or PUT depending on
move direction). Intended to complement MomentumSignal, not replace
it — MomentumSignal catches slow drifts, this catches shocks.

Off by default. Enable in settings.yaml::signals.extreme_momentum.enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..core.types import Signal, Side, OptionRight, Bar
from .base import SignalSource, SignalContext


@dataclass
class ExtremeMomentumConfig:
    lookback_bars: int = 5            # recent bars for the move calc
    baseline_bars: int = 20           # bars for the volume baseline
    min_move_pct: float = 0.03        # 3% move in lookback window
    min_volume_multiple: float = 3.0  # 3× baseline volume
    # Confidence emitted when the pattern fires — high because the
    # setup is specific (shock move + volume). The ensemble weights
    # this against other signals; we don't want to dominate just
    # because we fire rarely. 0.85 puts us above MomentumSignal (0.75)
    # but below a hard-gate (1.0).
    confidence: float = 0.85


def compute_shock(
    bars: List[Bar],
    cfg: Optional[ExtremeMomentumConfig] = None,
) -> Optional[dict]:
    """Pure function — return a shock descriptor if the pattern is
    present in the bar sequence, else None. No signals emitted here;
    the SignalSource wraps this to produce Signal objects. Easier
    testing this way.

    Returns:
      {"direction": "bullish"|"bearish",
       "move_pct": float, "volume_multiple": float,
       "close": float, "start_close": float}
    """
    cfg = cfg or ExtremeMomentumConfig()
    need = cfg.lookback_bars + cfg.baseline_bars + 1
    if len(bars) < need:
        return None

    lb = cfg.lookback_bars
    window = bars[-lb:]
    start_close = bars[-(lb + 1)].close
    end_close = bars[-1].close
    if start_close <= 0:
        return None
    move_pct = (end_close - start_close) / start_close
    if abs(move_pct) < cfg.min_move_pct:
        return None

    # Volume multiple: recent sum vs prior baseline average per bar.
    window_vol = sum(b.volume for b in window)
    baseline = bars[-(lb + cfg.baseline_bars):-lb]
    if not baseline:
        return None
    baseline_avg_per_bar = sum(b.volume for b in baseline) / len(baseline)
    if baseline_avg_per_bar <= 0:
        return None
    # Compare recent-window AVERAGE per bar to baseline average per bar
    # (not sum to average — that would bias by window length).
    recent_avg_per_bar = window_vol / lb
    vol_mult = recent_avg_per_bar / baseline_avg_per_bar
    if vol_mult < cfg.min_volume_multiple:
        return None

    return {
        "direction": "bullish" if move_pct > 0 else "bearish",
        "move_pct": float(move_pct),
        "volume_multiple": float(vol_mult),
        "close": float(end_close),
        "start_close": float(start_close),
    }


class ExtremeMomentumSignal(SignalSource):
    """SignalSource wrapper around compute_shock()."""

    name = "extreme_momentum"

    def __init__(self, cfg: Optional[ExtremeMomentumConfig] = None):
        self.cfg = cfg or ExtremeMomentumConfig()

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        shock = compute_shock(ctx.bars, self.cfg)
        if shock is None:
            return None
        direction = shock["direction"]
        side = Side.BUY
        right = OptionRight.CALL if direction == "bullish" else OptionRight.PUT
        return Signal(
            source=self.name,
            symbol=ctx.symbol,
            side=side,
            option_right=right,
            confidence=self.cfg.confidence,
            rationale=(
                f"shock_move {shock['move_pct']:+.2%} "
                f"in {self.cfg.lookback_bars} bars "
                f"on {shock['volume_multiple']:.1f}× vol"
            ),
            meta={
                "direction": direction,
                "entry_tag": "extreme_momentum",
                "move_pct": round(shock["move_pct"], 4),
                "volume_multiple": round(shock["volume_multiple"], 2),
                "trigger_close": round(shock["close"], 4),
            },
        )
