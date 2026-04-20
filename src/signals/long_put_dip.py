"""Long-put-dip signal — explicit "profit from the fall" trigger.

The bot historically had no signal source that explicitly said:
  "price just dropped hard below VWAP on elevated RSI-oversold + VIX
   expanding → buy puts to ride the continuation."

The classical momentum/ORB/candle-pattern stack catches breakouts and
reversals but doesn't specifically flag the high-edge setup an
operator watches for: a fear-spike dip with breadth confirming.

Triggers (ALL must hold):
  - Price < VWAP by `vwap_dip_pct` (default 0.4%)
  - RSI(5) <= `rsi_ceiling` (default 35; oversold)
  - EITHER VIX change > `vix_spike_pct` OR
    breadth decliners > `breadth_decliners_mult` × advancers
  - Volume on current bar >= `min_vol_ratio` × 20-bar avg (participation)

Emits BUY PUT at delta ~0.45, ~7-14 DTE (sizing handled by portfolio
layer). Confidence scaled by:
  - Depth of dip vs VWAP (bigger miss = higher conf)
  - RSI depth (lower = higher conf, capped)
  - VIX spike magnitude

Rationale lives in `signal.rationale` so the 8B brain's reviewer
can see exactly why the signal fired when it audits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.types import Signal, Side, OptionRight
from .base import SignalSource, SignalContext
from .technical_analysis import rsi as _rsi


@dataclass
class LongPutDipConfig:
    vwap_dip_pct: float = 0.004          # price below VWAP by 0.4%
    rsi_ceiling: float = 35.0             # 5-bar RSI below this
    rsi_period: int = 5
    vix_spike_pct: float = 0.05          # VIX up 5%+ on the day
    breadth_decliners_mult: float = 2.0   # decliners > 2× advancers
    min_vol_ratio: float = 1.1            # ~at-avg volume or higher
    min_bars: int = 40
    max_confidence: float = 0.92


class LongPutDipSignal(SignalSource):
    """Explicit dip-buy-put setup. Looks for panic-selling with
    participation + macro-fear confirmation."""

    name = "long_put_dip"

    def __init__(
        self,
        cfg: Optional[LongPutDipConfig] = None,
        *,
        get_vix_fn=None,          # callable() -> dict | None
        get_breadth_fn=None,      # callable() -> dict | None
    ):
        self.cfg = cfg or LongPutDipConfig()
        self._get_vix = get_vix_fn
        self._get_breadth = get_breadth_fn

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        cfg = self.cfg
        if len(ctx.bars) < cfg.min_bars:
            return None

        closes = np.asarray([b.close for b in ctx.bars], dtype=float)
        volumes = np.asarray([b.volume for b in ctx.bars], dtype=float)

        # 1. VWAP dip check
        if ctx.vwap <= 0:
            return None
        dip = (ctx.vwap - closes[-1]) / ctx.vwap
        if dip < cfg.vwap_dip_pct:
            return None

        # 2. RSI oversold
        rsi_vals = _rsi(closes, cfg.rsi_period)
        if np.isnan(rsi_vals[-1]):
            return None
        rsi_now = float(rsi_vals[-1])
        if rsi_now > cfg.rsi_ceiling:
            return None

        # 3. Macro confirmation (VIX spike OR breadth deterioration).
        # Both optional: if neither data source is wired, fall back to
        # requiring a stronger price read (deeper dip + deeper RSI).
        vix_change_pct = None
        decliners_over_advancers = None
        if self._get_vix is not None:
            try:
                v = self._get_vix() or {}
                vix_change_pct = v.get("change_pct")
            except Exception:
                pass
        if self._get_breadth is not None:
            try:
                b = self._get_breadth() or {}
                adv = float(b.get("advancers", 0))
                dec = float(b.get("decliners", 0))
                if adv > 0:
                    decliners_over_advancers = dec / adv
            except Exception:
                pass

        macro_ok = False
        macro_tag = ""
        if vix_change_pct is not None and vix_change_pct >= cfg.vix_spike_pct:
            macro_ok = True
            macro_tag = f"vix+{vix_change_pct*100:.1f}%"
        elif (decliners_over_advancers is not None
                and decliners_over_advancers >= cfg.breadth_decliners_mult):
            macro_ok = True
            macro_tag = f"dec/adv={decliners_over_advancers:.1f}"
        else:
            # No macro confirm — require a deeper stand-alone trigger.
            if dip >= 2 * cfg.vwap_dip_pct and rsi_now <= (cfg.rsi_ceiling - 5):
                macro_ok = True
                macro_tag = "price-only"
        if not macro_ok:
            return None

        # 4. Participation: current bar volume vs 20-bar avg.
        window = min(20, len(volumes) - 1)
        if window < 5:
            return None
        avg_vol = volumes[-(window + 1):-1].mean()
        vol_ratio = volumes[-1] / max(1e-9, avg_vol)
        if vol_ratio < cfg.min_vol_ratio:
            return None

        # Confidence shaping — stack the dip depth, rsi depth, vol, macro.
        conf = 0.60
        conf += min(0.15, (dip / cfg.vwap_dip_pct - 1.0) * 0.05)
        conf += min(0.10, (cfg.rsi_ceiling - rsi_now) / 100.0)
        conf += min(0.08, (vol_ratio - 1.0) * 0.08)
        if macro_tag.startswith("vix") or macro_tag.startswith("dec"):
            conf += 0.05
        conf = min(cfg.max_confidence, max(0.55, conf))

        return Signal(
            source=self.name,
            symbol=ctx.symbol,
            side=Side.BUY,
            option_right=OptionRight.PUT,
            confidence=conf,
            rationale=(
                f"dip={dip*100:.2f}% vs vwap · rsi{cfg.rsi_period}={rsi_now:.1f} "
                f"· vol×{vol_ratio:.2f} · {macro_tag}"
            ),
            meta={
                "direction": "bearish",
                "setup": "dip_buy_put",
                "dip_pct": round(dip, 5),
                "rsi": round(rsi_now, 2),
                "vol_ratio": round(vol_ratio, 2),
                "macro_tag": macro_tag,
            },
        )
