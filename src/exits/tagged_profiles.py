"""Tagged exit profiles: each entry tag has specific exit rules.

- zone_breach: spot >3% OTM + held >=15 min + P&L < -15%  → close
- vix_protection: VIX > 25  → close
- 0dte_time_stop: DTE=0 + time >=3:50 PM  → close
- theta_decay: DTE <= 1 + afternoon + flat P&L  → close
- scalp: 20% stop, 1 PM time stop, 60-min TTL
- vwap_reversion: mean-reversion + VWAP reverted  → close
- directional_momentum: 3 PM time stop if held > 2 hours
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional

from ..core.types import Position, ExitDecision


class TaggedProfileEvaluator:
    def __init__(self, vix_protection_threshold: float = 25.0):
        self.vix_threshold = vix_protection_threshold

    def evaluate(self, pos: Position, current_price: float, now: datetime,
                 vix: float, spot: float, vwap: float,
                 flat_pnl_abs_threshold: float = 0.05) -> Optional[ExitDecision]:
        tag = pos.entry_tags.get("tag") if pos.entry_tags else None
        dte = pos.dte()
        minutes_held = (now.timestamp() - pos.entry_ts) / 60.0
        pnl = pos.unrealized_pnl_pct(current_price)

        if vix > self.vix_threshold:
            return ExitDecision(True, f"vix_protection:{vix:.1f}", layer=2)

        if dte == 0 and now.time() >= time(15, 50):
            return ExitDecision(True, "0dte_time_stop", layer=2)

        if dte <= 1 and now.time() >= time(14, 0) and abs(pnl) < flat_pnl_abs_threshold:
            return ExitDecision(True, f"theta_decay_flat_pnl:{pnl:.2%}", layer=2)

        if tag == "scalp":
            if pnl <= -0.20:
                return ExitDecision(True, "scalp_stop", layer=2)
            if now.time() >= time(13, 0):
                return ExitDecision(True, "scalp_time_stop", layer=2)
            if minutes_held >= 60:
                return ExitDecision(True, "scalp_ttl", layer=2)

        if tag == "vwap_reversion" and vwap > 0 and spot > 0:
            # if price has reverted back within 10 bps of VWAP, take the trade off
            diff = abs(spot - vwap) / vwap
            if diff < 0.001:
                return ExitDecision(True, "vwap_reverted", layer=2)

        if tag == "directional_momentum":
            if now.time() >= time(15, 0) and minutes_held >= 120:
                return ExitDecision(True, "momo_3pm_time_stop", layer=2)

        # zone_breach: proxy check on held time + P&L + OTM distance using strike
        if pos.is_option and pos.strike is not None and pos.right is not None:
            if minutes_held >= 15 and pnl < -0.15:
                if pos.right.value == "call" and pos.strike > 0:
                    otm_dist = (pos.strike - spot) / spot if spot > 0 else 0
                else:
                    otm_dist = (spot - pos.strike) / spot if spot > 0 else 0
                if otm_dist > 0.03:
                    return ExitDecision(True, "zone_breach", layer=2)

        return None
