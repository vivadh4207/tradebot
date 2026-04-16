"""6-layer exit engine orchestrator.

Layer 1: 0DTE force close
Layer 2: Tagged exit profiles (scalp, vwap_reversion, zone_breach, vix_protection, ...)
Layer 3: Auto stop prices (computed at entry)
Layer 4: Global profit target + (optional) Claude AI hold check
Layer 5: Global stop loss
Layer 6: EOD sweep (3:45 PM ET)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Optional, Dict, Callable

from ..core.types import Position, ExitDecision, Bar
from .fast_exit import FastExitEvaluator, FastExitConfig
from .tagged_profiles import TaggedProfileEvaluator
from .momentum_boost import MomentumBoost, BoostConfig


@dataclass
class ExitEngineConfig:
    pt_short_pct: float = 0.35
    pt_multi_pct: float = 0.50
    sl_short_pct: float = 0.20
    sl_multi_pct: float = 0.30
    eod_force_close_time: time = time(15, 45)
    zero_dte_force_close_time: time = time(15, 45)
    hard_profit_cap_pct: float = 1.50
    max_consecutive_holds: int = 3
    claude_hold_conf_min: float = 0.70


class ExitEngine:
    def __init__(self, cfg: ExitEngineConfig = ExitEngineConfig(),
                 fast_cfg: Optional[FastExitConfig] = None,
                 boost_cfg: Optional[BoostConfig] = None,
                 claude_hold_hook: Optional[Callable[[Position, Dict], Dict]] = None):
        self.cfg = cfg
        self.fast = FastExitEvaluator(fast_cfg or FastExitConfig(
            pt_short_pct=cfg.pt_short_pct, pt_multi_pct=cfg.pt_multi_pct,
            sl_short_pct=cfg.sl_short_pct, sl_multi_pct=cfg.sl_multi_pct,
        ))
        self.tagged = TaggedProfileEvaluator()
        self.boost = MomentumBoost(boost_cfg or BoostConfig(
            initial_target_pct=cfg.pt_short_pct,
            boosted_target_pct=cfg.pt_multi_pct + 0.10,
            hard_cap_pct=cfg.hard_profit_cap_pct,
        ))
        self.claude_hold = claude_hold_hook

    def decide(self, pos: Position, current_price: float, now: datetime,
               vix: float, spot: float, vwap: float,
               bars: List[Bar]) -> ExitDecision:
        # Layer 1: 0DTE force close
        if pos.dte() == 0 and now.time() >= self.cfg.zero_dte_force_close_time:
            return ExitDecision(True, "layer1_0dte_force_close", layer=1)

        # Layer 2: Tagged profiles (also handles vix_protection, scalp etc.)
        tagged = self.tagged.evaluate(pos, current_price, now, vix, spot, vwap)
        if tagged and tagged.should_close:
            return tagged

        # Layer 3: Auto stops (set at entry on Position.auto_profit_target / auto_stop_loss)
        if pos.auto_profit_target is not None and pos.auto_stop_loss is not None:
            if pos.is_long:
                if current_price >= pos.auto_profit_target:
                    # Allow Claude hold check before closing (handled at layer 4)
                    pass
                if current_price <= pos.auto_stop_loss:
                    return ExitDecision(True, "layer3_auto_stop", layer=3)
            else:
                if current_price <= pos.auto_profit_target:
                    pass
                if current_price >= pos.auto_stop_loss:
                    return ExitDecision(True, "layer3_auto_stop", layer=3)

        # Layer 4: Global profit target (with momentum boost + optional Claude hold check)
        pnl = pos.unrealized_pnl_pct(current_price)
        is_short_dte = pos.dte() <= 1
        base_pt = self.cfg.pt_short_pct if is_short_dte else self.cfg.pt_multi_pct
        boosted_pt = self.boost.evaluate(pos, bars)
        effective_pt = max(base_pt, boosted_pt)
        if pnl >= effective_pt:
            if pnl >= self.cfg.hard_profit_cap_pct:
                return ExitDecision(True, f"layer4_hard_cap:{pnl:.2%}", layer=4)
            # EOD cutoff: don't hold into the close
            if now.time() >= time(15, 30):
                return ExitDecision(True, f"layer4_target_cutoff:{pnl:.2%}", layer=4)
            # Consider Claude hold vote
            if self.claude_hold and pos.consecutive_holds < self.cfg.max_consecutive_holds:
                vote = self.claude_hold(pos, {"pnl": pnl, "pt": effective_pt})
                if (vote and vote.get("decision") == "hold"
                        and float(vote.get("confidence", 0)) >= self.cfg.claude_hold_conf_min):
                    pos.consecutive_holds += 1
                    return ExitDecision(False, f"layer4_claude_hold conf={vote.get('confidence')}",
                                        layer=4, allow_hold=True)
            return ExitDecision(True, f"layer4_target_hit:{pnl:.2%}", layer=4)

        # Layer 5: Global stop loss
        sl = self.cfg.sl_short_pct if is_short_dte else self.cfg.sl_multi_pct
        if pnl <= -sl:
            return ExitDecision(True, f"layer5_global_stop:{pnl:.2%}", layer=5)

        # Layer 6: EOD sweep
        if now.time() >= self.cfg.eod_force_close_time:
            return ExitDecision(True, "layer6_eod_sweep", layer=6)

        return ExitDecision(False, "no_exit", layer=0)
