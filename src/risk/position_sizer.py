"""PositionSizer — wraps the Hybrid Kelly+VIX model for contract sizing.

Optional regime-aware multiplier: shrink size in regimes with poor
realized win rate, grow it in regimes with proven edge. Multipliers are
clipped to [0, 2]; set to 0.0 to forbid entries in that regime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..core.types import OptionContract
from ..math_tools.sizing import hybrid_sizing


@dataclass
class SizingInputs:
    equity: float
    contract: OptionContract
    win_rate_est: float
    avg_win: float
    avg_loss: float
    vix_today: float
    vix_52w_low: float
    vix_52w_high: float
    vrp_zscore: float
    is_0dte: bool
    is_long: bool = True                # BUY side → long option → capped loss = premium


class PositionSizer:
    def __init__(self, kelly_fraction_cap: float = 0.25,
                 kelly_hard_cap: float = 0.05,
                 max_0dte: int = 5, max_multiday: int = 10,
                 regime_multipliers: Optional[Dict[str, float]] = None):
        self.kelly_f = kelly_fraction_cap
        self.kelly_hc = kelly_hard_cap
        self.max_0dte = max_0dte
        self.max_multiday = max_multiday
        # Regime name → multiplier on the final contract count. Clipped [0, 2].
        self.regime_multipliers: Dict[str, float] = {
            k: max(0.0, min(2.0, float(v)))
            for k, v in (regime_multipliers or {}).items()
        }

    def contracts(self, inp: SizingInputs, regime: Optional[str] = None) -> int:
        premium_risk = max(inp.contract.ask, 0.01) * 100
        if inp.is_long:
            max_loss = premium_risk
        else:
            strike_risk = inp.contract.strike * 100
            max_loss = max(strike_risk, premium_risk)
        max_c = self.max_0dte if inp.is_0dte else self.max_multiday
        n = hybrid_sizing(
            equity=inp.equity,
            max_loss_per_contract=max_loss,
            win_rate_est=inp.win_rate_est,
            avg_win=inp.avg_win,
            avg_loss=inp.avg_loss,
            vix_today=inp.vix_today,
            vix_52w_low=inp.vix_52w_low,
            vix_52w_high=inp.vix_52w_high,
            vrp_zscore=inp.vrp_zscore,
            kelly_fraction_cap=self.kelly_f,
            kelly_hard_cap=self.kelly_hc,
            max_contracts=max_c,
        )
        if regime and self.regime_multipliers:
            mult = self.regime_multipliers.get(regime, 1.0)
            n = int(n * mult)
        return max(0, min(n, max_c))
