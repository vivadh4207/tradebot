"""PositionSizer — wraps the Hybrid Kelly+VIX model for contract sizing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
                 max_0dte: int = 5, max_multiday: int = 10):
        self.kelly_f = kelly_fraction_cap
        self.kelly_hc = kelly_hard_cap
        self.max_0dte = max_0dte
        self.max_multiday = max_multiday

    def contracts(self, inp: SizingInputs) -> int:
        # Max loss per contract:
        #   LONG option:  premium × 100
        #   SHORT option: naked = strike × 100 - credit (approx strike × 100)
        # For short defined-risk spreads, callers should pass a premium_risk-style
        # contract.ask equal to the net debit of the spread.
        premium_risk = max(inp.contract.ask, 0.01) * 100
        if inp.is_long:
            max_loss = premium_risk
        else:
            strike_risk = inp.contract.strike * 100
            max_loss = max(strike_risk, premium_risk)
        max_c = self.max_0dte if inp.is_0dte else self.max_multiday
        return hybrid_sizing(
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
