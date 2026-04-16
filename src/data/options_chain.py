"""Options-chain provider.

SyntheticOptionsChain: builds a plausible chain around spot for backtests.
For live, wire an Alpaca/Tradier/Polygon chain provider behind this interface.
"""
from __future__ import annotations

import abc
from datetime import date, timedelta
from typing import List, Optional

import numpy as np

from ..core.types import OptionContract, OptionRight
from ..math_tools.pricer import bs_price


class OptionsChainProvider(abc.ABC):
    @abc.abstractmethod
    def chain(self, underlying: str, spot: float, *,
              target_dte: int = 1) -> List[OptionContract]: ...


class SyntheticOptionsChain(OptionsChainProvider):
    def __init__(self, base_iv: float = 0.22, r: float = 0.045, q: float = 0.015,
                 strike_step_pct: float = 0.01, strikes_each_side: int = 10):
        self.base_iv = base_iv
        self.r = r
        self.q = q
        self.strike_step_pct = strike_step_pct
        self.strikes_each_side = strikes_each_side

    def chain(self, underlying: str, spot: float, *,
              target_dte: int = 1) -> List[OptionContract]:
        today = date.today()
        expiry = today + timedelta(days=max(1, target_dte))
        T = max(target_dte, 0) / 365.0 or 1 / 365.0

        contracts: List[OptionContract] = []
        step = max(spot * self.strike_step_pct, 0.5)
        for i in range(-self.strikes_each_side, self.strikes_each_side + 1):
            K = round(spot + i * step, 2)
            log_m = np.log(K / spot)
            # simple smile: puts more expensive than calls, OTM wings lifted
            iv = self.base_iv + 0.05 * (log_m * log_m) + (-0.02 if log_m < 0 else 0.0)
            for right in (OptionRight.CALL, OptionRight.PUT):
                rt = right.value
                mid = bs_price(spot, K, T, self.r, iv, self.q, rt)
                spread = max(0.02, mid * 0.04)
                bid = max(0.01, mid - spread / 2)
                ask = mid + spread / 2
                occ = f"{underlying}{expiry.strftime('%y%m%d')}{rt[0].upper()}{int(K*1000):08d}"
                contracts.append(OptionContract(
                    symbol=occ, underlying=underlying,
                    strike=K, expiry=expiry, right=right,
                    multiplier=100,
                    open_interest=max(200, int(5000 * np.exp(-abs(i) / 3))),
                    today_volume=max(50, int(500 * np.exp(-abs(i) / 3))),
                    bid=float(bid), ask=float(ask), last=float(mid),
                    iv=float(iv),
                ))
        return contracts

    @staticmethod
    def find_atm(contracts: List[OptionContract], spot: float,
                 right: OptionRight) -> Optional[OptionContract]:
        rights = [c for c in contracts if c.right == right]
        if not rights:
            return None
        rights.sort(key=lambda c: abs(c.strike - spot))
        return rights[0]

    @staticmethod
    def find_otm(contracts: List[OptionContract], spot: float,
                 right: OptionRight, otm_pct: float = 0.05) -> Optional[OptionContract]:
        rights = [c for c in contracts if c.right == right]
        if not rights:
            return None
        if right == OptionRight.CALL:
            target = spot * (1 + otm_pct)
        else:
            target = spot * (1 - otm_pct)
        rights.sort(key=lambda c: abs(c.strike - target))
        return rights[0]
