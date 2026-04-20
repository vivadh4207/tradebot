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
    def find_atm_liquid(contracts: List[OptionContract], spot: float,
                         right: OptionRight,
                         min_oi: int = 500,
                         min_today_volume: int = 100,
                         max_strike_dist_pct: float = 0.05) -> Optional[OptionContract]:
        """Like find_atm but filters for real tradable liquidity.

        Three tiers:
          (1) Fully-liquid: OI >= min_oi AND volume >= min_today_volume
              AND non-zero bid/ask AND strike within max_strike_dist_pct.
          (2) Quote-only: bid > 0 AND ask > 0 AND strike within distance,
              BUT OI and volume are BOTH zero (provider didn't report).
              Treated as "liquidity unknown" — pickable when no tier-1
              exists. Common with Alpaca's snapshot endpoint which
              doesn't populate OI reliably.
          (3) Nearest: whatever the chain's find_atm returns. Caller
              should treat this as advisory-only and typically skip.

        The tier-2 fallback is what makes the bot actually trade when
        Alpaca's OI data is missing — otherwise we'd reject every single
        contract despite a valid bid/ask being visible.
        """
        def _within(c):
            return abs(c.strike - spot) / max(spot, 1e-9) <= max_strike_dist_pct
        tier1 = [
            c for c in contracts
            if c.right == right
            and c.open_interest >= min_oi
            and c.today_volume >= min_today_volume
            and _within(c)
            and c.bid > 0 and c.ask > 0
        ]
        if tier1:
            tier1.sort(key=lambda c: abs(c.strike - spot))
            return tier1[0]
        # Tier 2: has a real quote (both sides > 0), within distance,
        # OI/volume unknown (all zeros). Better than blind-nearest.
        tier2 = [
            c for c in contracts
            if c.right == right
            and _within(c)
            and c.bid > 0 and c.ask > 0
            and c.open_interest == 0 and c.today_volume == 0
        ]
        if tier2:
            tier2.sort(key=lambda c: abs(c.strike - spot))
            return tier2[0]
        # Tier 3: nothing usable — let find_atm return raw nearest so
        # callers can log what they got and bail.
        return SyntheticOptionsChain.find_atm(contracts, spot, right)

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
