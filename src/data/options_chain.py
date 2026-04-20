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


# Real-world strike increments (in dollars) for liquid ETFs / indices.
# Adding a symbol here makes the synthetic chain generate strikes that
# match what's actually listed at CBOE — otherwise Alpaca rejects with
# "asset not found" when the bot tries to submit a fractional strike.
_FIXED_STRIKE_STEP: dict = {
    "SPY": 1.0,
    "QQQ": 1.0,
    "IWM": 1.0,
    "DIA": 1.0,
    "XLF": 0.5,
    "XLE": 0.5,
    "XLK": 1.0,
    "TLT": 1.0,
    "GLD": 1.0,
    "SLV": 0.5,
}


def _strike_step_for(underlying: str, spot: float,
                      default_pct: float) -> float:
    """Pick a realistic strike increment. Liquid ETFs use fixed $1
    (or $0.50) increments regardless of spot. Others fall back to a
    percentage of spot, clamped to at least $0.50 and rounded to a
    clean $0.50 multiple so the resulting strikes are plausible."""
    fixed = _FIXED_STRIKE_STEP.get(underlying.upper())
    if fixed is not None:
        return float(fixed)
    # Stocks: derive step from spot but quantize to $0.50 / $1 / $2.50
    # grid so we never produce fractional strikes like $87.37.
    raw = max(spot * default_pct, 0.5)
    if raw <= 0.75:
        return 0.5
    if raw <= 1.5:
        return 1.0
    if raw <= 3.5:
        return 2.5
    return 5.0


def _snap_strike(raw: float, step: float) -> float:
    """Snap a raw strike to the nearest valid step on the real-world
    grid (anchored at 0, i.e. strikes at k*step)."""
    return round(round(raw / step) * step, 2)


class SyntheticOptionsChain(OptionsChainProvider):
    def __init__(self, base_iv: float = 0.22, r: float = 0.045, q: float = 0.015,
                 strike_step_pct: float = 0.01, strikes_each_side: int = 10):
        self.base_iv = base_iv
        self.r = r
        self.q = q
        # strike_step_pct is now a FALLBACK — liquid names use the
        # fixed-step table in _strike_step_for.
        self.strike_step_pct = strike_step_pct
        self.strikes_each_side = strikes_each_side

    def chain(self, underlying: str, spot: float, *,
              target_dte: int = 1) -> List[OptionContract]:
        today = date.today()
        expiry = today + timedelta(days=max(1, target_dte))
        T = max(target_dte, 0) / 365.0 or 1 / 365.0

        contracts: List[OptionContract] = []
        step = _strike_step_for(underlying, spot, self.strike_step_pct)
        # Anchor the chain at the nearest-valid-strike to spot so every
        # generated K is on the real-world grid (e.g. $708, $709, $710).
        atm = _snap_strike(spot, step)
        seen: set = set()
        for i in range(-self.strikes_each_side, self.strikes_each_side + 1):
            K = _snap_strike(atm + i * step, step)
            if K <= 0 or K in seen:
                continue
            seen.add(K)
            log_m = np.log(K / spot)
            # simple smile: puts more expensive than calls, OTM wings lifted
            iv = self.base_iv + 0.05 * (log_m * log_m) + (-0.02 if log_m < 0 else 0.0)
            for right in (OptionRight.CALL, OptionRight.PUT):
                rt = right.value
                mid = bs_price(spot, K, T, self.r, iv, self.q, rt)
                spread = max(0.02, mid * 0.04)
                bid = max(0.01, mid - spread / 2)
                ask = mid + spread / 2
                occ = f"{underlying}{expiry.strftime('%y%m%d')}{rt[0].upper()}{int(round(K*1000)):08d}"
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
