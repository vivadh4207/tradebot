"""Locks in the liquid-strike picker behavior surfaced in the real
session audit: the bot was picking illiquid far-OTM strikes because
`find_atm` only considered distance to spot, ignoring OI + today_volume.

The new `find_atm_liquid` must:
  - Prefer liquid strikes (OI >= min, volume >= min, non-zero bid/ask)
  - Stay within max_strike_dist_pct of spot
  - Return nearest liquid contract among those that pass
  - Fall back to find_atm only when no liquid contract exists (caller
    is expected to detect this via the same filter and skip the entry)
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.core.types import OptionContract, OptionRight
from src.data.options_chain import SyntheticOptionsChain


def _mk(symbol: str, strike: float, right: OptionRight, *,
         oi: int = 1000, vol: int = 200,
         bid: float = 0.10, ask: float = 0.12) -> OptionContract:
    return OptionContract(
        symbol=symbol, underlying="TEST",
        strike=strike, expiry=date.today() + timedelta(days=1),
        right=right, multiplier=100,
        open_interest=oi, today_volume=vol,
        bid=bid, ask=ask, last=(bid + ask) / 2,
        iv=0.30,
    )


def test_picks_closest_among_liquid_only():
    """3 calls within 5% of spot; only the farther 2 are liquid. The
    nearest-liquid one must win, not the illiquid closest-to-spot."""
    spot = 100.0
    chain = [
        _mk("T_100", 100.0, OptionRight.CALL, oi=0, vol=0),          # ILLIQUID, closest
        _mk("T_102", 102.0, OptionRight.CALL, oi=800, vol=150),      # liquid, farther
        _mk("T_104", 104.0, OptionRight.CALL, oi=900, vol=200),      # liquid, farthest
    ]
    picked = SyntheticOptionsChain.find_atm_liquid(
        chain, spot, OptionRight.CALL, min_oi=500, min_today_volume=100,
        max_strike_dist_pct=0.05,
    )
    assert picked is not None
    assert picked.strike == 102.0  # nearest LIQUID, skipping 100


def test_enforces_strike_distance_cap():
    """A liquid strike 10% from spot should NOT be picked when a 3%
    liquid strike exists."""
    spot = 100.0
    chain = [
        _mk("T_110", 110.0, OptionRight.CALL, oi=5000, vol=2000),    # liquid but >5%
        _mk("T_103", 103.0, OptionRight.CALL, oi=600, vol=120),      # liquid, close
    ]
    picked = SyntheticOptionsChain.find_atm_liquid(
        chain, spot, OptionRight.CALL, min_oi=500, min_today_volume=100,
        max_strike_dist_pct=0.05,
    )
    assert picked.strike == 103.0


def test_zero_bid_or_ask_disqualifies():
    """Contracts with bid=0 or ask=0 are stale / no market — can't be
    traded. Must be excluded from the liquid set."""
    spot = 100.0
    chain = [
        _mk("T_100", 100.0, OptionRight.CALL, oi=5000, vol=1000, bid=0.0, ask=0.0),
        _mk("T_102", 102.0, OptionRight.CALL, oi=5000, vol=1000, bid=0.10, ask=0.12),
    ]
    picked = SyntheticOptionsChain.find_atm_liquid(
        chain, spot, OptionRight.CALL, min_oi=500, min_today_volume=100,
        max_strike_dist_pct=0.05,
    )
    assert picked.strike == 102.0


def test_falls_back_to_nearest_when_no_liquid_exists():
    """When NOTHING in the chain meets the liquidity bar, fall back to
    `find_atm` nearest-to-spot. Caller is expected to detect this via
    the same OI/vol check and skip the entry."""
    spot = 100.0
    chain = [
        _mk("T_105", 105.0, OptionRight.CALL, oi=50, vol=5),   # illiquid
        _mk("T_100", 100.0, OptionRight.CALL, oi=50, vol=5),   # illiquid
        _mk("T_110", 110.0, OptionRight.CALL, oi=50, vol=5),   # illiquid
    ]
    picked = SyntheticOptionsChain.find_atm_liquid(
        chain, spot, OptionRight.CALL, min_oi=500, min_today_volume=100,
    )
    # Fell back to nearest — caller must still reject because liquidity is low.
    assert picked is not None
    assert picked.strike == 100.0
    # Proof the caller would reject (OI < threshold):
    assert picked.open_interest < 500


def test_empty_chain_returns_none():
    picked = SyntheticOptionsChain.find_atm_liquid(
        [], 100.0, OptionRight.CALL,
    )
    assert picked is None


def test_tier2_picks_quote_only_when_oi_and_vol_unknown():
    """Critical real-session bug: Alpaca's snapshot endpoint doesn't
    populate OI/volume → every contract returns with oi=0 vol=0. Without
    tier-2 fallback, find_atm_liquid would return nothing usable and
    the bot would refuse every trade despite valid bid/ask being visible."""
    spot = 100.0
    chain = [
        # nearest but no bid/ask — unusable
        _mk("T_100", 100.0, OptionRight.CALL, oi=0, vol=0, bid=0, ask=0),
        # farther, OI/vol=0 (unknown), real bid/ask — TIER 2 wins
        _mk("T_102", 102.0, OptionRight.CALL, oi=0, vol=0, bid=0.50, ask=0.55),
    ]
    picked = SyntheticOptionsChain.find_atm_liquid(
        chain, spot, OptionRight.CALL, min_oi=500, min_today_volume=100,
        max_strike_dist_pct=0.05,
    )
    assert picked is not None
    assert picked.strike == 102.0
    assert picked.bid > 0 and picked.ask > 0


def test_tier1_still_beats_tier2_when_both_available():
    """If one contract has REAL OI/volume data, pick it even over a
    closer contract that has unknown (0/0) data."""
    spot = 100.0
    chain = [
        _mk("T_101", 101.0, OptionRight.CALL, oi=0, vol=0, bid=0.30, ask=0.35),
        _mk("T_103", 103.0, OptionRight.CALL, oi=5000, vol=1000, bid=0.20, ask=0.25),
    ]
    picked = SyntheticOptionsChain.find_atm_liquid(
        chain, spot, OptionRight.CALL, min_oi=500, min_today_volume=100,
        max_strike_dist_pct=0.05,
    )
    assert picked.strike == 103.0  # tier-1 wins, closer tier-2 loses


def test_only_considers_correct_right():
    spot = 100.0
    chain = [
        _mk("T_C100", 100.0, OptionRight.CALL, oi=5000, vol=1000),
        _mk("T_P100", 100.0, OptionRight.PUT, oi=5000, vol=1000),
        _mk("T_P99",   99.0, OptionRight.PUT, oi=5000, vol=1000),
    ]
    put = SyntheticOptionsChain.find_atm_liquid(chain, spot, OptionRight.PUT)
    assert put is not None
    assert put.right == OptionRight.PUT
    assert put.strike == 100.0
