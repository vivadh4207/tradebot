"""Put-call parity sanity check.

For European options (or approximately for American options on
non-dividend names near ATM):

    C - P ≈ S · exp(-qT) - K · exp(-rT)

A violation beyond a small tolerance usually means one of:
  - stale quote on one leg (call updated, put hasn't)
  - corporate action mispriced
  - measurement error (wide spreads, illiquid strikes)

Use this as a LIVE DATA QUALITY check — reject chains with parity
violations before using them to derive VRP / skew / IV rank.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from ..core.types import OptionContract, OptionRight


@dataclass
class ParityCheckResult:
    strike: float
    call_mid: float
    put_mid: float
    expected: float                       # S·e^{-qT} − K·e^{-rT}
    observed: float                       # call_mid − put_mid
    abs_violation: float                  # |observed − expected|
    ok: bool                              # |violation| < tolerance
    tolerance: float

    @property
    def violation_pct(self) -> float:
        if self.call_mid + self.put_mid <= 0:
            return 0.0
        return self.abs_violation / ((self.call_mid + self.put_mid) / 2.0)


def check_parity(call: OptionContract, put: OptionContract,
                  spot: float, r: float = 0.045, q: float = 0.015,
                  today=None, abs_tolerance: float = 0.05,
                  pct_tolerance: float = 0.10) -> ParityCheckResult:
    """Check parity for a single (call, put) pair at the same strike/expiry.

    Fails (ok=False) when the violation exceeds EITHER tolerance — absolute
    (cents) OR as a percent of the premium.
    """
    from datetime import date
    today = today or date.today()
    if call.strike != put.strike or call.expiry != put.expiry:
        raise ValueError("call and put must share strike + expiry")
    T = max((call.expiry - today).days, 0) / 365.0 or 1e-4

    call_mid = call.mid
    put_mid = put.mid
    expected = spot * math.exp(-q * T) - call.strike * math.exp(-r * T)
    observed = call_mid - put_mid
    abs_violation = abs(observed - expected)
    # Accept if within ABS tolerance OR if the percentage violation is small
    # (wide spreads on far-OTM wings naturally have larger abs errors).
    mid_avg = (call_mid + put_mid) / 2.0
    pct_v = abs_violation / mid_avg if mid_avg > 0 else 0.0
    ok = abs_violation <= abs_tolerance or pct_v <= pct_tolerance
    return ParityCheckResult(
        strike=call.strike, call_mid=call_mid, put_mid=put_mid,
        expected=expected, observed=observed,
        abs_violation=abs_violation, ok=ok, tolerance=abs_tolerance,
    )


def violations_in_chain(contracts: List[OptionContract], spot: float,
                         r: float = 0.045, q: float = 0.015,
                         abs_tolerance: float = 0.05,
                         pct_tolerance: float = 0.10,
                         today=None) -> List[ParityCheckResult]:
    """Pair up call+put at each (strike, expiry) and check all pairs.

    Returns ONLY the violations (ok=False). An empty list is a clean chain.
    Useful gate before running VRP / skew calculations.
    """
    # Index by (strike, expiry) → (call, put)
    by_key: dict = {}
    for c in contracts:
        key = (c.strike, c.expiry)
        rec = by_key.setdefault(key, {})
        if c.right == OptionRight.CALL:
            rec["call"] = c
        else:
            rec["put"] = c

    violations: List[ParityCheckResult] = []
    for (strike, expiry), rec in by_key.items():
        call = rec.get("call")
        put = rec.get("put")
        if call is None or put is None:
            continue
        # Skip wings with 0 bid — parity breaks down on illiquid contracts
        if call.bid <= 0 or put.bid <= 0:
            continue
        res = check_parity(call, put, spot, r=r, q=q, today=today,
                            abs_tolerance=abs_tolerance,
                            pct_tolerance=pct_tolerance)
        if not res.ok:
            violations.append(res)
    return violations
