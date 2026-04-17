"""AlpacaOptionsChain — live options chain from Alpaca.

Lazy-imports `alpaca-py` so the package imports cleanly without it. Falls
back to `SyntheticOptionsChain` on any failure so the trading loop never
dies because a chain call timed out.

Alpaca Options data requires an options-enabled paper account (free to
enable) and the `OptionHistoricalDataClient`.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List, Optional

from ..core.types import OptionContract, OptionRight
from .options_chain import OptionsChainProvider, SyntheticOptionsChain


class AlpacaOptionsChain(OptionsChainProvider):
    # US equity options conventionally expire Friday. Ultra-liquid ETFs
    # (SPY / QQQ / IWM / some mega-caps) also have Monday and Wednesday
    # weeklies. Anything OUTSIDE {Mon, Wed, Fri} on a normal week is either
    # bad data or a non-standard quarterly/end-of-month that we don't trade.
    _STANDARD_WEEKDAYS = {0, 2, 4}   # Mon, Wed, Fri

    def __init__(self, api_key: str, api_secret: str,
                 fallback: Optional[OptionsChainProvider] = None,
                 max_strikes_each_side: int = 10,
                 standard_cycles_only: bool = True):
        self._api_key = api_key
        self._api_secret = api_secret
        self._fallback = fallback or SyntheticOptionsChain()
        self._max_strikes = max_strikes_each_side
        self._standard_cycles_only = bool(standard_cycles_only)
        self._client = None
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            self._client = OptionHistoricalDataClient(api_key, api_secret)
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "alpaca_options_client_init_failed: %s", _e
            )
            self._client = None

    def chain(self, underlying: str, spot: float, *,
              target_dte: int = 1) -> List[OptionContract]:
        if not self._client:
            return self._fallback.chain(underlying, spot, target_dte=target_dte)
        try:
            from alpaca.data.requests import OptionChainRequest
            req = OptionChainRequest(underlying_symbol=underlying)
            resp = self._client.get_option_chain(req)
        except Exception:
            return self._fallback.chain(underlying, spot, target_dte=target_dte)

        # Pick the expiry closest to target_dte
        target_expiry = date.today() + timedelta(days=max(0, target_dte))
        by_sym = resp if isinstance(resp, dict) else getattr(resp, "data", {}) or {}
        parsed: List[OptionContract] = []
        for occ_symbol, snap in by_sym.items():
            try:
                # OCC: UNDER + YYMMDD + C/P + STRIKE*1000 padded to 8
                parts = _parse_occ(occ_symbol, underlying)
                if parts is None:
                    continue
                expiry_d, right, strike = parts
                bid = float(getattr(getattr(snap, "latest_quote", None), "bid_price", 0) or 0)
                ask = float(getattr(getattr(snap, "latest_quote", None), "ask_price", 0) or 0)
                oi = int(getattr(snap, "open_interest", 0) or 0)
                tv = int(getattr(snap, "day_volume", 0) or 0)
                parsed.append(OptionContract(
                    symbol=occ_symbol, underlying=underlying,
                    strike=strike, expiry=expiry_d, right=right,
                    multiplier=100,
                    open_interest=oi, today_volume=tv,
                    bid=bid, ask=ask, last=(bid + ask) / 2 if bid and ask else 0,
                    iv=float(getattr(snap, "implied_volatility", 0.0) or 0.0),
                ))
            except Exception:
                continue

        if not parsed:
            return self._fallback.chain(underlying, spot, target_dte=target_dte)

        # Filter to the expiry closest to target, then N strikes either side of spot.
        expiries = sorted({c.expiry for c in parsed})
        if self._standard_cycles_only:
            # Keep only {Mon, Wed, Fri} expiries. If that filter leaves us
            # empty (rare — typically means synthetic data or an ill-formed
            # feed), fall through to the full set.
            std = [d for d in expiries if d.weekday() in self._STANDARD_WEEKDAYS]
            if std:
                expiries = std
        best_expiry = min(expiries, key=lambda d: abs((d - target_expiry).days))
        at_expiry = [c for c in parsed if c.expiry == best_expiry]
        at_expiry.sort(key=lambda c: abs(c.strike - spot))
        return at_expiry[: self._max_strikes * 2 * 2]   # both rights × 2 sides


def _parse_occ(sym: str, underlying: str) -> Optional[tuple]:
    """Parse a standard OCC option symbol into (expiry_date, right, strike)."""
    if not sym.startswith(underlying):
        return None
    tail = sym[len(underlying):]
    if len(tail) < 15:
        return None
    try:
        ymd = tail[:6]
        y = 2000 + int(ymd[:2])
        m = int(ymd[2:4])
        d = int(ymd[4:6])
        right_char = tail[6]
        strike_raw = tail[7:15]
        strike = int(strike_raw) / 1000.0
        right = OptionRight.CALL if right_char == "C" else OptionRight.PUT
        return (date(y, m, d), right, strike)
    except (ValueError, IndexError):
        return None
