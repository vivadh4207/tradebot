"""Signal base class + shared context."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from ..core.types import Bar, Signal, OptionContract


@dataclass
class SignalContext:
    symbol: str
    now: datetime
    bars: List[Bar]
    spot: float
    vwap: float = 0.0
    opening_range_high: float = 0.0
    opening_range_low: float = 0.0
    atm_iv_30d: float = 0.25
    rv_20d: float = 0.20
    iv_52w_low: float = 0.10
    iv_52w_high: float = 0.50
    iv_30d: float = 0.25
    iv_90d: float = 0.27
    skew_zscore: float = 0.0
    rv_percentile_252d: float = 0.5
    chain: Optional[List[OptionContract]] = None


class SignalSource(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def emit(self, ctx: SignalContext) -> Optional[Signal]: ...
