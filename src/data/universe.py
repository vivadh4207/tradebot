"""Symbol universe: top-10 highest liquidity US names for options trading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

DEFAULT_UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT",
    "NVDA", "TSLA", "META", "AMZN", "GOOGL",
]


@dataclass
class Universe:
    symbols: List[str]

    @classmethod
    def default(cls) -> "Universe":
        return cls(symbols=list(DEFAULT_UNIVERSE))

    def is_etf(self, symbol: str) -> bool:
        return symbol.upper() in {"SPY", "QQQ", "IWM", "DIA", "VTI"}
