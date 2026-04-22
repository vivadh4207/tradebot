"""Shared interface + dataclasses for all market-data providers.

Every provider (Polygon, Tradier, Finnhub, Alpaca) implements the
MarketDataProvider protocol so the MultiProvider aggregator can fan
out uniformly. Missing methods return None; the aggregator falls back
to the next provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol


@dataclass
class ProviderQuote:
    symbol: str
    bid: float
    ask: float
    mid: float
    ts: Optional[str] = None
    source: str = ""        # "polygon" | "tradier" | "alpaca" | "finnhub"


@dataclass
class ProviderOptionRow:
    symbol: str             # OCC style, e.g. SPY260427P00716000
    underlying: str
    strike: float
    expiry: str             # ISO date string
    right: str              # "call" | "put"
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_vol: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    source: str = ""


@dataclass
class ProviderNewsItem:
    ts: str                 # ISO
    headline: str
    summary: str = ""
    url: str = ""
    tickers: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None    # -1..+1 when provider tags it
    source: str = ""        # "finnhub" | "polygon" | "alpaca" | ...


class MarketDataProvider(Protocol):
    """Every provider MUST implement `name` and `is_enabled`. Other
    methods are optional — return None when the provider doesn't
    support that data type."""

    name: str

    def is_enabled(self) -> bool: ...

    # Optional — return None if not supported / disabled.
    def latest_quote(self, symbol: str) -> Optional[ProviderQuote]: ...
    def option_chain(self, underlying: str, expiry: Optional[str] = None
                     ) -> Optional[List[ProviderOptionRow]]: ...
    def news(self, symbol: Optional[str] = None, limit: int = 20
             ) -> Optional[List[ProviderNewsItem]]: ...
