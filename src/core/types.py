"""Domain types used across the bot. Dataclasses with minimal logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, Dict, Any
import time


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OptionRight(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class Bar:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None


@dataclass
class Quote:
    symbol: str
    ts: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if self.bid > 0 and self.ask > 0 else 0.0

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return self.spread / m if m > 0 else 1.0


@dataclass
class OptionContract:
    symbol: str              # OCC symbol e.g. SPY240419C00500000
    underlying: str
    strike: float
    expiry: date
    right: OptionRight
    multiplier: int = 100
    open_interest: int = 0
    today_volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if self.bid > 0 and self.ask > 0 else 0.0

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return self.spread / m if m > 0 else 1.0


@dataclass
class Signal:
    """A trade intention emitted by a strategy module."""
    source: str                                    # e.g. "momentum", "orb", "claude_ai"
    symbol: str
    side: Side                                     # BUY or SELL (for options: direction of the option position)
    option_right: Optional[OptionRight] = None
    strike: Optional[float] = None
    expiry: Optional[date] = None
    confidence: float = 0.5                        # 0..1
    rationale: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def is_stale(self, ttl_sec: float = 5.0) -> bool:
        return (time.time() - self.ts) > ttl_sec


@dataclass
class Order:
    symbol: str                 # OCC for options, ticker for equities
    side: Side
    qty: int
    is_option: bool = False
    limit_price: Optional[float] = None
    tif: str = "DAY"            # DAY, IOC, GTC
    tag: str = ""
    ts: float = field(default_factory=time.time)

    def is_stale(self, ttl_sec: float = 5.0) -> bool:
        return (time.time() - self.ts) > ttl_sec


@dataclass
class Fill:
    order: Order
    price: float
    qty: int
    fee: float = 0.0
    ts: float = field(default_factory=time.time)


@dataclass
class Position:
    symbol: str
    qty: int                                        # positive long, negative short
    avg_price: float
    is_option: bool = False
    underlying: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[date] = None
    right: Optional[OptionRight] = None
    multiplier: int = 1
    entry_ts: float = field(default_factory=time.time)
    entry_tags: Dict[str, Any] = field(default_factory=dict)   # scalp, vwap_reversion, etc.
    auto_profit_target: Optional[float] = None                 # absolute price
    auto_stop_loss: Optional[float] = None                     # absolute price
    consecutive_holds: int = 0
    # Trailing-stop / scale-out state. `peak_price` = highest mark
    # seen since entry (for longs). `scaled_out` flips true after the
    # first 50% close at initial PT — from that tick forward the
    # remainder rides a trailing stop instead of the static PT.
    peak_price: Optional[float] = None
    scaled_out: bool = False

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.avg_price * self.multiplier

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.avg_price == 0:
            return 0.0
        sign = 1 if self.is_long else -1
        return sign * (current_price - self.avg_price) / self.avg_price

    def dte(self, today: Optional[date] = None) -> int:
        if not self.expiry:
            return 9999
        today = today or date.today()
        return (self.expiry - today).days


class ExitTag(str, Enum):
    SCALP = "scalp"
    VWAP_REVERSION = "vwap_reversion"
    DIRECTIONAL_MOMENTUM = "directional_momentum"
    ZONE_BREACH = "zone_breach"
    VIX_PROTECTION = "vix_protection"
    THETA_DECAY = "theta_decay"
    ZERO_DTE_TIME_STOP = "0dte_time_stop"
    IRON_CONDOR = "iron_condor"
    CREDIT_SPREAD = "credit_spread"


@dataclass
class ExitDecision:
    should_close: bool
    reason: str
    layer: int
    allow_hold: bool = False
    # Partial-close support: when not None, close ONLY this many
    # contracts/shares instead of the full position. Used by the
    # scale-out logic to close 50% at first PT and trail the rest.
    close_qty: Optional[int] = None
