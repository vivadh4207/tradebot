"""Broker adapter interface + account snapshot."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional

from ..core.types import Order, Fill, Position


@dataclass
class AccountState:
    equity: float
    cash: float
    buying_power: float
    day_pnl: float = 0.0
    total_pnl: float = 0.0


class BrokerAdapter(abc.ABC):
    """Minimum interface every broker must implement."""

    @abc.abstractmethod
    def account(self) -> AccountState: ...

    @abc.abstractmethod
    def positions(self) -> List[Position]: ...

    @abc.abstractmethod
    def submit(self, order: Order) -> Optional[Fill]:
        """Send an order. Return Fill on immediate fill (paper/IOC), None if resting."""

    @abc.abstractmethod
    def cancel_all(self) -> None: ...

    @abc.abstractmethod
    def flatten_all(self) -> None:
        """Force-close all open positions (EOD sweep)."""
