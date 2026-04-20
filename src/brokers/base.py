"""Broker adapter interface + account snapshot."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional

from ..core.types import Order, Fill, Position, ComboOrder


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
    def flatten_all(self, mark_prices: Optional[dict] = None) -> None:
        """Force-close all open positions (EOD sweep).

        `mark_prices` is an optional {symbol: price} hint used by simulated
        brokers to close at the last observed price. Live brokers ignore it
        and close at market.
        """

    def submit_combo(self, combo: ComboOrder) -> List[Fill]:
        """Submit a multi-leg option order.

        Default behavior: leg the combo as individual orders. This is
        NOT atomic — one leg could fill and the other not. Subclasses
        should override to use native mleg / combo-order APIs when
        available (Alpaca's `order_class=mleg`, for example).

        Callers must inspect `len(fills) == len(combo.legs)` and reconcile
        (e.g. close the lone filled leg) if they require all-or-nothing.
        """
        from ..core.types import Order
        import inspect
        # Sniff whether the concrete submit accepts `contract=` (paper
        # broker does; live brokers don't need it). This avoids two
        # different broker classes diverging on the combo path.
        sig = inspect.signature(self.submit)
        accepts_contract = "contract" in sig.parameters
        fills: List[Fill] = []
        for leg in combo.legs:
            leg_price = (abs(leg.contract.mid) if leg.contract.mid > 0
                          else abs(combo.net_limit) / max(1, len(combo.legs)))
            o = Order(
                symbol=leg.contract.symbol,
                side=leg.side,
                qty=combo.qty * leg.ratio,
                is_option=True,
                limit_price=leg_price,
                tif=combo.tif,
                tag=combo.tag,
            )
            fill = (self.submit(o, contract=leg.contract)
                    if accepts_contract
                    else self.submit(o))
            if fill is not None:
                fills.append(fill)
        return fills
