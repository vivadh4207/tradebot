"""AlpacaBroker — routes orders to Alpaca's REST API.

Imports alpaca-py lazily; raises a clear error if the package or credentials
are missing. Paper endpoint is default.

GUARDRAIL: requires settings.live_trading == True to route anything other
than paper orders. This is checked one level up in the orchestrator.
"""
from __future__ import annotations

from typing import List, Optional

from ..core.types import Order, Fill, Position, Side
from .base import BrokerAdapter, AccountState


class AlpacaBroker(BrokerAdapter):
    def __init__(self, api_key: str, api_secret: str,
                 paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as e:
            raise ImportError(
                "alpaca-py is required for AlpacaBroker. Run: pip install alpaca-py"
            ) from e
        self._client = TradingClient(api_key, api_secret, paper=paper)
        self._paper = paper

    def account(self) -> AccountState:
        a = self._client.get_account()
        return AccountState(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            day_pnl=float(getattr(a, "day_pnl", 0.0) or 0.0),
            total_pnl=0.0,
        )

    def positions(self) -> List[Position]:
        from ..core.types import Position as Pos
        out: List[Position] = []
        for p in self._client.get_all_positions():
            out.append(Pos(
                symbol=p.symbol,
                qty=int(float(p.qty)),
                avg_price=float(p.avg_entry_price),
                is_option=False,  # alpaca options are a separate feed
                multiplier=1,
            ))
        return out

    def submit(self, order: Order) -> Optional[Fill]:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        tif = {
            "DAY": TimeInForce.DAY, "IOC": TimeInForce.IOC, "GTC": TimeInForce.GTC,
        }.get(order.tif, TimeInForce.DAY)
        side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL
        req = LimitOrderRequest(
            symbol=order.symbol, qty=order.qty,
            side=side, time_in_force=tif,
            limit_price=float(order.limit_price or 0),
        )
        self._client.submit_order(req)
        return None  # rely on webhook/poll for fills

    def cancel_all(self) -> None:
        self._client.cancel_orders()

    def flatten_all(self) -> None:
        self._client.close_all_positions(cancel_orders=True)
