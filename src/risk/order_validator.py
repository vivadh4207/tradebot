"""OrderValidator — final sanity check before submission.

- ask > 0, strike > 0, expiration set
- option_type must be 'call' or 'put'
- quantity 1-50
- price rounding: >=$1 to $0.05, <$1 to $0.01
- budget check
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..core.types import Order, OptionContract, Side


@dataclass
class ValidationResult:
    ok: bool
    reason: str
    adjusted_order: Optional[Order] = None


def round_option_price(p: float) -> float:
    if p <= 0:
        return 0.0
    if p >= 1.0:
        return round(round(p / 0.05) * 0.05, 2)
    return round(round(p / 0.01) * 0.01, 2)


class OrderValidator:
    def __init__(self, min_qty: int = 1, max_qty: int = 50,
                 max_pct_buying_power_single: float = 0.50):
        self.min_qty = min_qty
        self.max_qty = max_qty
        self.max_pct_bp = max_pct_buying_power_single

    def validate(self, order: Order, contract: Optional[OptionContract],
                 buying_power: float, open_slots: int) -> ValidationResult:
        if order.qty < self.min_qty or order.qty > self.max_qty:
            return ValidationResult(False, f"qty_out_of_range: {order.qty}")
        if order.is_option:
            if contract is None:
                return ValidationResult(False, "missing_contract")
            if contract.ask <= 0 or contract.strike <= 0:
                return ValidationResult(False, "bad_option_data")
            if contract.expiry is None:
                return ValidationResult(False, "no_expiry")
            if contract.right.value not in {"call", "put"}:
                return ValidationResult(False, "bad_right")

        price = order.limit_price
        if price is None or price <= 0:
            return ValidationResult(False, "bad_limit_price")

        price = round_option_price(price) if order.is_option else round(price, 2)
        order.limit_price = price

        # budget check: max(buying_power/open_slots, buying_power * max_pct_bp)
        slots = max(open_slots, 1)
        per_slot = buying_power / slots
        hard_cap = buying_power * self.max_pct_bp
        budget = max(per_slot, hard_cap)
        mul = 100 if order.is_option else 1
        cost = price * order.qty * mul
        if cost > budget:
            return ValidationResult(False, f"budget_exceeded: {cost:.2f}>{budget:.2f}")

        return ValidationResult(True, "ok", adjusted_order=order)
