"""TWAP order slicer — break a large order into N child orders over a
fixed time window. Reduces market impact for size > displayed_size.

Use:
    slicer = TWAPSlicer(broker, slices=5, interval_sec=60)
    slicer.submit(parent_order, now=datetime.now())

For retail-scale paper we don't need POV or VWAP; TWAP is fine.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from ..core.types import Fill, Order, Side


@dataclass
class SliceResult:
    fills: List[Fill]
    total_filled: int
    total_cost: float
    parent_order: Order


class TWAPSlicer:
    def __init__(self, broker, slices: int = 5, interval_sec: float = 60.0,
                 min_slice_qty: int = 1):
        if slices < 1:
            raise ValueError("slices must be >= 1")
        self.broker = broker
        self.slices = int(slices)
        self.interval_sec = float(interval_sec)
        self.min_slice_qty = int(min_slice_qty)

    def submit(self, parent: Order, *, blocking: bool = True) -> SliceResult:
        """Slice `parent` into `self.slices` children and submit each one
        `interval_sec` apart. If blocking=False, spawn a background thread
        and return immediately with what's been filled so far.
        """
        if blocking:
            return self._run(parent)
        t = threading.Thread(
            target=self._run, args=(parent,),
            name=f"twap-slicer-{parent.symbol}", daemon=True,
        )
        t.start()
        return SliceResult(fills=[], total_filled=0, total_cost=0.0,
                            parent_order=parent)

    def _run(self, parent: Order) -> SliceResult:
        total_qty = int(parent.qty)
        base = max(total_qty // self.slices, self.min_slice_qty)
        remainder = total_qty - base * (self.slices - 1)
        fills: List[Fill] = []
        total_cost = 0.0
        total_filled = 0
        for i in range(self.slices):
            this_qty = base if i < self.slices - 1 else remainder
            if this_qty <= 0:
                continue
            child = Order(
                symbol=parent.symbol, side=parent.side, qty=this_qty,
                is_option=parent.is_option, limit_price=parent.limit_price,
                tif=parent.tif, tag=f"{parent.tag}|twap_{i + 1}/{self.slices}",
            )
            fill = self.broker.submit(child)
            if fill is not None:
                fills.append(fill)
                total_filled += fill.qty
                mul = 100 if parent.is_option else 1
                total_cost += fill.price * fill.qty * mul
            if i < self.slices - 1:
                time.sleep(self.interval_sec)
        return SliceResult(
            fills=fills, total_filled=total_filled,
            total_cost=total_cost, parent_order=parent,
        )
