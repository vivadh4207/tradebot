"""Walk-forward backtest pattern (playbook Section 9).

Refits parameters every `test_window` days using the prior `train_window`.
Callable shape:
    strategy.fit(train_data) -> params
    strategy.run(test_data, params) -> list of per-day returns/pnls
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List


def walk_forward_backtest(
    strategy: Any,
    data: List[Any],
    train_window: int = 252,
    test_window: int = 63,
) -> List[float]:
    results: List[float] = []
    n = len(data)
    if n < train_window + test_window:
        return results
    start = train_window
    while start + test_window <= n:
        train = data[start - train_window:start]
        test = data[start:start + test_window]
        params = strategy.fit(train)
        pnl = strategy.run(test, params)
        results.extend(pnl)
        start += test_window
    return results
