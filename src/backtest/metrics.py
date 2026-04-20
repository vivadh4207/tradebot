"""Performance metrics: Sharpe, max DD, tail ratio, hit rate, Calmar."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import math
import numpy as np


@dataclass
class PerformanceReport:
    n_trades: int
    total_return_pct: float
    cagr: float
    sharpe: float
    max_drawdown_pct: float
    calmar: float
    win_rate: float
    avg_win: float
    avg_loss: float
    tail_ratio_95_5: float
    worst_day_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def performance_report(equity_curve: List[float],
                       trade_pnls: List[float],
                       days_traded: int = 252) -> PerformanceReport:
    if not equity_curve or equity_curve[0] <= 0:
        return PerformanceReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    eq = np.asarray(equity_curve, dtype=float)
    returns = np.diff(eq) / eq[:-1]
    total = eq[-1] / eq[0] - 1.0
    cagr = (eq[-1] / eq[0]) ** (252.0 / max(days_traded, 1)) - 1.0 if days_traded > 0 else 0.0
    mean = float(np.mean(returns)) if returns.size else 0.0
    sd = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = (mean / sd) * math.sqrt(252) if sd > 0 else 0.0

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(np.min(dd))

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    if returns.size > 10:
        q95 = float(np.quantile(returns, 0.95))
        q05 = float(np.quantile(returns, 0.05))
        tail = (q95 / abs(q05)) if q05 != 0 else 0.0
    else:
        tail = 0.0
    worst_day = float(np.min(returns)) if returns.size else 0.0

    return PerformanceReport(
        n_trades=len(trade_pnls),
        total_return_pct=total * 100,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown_pct=max_dd * 100,
        calmar=calmar,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        tail_ratio_95_5=tail,
        worst_day_pct=worst_day * 100,
    )
