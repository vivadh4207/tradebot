"""Refit sizing priors from a set of closed trades (training window)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..storage.journal import ClosedTrade


@dataclass
class PriorFit:
    win_rate: float
    avg_win: float
    avg_loss: float
    n: int
    n_wins: int
    n_losses: int
    ev_per_trade: float

    @classmethod
    def from_trades(cls, trades: Iterable[ClosedTrade]) -> "PriorFit":
        wins: List[float] = []
        losses: List[float] = []
        for t in trades:
            pnl_pct = t.pnl_pct if t.pnl_pct is not None else 0.0
            if pnl_pct > 0:
                wins.append(pnl_pct)
            elif pnl_pct < 0:
                losses.append(abs(pnl_pct))
        n_d = len(wins) + len(losses)
        wr = (len(wins) / n_d) if n_d else 0.0
        aw = (sum(wins) / len(wins)) if wins else 0.0
        al = (sum(losses) / len(losses)) if losses else 0.0
        ev = wr * aw - (1 - wr) * al
        return cls(win_rate=wr, avg_win=aw, avg_loss=al,
                    n=n_d, n_wins=len(wins), n_losses=len(losses),
                    ev_per_trade=ev)

    def is_tradable(self, min_n: int = 30, min_ev: float = 0.0) -> bool:
        return self.n >= min_n and self.ev_per_trade > min_ev
