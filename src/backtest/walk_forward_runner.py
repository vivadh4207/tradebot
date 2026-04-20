"""End-to-end walk-forward: refit priors each window from realized trades.

For each window:
  1. Take closed trades in the TRAIN window from the journal.
  2. Fit `PriorFit` (win_rate, avg_win, avg_loss).
  3. Gate: only deploy if `is_tradable()` (n >= min_trades, ev > 0).
  4. Record the fit + the test window to a report; caller can use that
     fit for the corresponding `BacktestSimulator` or live paper period.

This is the production pattern: you refit quarterly from the prior year's
trades, and you don't trade in a window where the fit is statistically
thin or negative-EV.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..storage.journal import TradeJournal
from .prior_refitter import PriorFit


@dataclass
class WFWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    prior: PriorFit
    tradable: bool


def generate_windows(
    journal: TradeJournal,
    train_days: int = 365,
    test_days: int = 63,
    end: Optional[datetime] = None,
    min_trades: int = 30,
    min_ev: float = 0.0,
    max_windows: int = 20,
) -> List[WFWindow]:
    """Generate walk-forward windows with refitted priors from journal trades."""
    end = end or datetime.now(tz=timezone.utc)
    out: List[WFWindow] = []
    test_end = end
    for _ in range(max_windows):
        test_start = test_end - timedelta(days=test_days)
        train_end = test_start
        train_start = train_end - timedelta(days=train_days)
        ts = journal.closed_trades(since=train_start)
        ts = [t for t in ts if t.closed_at and t.closed_at < train_end]
        prior = PriorFit.from_trades(ts)
        out.append(WFWindow(
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            prior=prior,
            tradable=prior.is_tradable(min_n=min_trades, min_ev=min_ev),
        ))
        test_end = test_start
        if train_start < end - timedelta(days=(train_days + test_days) * max_windows):
            break
    return list(reversed(out))


def summarize(windows: List[WFWindow]) -> Dict[str, Any]:
    return {
        "windows": [
            {"train": f"{w.train_start.date()} → {w.train_end.date()}",
             "test": f"{w.test_start.date()} → {w.test_end.date()}",
             "n": w.prior.n,
             "win_rate": round(w.prior.win_rate, 4),
             "avg_win": round(w.prior.avg_win, 4),
             "avg_loss": round(w.prior.avg_loss, 4),
             "ev": round(w.prior.ev_per_trade, 4),
             "tradable": w.tradable}
            for w in windows
        ]
    }
