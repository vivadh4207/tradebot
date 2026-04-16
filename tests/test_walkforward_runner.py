from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.storage.journal import SqliteJournal, ClosedTrade
from src.backtest.walk_forward_runner import generate_windows
from src.backtest.prior_refitter import PriorFit


def _mk_trade(symbol: str, closed_at: datetime, pnl_pct: float,
               entry_price: float = 100.0, qty: int = 1) -> ClosedTrade:
    return ClosedTrade(
        symbol=symbol,
        opened_at=closed_at - timedelta(minutes=30),
        closed_at=closed_at,
        side="long", qty=qty,
        entry_price=entry_price,
        exit_price=entry_price * (1 + pnl_pct),
        pnl=entry_price * pnl_pct * qty,
        pnl_pct=pnl_pct,
        entry_tag="test", exit_reason="test", is_option=False,
    )


def test_prior_fit_empty_is_zeroed():
    fit = PriorFit.from_trades([])
    assert fit.n == 0 and fit.win_rate == 0


def test_prior_fit_with_edge():
    now = datetime.now(tz=timezone.utc)
    ts = [_mk_trade("X", now - timedelta(hours=i), 0.02) for i in range(6)] + \
         [_mk_trade("X", now - timedelta(hours=i + 20), -0.01) for i in range(4)]
    fit = PriorFit.from_trades(ts)
    assert fit.n == 10
    assert fit.win_rate == pytest.approx(0.6)
    assert fit.avg_win == pytest.approx(0.02)
    assert fit.avg_loss == pytest.approx(0.01)
    assert fit.is_tradable(min_n=5, min_ev=0.0)


def test_generate_windows_journal_roundtrip(tmp_path):
    db = tmp_path / "j.sqlite"
    j = SqliteJournal(str(db))
    j.init_schema()
    now = datetime.now(tz=timezone.utc)
    # seed some trades in the last 200 days
    for i in range(40):
        pnl = 0.02 if (i % 3 != 0) else -0.015
        j.record_trade(_mk_trade("SPY", now - timedelta(days=200 - i * 5), pnl))

    windows = generate_windows(
        j, train_days=90, test_days=30,
        min_trades=5, min_ev=0.0, max_windows=3, end=now,
    )
    assert len(windows) >= 1
    assert all(w.prior.n >= 0 for w in windows)
    # at least one window should be tradable given the seeded data
    assert any(w.tradable for w in windows)
    j.close()
