from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core.types import Order, Fill, Side
from src.storage.journal import SqliteJournal, ClosedTrade


def _tmp_path(tmp_path_factory, name="j.sqlite"):
    p = tmp_path_factory.mktemp("journal") / name
    return str(p)


def test_sqlite_roundtrip_fill_and_trade(tmp_path_factory):
    path = _tmp_path(tmp_path_factory)
    j = SqliteJournal(path)
    j.init_schema()

    o = Order(symbol="SPY", side=Side.BUY, qty=10, is_option=False,
              limit_price=500.0, tag="entry:momentum")
    f = Fill(order=o, price=500.01, qty=10, fee=0.0, ts=1_700_000_000.0)
    j.record_fill(f)

    now = datetime.now(tz=timezone.utc)
    ct = ClosedTrade(
        symbol="SPY", opened_at=now - timedelta(minutes=30), closed_at=now,
        side="long", qty=10, entry_price=500.0, exit_price=502.5,
        pnl=25.0, pnl_pct=0.005, entry_tag="momentum",
        exit_reason="exit:layer4_target_hit", is_option=False,
    )
    j.record_trade(ct)
    rows = j.closed_trades()
    assert len(rows) == 1
    r = rows[0]
    assert r.symbol == "SPY"
    assert r.side == "long"
    assert r.pnl == pytest.approx(25.0)
    assert r.pnl_pct == pytest.approx(0.005)
    j.close()


def test_paperbroker_logs_to_journal(tmp_path_factory):
    from src.brokers.paper import PaperBroker
    path = _tmp_path(tmp_path_factory, "broker.sqlite")
    j = SqliteJournal(path)
    j.init_schema()

    b = PaperBroker(starting_equity=10_000, slippage_bps=0, journal=j)
    buy = Order(symbol="SPY", side=Side.BUY, qty=1, is_option=False,
                limit_price=500.00, tag="entry:momentum")
    b.submit(buy)
    sell = Order(symbol="SPY", side=Side.SELL, qty=1, is_option=False,
                 limit_price=505.00, tag="exit:layer4_target_hit")
    b.submit(sell)

    trades = j.closed_trades()
    assert len(trades) == 1
    t = trades[0]
    assert t.symbol == "SPY"
    assert t.pnl == pytest.approx(5.0)     # 1 share × $5 move, 0 slippage, 0 fees
    j.close()


def test_equity_snapshot_upserts(tmp_path_factory):
    path = _tmp_path(tmp_path_factory, "eq.sqlite")
    j = SqliteJournal(path)
    j.init_schema()
    t = datetime.now(tz=timezone.utc)
    j.record_equity(t, 10_000.0, 9_500.0, -100.0)
    j.record_equity(t, 10_050.0, 9_550.0, -50.0)   # same ts → replace
    cur = j._conn.execute("SELECT equity FROM equity_curve").fetchall()
    assert len(cur) == 1
    assert cur[0][0] == 10_050.0
    j.close()
