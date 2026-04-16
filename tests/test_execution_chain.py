from datetime import datetime, timedelta, time
from src.core.clock import MarketClock, ET
from src.core.types import Signal, Side, OptionContract, OptionRight
from src.risk.execution_chain import ExecutionChain, ExecutionContext


SETTINGS = {
    "account": {"max_daily_loss_pct": 0.02, "max_open_positions": 5},
    "vix": {"halt_above": 40, "no_short_premium_above": 30, "no_0dte_long_below": 12},
    "iv_rank": {"block_sell_below": 0.30, "block_buy_above": 0.70},
    "execution": {
        "min_volume_confirmation": 1.2,
        "max_spread_pct_etf": 0.05, "max_spread_pct_stock": 0.10,
        "min_open_interest": 500, "min_today_option_volume": 100,
        "max_0dte_per_day": 50,
    },
    "mi_edge": {"block_below_combined_score": -5},
    "session": {
        "market_open": "09:30", "market_close": "16:00",
        "no_new_entries_after": "15:30", "eod_force_close": "15:45",
    },
}


def _ctx(**over):
    now = ET.localize(datetime(2026, 4, 16, 11, 0, 0))
    sig = Signal(source="momentum", symbol="SPY", side=Side.BUY,
                 option_right=OptionRight.CALL,
                 meta={"direction": "bullish"})
    base = dict(
        signal=sig, now=now, account_equity=10000, day_pnl=0,
        open_positions_count=0, current_bar_volume=200, avg_bar_volume=100,
        opening_range_high=500, opening_range_low=490,
        spot=501.0, vwap=499.0, vix=15, current_iv=0.25,
        iv_52w_low=0.1, iv_52w_high=0.5, is_etf=True,
        contract=OptionContract(symbol="SPY", underlying="SPY", strike=500,
                                expiry=now.date() + timedelta(days=2),
                                right=OptionRight.CALL, bid=1.0, ask=1.05,
                                open_interest=1000, today_volume=200),
    )
    base.update(over)
    return ExecutionContext(**base)


def test_happy_path_passes():
    clock = MarketClock()
    chain = ExecutionChain(SETTINGS, clock)
    results = chain.run(_ctx())
    assert ExecutionChain.decided_pass(results)


def test_daily_loss_blocks():
    clock = MarketClock()
    chain = ExecutionChain(SETTINGS, clock)
    results = chain.run(_ctx(day_pnl=-500))
    assert not ExecutionChain.decided_pass(results)


def test_vix_halt_blocks():
    clock = MarketClock()
    chain = ExecutionChain(SETTINGS, clock)
    results = chain.run(_ctx(vix=45))
    assert not ExecutionChain.decided_pass(results)


def test_spread_too_wide_blocks():
    clock = MarketClock()
    chain = ExecutionChain(SETTINGS, clock)
    ctx = _ctx()
    ctx.contract.bid = 1.0
    ctx.contract.ask = 1.20   # 18% spread > 5% ETF cap
    results = chain.run(ctx)
    assert not ExecutionChain.decided_pass(results)


def test_late_session_blocks_new_entry():
    clock = MarketClock()
    chain = ExecutionChain(SETTINGS, clock)
    late = ET.localize(datetime(2026, 4, 16, 15, 45, 0))
    results = chain.run(_ctx(now=late))
    assert not ExecutionChain.decided_pass(results)
