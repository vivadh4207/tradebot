from datetime import datetime, timedelta
from src.core.clock import MarketClock, ET
from src.core.types import Signal, Side, OptionRight, OptionContract
from src.risk.execution_chain import ExecutionChain, ExecutionContext
from src.intelligence.news import (
    StaticNewsProvider, NewsItem, CachedNewsSentiment,
)
from src.intelligence.news_classifier import KeywordClassifier


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
    "news": {"block_score": 0.5, "premium_harvest_block_score": 0.75},
}


def _ctx(**over):
    now = ET.localize(datetime(2026, 4, 16, 11, 0, 0))
    sig = Signal(source="momentum", symbol="SPY", side=Side.BUY,
                 option_right=OptionRight.CALL, meta={"direction": "bullish"})
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
        news_score=0.0, news_label="neutral", news_rationale="",
    )
    base.update(over)
    return ExecutionContext(**base)


def test_negative_news_blocks_bullish_call():
    clock = MarketClock()
    chain = ExecutionChain(SETTINGS, clock)
    ctx = _ctx(news_score=-0.8, news_label="negative", news_rationale="downgrade")
    results = chain.run(ctx)
    assert not ExecutionChain.decided_pass(results)
    assert any(r.reason.startswith("news_negative_for_long") for r in results)


def test_positive_news_blocks_bearish_put():
    clock = MarketClock()
    chain = ExecutionChain(SETTINGS, clock)
    sig = Signal(source="momentum", symbol="SPY", side=Side.BUY,
                 option_right=OptionRight.PUT, meta={"direction": "bearish"})
    ctx = _ctx(signal=sig, news_score=+0.7, news_label="positive",
                news_rationale="beat")
    results = chain.run(ctx)
    assert not ExecutionChain.decided_pass(results)
    assert any(r.reason.startswith("news_positive_for_short") for r in results)


def test_mildly_negative_news_advisory_only():
    clock = MarketClock()
    chain = ExecutionChain(SETTINGS, clock)
    ctx = _ctx(news_score=-0.3, news_label="negative", news_rationale="mild")
    results = chain.run(ctx)
    assert ExecutionChain.decided_pass(results)   # advisory, not blocking


def test_premium_harvest_requires_big_shock_to_block():
    clock = MarketClock()
    chain = ExecutionChain(SETTINGS, clock)
    sig = Signal(source="vrp", symbol="SPY", side=Side.SELL,
                 option_right=OptionRight.PUT,
                 meta={"direction": "premium_harvest",
                       "premium_action": "sell"})
    ctx_mild = _ctx(signal=sig, news_score=-0.6, news_label="negative")
    ctx_shock = _ctx(signal=sig, news_score=-0.9, news_label="negative")
    # 0.6 is below premium-harvest threshold of 0.75 → pass (advisory only)
    assert ExecutionChain.decided_pass(chain.run(ctx_mild))
    # 0.9 exceeds it → block
    assert not ExecutionChain.decided_pass(chain.run(ctx_shock))


def test_cached_sentiment_reuses_within_ttl():
    from datetime import datetime as _dt
    now = _dt.utcnow()
    items = {"SPY": [NewsItem(symbol="SPY", headline="Company X beats expectations",
                               source="t", published_at=now)]}
    provider = StaticNewsProvider(items)
    cache = CachedNewsSentiment(provider, KeywordClassifier(), ttl_seconds=60)
    first = cache.sentiment("SPY")
    # mutate the underlying data; cached call should ignore the change
    items["SPY"].append(NewsItem(symbol="SPY", headline="downgrade lawsuit fraud",
                                   source="t", published_at=now))
    second = cache.sentiment("SPY")
    assert first.score == second.score == cache.sentiment("SPY").score
