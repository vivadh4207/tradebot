"""Regression tests for the hedge-fund-style audit fixes.

One test per fix. If any of these fail, the bug that was fixed has
resurfaced.
"""
from __future__ import annotations

import threading
from datetime import date, datetime, time, timedelta, timezone

import pytest

from src.core.clock import ET
from src.core.types import (
    Order, Side, OptionContract, OptionRight, Position,
)
from src.brokers.paper import PaperBroker
from src.brokers.quote_validator import QuoteValidator
from src.risk.order_validator import OrderValidator
from src.risk.portfolio_risk import PortfolioRiskManager
from src.exits.exit_engine import ExitEngine, ExitEngineConfig
from src.exits.tagged_profiles import TaggedProfileEvaluator
from src.exits.auto_stops import compute_auto_stops
from src.signals.ensemble import EnsembleCoordinator, Contribution
from src.intelligence.regime import Regime


# ---- P0: order_validator budget cap inversion (single slot) ----
def test_single_slot_budget_cap_binds():
    """Regression: BP=10k, slots=1, max_pct=0.2 → must enforce 2k cap."""
    v = OrderValidator(max_pct_buying_power_single=0.20)
    c = OptionContract(symbol="X", underlying="X", strike=500.0,
                        expiry=date.today(), right=OptionRight.CALL,
                        ask=100.0, bid=99.0, open_interest=1000, today_volume=200)
    order = Order(symbol="X", side=Side.BUY, qty=5, is_option=True, limit_price=100.0)
    r = v.validate(order, c, buying_power=10_000, open_slots=1)
    assert not r.ok
    assert "budget" in r.reason


# ---- P0: portfolio_risk notional enforcement ----
def test_portfolio_notional_blocks_when_exceeded():
    """Disable the other Greek limits so we isolate the notional check."""
    prm = PortfolioRiskManager(
        max_dollar_delta_per_100k=1e12,
        max_dollar_gamma_per_100k=1e12,
        max_vega=1e12,
        max_theta_daily=-1e12,
        max_notional_pct=0.10,     # 10% of equity
    )
    existing = [
        Position(symbol="SPY", qty=10, avg_price=500.0, is_option=False,
                  multiplier=1),
    ]
    proposed = Position(symbol="QQQ", qty=5, avg_price=400.0, is_option=False,
                         multiplier=1)
    # total notional (equity valuation uses spot) = 10*500 + 5*500 = 7500
    # equity = 50k → 10% cap = 5000 → 7500 > 5000 → block
    ok, reason = prm.check(proposed, existing, spot=500.0, equity=50_000)
    assert not ok
    assert "notional" in reason


def test_portfolio_notional_allows_small_position():
    """Notional check should PASS for a position comfortably within the cap."""
    prm = PortfolioRiskManager(
        max_dollar_delta_per_100k=1e12,
        max_dollar_gamma_per_100k=1e12,
        max_vega=1e12,
        max_theta_daily=-1e12,
        max_notional_pct=0.50,
    )
    proposed = Position(symbol="SPY", qty=1, avg_price=500.0, is_option=False,
                         multiplier=1)
    ok, reason = prm.check(proposed, [], spot=500.0, equity=100_000)
    assert ok, reason


# ---- P0: PaperBroker thread safety under contention ----
def test_paper_broker_thread_safe_under_contention():
    """Hammer submit() from multiple threads. CASH (the authoritative
    post-close ledger) must be exactly preserved; positions must all close.

    Equity itself is only recomputed by mark_to_market(); after a sequence of
    round-trip fills it can be stale — that's by design and not the subject
    of this regression test.
    """
    broker = PaperBroker(starting_equity=1_000_000.0, slippage_bps=0)

    def worker(sym: str, n: int):
        for i in range(n):
            side = Side.BUY if i % 2 == 0 else Side.SELL
            broker.submit(Order(symbol=sym, side=side, qty=1,
                                 is_option=False, limit_price=100.0))

    threads = [threading.Thread(target=worker, args=(f"S{i}", 50))
               for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    acct = broker.account()
    # Cash is the definitive ledger: net zero P&L (no slippage, no fees,
    # every BUY matched by a SELL) → cash unchanged.
    assert acct.cash == pytest.approx(1_000_000.0, abs=1e-6)
    assert len(broker.positions()) == 0
    # After a mark-to-market equity matches cash (no open positions).
    broker.mark_to_market({})
    assert broker.account().equity == pytest.approx(1_000_000.0, abs=1e-6)


# ---- P0: backtest look-ahead — verify simulator uses next-bar open ----
def test_backtest_uses_next_bar_fill():
    """Scan the simulator source for the look-ahead fix markers."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "backtest" / "simulator.py"
    text = src.read_text()
    assert "fill_price" in text, "fill_price arg must exist in _try_enter"
    assert "next_bar.open" in text, "next_bar.open must be used for fills"
    assert "look-ahead" in text.lower(), "intent documented in comments"


# ---- P1: ensemble counts empty-direction signals as contributions ----
def test_ensemble_records_directionless_contributions():
    from src.core.types import Signal
    sig_good = Signal(source="momentum", symbol="SPY", side=Side.BUY,
                      confidence=0.7, meta={"direction": "bullish"})
    sig_bad = Signal(source="unknown", symbol="SPY", side=Side.BUY,
                     confidence=0.5, meta={})  # no direction
    c = EnsembleCoordinator(min_weighted_confidence=0.3, dominance_ratio=1.0)
    d = c.aggregate([sig_good, sig_bad], Regime.TREND_LOWVOL)
    # both recorded
    sources = {x.source for x in d.contributions}
    assert sources == {"momentum", "unknown"}
    # only directed one contributed to the score
    assert d.emitted is True
    assert d.dominant_direction == "bullish"


# ---- P1: exit_engine layer 3 uses stop-only, not profit target ----
def test_exit_engine_layer3_stop_only_for_shorts():
    """Short option: price dropping to profit target must NOT close at layer 3
    (layer 4 handles PT via pnl_pct); but price rising to stop MUST close."""
    pos = Position(
        symbol="SPY240416P500", qty=-1, avg_price=2.00,
        is_option=True, underlying="SPY", strike=500.0,
        expiry=date.today() + timedelta(days=1),
        right=OptionRight.PUT, multiplier=100,
        entry_ts=datetime.now().timestamp(),
    )
    pos.auto_profit_target = 1.00    # short: price DOWN = good
    pos.auto_stop_loss = 3.00        # short: price UP = bad

    ee = ExitEngine()
    now = ET.localize(datetime.combine(date.today(), time(11, 0)))

    # Price hit profit target. Layer 3 should NOT close (pass-through → layer 4
    # will decide on pnl%).
    d_pt = ee.decide(pos, 1.00, now, vix=15, spot=500, vwap=500, bars=[])
    # Could still close at layer 4 since pnl_pct = 50% target is hit. But must
    # not have layer=3 in the decision.
    if d_pt.should_close:
        assert d_pt.layer != 3

    # Price hit stop loss. Layer 3 MUST fire.
    d_sl = ee.decide(pos, 3.00, now, vix=15, spot=500, vwap=500, bars=[])
    assert d_sl.should_close
    assert d_sl.layer == 3


# ---- P1: pin-risk flatten at 15:45 ET for 0DTE ITM ----
def test_pin_risk_flatten_for_0dte_near_strike():
    tag = TaggedProfileEvaluator()
    # 0DTE long call, spot 499.75, strike 500.00 → dist_pct 0.0005 < 0.0025
    pos = Position(
        symbol="SPY", qty=1, avg_price=1.00,
        is_option=True, underlying="SPY", strike=500.0,
        expiry=date.today(),
        right=OptionRight.CALL, multiplier=100,
        entry_ts=datetime.now().timestamp(),
    )
    now = ET.localize(datetime.combine(date.today(), time(15, 46)))
    d = tag.evaluate(pos, current_price=1.0, now=now,
                      vix=15, spot=499.75, vwap=500.0)
    assert d is not None and d.should_close
    assert "pin_risk" in d.reason


def test_pin_risk_does_not_fire_when_far_from_strike():
    tag = TaggedProfileEvaluator()
    pos = Position(
        symbol="SPY", qty=1, avg_price=1.00,
        is_option=True, underlying="SPY", strike=500.0,
        expiry=date.today(),
        right=OptionRight.CALL, multiplier=100,
        entry_ts=datetime.now().timestamp(),
    )
    now = ET.localize(datetime.combine(date.today(), time(15, 46)))
    d = tag.evaluate(pos, current_price=1.0, now=now,
                      vix=15, spot=498.0, vwap=500.0)
    # spot 498 vs strike 500 → dist_pct = 0.004 > 0.0025 → no pin-risk
    # Other 0dte checks at 15:46 only trigger at 15:50; so decision is None.
    assert d is None or "pin_risk" not in d.reason


# ---- P1: QuoteValidator LRU cap ----
def test_quote_validator_lru_cap_evicts_oldest():
    qv = QuoteValidator(max_symbols=8, history_len=5)
    # Fill with 10 distinct symbols; cap is 8 → 2 oldest should be evicted.
    from src.core.types import Quote
    now = datetime.now()
    for i in range(10):
        qv.is_valid(Quote(symbol=f"S{i}", ts=now,
                           bid=1.00, ask=1.01, bid_size=100, ask_size=100))
    assert len(qv._spread_history) == 8
    assert "S0" not in qv._spread_history
    assert "S1" not in qv._spread_history
    assert "S9" in qv._spread_history


# ---- P1: Alpaca retry helper — retriable classification ----
def test_alpaca_retry_classification():
    from src.brokers.alpaca_adapter import _is_retriable

    class Http429(Exception):
        status_code = 429

    class Http401(Exception):
        status_code = 401

    class Http503(Exception):
        status_code = 503

    assert _is_retriable(Http429())
    assert not _is_retriable(Http401())
    assert _is_retriable(Http503())
    assert _is_retriable(ConnectionError("timeout"))
    assert not _is_retriable(ValueError("bad input"))
