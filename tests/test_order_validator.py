from datetime import date
from src.core.types import Order, Side, OptionContract, OptionRight
from src.risk.order_validator import OrderValidator, round_option_price


def test_round_option_price_over_dollar():
    assert round_option_price(1.234) == 1.25
    assert round_option_price(1.20) == 1.20


def test_round_option_price_under_dollar():
    assert round_option_price(0.236) == 0.24
    assert round_option_price(0.31) == 0.31


def test_qty_out_of_range():
    v = OrderValidator(min_qty=1, max_qty=10)
    o = Order(symbol="X", side=Side.BUY, qty=11, limit_price=1.0)
    r = v.validate(o, None, 1_000_000, 5)
    assert not r.ok and "qty" in r.reason


def test_budget_exceeded_rejects():
    v = OrderValidator(max_pct_buying_power_single=0.5)
    c = OptionContract(symbol="X", underlying="X", strike=100.0,
                        expiry=date.today(), right=OptionRight.CALL,
                        ask=2.0, bid=1.9, open_interest=1000, today_volume=200)
    o = Order(symbol="X", side=Side.BUY, qty=50, is_option=True, limit_price=2.0)
    # BP=1000, qty=50 × 2.0 × 100 = 10,000 → exceeds min(1000/5=200, 1000*.5=500) = 200
    r = v.validate(o, c, buying_power=1000, open_slots=5)
    assert not r.ok and "budget" in r.reason


def test_valid_equity_order_accepts():
    v = OrderValidator()
    o = Order(symbol="SPY", side=Side.BUY, qty=5, is_option=False,
              limit_price=500.234, tag="x")
    r = v.validate(o, None, buying_power=100_000, open_slots=5)
    assert r.ok
    assert r.adjusted_order.limit_price == 500.23


def test_single_slot_capped_at_hard_pct():
    """Regression: with open_slots=1, per_slot=full_BP. Pre-fix used max(),
    allowing 100% of BP into one trade. After fix, hard_cap must bind."""
    v = OrderValidator(max_pct_buying_power_single=0.20)
    # buying_power=10000, cost would be 5 × 100 × 100 = 50,000 if it slipped
    c = OptionContract(symbol="SPY", underlying="SPY", strike=500.0,
                        expiry=date.today(), right=OptionRight.CALL,
                        ask=100.0, bid=99.0, open_interest=1000, today_volume=200)
    big = Order(symbol="SPY", side=Side.BUY, qty=5, is_option=True, limit_price=100.0)
    r = v.validate(big, c, buying_power=10_000, open_slots=1)
    # hard_cap = 10k * .20 = 2k; cost = 50k → blocked
    assert not r.ok and "budget" in r.reason

    # A small order (cost 100) must still pass the hard cap of 2k
    small = Order(symbol="SPY", side=Side.BUY, qty=1, is_option=True, limit_price=1.0)
    c_small = OptionContract(symbol="SPY", underlying="SPY", strike=500.0,
                              expiry=date.today(), right=OptionRight.CALL,
                              ask=1.0, bid=0.95, open_interest=1000, today_volume=200)
    r2 = v.validate(small, c_small, buying_power=10_000, open_slots=1)
    assert r2.ok


def test_multi_slot_per_slot_binds_when_tighter():
    """When per_slot is tighter than hard_cap, per_slot should bind."""
    v = OrderValidator(max_pct_buying_power_single=0.80)  # loose hard cap
    c = OptionContract(symbol="SPY", underlying="SPY", strike=500.0,
                        expiry=date.today(), right=OptionRight.CALL,
                        ask=5.0, bid=4.9, open_interest=1000, today_volume=200)
    # BP=10k, slots=5 → per_slot=2k, hard_cap=8k. min=2k.
    # cost = 5 × 100 × 5 = 2500 → exceeds 2k → blocked
    o = Order(symbol="SPY", side=Side.BUY, qty=5, is_option=True, limit_price=5.0)
    r = v.validate(o, c, buying_power=10_000, open_slots=5)
    assert not r.ok and "budget" in r.reason
