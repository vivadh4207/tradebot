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
    # buying_power=1000, qty=50 × 2.0 × 100 = 10,000 → exceeds max(1000/5, 1000*.5)=500
    r = v.validate(o, c, buying_power=1000, open_slots=5)
    assert not r.ok and "budget" in r.reason


def test_valid_equity_order_accepts():
    v = OrderValidator()
    o = Order(symbol="SPY", side=Side.BUY, qty=5, is_option=False,
              limit_price=500.234, tag="x")
    r = v.validate(o, None, buying_power=100_000, open_slots=5)
    assert r.ok
    assert r.adjusted_order.limit_price == 500.23
