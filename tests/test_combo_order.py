"""ComboOrder + multi-leg broker path tests."""
from __future__ import annotations

from datetime import date
import pytest

from src.core.types import (
    ComboOrder, OptionLeg, OptionContract, OptionRight, Side,
)
from src.brokers.base import BrokerAdapter


def _contract(symbol, strike, right, bid=1.0, ask=1.1):
    return OptionContract(
        symbol=symbol,
        underlying="SPY",
        strike=strike,
        expiry=date(2026, 5, 15),
        right=right,
        bid=bid,
        ask=ask,
    )


# ---------- ComboOrder math ----------


def test_short_put_credit_spread_classification():
    """A short put credit spread: sell 450P, buy 440P. Net credit of $0.85."""
    short_put = _contract("SPY260515P00450000", 450, OptionRight.PUT, 1.20, 1.30)
    long_put  = _contract("SPY260515P00440000", 440, OptionRight.PUT, 0.35, 0.45)
    combo = ComboOrder(
        legs=[
            OptionLeg(contract=short_put, side=Side.SELL, ratio=1),
            OptionLeg(contract=long_put,  side=Side.BUY,  ratio=1),
        ],
        qty=1,
        net_limit=-0.85,
        tag="test_credit_spread",
    )
    assert combo.is_credit
    assert not combo.is_debit
    # Max loss = wing_width × 100 - credit_received × 100 = 10×100 - 0.85×100 = 915
    assert combo.max_loss_per_combo == pytest.approx(915.0)


def test_long_call_debit_spread_classification():
    """Long call debit spread: buy 500C, sell 510C. Net debit $3."""
    long_c  = _contract("SPY260515C00500000", 500, OptionRight.CALL, 4.10, 4.20)
    short_c = _contract("SPY260515C00510000", 510, OptionRight.CALL, 1.10, 1.20)
    combo = ComboOrder(
        legs=[
            OptionLeg(contract=long_c,  side=Side.BUY,  ratio=1),
            OptionLeg(contract=short_c, side=Side.SELL, ratio=1),
        ],
        qty=2,
        net_limit=3.0,
        tag="test_debit_spread",
    )
    assert combo.is_debit
    # Max loss for debit spread = debit × 100 = 300
    assert combo.max_loss_per_combo == pytest.approx(300.0)


def test_non_vertical_structure_returns_zero_max_loss():
    """Straddle (call + put, different rights) — base helper can't
    compute max loss; strategy code must handle its own risk math."""
    call = _contract("SPY260515C00500000", 500, OptionRight.CALL, 4.00, 4.10)
    put  = _contract("SPY260515P00500000", 500, OptionRight.PUT,  3.00, 3.10)
    combo = ComboOrder(
        legs=[
            OptionLeg(contract=call, side=Side.BUY, ratio=1),
            OptionLeg(contract=put,  side=Side.BUY, ratio=1),
        ],
        qty=1,
        net_limit=7.1,
    )
    assert combo.max_loss_per_combo == 0.0   # caller must compute


# ---------- Base broker submit_combo fallback path ----------


class _RecordingBroker(BrokerAdapter):
    """Minimal broker that records every submitted leg. Used to verify
    that submit_combo correctly decomposes a combo into individual
    leg orders when native mleg isn't available."""

    def __init__(self):
        self.submitted = []

    def account(self):
        from src.brokers.base import AccountState
        return AccountState(equity=100_000, cash=100_000, buying_power=100_000)

    def positions(self):
        return []

    def submit(self, order, **_kw):
        from src.core.types import Fill
        self.submitted.append(order)
        return Fill(order=order, price=order.limit_price, qty=order.qty)

    def cancel_all(self):
        pass

    def flatten_all(self, mark_prices=None):
        pass


def test_submit_combo_legs_individually_when_no_native_mleg():
    """Default submit_combo path: one broker submission per leg."""
    short_put = _contract("SPY260515P00450000", 450, OptionRight.PUT, 1.20, 1.30)
    long_put  = _contract("SPY260515P00440000", 440, OptionRight.PUT, 0.35, 0.45)
    combo = ComboOrder(
        legs=[
            OptionLeg(contract=short_put, side=Side.SELL, ratio=1),
            OptionLeg(contract=long_put,  side=Side.BUY,  ratio=1),
        ],
        qty=2,
        net_limit=-0.85,
        tag="spread_test",
    )

    b = _RecordingBroker()
    fills = b.submit_combo(combo)

    assert len(fills) == 2
    assert len(b.submitted) == 2
    # Each leg should have qty = combo.qty * leg.ratio
    assert all(o.qty == 2 for o in b.submitted)
    # Short put should be SELL, long put BUY
    sides = {o.symbol: o.side for o in b.submitted}
    assert sides["SPY260515P00450000"] == Side.SELL
    assert sides["SPY260515P00440000"] == Side.BUY
    # Both orders are tagged with the combo tag
    assert all(o.tag == "spread_test" for o in b.submitted)


def test_submit_combo_preserves_ratio_on_unequal_legs():
    """Ratio spread (2:1): closing should place twice the qty on the
    ratio-2 leg. Not common for our strategies but the primitive must
    support it correctly."""
    c1 = _contract("SPY260515C00500000", 500, OptionRight.CALL)
    c2 = _contract("SPY260515C00510000", 510, OptionRight.CALL)
    combo = ComboOrder(
        legs=[
            OptionLeg(contract=c1, side=Side.BUY,  ratio=1),
            OptionLeg(contract=c2, side=Side.SELL, ratio=2),
        ],
        qty=3,
        net_limit=2.0,
    )
    b = _RecordingBroker()
    b.submit_combo(combo)
    # ratio-1 leg → 3 contracts, ratio-2 leg → 6 contracts
    by_sym = {o.symbol: o.qty for o in b.submitted}
    assert by_sym["SPY260515C00500000"] == 3
    assert by_sym["SPY260515C00510000"] == 6
