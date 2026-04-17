"""Regression tests for the options hot-path bugs surfaced in the live-session audit.

Each test locks in one of the five bugs we just fixed so nothing can
quietly slide back:

  B1  Position metadata populated from OptionContract at entry
  B2  EOD flatten of options uses chain-derived option prices, not spot
  B3  mark_to_market for options uses option prices, not spot
  B4  auto_profit_target / auto_stop_loss set on Position at entry
  B6  fast_loop evaluates exits using per-option marks, not underlying spot
"""
from __future__ import annotations

from datetime import date, timedelta, timezone, datetime
from pathlib import Path

import pytest


def _build_paper_broker(tmp_path):
    from src.brokers.paper import PaperBroker
    return PaperBroker(starting_equity=10_000.0,
                        snapshot_path=str(tmp_path / "broker_state.json"))


def _make_atm_call(symbol: str, spot: float, dte: int = 1):
    from src.data.options_chain import SyntheticOptionsChain
    from src.core.types import OptionRight
    chain = SyntheticOptionsChain().chain(symbol, spot, target_dte=dte)
    return SyntheticOptionsChain.find_atm(chain, spot, OptionRight.CALL)


# ---------- B1: position metadata populated from OptionContract ----------
def test_paper_broker_populates_option_metadata_from_contract(tmp_path):
    from src.core.types import Order, Side
    contract = _make_atm_call("MAR", 250.0, dte=1)
    broker = _build_paper_broker(tmp_path)
    order = Order(symbol=contract.symbol, side=Side.BUY, qty=2,
                   is_option=True, limit_price=contract.ask * 1.02,
                   tag="entry:test:MAR")
    fill = broker.submit(order, contract=contract)
    assert fill is not None
    pos = next(p for p in broker.positions() if p.symbol == contract.symbol)
    # Before the fix, underlying / strike / expiry / right were all None.
    assert pos.underlying == "MAR"
    assert pos.strike == contract.strike
    assert pos.expiry == contract.expiry
    assert pos.right == contract.right
    # And `dte()` now returns the real number, not 9999.
    assert pos.dte() == (contract.expiry - date.today()).days


# ---------- B4: auto PT/SL set at entry (CLAUDE.md hard rule) ----------
def test_paper_broker_sets_auto_pt_and_sl_at_entry(tmp_path):
    from src.core.types import Order, Side
    contract = _make_atm_call("AAPL", 200.0, dte=1)
    broker = _build_paper_broker(tmp_path)
    entry_px = contract.ask * 1.02
    auto_pt = round(entry_px * 1.35, 4)
    auto_sl = round(entry_px * 0.80, 4)
    order = Order(symbol=contract.symbol, side=Side.BUY, qty=1,
                   is_option=True, limit_price=entry_px, tag="entry:test:AAPL")
    broker.submit(order, contract=contract,
                   auto_profit_target=auto_pt, auto_stop_loss=auto_sl)
    pos = next(p for p in broker.positions() if p.symbol == contract.symbol)
    assert pos.auto_profit_target == pytest.approx(auto_pt)
    assert pos.auto_stop_loss == pytest.approx(auto_sl)


# ---------- B1+B2+B3: _build_mark_prices uses chain for options ----------
def test_build_mark_prices_uses_chain_for_options_not_underlying_spot(tmp_path, monkeypatch):
    """Before the fix, EOD flatten would feed the underlying's spot ($250)
    as the limit price for closing a $1.20 option → absurd fill prices
    and fictitious P&L. The fix routes options through the chain."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src.main import TradeBot
    from src.core.config import load_settings
    from src.core.types import Order, Side
    s = load_settings()
    bot = TradeBot(s)
    # Stub latest_price so _build_mark_prices has a spot to base the chain
    # fetch on (the real data adapter needs bars to be ingested first).
    monkeypatch.setattr(bot.data, "latest_price",
                         lambda sym: 250.0 if sym == "MAR" else None)
    contract = _make_atm_call("MAR", 250.0, dte=1)
    order = Order(symbol=contract.symbol, side=Side.BUY, qty=2,
                   is_option=True, limit_price=contract.ask * 1.02,
                   tag="entry:test:MAR")
    bot.broker.submit(order, contract=contract)
    positions = bot.broker.positions()
    assert any(p.symbol == contract.symbol for p in positions)

    marks = bot._build_mark_prices(positions)
    m = marks.get(contract.symbol)
    assert m is not None, "options position must have a chain-derived mark"
    # Mark must be an OPTION price (few dollars per share), not the
    # underlying's spot (~$250).
    assert 0.0 < m < 20.0, (
        f"expected option-priced mark, got {m} — likely feeding underlying spot"
    )


# ---------- B6: fast_loop path closes options with option-priced limit ----------
def test_fast_loop_limit_uses_option_mark_not_spot(tmp_path, monkeypatch):
    """Simulate what fast_loop does: build mark_prices for open option
    positions, then evaluate. The limit fed into the closing order must
    be an option price, not the underlying's spot."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    from src.main import TradeBot
    from src.core.config import load_settings
    from src.core.types import Order, Side
    s = load_settings()
    bot = TradeBot(s)
    monkeypatch.setattr(bot.data, "latest_price",
                         lambda sym: 900.0 if sym == "NVDA" else None)
    contract = _make_atm_call("NVDA", 900.0, dte=1)
    order = Order(symbol=contract.symbol, side=Side.BUY, qty=1,
                   is_option=True, limit_price=contract.ask * 1.02,
                   tag="entry:test:NVDA")
    bot.broker.submit(order, contract=contract)
    positions = bot.broker.positions()
    marks = bot._build_mark_prices(positions)
    mark = marks[contract.symbol]
    # The mark must NOT be close to NVDA's spot ($900). If it is, we've
    # regressed and fast_loop will close options at spot-level limits.
    assert abs(mark - 900.0) > 100.0, (
        f"mark too close to underlying spot: {mark} — fast_loop will "
        f"generate fictitious P&L if this ever regresses"
    )


# ---------- B1: Position.dte() works after fill ----------
def test_position_dte_is_correct_after_option_fill(tmp_path):
    from src.core.types import Order, Side
    contract = _make_atm_call("SPY", 500.0, dte=1)
    broker = _build_paper_broker(tmp_path)
    broker.submit(Order(symbol=contract.symbol, side=Side.BUY, qty=1,
                         is_option=True, limit_price=contract.ask,
                         tag="entry:test:SPY"),
                   contract=contract)
    pos = broker.positions()[0]
    # Before fix: dte() was 9999 because expiry was never populated.
    assert pos.dte() <= 3, f"dte should reflect contract expiry; got {pos.dte()}"
