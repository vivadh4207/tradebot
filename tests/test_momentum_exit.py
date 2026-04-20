"""Tests for the momentum/volume exit — closes long calls/puts when
trend reverses or volume dries up, provided already in profit."""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import pytest


def _mkbars(specs):
    """specs: list of (open, close, volume). Returns Bar list."""
    from src.core.types import Bar
    now = datetime.now(tz=timezone.utc)
    out = []
    for i, (o, c, v) in enumerate(specs):
        high = max(o, c) * 1.0005
        low = min(o, c) * 0.9995
        out.append(Bar(symbol="X", ts=now + timedelta(minutes=i),
                        open=float(o), high=float(high),
                        low=float(low), close=float(c),
                        volume=float(v)))
    return out


def _long_call(*, entry_premium: float = 1.00, dte_days: int = 7):
    from src.core.types import Position, OptionRight
    return Position(
        symbol="X260424C00100000", qty=1, avg_price=entry_premium,
        is_option=True, multiplier=100,
        underlying="X", strike=100.0,
        expiry=date.today() + timedelta(days=dte_days),
        right=OptionRight.CALL,
        entry_ts=time.time(),
    )


def _long_put(*, entry_premium: float = 1.00, dte_days: int = 7):
    from src.core.types import Position, OptionRight
    return Position(
        symbol="X260424P00100000", qty=1, avg_price=entry_premium,
        is_option=True, multiplier=100,
        underlying="X", strike=100.0,
        expiry=date.today() + timedelta(days=dte_days),
        right=OptionRight.PUT,
        entry_ts=time.time(),
    )


# ---------- reverse-bars trigger ----------
def test_momentum_exit_fires_on_3_red_bars_against_call():
    """Long call + 3 red bars = momentum reversed → close if in profit."""
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(momentum_exit_enabled=True))
    pos = _long_call(entry_premium=1.00, dte_days=7)
    # 20 bars: 17 green (trend) + 3 red (reversal)
    specs = [(100+i*0.1, 100+i*0.1+0.05, 10000) for i in range(17)]
    # Add 3 red at the end
    last_close = specs[-1][1]
    for _ in range(3):
        specs.append((last_close, last_close - 0.1, 10000))
        last_close -= 0.1
    bars = _mkbars(specs)
    # Current option price up 20% (in profit, above the 10% threshold)
    d = ev.evaluate(pos, current_price=1.20, bars=bars)
    assert d is not None
    assert d.should_close is True
    assert "momentum_reversal" in d.reason
    assert "red_bars" in d.reason


def test_momentum_exit_fires_on_3_green_bars_against_put():
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(momentum_exit_enabled=True))
    pos = _long_put(entry_premium=1.00, dte_days=7)
    # 17 red (trend down = put profitable) + 3 green (reversal)
    specs = [(100-i*0.1, 100-i*0.1-0.05, 10000) for i in range(17)]
    last_close = specs[-1][1]
    for _ in range(3):
        specs.append((last_close, last_close + 0.1, 10000))
        last_close += 0.1
    bars = _mkbars(specs)
    d = ev.evaluate(pos, current_price=1.20, bars=bars)
    assert d is not None
    assert "momentum_reversal" in d.reason
    assert "green_bars" in d.reason


def test_momentum_exit_does_not_fire_without_profit():
    """If position is underwater, the momentum-exit should NOT fire —
    that's the stop loss's job."""
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(
        momentum_exit_enabled=True,
        momentum_exit_min_profit=0.10,
    ))
    pos = _long_call(entry_premium=1.00, dte_days=7)
    specs = [(100+i*0.1, 100+i*0.1+0.05, 10000) for i in range(17)]
    last = specs[-1][1]
    for _ in range(3):
        specs.append((last, last - 0.1, 10000))
        last -= 0.1
    bars = _mkbars(specs)
    # P&L flat (not in profit) — momentum exit should skip
    d = ev.evaluate(pos, current_price=1.00, bars=bars)
    # Neither SL (no -10%) nor PT (no +35%) should fire either
    assert d is None


# ---------- volume dry-up trigger ----------
def test_volume_dry_up_and_no_new_high_closes_call():
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(
        momentum_exit_enabled=True,
        momentum_exit_reverse_bars=3,
        momentum_exit_vol_multiple=0.50,
        momentum_exit_vol_lookback=15,
        momentum_exit_stall_lookback=5,
    ))
    pos = _long_call(entry_premium=1.00, dte_days=7)
    # 15 bars of high volume + rising prices, then 5 bars low volume stalled
    specs = [(100+i*0.1, 100+i*0.1+0.05, 50000) for i in range(15)]
    # Now 5 bars: volume drops to 15k (<25k = 50% of 50k), no new high
    stall_close = specs[-1][1]  # last high
    for _ in range(5):
        specs.append((stall_close, stall_close, 15000))
    bars = _mkbars(specs)
    d = ev.evaluate(pos, current_price=1.20, bars=bars)
    assert d is not None
    assert "volume_dry_up" in d.reason


def test_volume_dry_up_does_not_fire_with_new_high():
    """Volume dropped but price is still making new highs → don't exit."""
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(
        momentum_exit_enabled=True,
        momentum_exit_reverse_bars=3,
        momentum_exit_vol_lookback=15,
        momentum_exit_stall_lookback=5,
    ))
    pos = _long_call(entry_premium=1.00, dte_days=7)
    specs = [(100+i*0.1, 100+i*0.1+0.05, 50000) for i in range(15)]
    # Low volume but STILL making new highs
    last = specs[-1][1]
    for i in range(5):
        specs.append((last + i*0.2, last + i*0.2 + 0.1, 15000))
    bars = _mkbars(specs)
    d = ev.evaluate(pos, current_price=1.30, bars=bars)
    # No volume-dry-up exit because we're still making new highs.
    # Also no reverse-bars (they're green).
    # So d is None (and the scale-out path would only fire at +35%).
    assert d is None or "volume" not in d.reason


# ---------- config toggle ----------
def test_momentum_exit_disabled_never_fires():
    from src.exits.fast_exit import FastExitEvaluator, FastExitConfig
    ev = FastExitEvaluator(FastExitConfig(momentum_exit_enabled=False))
    pos = _long_call(entry_premium=1.00, dte_days=7)
    specs = [(100+i*0.1, 100+i*0.1-0.1, 10000) for i in range(20)]  # all red
    bars = _mkbars(specs)
    d = ev.evaluate(pos, current_price=1.20, bars=bars)
    # Even though bars are reversed, the check is disabled → no exit
    # (unless hard cap or SL fire, which they don't at +20%/0% stops)
    # BUT the existing pt at 35% would not fire either at 1.20 (20%)
    assert d is None
