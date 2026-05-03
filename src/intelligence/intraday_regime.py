"""Intraday regime detector — classifies the current session into one
of four states based on SPY's price action:

  CRASH   — sharp drop in last 5-30 min (>= configured thresholds).
            Bot OVERRIDES session floors, biases bearish entries,
            and SCALES SIZE UP. This is when puts pay big.
  RUSH    — sharp rally (mirror of CRASH).
  CHOP    — first hour realized range below threshold.
            Bot PAUSES new entries — no edge to extract.
  NORMAL  — standard market, default behavior.

Returned by `evaluate_intraday_state(bars)` so callers can adjust
risk parameters live.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from ..core.types import Bar


@dataclass
class IntradayState:
    label: str                  # "crash" | "rush" | "chop" | "normal"
    pct_change_5m: float        # SPY 5-min % move
    pct_change_30m: float       # SPY 30-min % move
    realized_range_60m: float   # SPY 60-min range as % of price
    reason: str                 # human-readable
    size_mult_bias: float       # 1.0 = normal, 1.5 = boost, 0.0 = halt
    override_floors: bool       # True = ignore session-floor bonus
    bearish_bias: bool          # True = prefer puts over calls
    bullish_bias: bool          # True = prefer calls over puts


def evaluate_intraday_state(
    spy_bars: List[Bar],
    *,
    crash_5m_pct: float = -0.008,    # -0.8% in 5 min = crash trigger
    crash_30m_pct: float = -0.020,   # -2.0% in 30 min = sustained crash
    rush_5m_pct: float = 0.008,      # +0.8% / 5 min
    rush_30m_pct: float = 0.020,     # +2.0% / 30 min
    chop_60m_range_pct: float = 0.003,  # 0.3% range over 1 hour = chop
    range_expansion_5m_pct: float = 0.006,  # 0.6% range/5min = vol expansion
) -> IntradayState:
    """Classify intraday regime from recent SPY bars (1-min preferred).

    All thresholds tunable. Defaults are intentionally aggressive on
    crash detection so the bot doesn't miss the big move while too-strict
    rules debate whether it's "really" a crash.
    """
    if not spy_bars or len(spy_bars) < 5:
        return IntradayState(
            label="normal",
            pct_change_5m=0.0, pct_change_30m=0.0,
            realized_range_60m=0.0,
            reason="not_enough_bars",
            size_mult_bias=1.0,
            override_floors=False,
            bearish_bias=False, bullish_bias=False,
        )

    last_close = spy_bars[-1].close
    if last_close <= 0:
        return IntradayState(
            label="normal", pct_change_5m=0.0, pct_change_30m=0.0,
            realized_range_60m=0.0, reason="bad_price",
            size_mult_bias=1.0, override_floors=False,
            bearish_bias=False, bullish_bias=False,
        )

    # 5-min change: last bar close vs bar 5 min ago
    bar_5m_ago = spy_bars[-min(5, len(spy_bars))]
    pct_5m = (
        (last_close - bar_5m_ago.close) / bar_5m_ago.close
        if bar_5m_ago.close > 0 else 0.0
    )

    # 30-min change
    bar_30m_ago = spy_bars[-min(30, len(spy_bars))]
    pct_30m = (
        (last_close - bar_30m_ago.close) / bar_30m_ago.close
        if bar_30m_ago.close > 0 else 0.0
    )

    # 60-min realized range as % of last price
    last_60 = spy_bars[-min(60, len(spy_bars)):]
    rng_60 = max(b.high for b in last_60) - min(b.low for b in last_60)
    rng_pct = rng_60 / last_close if last_close > 0 else 0.0

    # ---- range-expansion detection (volatility regardless of direction) ----
    # If SPY's last 5 min has a wider range than threshold, market is
    # MOVING — even if direction unclear. Loosens filters proactively
    # so the bot captures fast moves like today's QQQ/SPX runs.
    last_5 = spy_bars[-min(5, len(spy_bars)):]
    if last_5:
        rng_5m = max(b.high for b in last_5) - min(b.low for b in last_5)
        rng_5m_pct = rng_5m / last_close
        if rng_5m_pct >= range_expansion_5m_pct and abs(pct_5m) < crash_5m_pct:
            return IntradayState(
                label="vol_expansion",
                pct_change_5m=pct_5m, pct_change_30m=pct_30m,
                realized_range_60m=rng_pct,
                reason=(f"SPY 5-min range {rng_5m_pct:.2%} "
                         f">= {range_expansion_5m_pct:.2%} — vol expansion, "
                         f"trade aggressively"),
                size_mult_bias=1.25,         # 25% size boost
                override_floors=True,         # ignore session floor
                bearish_bias=False,
                bullish_bias=False,
            )

    # ---- crash detection (aggressive bearish bias) ----
    if pct_5m <= crash_5m_pct or pct_30m <= crash_30m_pct:
        return IntradayState(
            label="crash",
            pct_change_5m=pct_5m, pct_change_30m=pct_30m,
            realized_range_60m=rng_pct,
            reason=(f"SPY 5m={pct_5m:+.2%} 30m={pct_30m:+.2%} "
                     f"— crash trigger fired"),
            size_mult_bias=1.50,           # 50% larger on bearish entries
            override_floors=True,           # ignore session-floor penalty
            bearish_bias=True,
            bullish_bias=False,
        )

    # ---- rush (mirror — strong rally) ----
    if pct_5m >= rush_5m_pct or pct_30m >= rush_30m_pct:
        return IntradayState(
            label="rush",
            pct_change_5m=pct_5m, pct_change_30m=pct_30m,
            realized_range_60m=rng_pct,
            reason=(f"SPY 5m={pct_5m:+.2%} 30m={pct_30m:+.2%} "
                     f"— rush trigger fired"),
            size_mult_bias=1.50,
            override_floors=True,
            bearish_bias=False,
            bullish_bias=True,
        )

    # ---- chop detection (sit out) ----
    if len(last_60) >= 60 and rng_pct < chop_60m_range_pct:
        return IntradayState(
            label="chop",
            pct_change_5m=pct_5m, pct_change_30m=pct_30m,
            realized_range_60m=rng_pct,
            reason=(f"SPY 60m range={rng_pct:.2%} "
                     f"< {chop_60m_range_pct:.2%} — chop, sit out"),
            size_mult_bias=0.50,             # halve size when chop allowed
            override_floors=False,
            bearish_bias=False, bullish_bias=False,
        )

    # ---- normal ----
    return IntradayState(
        label="normal",
        pct_change_5m=pct_5m, pct_change_30m=pct_30m,
        realized_range_60m=rng_pct,
        reason="normal_session",
        size_mult_bias=1.0,
        override_floors=False,
        bearish_bias=False, bullish_bias=False,
    )
