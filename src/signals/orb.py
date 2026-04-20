"""Opening Range Breakout (ORB) scalp signal.

Two modes:

  - **immediate** (default): fire on the first bar that breaks the
    opening range. Simple, catches all real breakouts, but eats
    whipsaws/fake-outs near the boundary.

  - **retest**: after a break, wait for a pullback to within
    `retest_band_pct` of the breached boundary, then require a bar
    that closes back in the breakout direction before firing. Cuts
    false breakouts substantially; misses pure runaway breaks.

The retest mode is stateful (per-symbol). State is kept in-memory;
restart-tolerant because the bot re-enters the same state on the
first few bars of trading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from ..core.types import Signal, Side, OptionRight
from .base import SignalSource, SignalContext


class _BreakState(str, Enum):
    """Per-symbol breakout state machine."""
    IDLE = "idle"                    # inside the range
    BROKE_UP = "broke_up"            # bar closed above OR high
    BROKE_DN = "broke_dn"            # bar closed below OR low
    RETESTED_UP = "retested_up"      # pulled back to ORH band
    RETESTED_DN = "retested_dn"      # pulled back to ORL band
    FIRED = "fired"                  # signal emitted — don't re-fire


@dataclass
class _SymbolState:
    state: _BreakState = _BreakState.IDLE


class OpeningRangeBreakout(SignalSource):
    name = "orb"

    def __init__(self, range_minutes: int = 30, *,
                 retest_required: bool = False,
                 retest_band_pct: float = 0.0015,
                 confidence_immediate: float = 0.7,
                 confidence_retest: float = 0.85):
        """
        Args:
          retest_required:       True enables the retest state machine
          retest_band_pct:       how close to OR boundary the pullback
                                 must get (default 0.15%)
          confidence_immediate:  confidence for a vanilla breakout emit
          confidence_retest:     confidence for a retest-confirmed emit
                                 (higher because it's a cleaner pattern)
        """
        self.range_minutes = range_minutes
        self.retest_required = retest_required
        self.retest_band_pct = retest_band_pct
        self.conf_imm = confidence_immediate
        self.conf_retest = confidence_retest
        self._state: Dict[str, _SymbolState] = {}

    # ---------- internal helpers ----------

    def _s(self, symbol: str) -> _SymbolState:
        if symbol not in self._state:
            self._state[symbol] = _SymbolState()
        return self._state[symbol]

    def _within_band(self, spot: float, level: float) -> bool:
        if level <= 0:
            return False
        return abs(spot - level) / level <= self.retest_band_pct

    def reset(self, symbol: Optional[str] = None) -> None:
        """Clear state — called at EOD or after a successful entry."""
        if symbol is None:
            self._state.clear()
        elif symbol in self._state:
            del self._state[symbol]

    # ---------- emit ----------

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        orh, orl = ctx.opening_range_high, ctx.opening_range_low
        if orh <= 0 or orl <= 0 or ctx.spot <= 0:
            return None

        if not self.retest_required:
            # --- immediate mode (legacy behavior) ---
            if ctx.spot > orh:
                return Signal(source=self.name, symbol=ctx.symbol,
                              side=Side.BUY, option_right=OptionRight.CALL,
                              confidence=self.conf_imm,
                              rationale=f"close>{orh:.2f}",
                              meta={"direction": "bullish", "entry_tag": "scalp"})
            if ctx.spot < orl:
                return Signal(source=self.name, symbol=ctx.symbol,
                              side=Side.BUY, option_right=OptionRight.PUT,
                              confidence=self.conf_imm,
                              rationale=f"close<{orl:.2f}",
                              meta={"direction": "bearish", "entry_tag": "scalp"})
            return None

        # --- retest mode: state machine ---
        st = self._s(ctx.symbol)

        if st.state in (_BreakState.IDLE, _BreakState.FIRED):
            # look for a fresh breakout (a bar closed outside OR)
            if ctx.spot > orh:
                st.state = _BreakState.BROKE_UP
            elif ctx.spot < orl:
                st.state = _BreakState.BROKE_DN
            return None

        if st.state == _BreakState.BROKE_UP:
            # Waiting for a pullback to within band of ORH (but not
            # below ORH — a full re-entry into the range invalidates)
            if ctx.spot < orl:
                st.state = _BreakState.BROKE_DN
                return None
            if ctx.spot < orh or self._within_band(ctx.spot, orh):
                # Pullback reached — now armed for retest fire
                st.state = _BreakState.RETESTED_UP
            return None

        if st.state == _BreakState.BROKE_DN:
            if ctx.spot > orh:
                st.state = _BreakState.BROKE_UP
                return None
            if ctx.spot > orl or self._within_band(ctx.spot, orl):
                st.state = _BreakState.RETESTED_DN
            return None

        if st.state == _BreakState.RETESTED_UP:
            # Fire when price reclaims the breakout direction — close
            # back above ORH confirms buyers absorbed the retest
            if ctx.spot > orh:
                st.state = _BreakState.FIRED
                return Signal(source=self.name, symbol=ctx.symbol,
                              side=Side.BUY, option_right=OptionRight.CALL,
                              confidence=self.conf_retest,
                              rationale=f"retest_hold>{orh:.2f}",
                              meta={"direction": "bullish", "entry_tag": "scalp",
                                    "mode": "retest"})
            if ctx.spot < orl:
                # Full range reversal — flip state
                st.state = _BreakState.BROKE_DN
            return None

        if st.state == _BreakState.RETESTED_DN:
            if ctx.spot < orl:
                st.state = _BreakState.FIRED
                return Signal(source=self.name, symbol=ctx.symbol,
                              side=Side.BUY, option_right=OptionRight.PUT,
                              confidence=self.conf_retest,
                              rationale=f"retest_hold<{orl:.2f}",
                              meta={"direction": "bearish", "entry_tag": "scalp",
                                    "mode": "retest"})
            if ctx.spot > orh:
                st.state = _BreakState.BROKE_UP
            return None

        return None
