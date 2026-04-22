"""Fast exit evaluator — runs every 5 seconds, independent of main loop.

- 0DTE/1DTE: sells at +35% profit or -20% loss
- 2+ DTE: sells at +50% profit or -30% loss

Applies BEFORE the main 6-layer engine. Designed to lock gains and stop
bleeding with minimum latency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.types import Position, ExitDecision, OptionRight


@dataclass
class FastExitConfig:
    pt_short_pct: float = 0.35
    pt_multi_pct: float = 0.50
    sl_short_pct: float = 0.20
    sl_multi_pct: float = 0.30
    # Force-close same-day expiry positions after this many minutes of
    # being held. 0DTE options decay fast — holding past the scalp
    # window just donates premium to theta.
    zero_dte_max_hold_minutes: float = 30.0
    # Dynamic scalp-window — tightens when the trade is moving against
    # you (underlying going the wrong way + theta bleeding you → cut fast)
    # and loosens when it's working (let the winner run).
    #
    # The effective timeout is:
    #   base + (extension if pnl >= +favorable_threshold
    #           else -reduction if pnl <= -unfavorable_threshold
    #           else 0)
    #
    # P&L is used as the direction-alignment signal because for long
    # directional options (long call = bullish bet, long put = bearish
    # bet), unrealized P&L is equivalent to "is the underlying moving
    # the way I bet?" The option's price captures both direction AND
    # theta, which is what we actually care about.
    zero_dte_favorable_pnl_threshold: float = 0.10      # +10% unrealized → trade is working
    zero_dte_unfavorable_pnl_threshold: float = 0.10    # -10% unrealized → trade is failing
    zero_dte_favorable_extension_minutes: float = 15.0  # add time when winning
    zero_dte_unfavorable_reduction_minutes: float = 20.0  # subtract time when losing
    # Scale-out + trailing stop — lets winners run.
    # At first PT hit, close `scale_out_fraction` of the position. The
    # remainder then rides a trailing stop at peak * (1 - trailing_stop_pct).
    # This is what momentum-options prop desks do: lock half the gain,
    # let the rest chase higher highs, exit on first meaningful retrace.
    scale_out_fraction: float = 0.50
    trailing_stop_pct: float = 0.10
    trailing_stop_enabled: bool = True
    hard_cap_pct: float = 1.50
    # Momentum-exit: close when the underlying's momentum reverses or
    # volume dries up, even if we haven't hit the fixed PT. Only fires
    # when the position is in profit (> min_profit_to_exit) — we don't
    # exit losers on noise; stop-loss handles those.
    momentum_exit_enabled: bool = True
    momentum_exit_min_profit: float = 0.10      # must be up 10% first
    momentum_exit_reverse_bars: int = 3          # N consecutive bars against direction
    momentum_exit_vol_multiple: float = 0.50     # volume < 0.5× avg = dry-up
    momentum_exit_vol_lookback: int = 15         # baseline volume window
    momentum_exit_stall_lookback: int = 5        # "no new high" window

    # ---- Intelligent profit protection (operator-requested) ----
    # "If we're in profit and price moves the other way we can lose
    #  even if SL didn't hit. Need smarter detection. +75% went back
    #  to +1% — we need to detect trend reversal from charts."
    #
    # Profit-lock trailing: activates at `profit_lock_arm_pct`, closes
    # when pnl retraces past the adaptive give-back threshold. Works
    # for single-contract positions (doesn't require scale-out to arm).
    # Bigger peaks get tighter give-backs so massive winners don't
    # evaporate.
    profit_lock_enabled: bool = True
    profit_lock_arm_pct: float = 0.03        # arm once we've been +3%
    profit_lock_give_back_pct: float = 0.30  # base give-back (adaptive below)
    # DTE-aware tightening: 0DTE can't recover, so exits must fire
    # FASTER than for swing trades. Multipliers applied to thresholds.
    # Operator: "0DTE we need to act fast. Fine with longer ones since
    # we still have ways to go."
    #   0DTE:     arm +2%, give-back multiplied 0.5x (tighter)
    #   short:    default (1-7 DTE)
    #   swing:    arm +4%, give-back multiplied 1.5x (looser)
    dte_aware_enabled: bool = True
    # Adaptive give-back by peak tier — SMALL WINNERS TOO, because
    # operator saw +7% -> +1% -> negative. Rule: once we've touched a
    # profit peak, the position must NEVER cross back to zero pnl.
    #   peak >= +100% → 15% give-back (floor +85%)
    #   peak >= +50%  → 20% give-back (floor +40%)
    #   peak >= +25%  → 25% give-back (floor +19%)
    #   peak >= +10%  → 40% give-back (floor +6%)
    #   peak >= +5%   → 50% give-back (floor +2.5%)
    #   peak >= +3%   → 60% give-back (floor +1.2%) — saves small winners
    profit_lock_tiers_enabled: bool = True
    # Zero-tolerance floor: once we've been +3%+, never let the
    # position cross back below +0.5% (minus round-trip slippage cost).
    # This is the invariant the operator wants: "if +7% drops, we sell
    # while still in profit, not after it goes negative."
    zero_tolerance_floor_pct: float = 0.005

    # Ratcheting profit floor: once we've been above these thresholds,
    # SL tightens up. Ensures we don't give back big gains.
    #   Been above +25% → never exit below +10% (hard floor on that trade)
    #   Been above +50% → never exit below +25%
    #   Been above +100% → never exit below +50%
    profit_floor_enabled: bool = True

    # Support-break exit: when in profit AND underlying breaks VWAP /
    # prior bar low (for calls) or high (for puts), take profit. The
    # thesis for going long is broken once the underlying starts making
    # lower highs / higher lows against us.
    support_break_enabled: bool = True
    support_break_min_profit: float = 0.08   # only fires if up 8%+

    # Chart-reversal detectors — fire while in profit when the
    # underlying's bar structure screams "trend changed":
    #
    #   a. Lower-high pattern (for long calls): the most recent bar's
    #      high is lower than the prior bar's high which was lower
    #      than the bar before that. Classic trend-break signature.
    #      For long puts: higher-low mirror.
    #   b. VWAP break: underlying closes below its session VWAP (for
    #      long calls). Institutional "value" line is lost — most
    #      day-traders close here.
    chart_reversal_enabled: bool = True
    chart_reversal_min_profit: float = 0.05    # only fires +5%+
    # Lower-high pattern: 2 consecutive lower highs = trend weakening,
    # 3 = trend reversing. Conservative default = 2 (faster protect).
    lower_high_bars: int = 2
    # VWAP break: close crosses below session VWAP by this margin (in
    # percent of price) before we declare thesis broken. Tiny margin
    # avoids thrashing on sideways chop at VWAP.
    vwap_break_margin_pct: float = 0.001      # 0.1% (e.g. $0.50 on $500)


class FastExitEvaluator:
    def __init__(self, cfg: FastExitConfig = FastExitConfig()):
        self.cfg = cfg

    def _momentum_exit(self, pos: Position, bars) -> Optional[ExitDecision]:
        """Close-on-trend-reversal exit for long calls/puts.

        Two independent triggers (either fires = close):
          A. **Reverse bars** — N consecutive candles against our
             direction (red for calls, green for puts). Default N=3.
             Indicates momentum died.
          B. **Volume dry-up + stall** — recent avg volume < 0.5× of the
             prior baseline window, AND the underlying hasn't made a
             new extreme (high for calls, low for puts) in the last K
             bars. Indicates buying/selling interest faded.

        The caller guards with a min-profit-threshold so we only fire
        on positions already in the green.
        """
        cfg = self.cfg
        # Direction derived from the option right (we only go long).
        is_bullish = (pos.right == OptionRight.CALL)
        # --- A. reverse bars ---
        n = cfg.momentum_exit_reverse_bars
        recent = bars[-n:]
        if is_bullish:
            if all(b.close < b.open for b in recent):
                return ExitDecision(
                    True,
                    f"momentum_reversal:{n}_red_bars_against_call",
                    layer=0,
                )
        else:
            if all(b.close > b.open for b in recent):
                return ExitDecision(
                    True,
                    f"momentum_reversal:{n}_green_bars_against_put",
                    layer=0,
                )
        # --- B. volume dry-up + stall ---
        # Baseline = the older portion of the volume lookback window,
        # excluding the very latest bars (so we compare "recent" vs
        # "before").
        lb = cfg.momentum_exit_vol_lookback
        stall = cfg.momentum_exit_stall_lookback
        if len(bars) >= lb + stall:
            baseline = bars[-(lb + stall):-stall]
            recent_window = bars[-stall:]
            avg_baseline = sum(b.volume for b in baseline) / max(1, len(baseline))
            avg_recent = sum(b.volume for b in recent_window) / max(1, len(recent_window))
            if avg_baseline > 0 and avg_recent < cfg.momentum_exit_vol_multiple * avg_baseline:
                if is_bullish:
                    recent_high = max(b.high for b in recent_window)
                    baseline_high = max(b.high for b in baseline)
                    if recent_high <= baseline_high:
                        return ExitDecision(
                            True,
                            f"volume_dry_up:{avg_recent/avg_baseline:.2f}x"
                            f"_and_no_new_high",
                            layer=0,
                        )
                else:
                    recent_low = min(b.low for b in recent_window)
                    baseline_low = min(b.low for b in baseline)
                    if recent_low >= baseline_low:
                        return ExitDecision(
                            True,
                            f"volume_dry_up:{avg_recent/avg_baseline:.2f}x"
                            f"_and_no_new_low",
                            layer=0,
                        )
        return None

    def _effective_0dte_timeout(self, pnl_pct: float) -> float:
        """Compute the current max-hold window based on unrealized P&L.
        Winning trades get extra time; losing trades get cut faster."""
        base = self.cfg.zero_dte_max_hold_minutes
        if base <= 0:
            return 0.0
        if pnl_pct >= self.cfg.zero_dte_favorable_pnl_threshold:
            return base + self.cfg.zero_dte_favorable_extension_minutes
        if pnl_pct <= -self.cfg.zero_dte_unfavorable_pnl_threshold:
            return max(1.0, base - self.cfg.zero_dte_unfavorable_reduction_minutes)
        return base

    def evaluate(self, pos: Position, current_price: float,
                  bars=None) -> Optional[ExitDecision]:
        dte = pos.dte()
        pnl = pos.unrealized_pnl_pct(current_price)
        short_dte = dte <= 1
        # 0DTE scalp-window timeout — fires BEFORE PT/SL so it reliably
        # caps the hold time on same-day contracts. Timeout is DYNAMIC:
        # contracts faster when the trade is moving against you, expands
        # when it's working. Applies only to DTE==0 (not 1DTE, which has
        # overnight theta already priced in).
        if dte == 0 and self.cfg.zero_dte_max_hold_minutes > 0:
            import time as _time
            hold_min = max(0.0, (_time.time() - float(pos.entry_ts)) / 60.0)
            effective_max = self._effective_0dte_timeout(pnl)
            if hold_min > effective_max:
                return ExitDecision(
                    True,
                    f"fast_0dte_scalp_timeout:{hold_min:.0f}min>"
                    f"{effective_max:.0f}min@pnl={pnl:+.1%}",
                    layer=0,
                )
        if short_dte:
            pt, sl = self.cfg.pt_short_pct, self.cfg.sl_short_pct
        else:
            pt, sl = self.cfg.pt_multi_pct, self.cfg.sl_multi_pct

        # Update peak price for trailing-stop math. Mutating pos here
        # is intentional — main.py persists the state via the broker
        # snapshot, and the next fast_loop tick reads the updated peak.
        if pos.is_long:
            if pos.peak_price is None or current_price > pos.peak_price:
                pos.peak_price = current_price

        # Hard cap: close fully at runaway wins.
        if pnl >= self.cfg.hard_cap_pct:
            return ExitDecision(True, f"fast_hard_cap:{pnl:.2%}", layer=0)

        # Stop loss always fires first.
        if pnl <= -sl:
            return ExitDecision(True, f"fast_sl_hit:{pnl:.2%}", layer=0)

        # Momentum/volume exit — close when the underlying trend reverses
        # even if we haven't hit the fixed PT. Only fires when already
        # in profit so we don't exit on noise when the trade is still
        # underwater. Requires bars to be passed in from the caller.
        if (self.cfg.momentum_exit_enabled
                and bars is not None
                and len(bars) >= max(
                    self.cfg.momentum_exit_reverse_bars,
                    self.cfg.momentum_exit_vol_lookback,
                    self.cfg.momentum_exit_stall_lookback,
                )
                and pnl >= self.cfg.momentum_exit_min_profit):
            mx = self._momentum_exit(pos, bars)
            if mx is not None:
                return mx

        # ---- Profit-lock trailing (works for 1-contract positions) ----
        # Track peak unrealized PnL% on the position so single-contract
        # trades get trailing protection without waiting for scale-out.
        peak_pnl_attr = getattr(pos, "peak_pnl_pct", None)
        if peak_pnl_attr is None or pnl > peak_pnl_attr:
            try:
                pos.peak_pnl_pct = pnl
                peak_pnl_attr = pnl
            except Exception:
                peak_pnl_attr = pnl
        # DTE-aware arm threshold: 0DTE fires sooner, swing fires later.
        arm_pct = self.cfg.profit_lock_arm_pct
        if self.cfg.dte_aware_enabled:
            if dte == 0:
                arm_pct = 0.02     # 0DTE: arm at +2%
            elif dte >= 14:
                arm_pct = 0.04     # swing: arm at +4% (let it breathe)

        if (self.cfg.profit_lock_enabled
                and peak_pnl_attr is not None
                and peak_pnl_attr >= arm_pct):
            # Adaptive give-back: bigger winners get tighter protection.
            # Operator: "+7% went to +1% then negative — can't let that
            # happen. If we were up, we must NEVER close negative."
            if self.cfg.profit_lock_tiers_enabled:
                if peak_pnl_attr >= 1.00:
                    give_back = 0.15    # +100%+ peak → lock hard
                elif peak_pnl_attr >= 0.50:
                    give_back = 0.20
                elif peak_pnl_attr >= 0.25:
                    give_back = 0.25
                elif peak_pnl_attr >= 0.10:
                    give_back = 0.40
                elif peak_pnl_attr >= 0.05:
                    give_back = 0.50
                else:
                    give_back = 0.60    # +3-5% peak → loose but still armed
            else:
                give_back = self.cfg.profit_lock_give_back_pct
            # DTE-aware give-back: 0DTE tighter, swing looser
            if self.cfg.dte_aware_enabled:
                if dte == 0:
                    give_back = min(give_back * 0.5, 0.30)   # much tighter
                elif dte >= 14:
                    give_back = min(give_back * 1.5, 0.80)   # much looser
            close_threshold = peak_pnl_attr * (1.0 - give_back)
            # Zero-tolerance floor: if we've been above arm threshold
            # ever, the floor is at LEAST zero_tolerance_floor_pct.
            # Even if the adaptive give-back would drop below, we
            # never let a position that was +3%+ cross back to <= 0.
            close_threshold = max(close_threshold,
                                     self.cfg.zero_tolerance_floor_pct)
            if pnl <= close_threshold and pnl > 0:
                dte_tag = ("0dte" if dte == 0 else
                             "swing" if dte >= 14 else "short")
                return ExitDecision(
                    True,
                    f"profit_lock_{dte_tag}:peak={peak_pnl_attr:+.2%}"
                    f"_now={pnl:+.2%}_gave_back_{give_back:.0%}"
                    f"_floor={close_threshold:+.2%}",
                    layer=0,
                )

        # ---- Ratcheting profit floor ----
        # Once we've crossed a profit tier, never close below its floor.
        # Triggered when current pnl falls below the floor of the
        # highest tier we've ever touched.
        if self.cfg.profit_floor_enabled and peak_pnl_attr is not None:
            tiers = [
                (1.00, 0.50),    # >+100% peak → floor +50%
                (0.50, 0.25),    # >+50%  peak → floor +25%
                (0.25, 0.10),    # >+25%  peak → floor +10%
            ]
            for tier_peak, tier_floor in tiers:
                if peak_pnl_attr >= tier_peak and pnl <= tier_floor and pnl > 0:
                    return ExitDecision(
                        True,
                        f"profit_floor:tier={tier_peak:+.0%}"
                        f"_peak={peak_pnl_attr:+.2%}_floor={tier_floor:+.0%}"
                        f"_now={pnl:+.2%}",
                        layer=0,
                    )
                    # first matching tier wins; stop checking
                    break

        # ---- Support-break exit ----
        # Long call thesis breaks when underlying makes a lower low on
        # the close (support lost). Long put thesis breaks when it
        # makes a higher high. Only fires when in profit — we don't
        # cut losers on one red bar.
        if (self.cfg.support_break_enabled
                and bars is not None and len(bars) >= 5
                and pnl >= self.cfg.support_break_min_profit):
            is_bullish = (pos.right == OptionRight.CALL)
            recent = bars[-5:]
            closes = [b.close for b in recent]
            lows = [b.low for b in recent]
            highs = [b.high for b in recent]
            if is_bullish:
                # Support break = last close breaks the 5-bar low
                prior_low = min(lows[:-1])
                if closes[-1] < prior_low:
                    return ExitDecision(
                        True,
                        f"support_break:close={closes[-1]:.2f}<5bar_low"
                        f"={prior_low:.2f}_pnl={pnl:+.2%}",
                        layer=0,
                    )
            else:
                # Resistance break = last close breaks the 5-bar high
                prior_high = max(highs[:-1])
                if closes[-1] > prior_high:
                    return ExitDecision(
                        True,
                        f"resistance_break:close={closes[-1]:.2f}>5bar_high"
                        f"={prior_high:.2f}_pnl={pnl:+.2%}",
                        layer=0,
                    )

        # ---- Chart-reversal detectors ----
        # Operator: "the algo needs to be intelligent enough monitoring
        # charts that it's heading to another way — close and get
        # profit." Two new layers:
        #   (1) Lower-high / higher-low sequence — trend change
        #   (2) VWAP break — institutional value line lost
        if (self.cfg.chart_reversal_enabled
                and bars is not None
                and len(bars) >= max(self.cfg.lower_high_bars + 1, 10)
                and pnl >= self.cfg.chart_reversal_min_profit):
            is_bullish = (pos.right == OptionRight.CALL)
            lh_n = int(self.cfg.lower_high_bars)
            tail = bars[-(lh_n + 1):]        # need lh_n+1 bars to compare
            # (1) Lower-high pattern for long calls
            if is_bullish:
                highs_seq = [b.high for b in tail]
                # Strictly decreasing high sequence = lower-high pattern
                if all(highs_seq[i] > highs_seq[i + 1]
                        for i in range(len(highs_seq) - 1)):
                    return ExitDecision(
                        True,
                        f"chart_lower_highs:{lh_n}_consec_lower_highs"
                        f"_pnl={pnl:+.2%}_take_profit",
                        layer=0,
                    )
            else:
                # Higher-low pattern for long puts = uptrend resuming
                lows_seq = [b.low for b in tail]
                if all(lows_seq[i] < lows_seq[i + 1]
                        for i in range(len(lows_seq) - 1)):
                    return ExitDecision(
                        True,
                        f"chart_higher_lows:{lh_n}_consec_higher_lows"
                        f"_pnl={pnl:+.2%}_take_profit",
                        layer=0,
                    )
            # (2) VWAP break — compute rolling VWAP on the supplied bars
            # (approximates session VWAP well over 30+ bars). If the
            # last close crosses below VWAP for a long call (above VWAP
            # for a long put) by more than vwap_break_margin_pct,
            # institutional value line is lost → take profit.
            try:
                typical = [(b.high + b.low + b.close) / 3 for b in bars]
                vols = [max(1.0, b.volume or 0) for b in bars]
                tp_vol_sum = sum(t * v for t, v in zip(typical, vols))
                vol_sum = sum(vols) or 1.0
                vwap = tp_vol_sum / vol_sum
                last_close = bars[-1].close
                margin = self.cfg.vwap_break_margin_pct * max(vwap, 1e-9)
                if is_bullish and last_close < (vwap - margin):
                    return ExitDecision(
                        True,
                        f"vwap_break:close={last_close:.2f}<vwap={vwap:.2f}"
                        f"_pnl={pnl:+.2%}_take_profit",
                        layer=0,
                    )
                if (not is_bullish) and last_close > (vwap + margin):
                    return ExitDecision(
                        True,
                        f"vwap_break_up:close={last_close:.2f}>vwap={vwap:.2f}"
                        f"_pnl={pnl:+.2%}_take_profit",
                        layer=0,
                    )
            except Exception:
                pass

        # Scale-out at first PT hit: close half, flag pos.scaled_out,
        # let the remainder trail the high. Requires at least 2 contracts
        # to split; with 1 contract we fall back to full close.
        can_scale = (
            self.cfg.scale_out_fraction > 0
            and not pos.scaled_out
            and abs(pos.qty) >= 2
        )
        if pnl >= pt:
            if can_scale:
                half = max(1, int(round(abs(pos.qty)
                                          * self.cfg.scale_out_fraction)))
                return ExitDecision(
                    True,
                    f"fast_scale_out_at_pt:{pnl:.2%}_closing_{half}of{abs(pos.qty)}",
                    layer=0,
                    close_qty=half,
                )
            # One contract, can't split → legacy full close at PT.
            return ExitDecision(True, f"fast_pt_hit:{pnl:.2%}", layer=0)

        # Trailing stop for the runner. Only active after scale-out.
        # Close the remainder once price retraces from the peak beyond
        # `trailing_stop_pct`. Captures the bulk of a big move, exits on
        # the first real retrace.
        if (self.cfg.trailing_stop_enabled
                and pos.scaled_out
                and pos.peak_price and pos.peak_price > 0):
            if pos.is_long:
                retrace = (pos.peak_price - current_price) / pos.peak_price
            else:
                retrace = (current_price - pos.peak_price) / pos.peak_price
            if retrace >= self.cfg.trailing_stop_pct:
                peak_pnl = pos.unrealized_pnl_pct(pos.peak_price)
                return ExitDecision(
                    True,
                    f"fast_trailing_stop:peak_pnl={peak_pnl:+.2%}_now="
                    f"{pnl:+.2%}_retrace={retrace:.2%}",
                    layer=0,
                )
        return None
