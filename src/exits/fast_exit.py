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
    #  even if SL didn't hit. Need smarter detection."
    #
    # Profit-lock trailing: activates at `profit_lock_arm_pct`, closes
    # when pnl retraces to `profit_lock_give_back_pct`. Works for
    # single-contract positions (doesn't require scale-out to arm).
    profit_lock_enabled: bool = True
    profit_lock_arm_pct: float = 0.15        # arm once we've been +15%
    profit_lock_give_back_pct: float = 0.30  # close if we give back 30% of peak gain
    # Example: peak = +40%, give_back = 30% → close at +28% (40 * 0.70).
    # Example: peak = +20%, give_back = 30% → close at +14%.
    # Never lets a +20%+ winner turn into a loser.

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
        if (self.cfg.profit_lock_enabled
                and peak_pnl_attr is not None
                and peak_pnl_attr >= self.cfg.profit_lock_arm_pct):
            give_back = self.cfg.profit_lock_give_back_pct
            close_threshold = peak_pnl_attr * (1.0 - give_back)
            if pnl <= close_threshold and pnl > 0:
                return ExitDecision(
                    True,
                    f"profit_lock:peak={peak_pnl_attr:+.2%}_now={pnl:+.2%}"
                    f"_gave_back_{give_back:.0%}",
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
