"""14-Filter Execution Chain.

Each filter is pure: (ctx) -> (passed: bool, reason: str).
The chain short-circuits on the first failure. Every decision is logged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional
from datetime import datetime

from ..core.clock import MarketClock
from ..core.types import Signal, OptionContract
from ..core.logger import get_logger
from .iv_rank import iv_rank


log = get_logger(__name__)


@dataclass
class FilterResult:
    passed: bool
    reason: str
    advisory: bool = False


@dataclass
class ExecutionContext:
    signal: Signal
    now: datetime
    account_equity: float
    day_pnl: float
    open_positions_count: int
    current_bar_volume: float
    avg_bar_volume: float
    opening_range_high: float
    opening_range_low: float
    contract: Optional[OptionContract] = None
    vix: float = 15.0
    current_iv: float = 0.25
    iv_52w_low: float = 0.10
    iv_52w_high: float = 0.50
    vwap: float = 0.0
    spot: float = 0.0
    is_etf: bool = False
    zero_dte_count_today: int = 0
    econ_blackout: bool = False
    news_score: float = 0.0           # -1..+1, 0 = no news or neutral
    news_label: str = "neutral"       # 'negative' | 'neutral' | 'positive'
    news_rationale: str = ""          # short, loggable
    # Recent price bars (last ~20-80) — used by the momentum-confirmation
    # filter to verify the underlying actually moved in signal direction.
    recent_bars: list = None          # type: ignore[assignment]


class ExecutionChain:
    def __init__(self, settings: Dict[str, Any], clock: MarketClock):
        self._s = settings
        self._clock = clock

    # ---------- filters ----------
    def f01_daily_loss_limit(self, ctx: ExecutionContext) -> FilterResult:
        cap = self._s["account"]["max_daily_loss_pct"] * ctx.account_equity
        if ctx.day_pnl <= -cap:
            return FilterResult(False, f"daily_loss_limit: day_pnl {ctx.day_pnl:.2f} <= -{cap:.2f}")
        return FilterResult(True, "ok")

    def f02_max_positions(self, ctx: ExecutionContext) -> FilterResult:
        cap = self._s["account"]["max_open_positions"]
        if ctx.open_positions_count >= cap:
            return FilterResult(False, f"max_positions: {ctx.open_positions_count}>={cap}")
        return FilterResult(True, "ok")

    def f03_session_window(self, ctx: ExecutionContext) -> FilterResult:
        if not self._clock.is_market_open(ctx.now):
            return FilterResult(False, "session_closed")
        return FilterResult(True, "ok")

    def f04_no_new_entries_late(self, ctx: ExecutionContext) -> FilterResult:
        if not self._clock.can_enter_new(ctx.now):
            return FilterResult(False, "no_new_entries_after_cutoff")
        return FilterResult(True, "ok")

    def f05_economic_calendar(self, ctx: ExecutionContext) -> FilterResult:
        if ctx.econ_blackout:
            return FilterResult(False, "econ_blackout")
        return FilterResult(True, "ok")

    def f06_vix_filter(self, ctx: ExecutionContext) -> FilterResult:
        v = ctx.vix
        halt = self._s["vix"]["halt_above"]
        if v > halt:
            return FilterResult(False, f"vix_halt: {v:.1f}>{halt}")
        return FilterResult(True, "ok")

    def f07_iv_rank_gate(self, ctx: ExecutionContext) -> FilterResult:
        rank = iv_rank(ctx.current_iv, ctx.iv_52w_low, ctx.iv_52w_high)
        meta = ctx.signal.meta
        action = meta.get("premium_action")   # 'sell' | 'buy' | None
        if action == "sell" and rank < self._s["iv_rank"]["block_sell_below"]:
            return FilterResult(False, f"iv_rank_too_low_for_sell: {rank:.2f}")
        if action == "buy" and rank > self._s["iv_rank"]["block_buy_above"]:
            return FilterResult(False, f"iv_rank_too_high_for_buy: {rank:.2f}")
        return FilterResult(True, "ok")

    def f08_vwap_bias(self, ctx: ExecutionContext) -> FilterResult:
        # advisory: logs but doesn't block (matches your spec)
        if ctx.vwap <= 0 or ctx.spot <= 0:
            return FilterResult(True, "vwap_bias: no_data", advisory=True)
        above = ctx.spot > ctx.vwap
        return FilterResult(True, f"vwap_bias: {'above' if above else 'below'}_vwap",
                            advisory=True)

    def f09_volume_confirmation(self, ctx: ExecutionContext) -> FilterResult:
        """Volume confirmation with ETF-aware thresholds.

        ETFs (SPY/QQQ/IWM + sector ETFs) are inherently high-liquidity;
        their per-minute bar volume doesn't need to exceed average to be
        tradable — the average itself is already huge. Individual stocks
        do need volume confirmation to separate genuine momentum from
        chop. Two separate thresholds:
          execution.min_volume_confirmation_etf   (default 0.80)
          execution.min_volume_confirmation_stock (default 1.20)

        `execution.min_volume_confirmation` (legacy) is used as a shared
        default if the split values aren't set, for backward compat.
        """
        ex = self._s["execution"]
        legacy = float(ex.get("min_volume_confirmation", 1.20))
        if ctx.is_etf:
            min_mult = float(ex.get("min_volume_confirmation_etf", 0.80))
        else:
            min_mult = float(ex.get("min_volume_confirmation_stock", legacy))
        if ctx.avg_bar_volume <= 0:
            return FilterResult(True, "vol_confirm: no_avg", advisory=True)
        mult = ctx.current_bar_volume / ctx.avg_bar_volume
        kind = "etf" if ctx.is_etf else "stock"
        if mult < min_mult:
            return FilterResult(
                False, f"vol_confirm[{kind}]: {mult:.2f}<{min_mult}"
            )
        return FilterResult(True, f"vol_confirm[{kind}]: {mult:.2f}>={min_mult}")

    def f10_orb_breakout(self, ctx: ExecutionContext) -> FilterResult:
        # Only apply to signals that explicitly require ORB confirmation.
        # Momentum is a separate strategy with its own entry logic.
        src = ctx.signal.source.lower()
        if src not in {"orb"}:
            return FilterResult(True, "orb_na", advisory=True)
        if ctx.opening_range_high <= 0 or ctx.opening_range_low <= 0:
            return FilterResult(True, "orb_no_range", advisory=True)
        bullish = ctx.signal.meta.get("direction", "").lower() == "bullish"
        bearish = ctx.signal.meta.get("direction", "").lower() == "bearish"
        if bullish and ctx.spot <= ctx.opening_range_high:
            return FilterResult(False, "orb_no_upper_breakout")
        if bearish and ctx.spot >= ctx.opening_range_low:
            return FilterResult(False, "orb_no_lower_breakout")
        return FilterResult(True, "ok")

    def f11_spread_validator(self, ctx: ExecutionContext) -> FilterResult:
        c = ctx.contract
        if c is None:
            return FilterResult(True, "spread_na_no_contract", advisory=True)
        cap = (self._s["execution"]["max_spread_pct_etf"] if ctx.is_etf
               else self._s["execution"]["max_spread_pct_stock"])
        if c.spread_pct > cap:
            return FilterResult(False, f"spread_too_wide: {c.spread_pct:.3f}>{cap}")
        return FilterResult(True, "ok")

    def f12_open_interest(self, ctx: ExecutionContext) -> FilterResult:
        c = ctx.contract
        if c is None:
            return FilterResult(True, "oi_na_no_contract", advisory=True)
        min_oi = self._s["execution"]["min_open_interest"]
        min_vol = self._s["execution"]["min_today_option_volume"]
        if c.open_interest < min_oi:
            return FilterResult(False, f"oi_too_low: {c.open_interest}<{min_oi}")
        if c.today_volume < min_vol:
            return FilterResult(False, f"vol_today_too_low: {c.today_volume}<{min_vol}")
        return FilterResult(True, "ok")

    def f13_0dte_cap(self, ctx: ExecutionContext) -> FilterResult:
        c = ctx.contract
        if c is None or c.expiry is None:
            return FilterResult(True, "0dte_na", advisory=True)
        dte = (c.expiry - ctx.now.date()).days
        if dte == 0:
            cap = self._s["execution"]["max_0dte_per_day"]
            if ctx.zero_dte_count_today >= cap:
                return FilterResult(False, f"0dte_cap: {ctx.zero_dte_count_today}>={cap}")
        return FilterResult(True, "ok")

    def f14_mi_edge_gate(self, ctx: ExecutionContext) -> FilterResult:
        mi_edge = ctx.signal.meta.get("mi_edge_score")
        if mi_edge is None:
            return FilterResult(True, "mi_edge_absent", advisory=True)
        cutoff = self._s["mi_edge"]["block_below_combined_score"]
        if mi_edge < cutoff:
            return FilterResult(False, f"mi_edge_block: {mi_edge}<{cutoff}")
        return FilterResult(True, "ok")

    def f15_news_filter(self, ctx: ExecutionContext) -> FilterResult:
        """Block trades where strong recent news contradicts the signal.

        Defaults (tunable via settings['news']):
          block_score  = 0.50  magnitude required to block
        Bullish signals blocked by score <= -block_score.
        Bearish signals blocked by score >=  +block_score.
        Premium-harvest (directionless) only blocks on |score| >= 0.75.
        """
        cfg = self._s.get("news", {}) or {}
        block_score = float(cfg.get("block_score", 0.50))
        premium_block = float(cfg.get("premium_harvest_block_score", 0.75))

        direction = ctx.signal.meta.get("direction", "").lower()
        s = ctx.news_score
        if direction == "bullish" and s <= -block_score:
            return FilterResult(False, f"news_negative_for_long: {s:+.2f} {ctx.news_rationale}")
        if direction == "bearish" and s >= block_score:
            return FilterResult(False, f"news_positive_for_short: {s:+.2f} {ctx.news_rationale}")
        if direction == "premium_harvest" and abs(s) >= premium_block:
            return FilterResult(False, f"news_shock_premium_harvest: {s:+.2f} {ctx.news_rationale}")
        if ctx.news_label != "neutral":
            return FilterResult(True, f"news_{ctx.news_label}:{s:+.2f}", advisory=True)
        return FilterResult(True, "news_neutral_or_none", advisory=True)

    def f16_vwap_alignment(self, ctx: ExecutionContext) -> FilterResult:
        """Reject trades fighting the intraday VWAP trend.

        A bullish directional bet (long call) should not enter when the
        spot is *below* VWAP — that means the session is trending down
        and we'd be buying strength that isn't there. Conversely for
        bearish (long put) below-VWAP entries are preferred.

        Disabled when execution.vwap_alignment_enabled=false (default:
        true).
        """
        if not bool(self._s.get("execution", {}).get(
                "vwap_alignment_enabled", True)):
            return FilterResult(True, "vwap_align_disabled", advisory=True)
        if ctx.vwap <= 0 or ctx.spot <= 0:
            return FilterResult(True, "vwap_align_na", advisory=True)
        direction = (ctx.signal.meta.get("direction") or "").lower()
        diff_pct = (ctx.spot - ctx.vwap) / ctx.vwap
        if direction == "bullish" and diff_pct < 0:
            return FilterResult(
                False,
                f"vwap_align_wrong_side: bullish but spot {diff_pct:+.2%} vs VWAP",
            )
        if direction == "bearish" and diff_pct > 0:
            return FilterResult(
                False,
                f"vwap_align_wrong_side: bearish but spot {diff_pct:+.2%} vs VWAP",
            )
        return FilterResult(True, f"vwap_align_ok: {diff_pct:+.2%}")

    def f17_momentum_confirmation(self, ctx: ExecutionContext) -> FilterResult:
        """Require the underlying to have already moved in the signal
        direction over the last few bars. Blocks "entering on noise"
        signals where the model's score is positive but price action
        hasn't confirmed.

        Reads ctx.recent_bars (populated by main.py when available).
        Requires last-N-bar return >= min move threshold with correct sign.
        """
        if not bool(self._s.get("execution", {}).get(
                "momentum_confirmation_enabled", True)):
            return FilterResult(True, "momentum_conf_disabled", advisory=True)
        bars = getattr(ctx, "recent_bars", None) or []
        if len(bars) < 5:
            return FilterResult(True, "momentum_conf_na", advisory=True)
        direction = (ctx.signal.meta.get("direction") or "").lower()
        k = 5
        start = bars[-k].open
        end = bars[-1].close
        if start <= 0:
            return FilterResult(True, "momentum_conf_na", advisory=True)
        move = (end - start) / start
        min_move = float(self._s.get("execution", {}).get(
            "momentum_confirmation_min_move", 0.002))  # 0.2% over 5 bars
        if direction == "bullish" and move < min_move:
            return FilterResult(
                False,
                f"momentum_weak: +{move:.3%} < +{min_move:.1%} (5-bar move)",
            )
        if direction == "bearish" and move > -min_move:
            return FilterResult(
                False,
                f"momentum_weak: {move:+.3%} > -{min_move:.1%} (5-bar move)",
            )
        return FilterResult(True, f"momentum_ok: {move:+.3%}")

    # ---------- runner ----------
    def run(self, ctx: ExecutionContext) -> List[FilterResult]:
        filters: List[Callable[[ExecutionContext], FilterResult]] = [
            self.f01_daily_loss_limit, self.f02_max_positions,
            self.f03_session_window, self.f04_no_new_entries_late,
            self.f05_economic_calendar, self.f06_vix_filter,
            self.f07_iv_rank_gate, self.f08_vwap_bias,
            self.f09_volume_confirmation, self.f10_orb_breakout,
            self.f11_spread_validator, self.f12_open_interest,
            self.f13_0dte_cap, self.f14_mi_edge_gate,
            self.f15_news_filter,
            self.f16_vwap_alignment, self.f17_momentum_confirmation,
        ]
        results: List[FilterResult] = []
        for f in filters:
            r = f(ctx)
            results.append(r)
            if not r.passed and not r.advisory:
                log.info("exec_chain_block", filter=f.__name__, reason=r.reason)
                return results
        log.info("exec_chain_pass", signal=ctx.signal.source)
        return results

    @staticmethod
    def decided_pass(results: List[FilterResult]) -> bool:
        return all(r.passed or r.advisory for r in results)
