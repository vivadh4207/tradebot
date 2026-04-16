"""BacktestSimulator — runs equity-style signals through the paper broker
with realistic slippage and the full 14-filter + 6-layer stack.

Scope kept modest: equities-style simulation (each signal results in a
single-leg long option or an equity proxy). Options are simulated as
price-fractional positions for speed; upgrade to chain-level fills when
you have historical options data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional

from ..core.clock import MarketClock, ET
from ..core.types import Bar, Order, Position, Side, Signal, OptionRight, OptionContract
from ..brokers.paper import PaperBroker
from ..data.market_data import MarketDataAdapter, SyntheticDataAdapter
from ..risk.execution_chain import ExecutionChain, ExecutionContext
from ..risk.order_validator import OrderValidator
from ..risk.position_sizer import PositionSizer, SizingInputs
from ..exits.exit_engine import ExitEngine, ExitEngineConfig
from ..exits.auto_stops import compute_auto_stops
from ..signals.base import SignalContext
from ..signals.momentum import MomentumSignal
from ..signals.vwap_reversion import VwapReversionSignal
from ..signals.orb import OpeningRangeBreakout


@dataclass
class SimConfig:
    starting_equity: float = 10_000.0
    slippage_bps: float = 2.0
    vix_series: Optional[List[float]] = None
    vix_52w_low: float = 10.0
    vix_52w_high: float = 40.0
    win_rate_prior: float = 0.58
    avg_win_prior: float = 0.030
    avg_loss_prior: float = 0.020
    verbose: bool = False


class BacktestSimulator:
    def __init__(self, settings: Dict, data: MarketDataAdapter,
                 cfg: SimConfig = SimConfig()):
        self.settings = settings
        self.data = data
        self.cfg = cfg
        self.clock = MarketClock(
            market_open=settings["session"]["market_open"],
            market_close=settings["session"]["market_close"],
            no_new_entries_after=settings["session"]["no_new_entries_after"],
            eod_force_close=settings["session"]["eod_force_close"],
        )
        self.broker = PaperBroker(
            starting_equity=cfg.starting_equity,
            slippage_bps=cfg.slippage_bps,
        )
        self.chain = ExecutionChain(settings, self.clock)
        self.validator = OrderValidator()
        self.sizer = PositionSizer(
            kelly_fraction_cap=settings["sizing"]["kelly_fraction_cap"],
            kelly_hard_cap=settings["sizing"]["kelly_hard_cap_pct"],
            max_0dte=settings["sizing"]["max_contracts_0dte"],
            max_multiday=settings["sizing"]["max_contracts_multiday"],
        )
        self.exit_engine = ExitEngine(ExitEngineConfig(
            pt_short_pct=settings["exits"]["profit_target_short_dte_pct"],
            pt_multi_pct=settings["exits"]["profit_target_multi_dte_pct"],
            sl_short_pct=settings["exits"]["stop_loss_short_dte_pct"],
            sl_multi_pct=settings["exits"]["stop_loss_multi_dte_pct"],
            hard_profit_cap_pct=settings["exits"]["hard_profit_cap_pct"],
            max_consecutive_holds=settings["exits"]["max_consecutive_holds"],
            claude_hold_conf_min=settings["exits"]["claude_hold_conf_min"],
        ))
        self.strategies = [
            MomentumSignal(
                bars=settings["signal"]["momentum_bars"],
                slope_long=settings["signal"]["momentum_slope_long"],
                slope_short=settings["signal"]["momentum_slope_short"],
            ),
            VwapReversionSignal(),
            OpeningRangeBreakout(),
        ]
        self.equity_curve: List[float] = []
        self.trade_pnls: List[float] = []

    def _vix_at(self, i: int) -> float:
        if not self.cfg.vix_series:
            return 15.0
        return self.cfg.vix_series[min(i, len(self.cfg.vix_series) - 1)]

    def _snapshot_ctx(self, symbol: str, bars: List[Bar], i: int) -> SignalContext:
        spot = bars[-1].close if bars else 0.0
        vwap = bars[-1].vwap or spot
        or_bars = bars[:30] if len(bars) >= 30 else bars[:max(1, len(bars)//3)]
        or_hi = max((b.high for b in or_bars), default=0.0)
        or_lo = min((b.low for b in or_bars), default=0.0)
        return SignalContext(
            symbol=symbol, now=bars[-1].ts, bars=bars,
            spot=spot, vwap=vwap,
            opening_range_high=or_hi, opening_range_low=or_lo,
        )

    def _apply_exits(self, now: datetime, vix: float) -> None:
        for pos in list(self.broker.positions()):
            bars = self.data.get_bars(pos.underlying or pos.symbol, limit=30)
            spot = bars[-1].close if bars else pos.avg_price
            vwap = bars[-1].vwap or spot
            decision = self.exit_engine.decide(pos, spot, now, vix, spot, vwap, bars)
            if decision.should_close:
                side = Side.SELL if pos.qty > 0 else Side.BUY
                o = Order(symbol=pos.symbol, side=side, qty=abs(pos.qty),
                          is_option=pos.is_option, limit_price=spot,
                          tag=f"exit:{decision.reason}")
                fill = self.broker.submit(o)
                if fill and self.cfg.verbose:
                    print(f"[exit] {pos.symbol} reason={decision.reason} px={fill.price:.2f}")

    def _try_enter(self, sig: Signal, bars: List[Bar], now: datetime, vix: float) -> None:
        spot = bars[-1].close
        or_bars = bars[:30] if len(bars) >= 30 else bars[:max(1, len(bars)//3)]
        or_hi = max((b.high for b in or_bars), default=0.0)
        or_lo = min((b.low for b in or_bars), default=0.0)
        # We simulate option-equivalent entry as a price-fractional position on the underlying
        # to keep the backtest data-light. Upgrade path: load a historical options chain.
        ctx = ExecutionContext(
            signal=sig, now=now,
            account_equity=self.broker.account().equity,
            day_pnl=self.broker.account().day_pnl,
            open_positions_count=len(self.broker.positions()),
            current_bar_volume=bars[-1].volume,
            avg_bar_volume=sum(b.volume for b in bars[-20:]) / max(1, min(20, len(bars))),
            opening_range_high=or_hi, opening_range_low=or_lo,
            spot=spot, vwap=bars[-1].vwap or spot, vix=vix,
            current_iv=0.25, iv_52w_low=0.10, iv_52w_high=0.50,
            is_etf=sig.symbol in {"SPY", "QQQ", "IWM"},
        )
        results = self.chain.run(ctx)
        if not ExecutionChain.decided_pass(results):
            return
        size_in = SizingInputs(
            equity=self.broker.account().equity,
            contract=OptionContract(                          # synthetic contract stub for sizing
                symbol=sig.symbol, underlying=sig.symbol,
                strike=spot, expiry=date.today(),
                right=sig.option_right or OptionRight.CALL,
                # tight ETF-plausible spread (<3%)
                bid=max(spot * 0.020, 0.05),
                ask=max(spot * 0.0206, 0.0515),
                open_interest=1000, today_volume=200,
            ),
            win_rate_est=self.cfg.win_rate_prior,
            avg_win=self.cfg.avg_win_prior, avg_loss=self.cfg.avg_loss_prior,
            vix_today=vix, vix_52w_low=self.cfg.vix_52w_low,
            vix_52w_high=self.cfg.vix_52w_high, vrp_zscore=0.0, is_0dte=True,
            is_long=(sig.side == Side.BUY),
        )
        # Backtest uses an equity-proxy: size as shares-equivalent so the
        # P&L curve is interpretable without historical options data.
        # 5% notional per trade (equivalent to ~0.5% risk at a 10% stop).
        notional_budget = self.broker.account().equity * 0.05
        n = max(1, int(notional_budget / max(spot, 1e-6)))
        if n <= 0:
            return
        limit = spot * 1.001 if sig.side == Side.BUY else spot * 0.999
        order = Order(symbol=sig.symbol, side=sig.side, qty=n,
                      is_option=False, limit_price=limit,
                      tag=f"entry:{sig.source}")
        v = self.validator.validate(order, size_in.contract,
                                    self.broker.account().buying_power,
                                    self.settings["account"]["max_open_positions"])
        if not v.ok:
            return
        fill = self.broker.submit(v.adjusted_order or order)
        if fill and self.cfg.verbose:
            print(f"[entry] {sig.symbol} src={sig.source} qty={n} px={fill.price:.2f}")
        # attach auto-stops
        pos = next((p for p in self.broker.positions() if p.symbol == sig.symbol), None)
        if pos is not None:
            pt, sl = compute_auto_stops(pos, is_short_dte=True,
                                         pt_short_pct=self.settings["exits"]["profit_target_short_dte_pct"],
                                         sl_short_pct=self.settings["exits"]["stop_loss_short_dte_pct"])
            pos.auto_profit_target = pt
            pos.auto_stop_loss = sl
            pos.entry_tags = {"tag": sig.meta.get("entry_tag", sig.source)}

    def run(self, symbols: List[str], total_bars: int = 300) -> Dict:
        # Synthesize a single trading day's session: 09:30 → 14:30 ET on the
        # most recent weekday. First 30 bars ARE the true opening range so
        # ORB signals and filters operate sensibly.
        from datetime import datetime, timedelta
        now = datetime.now(tz=ET)
        anchor = now.replace(hour=14, minute=30, second=0, microsecond=0)
        if now < anchor:
            anchor = anchor - timedelta(days=1)
        while anchor.weekday() >= 5:
            anchor = anchor - timedelta(days=1)
        # ensure at most 300 minutes from 9:30 (09:30 + 300 min = 14:30)
        total_bars = min(total_bars, 300)
        bars_by_sym: Dict[str, List[Bar]] = {
            s: self.data.get_bars(s, limit=total_bars, timeframe_minutes=1, end=anchor)
            for s in symbols
        }
        # Align: walk index 50..N
        for i in range(50, total_bars):
            for sym in symbols:
                window = bars_by_sym[sym][:i + 1]
                if len(window) < 30:
                    continue
                vix = self._vix_at(i)
                ctx = self._snapshot_ctx(sym, window, i)
                for strat in self.strategies:
                    sig = strat.emit(ctx)
                    if sig is None or sig.is_stale(ttl_sec=60):
                        continue
                    sig.meta.setdefault("mi_edge_score", 0)  # neutral in synth sim
                    self._try_enter(sig, window, ctx.now, vix)
                # exit sweep after signals for this bar
                self._apply_exits(ctx.now, vix)
            # mark-to-market
            prices = {s: bars_by_sym[s][min(i, len(bars_by_sym[s]) - 1)].close for s in symbols}
            self.broker.mark_to_market(prices)
            self.equity_curve.append(self.broker.account().equity)
        # collect pnls from closed positions via day_pnl/total_pnl deltas
        return {
            "equity_curve": self.equity_curve,
            "total_pnl": self.broker._total_pnl,  # noqa: intentional peek
            "final_equity": self.broker.account().equity,
        }
