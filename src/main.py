"""Main orchestration loop.

- Main loop: 3-5 minute cycle. For each symbol: build SignalContext, run
  enabled strategies, pass through 14-filter chain, validate, size, submit.
  Then a portfolio exit sweep.
- Fast exit thread: every 5 seconds, iterate open positions and apply the
  fast profit/stop thresholds.

Paper only unless settings.live_trading AND broker != 'paper'.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import date
from typing import Dict, List, Optional

from .core.clock import MarketClock
from .core.config import load_settings, Settings
from .core.logger import configure_logging, get_logger
from .core.types import Signal, Order, Side, OptionRight
from .brokers.paper import PaperBroker
from .brokers.quote_validator import QuoteValidator
from .brokers.slippage_model import StochasticCostModel, LinearCostModel, MarketContext
from .brokers.auto_calibrating_model import (
    AutoCalibratingCostModel, start_calibration_scheduler,
)
from .risk.drawdown_guard import DrawdownGuard
from .risk.vol_scaling import vol_scale
from .data.market_data import SyntheticDataAdapter, AlpacaDataAdapter, MarketDataAdapter
from .data.options_chain import SyntheticOptionsChain, OptionsChainProvider
from .data.options_chain_alpaca import AlpacaOptionsChain
from .risk.execution_chain import ExecutionChain, ExecutionContext
from .risk.order_validator import OrderValidator
from .risk.position_sizer import PositionSizer, SizingInputs
from .exits.exit_engine import ExitEngine, ExitEngineConfig
from .exits.fast_exit import FastExitEvaluator
from .signals.momentum import MomentumSignal
from .signals.vwap_reversion import VwapReversionSignal
from .signals.orb import OpeningRangeBreakout
from .signals.lstm_signal import LSTMSignal
from .signals.ensemble import EnsembleCoordinator, DEFAULT_WEIGHTS
from .intelligence.regime import Regime, RegimeClassifier
from .intelligence.vix_probe import VixProbe
from .intelligence.dividend_yield import DividendYieldProvider
from .storage.journal import build_journal, TradeJournal
from .notify.base import build_notifier, Notifier
from .intelligence.news import NewsProvider, StaticNewsProvider, CachedNewsSentiment
from .intelligence.news_alpaca import AlpacaNewsProvider
from .intelligence.news_classifier import build_classifier
from .intelligence.econ_calendar import EconomicCalendar
from .intelligence.catalyst_calendar import (
    CatalystCalendar, build_default_catalyst_calendar,
)


log = get_logger(__name__)


def _have_alpaca_creds() -> tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY_ID", "").strip()
    secret = os.getenv("ALPACA_API_SECRET_KEY", "").strip()
    if key and secret and not key.startswith("your_"):
        return key, secret
    return "", ""


def _build_data_adapter(settings: Settings) -> MarketDataAdapter:
    """Use Alpaca if credentials are present, else synthetic fallback."""
    key, secret = _have_alpaca_creds()
    if key and secret:
        log.info("data_adapter", kind="alpaca")
        return AlpacaDataAdapter(api_key=key, api_secret=secret,
                                 fallback=SyntheticDataAdapter())
    log.info("data_adapter", kind="synthetic",
             reason="ALPACA_API_KEY_ID/SECRET not set")
    return SyntheticDataAdapter()


def _build_chain_provider(settings: Settings) -> OptionsChainProvider:
    key, secret = _have_alpaca_creds()
    if key and secret:
        log.info("options_chain", kind="alpaca")
        return AlpacaOptionsChain(api_key=key, api_secret=secret,
                                  fallback=SyntheticOptionsChain())
    log.info("options_chain", kind="synthetic")
    return SyntheticOptionsChain()


def _build_news_sentiment(settings: Settings) -> Optional[CachedNewsSentiment]:
    if not settings.get("news.enabled", True):
        return None
    key, secret = _have_alpaca_creds()
    if key and secret:
        provider: NewsProvider = AlpacaNewsProvider(
            api_key=key, api_secret=secret,
            default_lookback_hours=int(settings.get("news.lookback_hours", 2)),
        )
        log.info("news_provider", kind="alpaca")
    else:
        provider = StaticNewsProvider()
        log.info("news_provider", kind="static")
    classifier = build_classifier()
    log.info("news_classifier", kind=classifier.__class__.__name__)
    return CachedNewsSentiment(
        provider=provider,
        classifier=classifier,
        ttl_seconds=int(settings.get("news.cache_ttl_seconds", 300)),
    )


def _build_journal_from_settings(settings: Settings) -> Optional[TradeJournal]:
    backend = settings.get("storage.backend", "sqlite")
    sqlite_path = settings.get("storage.sqlite_path", "logs/tradebot.sqlite")
    dsn_env = settings.get("storage.cockroach_dsn_env", "COCKROACH_DSN")
    try:
        j = build_journal(backend=backend, sqlite_path=sqlite_path, dsn_env_var=dsn_env)
        log.info("journal", backend=backend)
        return j
    except Exception as e:
        log.warning("journal_init_failed_fallback_to_none", err=str(e))
        return None


class TradeBot:
    def __init__(self, settings: Settings):
        # Pre-init attributes that later blocks conditionally reference so
        # we never hit AttributeError when a block runs before another sets it.
        self._auto_cost_model = None
        self._autocal_mode = "manual"

        self.s = settings
        self.clock = MarketClock(
            market_open=settings.get("session.market_open"),
            market_close=settings.get("session.market_close"),
            no_new_entries_after=settings.get("session.no_new_entries_after"),
            eod_force_close=settings.get("session.eod_force_close"),
        )
        self.data = _build_data_adapter(settings)
        self.chain_provider = _build_chain_provider(settings)
        self.journal = _build_journal_from_settings(settings)
        self.notifier: Notifier = build_notifier()
        # Late-bind notifier to the auto-calibrator so it can push alerts
        # when realized vs. predicted slippage diverges meaningfully.
        if self._auto_cost_model is not None:
            self._auto_cost_model.notifier = self.notifier
        self.news = _build_news_sentiment(settings)
        self.econ_calendar = EconomicCalendar()
        self.catalyst_calendar: CatalystCalendar = build_default_catalyst_calendar(
            static_yaml_path="config/catalysts.yaml",
            lookahead_days=int(settings.get("catalysts.lookahead_days", 14)),
        )
        self._refresh_catalysts()
        snap_path = settings.get("broker.snapshot_path", "logs/broker_state.json")
        # Cost model: stochastic (recommended) or linear (legacy).
        # Stochastic makes backtest Sharpe match live Sharpe; linear overstates.
        model_type = str(settings.get("broker.cost_model", "stochastic")).lower()
        if model_type == "stochastic":
            base_model = StochasticCostModel(
                base_half_spread_mult=float(settings.get("broker.half_spread_mult", 1.0)),
                size_impact_coef=float(settings.get("broker.size_impact_coef", 0.25)),
                vix_impact_coef=float(settings.get("broker.vix_impact_coef", 0.015)),
                random_noise_bps=float(settings.get("broker.slip_noise_bps", 0.5)),
                min_slippage_bps=float(settings.get("broker.slip_floor_bps", 0.5)),
            )
            # Auto-calibration: 'hourly' (most reactive), 'daily', 'manual'.
            self._autocal_mode = str(
                settings.get("broker.auto_calibrate", "daily")
            ).lower()
            if self._autocal_mode in ("hourly", "daily"):
                slip_model = AutoCalibratingCostModel(
                    inner=base_model,
                    calibration_path=str(settings.get(
                        "broker.calibration_path",
                        "logs/slippage_calibration.jsonl",
                    )),
                    history_path=str(settings.get(
                        "broker.calibration_history",
                        "logs/calibration_history.jsonl",
                    )),
                    min_samples=int(settings.get("broker.calibration_min_samples", 30)),
                    max_step_per_cycle=float(
                        settings.get("broker.calibration_max_step", 0.30)
                    ),
                    max_drift_from_baseline=float(
                        settings.get("broker.calibration_max_drift", 2.0)
                    ),
                    notifier=None,  # attached after notifier init below
                )
                self._auto_cost_model = slip_model
            else:
                slip_model = base_model
                self._auto_cost_model = None
        else:
            slip_model = LinearCostModel(
                slippage_bps=float(settings.get("broker.slippage_bps", 2.0))
            )
            self._autocal_mode = "manual"
            self._auto_cost_model = None
        self.broker = PaperBroker(
            starting_equity=settings.paper_equity,
            slippage_bps=settings.get("broker.slippage_bps", 2),
            journal=self.journal,
            snapshot_path=snap_path,
            slippage_model=slip_model,
        )
        # Crash recovery: if a snapshot exists, restore state before we
        # accept the first tick. If this is a clean start, no-op.
        try:
            restored = self.broker.restore_from_snapshot(snap_path)
            if restored > 0:
                log.info("broker_state_restored", n_positions=restored,
                         path=snap_path)
                self.notifier.notify(
                    f"Restored {restored} position(s) from snapshot. "
                    f"Review vs broker before trading.",
                    level="warn", title="startup",
                )
        except Exception as e:                           # noqa: BLE001
            log.warning("broker_snapshot_restore_failed", err=str(e))

        # Per-symbol dividend yield cache (only dividend-payers get non-zero).
        self.dividend_yield = DividendYieldProvider(
            cache_path=settings.get("pricing.dividend_cache",
                                     "data_cache/div_yields.json"),
            max_age_hours=int(settings.get("pricing.dividend_refresh_hours", 24)),
        )
        try:
            self.dividend_yield.prime(self.s.universe)
            log.info("dividend_yield_primed", n=len(self.s.universe))
        except Exception as e:                           # noqa: BLE001
            log.warning("dividend_yield_prime_failed", err=str(e))

        # Load measured priors from the journal (last 30 days). If we don't
        # have enough trades yet, fall back to conservative defaults.
        self._win_rate, self._avg_win, self._avg_loss = self._load_measured_priors()
        log.info("priors_loaded", win_rate=self._win_rate,
                 avg_win=self._avg_win, avg_loss=self._avg_loss)
        self._last_daily_summary_date = None
        self._halted_today = False

        # Drawdown guard — reduces size / halts when peak-to-trough DD
        # exceeds tiered thresholds. Peak seeded from last known equity.
        self.drawdown_guard = DrawdownGuard()
        self._peak_equity: float = float(settings.paper_equity)
        self._dd_size_multiplier: float = 1.0      # updated each tick

        # Regime classifier kind: 'rule' (stable default) or 'hmm' (smoother).
        regime_kind = str(settings.get("regime.classifier", "rule")).lower()
        self.regime_kind = regime_kind
        self.hmm_regime_classifier = None
        if regime_kind == "hmm":
            try:
                from .intelligence.regime_hmm import HMMRegimeClassifier
                self.hmm_regime_classifier = HMMRegimeClassifier()
                log.info("regime_classifier", kind="hmm")
            except Exception as e:            # noqa: BLE001
                log.warning("hmm_init_failed_fallback_to_rule", err=str(e))

        # Regime classifier + ensemble coordinator + live VIX probe
        self.vix_probe = VixProbe(
            ttl_seconds=int(settings.get("vix.cache_seconds", 60)),
            fallback_vix=float(settings.get("vix.fallback", 15.0)),
            prefer=str(settings.get("vix.prefer", "auto")),
        )
        self.regime_classifier = RegimeClassifier()
        self.ensemble_enabled = bool(settings.get("ensemble.enabled", True))
        self.ensemble_log = bool(settings.get("ensemble.log_decisions", True))
        ens_weights_raw = settings.get("ensemble.weights", None)
        ens_weights = None
        if ens_weights_raw:
            try:
                ens_weights = {}
                for k, v in ens_weights_raw.items():
                    ens_weights[Regime(k)] = {sn: float(w) for sn, w in v.items()}
            except Exception as e:                     # noqa: BLE001
                log.warning("ensemble_weights_parse_failed", err=str(e))
                ens_weights = None
        self.ensemble = EnsembleCoordinator(
            weights=ens_weights,
            min_weighted_confidence=float(
                settings.get("ensemble.min_weighted_confidence", 0.70)
            ),
            dominance_ratio=float(settings.get("ensemble.dominance_ratio", 1.5)),
        )
        self.qv = QuoteValidator()
        self.exec_chain = ExecutionChain(settings.raw, self.clock)
        self.validator = OrderValidator()
        self.sizer = PositionSizer(
            kelly_fraction_cap=settings.get("sizing.kelly_fraction_cap", 0.25),
            kelly_hard_cap=settings.get("sizing.kelly_hard_cap_pct", 0.05),
            max_0dte=settings.get("sizing.max_contracts_0dte", 5),
            max_multiday=settings.get("sizing.max_contracts_multiday", 10),
            regime_multipliers=settings.get("sizing.regime_multipliers", {}) or {},
        )
        self.exits = ExitEngine(ExitEngineConfig(
            pt_short_pct=settings.get("exits.profit_target_short_dte_pct", 0.35),
            pt_multi_pct=settings.get("exits.profit_target_multi_dte_pct", 0.50),
            sl_short_pct=settings.get("exits.stop_loss_short_dte_pct", 0.20),
            sl_multi_pct=settings.get("exits.stop_loss_multi_dte_pct", 0.30),
            hard_profit_cap_pct=settings.get("exits.hard_profit_cap_pct", 1.50),
        ))
        self.fast = FastExitEvaluator()
        self.strategies = [
            MomentumSignal(), VwapReversionSignal(), OpeningRangeBreakout(),
        ]
        if settings.get("ml.lstm_enabled", True):
            lstm = LSTMSignal(
                checkpoint_path=settings.get("ml.lstm_checkpoint",
                                              "checkpoints/lstm_best.pt"),
                min_confidence=float(settings.get("ml.lstm_min_confidence", 0.55)),
                journal=self.journal,
                timeframe_minutes=int(settings.get("ml.lstm_timeframe_minutes", 5)),
                log_all_predictions=bool(settings.get("ml.lstm_log_predictions", True)),
            )
            # Only append when the model actually loaded; otherwise a no-op.
            if lstm._model is not None:
                self.strategies.append(lstm)
        self._stop = threading.Event()

    def _load_measured_priors(self, days: int = 30,
                                min_trades: int = 30) -> tuple:
        """Pull realized win_rate / avg_win / avg_loss from the journal.

        Fallback priors (conservative, slight positive edge):
          win_rate = 0.55, avg_win = 0.020, avg_loss = 0.025

        These are used ONLY when there's insufficient live trade history.
        Real bots should aim to have 100+ closed trades before trusting
        journal-derived priors.
        """
        fallback = (0.55, 0.020, 0.025)
        if self.journal is None:
            return fallback
        try:
            from datetime import datetime as _dt, timedelta as _td, timezone as _tz
            since = _dt.now(tz=_tz.utc) - _td(days=days)
            trades = self.journal.closed_trades(since=since)
        except Exception as e:                      # noqa: BLE001
            log.warning("priors_load_failed", err=str(e))
            return fallback
        wins = [t for t in trades if (t.pnl or 0) > 0]
        losses = [t for t in trades if (t.pnl or 0) < 0]
        if len(wins) + len(losses) < min_trades:
            log.info("priors_using_fallback",
                     n_trades=len(trades), needed=min_trades)
            return fallback
        win_rate = len(wins) / (len(wins) + len(losses))
        avg_win = sum((t.pnl_pct or 0) for t in wins) / max(1, len(wins))
        avg_loss = sum(abs(t.pnl_pct or 0) for t in losses) / max(1, len(losses))
        # Sanity-clamp so one noisy month can't push us into Kelly extremes.
        win_rate = max(0.30, min(0.80, win_rate))
        avg_win = max(0.005, min(0.10, avg_win))
        avg_loss = max(0.005, min(0.10, avg_loss))
        return (win_rate, avg_win, avg_loss)

    def _refresh_catalysts(self) -> None:
        """Pull earnings/FDA events and hydrate the EconomicCalendar with
        per-symbol blackouts. Swallows errors (network, missing deps)."""
        try:
            events = self.catalyst_calendar.refresh(self.s.universe)
            n = self.catalyst_calendar.hydrate_econ_calendar(self.econ_calendar)
            log.info("catalysts_refreshed", events=len(events), blackouts=n)
            if events:
                summary = ", ".join(
                    f"{e.symbol}:{e.event_type}:{e.when}" for e in events[:8]
                )
                self.notifier.notify(
                    f"{len(events)} upcoming catalysts — {summary}"
                    + ("..." if len(events) > 8 else ""),
                    title="catalysts",
                )
        except Exception as e:   # noqa: BLE001
            log.warning("catalyst_refresh_failed", err=str(e))

    def fast_loop(self) -> None:
        interval = float(self.s.get("exits.fast_thread_interval_sec", 5))
        kill_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "KILL",
        )
        while not self._stop.is_set():
            # Cooperative kill check — needs to be here too, not just main_loop.
            # At a 5s cadence we react within ~5s even if main_loop is blocked.
            if os.path.exists(kill_file):
                log.warning("kill_switch_seen_fast", path=kill_file)
                self._stop.set()
                break
            try:
                for pos in list(self.broker.positions()):
                    price = self.data.latest_price(pos.underlying or pos.symbol) or pos.avg_price
                    d = self.fast.evaluate(pos, price)
                    if d and d.should_close:
                        side = Side.SELL if pos.qty > 0 else Side.BUY
                        o = Order(symbol=pos.symbol, side=side, qty=abs(pos.qty),
                                  is_option=pos.is_option, limit_price=price,
                                  tag=f"fast:{d.reason}")
                        self.broker.submit(o)
                        log.info("fast_exit", symbol=pos.symbol, reason=d.reason)
            except Exception as e:  # noqa: BLE001
                log.warning("fast_loop_error", err=str(e))
            self._stop.wait(interval)

    def main_loop(self) -> None:
        interval = float(self.s.get("exits.main_loop_interval_sec", 180))
        kill_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "KILL",
        )
        while not self._stop.is_set():
            # cooperative kill switch: tradebotctl stop / `touch KILL`
            if os.path.exists(kill_file):
                log.warning("kill_switch_seen", path=kill_file)
                self.notifier.notify("kill switch triggered — shutting down",
                                     level="warn", title="tradebot")
                self._stop.set()
                break
            try:
                now = self.clock.now_et()
                if not self.clock.is_market_open(now):
                    self._maybe_daily_summary(now)
                    self._stop.wait(interval)
                    continue
                self._check_halt_conditions()
                if not self._halted_today:
                    for symbol in self.s.universe:
                        self._tick_symbol(symbol)
            except Exception as e:  # noqa: BLE001
                log.error("main_loop_error", err=str(e))
                self.notifier.notify(f"main_loop_error: {e}", level="error",
                                     title="tradebot")
            self._stop.wait(interval)

    def _check_halt_conditions(self) -> None:
        if self._halted_today:
            return
        acct = self.broker.account()
        # Track peak equity for peak-to-trough DD
        if acct.equity > self._peak_equity:
            self._peak_equity = acct.equity
        cap = self.s.max_daily_loss_pct * acct.equity
        if acct.day_pnl <= -cap:
            self._halted_today = True
            log.warning("daily_loss_halt", day_pnl=acct.day_pnl, cap=cap)
            self.notifier.notify(
                f"Daily loss halt hit: day_pnl={acct.day_pnl:.2f} "
                f"equity={acct.equity:.2f}. No new entries today.",
                level="warn", title="HALT",
            )
            return
        # Tiered drawdown guard: reduces size multiplier or HALTS if DD
        # exceeds thresholds. Multiplier applied in _try_enter when sizing.
        state = self.drawdown_guard.evaluate(acct.equity, self._peak_equity)
        prev_mult = self._dd_size_multiplier
        self._dd_size_multiplier = state.size_multiplier
        if state.halted:
            self._halted_today = True
            log.warning("drawdown_halt", dd=state.current_drawdown_pct,
                         peak=self._peak_equity, equity=acct.equity)
            self.notifier.notify(
                f"Drawdown halt: {state.current_drawdown_pct:.1%} from peak "
                f"${self._peak_equity:.2f}. No new entries today.",
                level="warn", title="HALT",
            )
        elif state.size_multiplier < 1.0 and prev_mult >= 1.0:
            log.warning("drawdown_scale_down", mult=state.size_multiplier,
                         dd=state.current_drawdown_pct)
            self.notifier.notify(
                f"Drawdown scale-down: ×{state.size_multiplier} "
                f"(DD {state.current_drawdown_pct:.1%} from peak).",
                level="warn", title="risk",
            )

    def _maybe_daily_summary(self, now) -> None:
        today = now.date()
        if self._last_daily_summary_date == today:
            return
        # Only post after close to avoid spamming between ticks pre-open
        if not self.clock.should_eod_force_close(now):
            return
        # 1. Force-close any open positions using the most recent bar close for
        #    each symbol — avoids the zero-slippage `avg_price` close.
        open_before = list(self.broker.positions())
        if open_before:
            log.info("eod_flatten_start", n=len(open_before))
            mark_prices = {}
            for p in open_before:
                underlying = p.underlying or p.symbol
                last = self.data.latest_price(underlying)
                if last is not None:
                    mark_prices[p.symbol] = last
            try:
                self.broker.flatten_all(mark_prices=mark_prices)
            except Exception as e:                 # noqa: BLE001
                log.error("eod_flatten_error", err=str(e))
                self.notifier.notify(f"EOD flatten error: {e}",
                                     level="error", title="HALT")
            # Verify
            remaining = self.broker.positions()
            if remaining:
                log.error("eod_flatten_incomplete",
                          remaining=[p.symbol for p in remaining])
                self.notifier.notify(
                    f"EOD flatten INCOMPLETE — {len(remaining)} positions "
                    f"remain. Reconcile manually.",
                    level="error", title="HALT",
                )

        # 2. Emit summary
        acct = self.broker.account()
        self.notifier.notify(
            f"EOD {today}: equity={acct.equity:.2f} "
            f"day_pnl={acct.day_pnl:+.2f} total_pnl={acct.total_pnl:+.2f} "
            f"flattened={len(open_before)}",
            title="daily",
        )
        self._last_daily_summary_date = today
        # roll the daily halt state at the end of day
        self.broker.reset_day()
        self._halted_today = False

    def _tick_symbol(self, symbol: str) -> None:
        bars = self.data.get_bars(symbol, limit=80)
        if len(bars) < 30:
            return
        spot = bars[-1].close
        vwap = bars[-1].vwap or spot
        or_bars = bars[:30]
        or_hi = max(b.high for b in or_bars)
        or_lo = min(b.low for b in or_bars)
        from .signals.base import SignalContext
        sctx = SignalContext(
            symbol=symbol, now=bars[-1].ts, bars=bars,
            spot=spot, vwap=vwap,
            opening_range_high=or_hi, opening_range_low=or_lo,
            chain=self.chain_provider.chain(symbol, spot, target_dte=1),
        )

        # 1. Collect raw signals from every enabled strategy
        raw_signals = []
        for strat in self.strategies:
            sig = strat.emit(sctx)
            if sig is None or sig.is_stale(ttl_sec=30):
                continue
            raw_signals.append(sig)

        # Fall back to legacy per-signal path if the ensemble is disabled.
        if not self.ensemble_enabled:
            for sig in raw_signals:
                self._try_enter(sig, bars, or_hi, or_lo, spot, vwap)
            return

        if not raw_signals:
            return

        # 2. Classify current regime from bars + live VIX
        vix_reading = self.vix_probe.get()
        regime_snap = self.regime_classifier.classify(
            vix=vix_reading.value,
            now=bars[-1].ts,
            recent_closes=[b.close for b in bars[-60:]],
        )
        regime = regime_snap.regime

        # HMM overlay: if the HMM says we're in a high-vol regime, override
        # the rule-based classifier's low-vol label. The rule-based one is
        # stable and fast; the HMM catches structural breaks earlier.
        if self.hmm_regime_classifier is not None:
            try:
                hmm = self.hmm_regime_classifier.classify(
                    [b.close for b in bars[-150:]]
                )
                if hmm is not None and hmm.current_label == "high_vol":
                    # Map rule-based regimes to their high-vol equivalents.
                    from .intelligence.regime import Regime
                    if regime == Regime.TREND_LOWVOL:
                        regime = Regime.TREND_HIGHVOL
                    elif regime == Regime.RANGE_LOWVOL:
                        regime = Regime.RANGE_HIGHVOL
            except Exception:
                pass

        # 3. Run the coordinator
        decision = self.ensemble.aggregate(raw_signals, regime)

        # 4. Log decision (for analyze_ensemble.py)
        if self.ensemble_log and self.journal is not None:
            try:
                import json as _json
                from .storage.journal import EnsembleRecord
                self.journal.record_ensemble_decision(EnsembleRecord(
                    id=None, ts=bars[-1].ts, symbol=symbol,
                    regime=regime.value, emitted=decision.emitted,
                    dominant_direction=decision.dominant_direction,
                    dominant_score=decision.dominant_score,
                    opposing_score=decision.opposing_score,
                    n_inputs=decision.n_inputs, reason=decision.reason,
                    contributors=_json.dumps(
                        [{"source": c.source, "direction": c.direction,
                          "raw": round(c.raw_confidence, 4),
                          "weight": round(c.weight, 3)}
                         for c in decision.contributions]
                    ),
                ))
            except Exception as e:                     # noqa: BLE001
                log.warning("ensemble_log_error", err=str(e))

        if not decision.emitted or decision.signal is None:
            log.info("ensemble_skip", symbol=symbol,
                      regime=regime.value, reason=decision.reason)
            return

        log.info("ensemble_emit", symbol=symbol, regime=regime.value,
                 direction=decision.dominant_direction,
                 score=round(decision.dominant_score, 3),
                 contributors=[c.source for c in decision.contributions
                                if c.direction == decision.dominant_direction])
        self._try_enter(decision.signal, bars, or_hi, or_lo, spot, vwap)

    def _try_enter(self, sig: Signal, bars, or_hi, or_lo, spot, vwap) -> None:
        acct = self.broker.account()
        news_score, news_label, news_rationale = 0.0, "neutral", ""
        if self.news is not None:
            try:
                ns = self.news.sentiment(sig.symbol)
                news_score = ns.score
                news_label = ns.label
                news_rationale = ns.rationale
            except Exception as e:
                log.warning("news_sentiment_error", symbol=sig.symbol, err=str(e))
        econ_blackout = self.econ_calendar.in_blackout(bars[-1].ts, symbol=sig.symbol)
        vix_now = self.vix_probe.value()
        ectx = ExecutionContext(
            signal=sig, now=bars[-1].ts,
            account_equity=acct.equity, day_pnl=acct.day_pnl,
            open_positions_count=len(self.broker.positions()),
            current_bar_volume=bars[-1].volume,
            avg_bar_volume=sum(b.volume for b in bars[-20:]) / 20,
            opening_range_high=or_hi, opening_range_low=or_lo,
            spot=spot, vwap=vwap, vix=vix_now,
            is_etf=sig.symbol in {"SPY", "QQQ", "IWM"},
            econ_blackout=econ_blackout,
            news_score=news_score, news_label=news_label,
            news_rationale=news_rationale,
        )
        results = self.exec_chain.run(ectx)
        if not ExecutionChain.decided_pass(results):
            # surface news-blocked trades to the operator; other blocks are noisy
            for r in results:
                if (not r.passed) and (not r.advisory) and r.reason.startswith("news_"):
                    self.notifier.notify(
                        f"{sig.symbol} {sig.source} blocked by news: {r.reason}",
                        level="warn", title="news block",
                    )
                    break
            return
        from .core.types import OptionContract
        contract = OptionContract(symbol=sig.symbol, underlying=sig.symbol,
                                  strike=spot, expiry=sig.expiry or date.today(),
                                  right=sig.option_right or OptionRight.CALL,
                                  ask=max(0.05, spot * 0.02), bid=max(0.04, spot * 0.018),
                                  open_interest=1000, today_volume=200)
        # Regime passes through via sig.meta when the ensemble emitted it;
        # otherwise None → sizer leaves size untouched.
        regime = sig.meta.get("regime") if isinstance(sig.meta, dict) else None

        # Per-symbol vol scaling: normalizes exposure so a $1 risk on NVDA
        # ≈ $1 risk on KO. Uses recent bars passed in from _tick_symbol.
        vs = vol_scale(bars, target_annual_vol=float(self.s.get(
            "sizing.target_annual_vol", 0.20)))
        vol_mult = vs.multiplier

        n = self.sizer.contracts(SizingInputs(
            equity=acct.equity, contract=contract,
            # Journal-measured priors — real observed performance, not guesses.
            win_rate_est=self._win_rate,
            avg_win=self._avg_win,
            avg_loss=self._avg_loss,
            vix_today=vix_now, vix_52w_low=10.0, vix_52w_high=40.0,
            vrp_zscore=0.0, is_0dte=True,
            is_long=(sig.side == Side.BUY),
        ), regime=regime)
        # Apply vol scaling + drawdown guard multiplier AFTER Kelly sizing
        n = int(round(n * vol_mult * self._dd_size_multiplier))
        if n <= 0:
            return
        limit = spot * (1.001 if sig.side == Side.BUY else 0.999)
        order = Order(symbol=sig.symbol, side=sig.side, qty=n,
                      is_option=False, limit_price=limit, tag=f"entry:{sig.source}")
        v = self.validator.validate(order, contract, acct.buying_power,
                                    self.s.get("account.max_open_positions", 5))
        if not v.ok:
            log.info("order_reject", reason=v.reason)
            return
        # Push the current quote + VIX into the slippage model so the fill
        # cost reflects real microstructure, not a fixed constant.
        self.broker.update_market_context(
            sig.symbol,
            MarketContext(
                bid=spot * 0.9998, ask=spot * 1.0002,
                bid_size=1000, ask_size=1000, vix=vix_now,
                recent_spread_pct=0.0004,
            ),
        )
        fill = self.broker.submit(v.adjusted_order or order)
        if fill:
            log.info("fill", symbol=sig.symbol, qty=n, price=fill.price, src=sig.source)
            self.notifier.notify(
                f"{sig.symbol} {sig.side.value} x{n} @ {fill.price:.2f} src={sig.source}",
                title="entry",
            )

    def run(self) -> None:
        if self.s.live_trading:
            log.error("live_trading_blocked_in_main_of_default_build — remove guard explicitly")
            return
        configure_logging("INFO")

        # SIGTERM / SIGINT handler so `systemctl stop` and Ctrl-C give us a
        # chance to flush the journal and trigger an EOD flatten.
        import signal as _signal

        def _shutdown(signum, _frame):
            name = {_signal.SIGTERM: "SIGTERM", _signal.SIGINT: "SIGINT"}.get(
                signum, f"signal={signum}"
            )
            log.warning("shutdown_signal", signal=name)
            try:
                self.notifier.notify(f"{name} received — flattening + flushing",
                                     level="warn", title="shutdown")
            except Exception:
                pass
            self._stop.set()

        try:
            _signal.signal(_signal.SIGTERM, _shutdown)
            _signal.signal(_signal.SIGINT, _shutdown)
        except ValueError:
            # Not on the main thread (e.g. called from a test runner) — skip.
            pass

        t = threading.Thread(target=self.fast_loop, name="fast_exit", daemon=True)
        t.start()

        # Kick off the auto-calibration scheduler if enabled.
        cal_thread = None
        if self._auto_cost_model is not None and self._autocal_mode in ("hourly", "daily"):
            cal_thread = start_calibration_scheduler(
                self._auto_cost_model,
                mode=self._autocal_mode,
                stop_event=self._stop,
            )
            log.info("calibration_scheduler_started", mode=self._autocal_mode)

        try:
            self.main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            t.join(timeout=2)
            if cal_thread is not None:
                cal_thread.join(timeout=2)
            # Best-effort flatten on shutdown — fail-soft so a broken broker
            # doesn't hang the exit path.
            try:
                open_now = list(self.broker.positions())
                if open_now:
                    log.warning("shutdown_flatten", n=len(open_now))
                    mark = {}
                    for p in open_now:
                        last = self.data.latest_price(p.underlying or p.symbol)
                        if last is not None:
                            mark[p.symbol] = last
                    self.broker.flatten_all(mark_prices=mark)
            except Exception as e:                    # noqa: BLE001
                log.error("shutdown_flatten_error", err=str(e))


def main() -> None:
    s = load_settings()
    bot = TradeBot(s)
    bot.run()


if __name__ == "__main__":
    main()
