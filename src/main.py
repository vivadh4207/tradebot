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
from datetime import date, datetime, timezone
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


# ETF universe classification. Used by f09_volume_confirmation to pick the
# right volume-confirmation threshold: ETFs are inherently high-liquidity,
# so per-bar volume often sits at ~1.0x avg (no surge) yet is still
# perfectly tradable. Individual stocks need the surge to confirm
# momentum is real.
#
# Prefix-based heuristic covers the major sector / thematic ETF families
# (X**, IY**, VO**, Q** index ones) plus an explicit allow-list of the
# top broad-market ETFs. Cheap to evaluate on every tick.
_ETF_EXPLICIT = {
    "SPY", "QQQ", "IWM", "DIA",       # broad index
    "VOO", "VTI", "VT", "VEA",        # Vanguard
    "IVV", "IJH", "IJR", "IWF",       # iShares
    "EEM", "EFA", "EWZ",              # international
    "TLT", "IEF", "SHY", "LQD", "HYG", # bonds
    "GLD", "SLV", "USO",              # commodities
    "UVXY", "VXX",                    # vol
}
_ETF_PREFIXES = ("XL", "XB", "XH", "XM", "XR", "XS", "XO", "XU")  # SPDR sectors


def _is_etf(symbol: str) -> bool:
    sym = (symbol or "").upper()
    if sym in _ETF_EXPLICIT:
        return True
    # SPDR sector ETFs: XLF, XLE, XLC, XLI, XLB, XLU, XLY, XLV, XLP, XLK
    # + XBI, XHE, XME, XRT, XSD, XOP, XHB, XUR, etc. from the user's universe
    if any(sym.startswith(p) for p in _ETF_PREFIXES) and len(sym) <= 4:
        return True
    return False


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
    """Build the local SQLite journal.

    Path resolution priority:
      1. TRADEBOT_SANDBOX_LOGS (used by tests — tmp dir)
      2. TRADEBOT_DATA_ROOT + storage.sqlite_path (Jetson SD-card mode)
      3. Repo-root + storage.sqlite_path (default laptop install)
    """
    from .core.data_paths import data_path
    sqlite_path = settings.get("storage.sqlite_path", "logs/tradebot.sqlite")
    _sandbox = os.getenv("TRADEBOT_SANDBOX_LOGS", "").strip()
    if _sandbox:
        resolved = os.path.join(_sandbox, "tradebot.sqlite")
    else:
        resolved = str(data_path(sqlite_path))
    try:
        j = build_journal(sqlite_path=resolved)
        log.info("journal", backend="sqlite", path=resolved)
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
        # Log the notifier flavor so operators can tell whether webhooks
        # are wired (NullNotifier means DISCORD_WEBHOOK_URL /
        # SLACK_WEBHOOK_URL was empty at import time — env var missing or
        # .env not loaded by the time build_notifier() ran).
        log.info("notifier", flavor=type(self.notifier).__name__)
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
        # Test harness: conftest.py sets TRADEBOT_SANDBOX_LOGS to a tmp
        # dir to keep tests from touching the real logs/ directory.
        _sandbox = os.getenv("TRADEBOT_SANDBOX_LOGS", "").strip()
        if _sandbox:
            snap_path = os.path.join(_sandbox, "broker_state.json")
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
        # Optional: mirror every order to Alpaca paper for UI visibility.
        # Our PaperBroker stays source of truth; the mirror is
        # fire-and-forget. The two books will drift over time
        # (different fill simulators) but that's a feature, not a bug —
        # we compare them to detect our model's bias.
        mirror_alpaca = bool(settings.get("broker.mirror_to_alpaca", False))
        alpaca_mirror_broker = None
        if mirror_alpaca:
            try:
                key = os.environ.get("ALPACA_API_KEY_ID", "").strip()
                sec = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
                if not key or not sec:
                    raise RuntimeError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing in env")
                from .brokers.alpaca_adapter import AlpacaBroker
                alpaca_mirror_broker = AlpacaBroker(
                    api_key=key, api_secret=sec, paper=True,
                )
                log.info("alpaca_mirror_initialized", paper=True)
            except Exception as e:                           # noqa: BLE001
                log.warning("alpaca_mirror_init_failed", err=str(e))
                alpaca_mirror_broker = None

        # Optional Tradier mirror — orders also fire into a Tradier
        # sandbox account for dual-broker paper validation. Activated
        # when TRADIER_TOKEN (or alias) is set; no settings.yaml flag
        # required. Silently no-op on auth failure.
        tradier_mirror_broker = None
        if bool(settings.get("broker.mirror_to_tradier", True)):
            try:
                from .brokers.tradier_adapter import build_tradier_broker
                tradier_mirror_broker = build_tradier_broker()
                if tradier_mirror_broker is not None:
                    log.info("tradier_mirror_initialized",
                             account=tradier_mirror_broker._account[:6] + "...")
            except Exception as e:                      # noqa: BLE001
                log.info("tradier_mirror_init_skipped", err=str(e))
                tradier_mirror_broker = None

        # MirrorAlpacaBroker is the multi-mirror wrapper — build it
        # whenever EITHER Alpaca or Tradier mirror is active. Earlier
        # logic only built it when Alpaca was set, which meant disabling
        # Alpaca silently disabled Tradier mirror too (trades fired
        # locally but never reached Tradier). This is the bug that let
        # kind=paper show up on logs after switching to Tradier-only.
        if alpaca_mirror_broker is not None or tradier_mirror_broker is not None:
            from .brokers.mirror_alpaca import MirrorAlpacaBroker
            self.broker = MirrorAlpacaBroker(
                starting_equity=settings.paper_equity,
                slippage_bps=settings.get("broker.slippage_bps", 2),
                journal=self.journal,
                snapshot_path=snap_path,
                slippage_model=slip_model,
                alpaca_broker=alpaca_mirror_broker,
                tradier_broker=tradier_mirror_broker,
            )
            self.broker._close_quote_fn = self._option_quote_for_close
            parts = ["paper"]
            if alpaca_mirror_broker is not None:
                parts.append("alpaca_mirror")
            if tradier_mirror_broker is not None:
                parts.append("tradier_mirror")
            kind = "+".join(parts)
            log.info("broker", kind=kind)
        else:
            self.broker = PaperBroker(
                starting_equity=settings.paper_equity,
                slippage_bps=settings.get("broker.slippage_bps", 2),
                journal=self.journal,
                snapshot_path=snap_path,
                slippage_model=slip_model,
            )
            log.info("broker", kind="paper")
        # Crash recovery: if a snapshot exists, restore state before we
        # accept the first tick. If this is a clean start, no-op.
        try:
            restored = self.broker.restore_from_snapshot(snap_path)
            if restored > 0:
                log.info("broker_state_restored", n_positions=restored,
                         path=snap_path)
                if bool(self.s.get("notifier.startup_notify", False)):
                    self.notifier.notify(
                        f"Restored {restored} position(s) from snapshot. "
                        f"Review vs broker before trading.",
                        level="warn", title="startup",
                    )
        except Exception as e:                           # noqa: BLE001
            log.warning("broker_snapshot_restore_failed", err=str(e))

        # Alpaca reconciliation: zombie cleanup. If the mirror broker
        # is active and Alpaca paper holds positions we don't know
        # about (usually from earlier sessions where close-orders
        # bounced due to the IOC-tif bug), auto-close them so Alpaca's
        # book matches ours. Fail-soft — reconcile errors never block
        # startup.
        try:
            from .brokers.mirror_alpaca import MirrorAlpacaBroker
            if isinstance(self.broker, MirrorAlpacaBroker):
                summary = self.broker.reconcile_with_alpaca()
                reconciled = summary.get("reconciled", 0)
                errors = summary.get("errors", 0)
                # Log to file always (useful for audit), but Discord
                # notify only when we ACTUALLY cleaned something up.
                # Errors-only cases (0 reconciled, N errors) happen on
                # every startup when zombies are expired/unclosable —
                # notifying would spam Discord with useless noise.
                if reconciled > 0 or errors > 0:
                    log.info("alpaca_reconcile", **summary)
                if reconciled > 0 and bool(self.s.get(
                        "notifier.reconcile_notify", False)):
                    self.notifier.notify(
                        f"Closed {reconciled} zombie position(s) on Alpaca paper.",
                        title="reconcile", level="info",
                        meta={
                            "reconciled": reconciled,
                            "errors": errors,
                            "symbols": ", ".join(
                                summary["alpaca_only_symbols"][:10]
                            ),
                        },
                    )
        except Exception as e:                           # noqa: BLE001
            log.warning("alpaca_reconcile_failed_nonfatal", err=str(e))

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
        _zero_dte_max_hold = float(settings.get(
            "exits.zero_dte_max_hold_minutes", 30.0
        ))
        self.exits = ExitEngine(ExitEngineConfig(
            pt_short_pct=settings.get("exits.profit_target_short_dte_pct", 0.35),
            pt_multi_pct=settings.get("exits.profit_target_multi_dte_pct", 0.50),
            sl_short_pct=settings.get("exits.stop_loss_short_dte_pct", 0.20),
            sl_multi_pct=settings.get("exits.stop_loss_multi_dte_pct", 0.30),
            hard_profit_cap_pct=settings.get("exits.hard_profit_cap_pct", 1.50),
            zero_dte_max_hold_minutes=_zero_dte_max_hold,
        ))
        from .exits.fast_exit import FastExitConfig as _FastExitConfig
        self.fast = FastExitEvaluator(_FastExitConfig(
            pt_short_pct=settings.get("exits.profit_target_short_dte_pct", 0.35),
            pt_multi_pct=settings.get("exits.profit_target_multi_dte_pct", 0.50),
            sl_short_pct=settings.get("exits.stop_loss_short_dte_pct", 0.20),
            sl_multi_pct=settings.get("exits.stop_loss_multi_dte_pct", 0.30),
            zero_dte_max_hold_minutes=_zero_dte_max_hold,
            entry_grace_sec=float(settings.get("exits.entry_grace_sec", 60.0)),
        ))
        # Position-fade advisor: LLM + chart review -> Discord with
        # action buttons when a winner starts fading. Runs alongside
        # the automatic fast-exit layers; this is the manual override
        # + second-opinion path.
        try:
            from .intelligence.position_advisor import PositionAdvisor
            self.position_advisor = PositionAdvisor(
                cooldown_sec=int(settings.get(
                    "exits.position_advisor_cooldown_sec", 600,
                )),
            )
        except Exception:
            self.position_advisor = None

        # Strategy mode: "directional" (long options on momentum/ORB
        # signals, original behavior) or "wheel" (sell cash-secured puts
        # on SPY/QQQ, positive-theta premium harvest). The wheel mode
        # bypasses the ensemble entirely — it's not a directional bet,
        # so competing with momentum signals makes no sense.
        self.strategy_mode = str(settings.get("strategy_mode", "directional")).lower()
        log.info("strategy_mode", mode=self.strategy_mode)
        if self.strategy_mode == "wheel":
            from .signals.wheel_runner import WheelRunner, WheelRunnerConfig
            from .exits.wheel_exits import WheelExitEvaluator, WheelExitConfig
            w = settings.raw.get("wheel", {}) or {}
            self.wheel_runner = WheelRunner(
                WheelRunnerConfig(
                    universe=list(w.get("universe", ["SPY", "QQQ"])),
                    target_dte=int(w.get("target_dte", 35)),
                    target_delta=float(w.get("target_delta", 0.30)),
                    min_premium_pct=float(w.get("min_premium_pct", 0.004)),
                    max_open_positions=int(w.get("max_open_positions", 2)),
                    min_cash_reserve_pct=float(w.get("min_cash_reserve_pct", 0.10)),
                ),
                bot=self,
            )
            self.wheel_exits = WheelExitEvaluator(WheelExitConfig(
                profit_target_pct=float(w.get("profit_target_pct", 0.50)),
                stop_loss_pct=float(w.get("stop_loss_pct", 1.00)),
                dte_roll_threshold=int(w.get("dte_roll_threshold", 21)),
            ))
            # Directional signal list is intentionally empty — wheel
            # runner is the only entry source in this mode.
            self.strategies = []
        else:
            self.wheel_runner = None
            self.wheel_exits = None
            # Build the ORB signal with the retest toggle from config
            orb_cfg = settings.raw.get("signals", {}).get("orb", {}) or {}
            orb = OpeningRangeBreakout(
                retest_required=bool(orb_cfg.get("retest_required", False)),
                retest_band_pct=float(orb_cfg.get("retest_band_pct", 0.0015)),
            )
            self.strategies = [MomentumSignal(), VwapReversionSignal(), orb]
            # Optional extreme-momentum (shock) signal — off by default.
            em_cfg = settings.raw.get("signals", {}).get("extreme_momentum", {}) or {}
            if em_cfg.get("enabled", False):
                from .signals.extreme_momentum import (
                    ExtremeMomentumSignal, ExtremeMomentumConfig,
                )
                self.strategies.append(ExtremeMomentumSignal(
                    ExtremeMomentumConfig(
                        lookback_bars=int(em_cfg.get("lookback_bars", 5)),
                        baseline_bars=int(em_cfg.get("baseline_bars", 20)),
                        min_move_pct=float(em_cfg.get("min_move_pct", 0.03)),
                        min_volume_multiple=float(em_cfg.get("min_volume_multiple", 3.0)),
                        confidence=float(em_cfg.get("confidence", 0.85)),
                    )
                ))
                log.info("extreme_momentum_signal_enabled")
            # Optional candle-pattern signal — chart-reading layer.
            # Detects bullish/bearish reversals + breakouts with volume
            # confirmation. Helps catch entries before slope-based
            # momentum triggers.
            cp_cfg = settings.raw.get("signals", {}).get("candle_patterns", {}) or {}
            if cp_cfg.get("enabled", False):
                from .signals.candle_patterns import CandlePatternSignal
                self.strategies.append(CandlePatternSignal(
                    min_bars=int(cp_cfg.get("min_bars", 20)),
                    volume_hi_ratio=float(cp_cfg.get("volume_hi_ratio", 1.5)),
                    volume_lo_ratio=float(cp_cfg.get("volume_lo_ratio", 0.6)),
                    low_vol_damp=float(cp_cfg.get("low_vol_damp", 0.20)),
                    high_vol_boost=float(cp_cfg.get("high_vol_boost", 0.10)),
                    near_vwap_bps=float(cp_cfg.get("near_vwap_bps", 50.0)),
                    near_vwap_boost=float(cp_cfg.get("near_vwap_boost", 0.05)),
                    continuation_require_volume=bool(
                        cp_cfg.get("continuation_require_volume", True)
                    ),
                ))
                log.info("candle_patterns_signal_enabled")
            # Optional technical-analysis signal — RSI divergence,
            # double top/bottom, median break, Bollinger reclaim,
            # multi-timeframe RSI confluence. Catches the kind of
            # setups discretionary chart traders watch.
            ta_cfg = settings.raw.get("signals", {}).get("technical_analysis", {}) or {}
            if ta_cfg.get("enabled", False):
                from .signals.technical_analysis import TechnicalAnalysisSignal
                self.strategies.append(TechnicalAnalysisSignal(
                    min_bars=int(ta_cfg.get("min_bars", 60)),
                    rsi_period=int(ta_cfg.get("rsi_period", 14)),
                    bb_period=int(ta_cfg.get("bb_period", 20)),
                    bb_std=float(ta_cfg.get("bb_std", 2.0)),
                    sma_period=int(ta_cfg.get("sma_period", 50)),
                    short_tf_group=int(ta_cfg.get("short_tf_group", 5)),
                    long_tf_group=int(ta_cfg.get("long_tf_group", 15)),
                    low_vol_damp=float(ta_cfg.get("low_vol_damp", 0.15)),
                ))
                log.info("technical_analysis_signal_enabled")
            # Optional long-put-dip signal — explicit "profit from the
            # fall" trigger. Fires on VWAP dip + RSI oversold + macro
            # confirmation (VIX spike or breadth deterioration).
            lpd_cfg = settings.raw.get("signals", {}).get("long_put_dip", {}) or {}
            if lpd_cfg.get("enabled", False):
                from .signals.long_put_dip import LongPutDipSignal, LongPutDipConfig
                # Wire VIX + breadth callbacks if those providers exist.
                _vix_fn = None
                _brd_fn = None
                try:
                    if self.real_time_intel is not None:
                        _vix_fn = lambda: self.real_time_intel.get_vix_snapshot()
                except Exception:
                    pass
                try:
                    if hasattr(self, "breadth_probe") and self.breadth_probe is not None:
                        _brd_fn = lambda: self.breadth_probe.latest_snapshot()
                except Exception:
                    pass
                self.strategies.append(LongPutDipSignal(
                    cfg=LongPutDipConfig(
                        vwap_dip_pct=float(lpd_cfg.get("vwap_dip_pct", 0.004)),
                        rsi_ceiling=float(lpd_cfg.get("rsi_ceiling", 35.0)),
                        rsi_period=int(lpd_cfg.get("rsi_period", 5)),
                        vix_spike_pct=float(lpd_cfg.get("vix_spike_pct", 0.05)),
                        breadth_decliners_mult=float(
                            lpd_cfg.get("breadth_decliners_mult", 2.0)
                        ),
                        min_vol_ratio=float(lpd_cfg.get("min_vol_ratio", 1.1)),
                        min_bars=int(lpd_cfg.get("min_bars", 40)),
                    ),
                    get_vix_fn=_vix_fn,
                    get_breadth_fn=_brd_fn,
                ))
                log.info("long_put_dip_signal_enabled")
            # Optional LLM-origination signal — the bot treats ideas
            # from the research agent as first-class signals. Queue is
            # file-backed so research can run in a separate process
            # and hand off async. Safety: env gate (LLM_AUTOTRADE=1) +
            # kill switch file + daily cap + confidence floor.
            llmo_cfg = settings.raw.get("signals", {}).get("llm_origination", {}) or {}
            if llmo_cfg.get("enabled", False):
                from .signals.llm_origination import LLMOriginationSignal
                self.strategies.append(LLMOriginationSignal(
                    max_age_min=int(llmo_cfg.get("max_age_min", 30)),
                    max_trades_per_day=int(llmo_cfg.get("max_trades_per_day", 3)),
                    min_confidence=str(llmo_cfg.get("min_confidence", "medium")),
                ))
                log.info("llm_origination_signal_enabled")
            # Optional TradingView webhook signal. The ingest side runs
            # in the dashboard process (POST /webhook/tradingview). This
            # source polls the file-backed queue each tick. The bot
            # doesn't need to know the dashboard is up — if the queue
            # file exists and has unconsumed alerts, they're emitted.
            if os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "").strip():
                try:
                    from .signals.tradingview_webhook import TradingViewWebhookSignal
                    self.strategies.append(TradingViewWebhookSignal())
                    log.info("tradingview_webhook_signal_enabled")
                except Exception as e:
                    log.warning("tradingview_webhook_load_failed err=%s", e)
            # Optional per-signal audit wrapping — enabled only when
            # TRADEBOT_SIGNAL_AUDIT=1 in env. Zero overhead otherwise.
            try:
                from .core.signal_audit import audit_source
                self.strategies = [audit_source(s) for s in self.strategies]
            except Exception:
                pass

        # --- Breadth probe (risk-on/off composite) ---
        # Used by the credit-spread runners + sizer to stand down
        # during risk-off regimes. Off by default; enable with
        # breadth.enabled: true in settings.yaml.
        breadth_cfg = settings.raw.get("breadth", {}) or {}
        self.breadth_probe = None
        if breadth_cfg.get("enabled", False):
            try:
                from .intelligence.breadth_probe import (
                    BreadthProbe, BreadthProbeConfig,
                )
                bp_cfg = BreadthProbeConfig(
                    cache_seconds=float(breadth_cfg.get("cache_seconds", 60.0)),
                    risk_off_threshold=float(breadth_cfg.get("risk_off_threshold", -0.3)),
                )
                self.breadth_probe = BreadthProbe(
                    cfg=bp_cfg,
                    spot_fetcher=self.data.latest_price,
                    open_fetcher=lambda s: (self.data.get_bars(s, limit=1)[0].open
                                             if self.data.get_bars(s, limit=1) else None),
                    vix_probe=getattr(self, "vix_probe", None),
                )
                log.info("breadth_probe_enabled")
            except Exception as e:
                log.warning("breadth_probe_init_failed err=%s", e)

        # --- Credit-spread runners (additive to any strategy_mode) ---
        # These are short-premium strategies that work ALONGSIDE the
        # directional or wheel paths. They don't compete with the
        # ensemble — each has its own tick cadence and risk math.
        # Toggled independently in settings.yaml under credit_spreads.*
        cs_cfg = settings.raw.get("credit_spreads", {}) or {}
        self.weekly_cs_runner = None
        self.zero_dte_cs_runner = None
        self.credit_spread_exits = None
        if cs_cfg.get("enabled", False):
            from .signals.credit_spread_runner import (
                WeeklyCreditSpreadRunner, WeeklyCreditSpreadConfig,
                ZeroDTECreditSpreadRunner, ZeroDTECreditSpreadConfig,
            )
            from .exits.credit_spread_exits import CreditSpreadExitConfig
            weekly_on = cs_cfg.get("weekly_enabled", True)
            zero_on   = cs_cfg.get("zero_dte_enabled", False)
            uni = list(cs_cfg.get("universe", ["SPY", "QQQ"]))
            if weekly_on:
                wk = cs_cfg.get("weekly", {}) or {}
                self.weekly_cs_runner = WeeklyCreditSpreadRunner(
                    WeeklyCreditSpreadConfig(
                        universe=uni,
                        dte_min=int(wk.get("dte_min", 30)),
                        dte_max=int(wk.get("dte_max", 45)),
                        short_delta=float(wk.get("short_delta", 0.20)),
                        long_delta=float(wk.get("long_delta", 0.08)),
                        max_wing_width=float(wk.get("max_wing_width", 10.0)),
                        min_credit_pct_of_wing=float(wk.get("min_credit_pct_of_wing", 0.15)),
                        max_open_positions=int(wk.get("max_open_positions", 2)),
                        profit_target_pct=float(wk.get("profit_target_pct", 0.50)),
                        dte_close_threshold=int(wk.get("dte_close_threshold", 21)),
                    ),
                    bot=self,
                )
            if zero_on:
                zd = cs_cfg.get("zero_dte", {}) or {}
                self.zero_dte_cs_runner = ZeroDTECreditSpreadRunner(
                    ZeroDTECreditSpreadConfig(
                        universe=uni,
                        short_delta=float(zd.get("short_delta", 0.12)),
                        wing_width=float(zd.get("wing_width", 5.0)),
                        max_open_positions=int(zd.get("max_open_positions", 3)),
                        rsi_oversold=float(zd.get("rsi_oversold", 35.0)),
                        support_band_pct=float(zd.get("support_band_pct", 0.005)),
                        support_lookback_bars=int(zd.get("support_lookback_bars", 20)),
                        profit_target_pct=float(zd.get("profit_target_pct", 0.50)),
                        stop_loss_pct=float(zd.get("stop_loss_pct", 1.50)),
                    ),
                    bot=self,
                )
            # Shared exit engine for both credit-spread variants
            ex = cs_cfg.get("exits", {}) or {}
            self.credit_spread_exits = CreditSpreadExitConfig(
                profit_target_pct=float(ex.get("profit_target_pct", 0.50)),
                stop_loss_pct=float(ex.get("stop_loss_pct", 1.50)),
                dte_close_threshold=int(ex.get("dte_close_threshold", 21)),
            )
            log.info("credit_spreads_enabled",
                     weekly=bool(self.weekly_cs_runner),
                     zero_dte=bool(self.zero_dte_cs_runner))
        # --- LLM brain (review layer, 8B on SD card) ---
        # Gets a compact structured summary of what the rule-based
        # signals concluded, returns {action, confidence_multiplier,
        # reason}. Soft mode (default): can only scale confidence.
        # Hard mode (opt-in): can veto. Off entirely when enabled=false.
        try:
            from .intelligence.llm_brain import build_llm_brain_from_settings
            self.llm_brain = build_llm_brain_from_settings(settings)
            if self.llm_brain is not None:
                log.info("llm_brain_enabled hard_gate=%s",
                          self.llm_brain.cfg.hard_gate)
        except Exception as e:
            log.warning("llm_brain_init_failed err=%s", e)
            self.llm_brain = None

        # --- Strategy auditor (70B on SD card, offline) ---
        # Not constructed in __init__ — it's triggered on demand by a
        # dashboard button + nightly cron so it doesn't compete for
        # GPU with the review brain during market hours.

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

        Fallback priors (small, honestly positive edge — +0.34% EV/trade):
          win_rate = 0.52, avg_win = 0.025, avg_loss = 0.020

        EV = 0.52*0.025 - 0.48*0.020 = +0.0034 per trade.
        Kelly fraction f = (1.25*0.52 - 0.48) / 1.25 = 0.136 → combined
        with `kelly_fraction_cap=0.25` and `kelly_hard_cap_pct=0.05`,
        sizer emits small-but-nonzero positions. That's important: with
        negative-EV fallbacks the bot never trades → priors never
        accumulate → it stays stuck forever (chicken-and-egg).

        These are used ONLY when there's insufficient live trade history.
        Real bots should aim to have 100+ closed trades before trusting
        journal-derived priors.
        """
        fallback = (0.52, 0.025, 0.020)
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

    def _option_quote_for_close(self, occ_symbol: str):
        """Return (bid, ask) for a specific OCC option symbol or None.
        Used by the mirror broker to price its close orders correctly.

        Parses the OCC symbol to get underlying / strike / expiry /
        right, fetches the current chain, and returns the matching
        contract's bid/ask. Returns None on any failure — callers
        interpret "no quote" as "skip the close, don't blind-dump".
        """
        try:
            from .data.options_chain_alpaca import _parse_occ
        except Exception:
            _parse_occ = None
        if _parse_occ is None:
            return None
        # OCC format: UNDER + YYMMDD + C/P + STRIKE*1000 padded to 8
        # Extract underlying by scanning until first digit.
        i = 0
        while i < len(occ_symbol) and not occ_symbol[i].isdigit():
            i += 1
        underlying = occ_symbol[:i]
        parts = _parse_occ(occ_symbol, underlying)
        if parts is None:
            return None
        expiry_d, right, strike = parts
        spot = self.data.latest_price(underlying)
        if spot is None or spot <= 0:
            return None
        try:
            from datetime import date as _d
            dte = max(1, (expiry_d - _d.today()).days)
            chain = self.chain_provider.chain(
                underlying, float(spot), target_dte=int(dte),
            )
        except Exception:
            return None
        for c in chain:
            if c.symbol == occ_symbol:
                if c.bid > 0 and c.ask > 0:
                    return (float(c.bid), float(c.ask))
                return None
            # fallback match by strike/right/expiry in case OCC symbol
            # format differs slightly between feeds
            if (c.strike == strike and c.right == right
                    and c.expiry == expiry_d
                    and c.bid > 0 and c.ask > 0):
                return (float(c.bid), float(c.ask))
        return None

    def _build_mark_prices(self, positions) -> Dict[str, float]:
        """Build {position.symbol → mark_price} for an arbitrary list of
        open positions. Options look up the current chain and use the
        conservative side (bid on long, ask on short); stocks use
        `latest_price(underlying)`. Missing quotes fall through silently
        so flatten_all can fall back to avg_price.
        """
        marks: Dict[str, float] = {}
        # Group by underlying so we fetch each chain at most once.
        option_pos_by_underlying: Dict[str, list] = {}
        for p in positions:
            if p.is_option and p.underlying:
                option_pos_by_underlying.setdefault(p.underlying, []).append(p)
            else:
                u = p.underlying or p.symbol
                last = self.data.latest_price(u)
                if last is not None:
                    marks[p.symbol] = float(last)
        for under, ops in option_pos_by_underlying.items():
            spot = self.data.latest_price(under)
            if spot is None:
                continue
            try:
                from datetime import date as _d
                # Pull the nearest chain; exit engine closes the specific
                # contract, so we need its exact (strike, expiry, right).
                dte_needed = max(1, min(
                    ((p.expiry - _d.today()).days for p in ops if p.expiry),
                    default=1,
                ))
                chain = self.chain_provider.chain(under, float(spot),
                                                   target_dte=int(dte_needed))
            except Exception as e:
                log.warning("mark_chain_fetch_failed", underlying=under, err=str(e))
                continue
            # index chain by (strike, expiry, right) for O(1) lookup
            idx = {(round(c.strike, 4), c.expiry, c.right): c for c in chain}
            for p in ops:
                key = (round(p.strike or 0.0, 4), p.expiry, p.right)
                c = idx.get(key)
                if c is None:
                    continue
                # Conservative mark: long closes at bid, short closes at ask.
                mark = c.bid if p.qty > 0 else c.ask
                if mark and mark > 0:
                    marks[p.symbol] = float(mark)
        return marks

    def _process_manual_close_intents(self, open_pos) -> None:
        """Read data/manual_close_intents.json, execute each queued
        close/trim, then clear the file. Intents come from the Discord
        `!close <aid|symbol>` and `!trim <aid>` commands and from
        button clicks on the position-fade advisory card.
        """
        from pathlib import Path as _P
        import json as _json
        from .core.data_paths import data_path
        intent_path = _P(data_path("manual_close_intents.json"))
        if not intent_path.exists():
            return
        try:
            intents = _json.loads(intent_path.read_text() or "[]")
        except Exception:
            intents = []
        if not isinstance(intents, list) or not intents:
            # Clear corrupt/empty file to avoid replay
            try:
                intent_path.write_text("[]")
            except Exception:
                pass
            return
        # Build lookup: symbol OR underlying -> Position
        by_sym = {}
        for p in open_pos:
            by_sym.setdefault(p.symbol, p)
            if p.underlying:
                by_sym.setdefault(p.underlying, p)
        remaining = []
        for it in intents:
            sym = str(it.get("symbol", "")).upper()
            kind = it.get("kind", "full_close")
            pos = by_sym.get(sym)
            if pos is None:
                log.info("manual_close_no_position", symbol=sym)
                continue     # drop — position already closed
            side = Side.SELL if pos.qty > 0 else Side.BUY
            qty_to_close = abs(pos.qty)
            if kind == "trim_half":
                qty_to_close = max(1, qty_to_close // 2)
            # Mark price (option: use chain; stock: spot)
            try:
                marks = self._build_mark_prices([pos])
                price = marks.get(pos.symbol, pos.avg_price)
            except Exception:
                price = pos.avg_price
            o = Order(
                symbol=pos.symbol, side=side, qty=qty_to_close,
                is_option=pos.is_option, limit_price=price,
                tag=f"manual:{kind}:{it.get('source', 'discord')}",
            )
            try:
                fill = self.broker.submit(o)
                log.info("manual_close_executed",
                          symbol=pos.symbol, kind=kind,
                          qty=qty_to_close,
                          fill_price=(float(fill.price) if fill else price))
                try:
                    self.notifier.notify(
                        f"✅ **Manual {kind}** executed — `{pos.symbol}` "
                        f"qty={qty_to_close} @ ~${price:.2f}",
                        level="info", title="manual_close",
                        meta={"Symbol": pos.symbol, "Kind": kind,
                               "Qty": qty_to_close},
                    )
                except Exception:
                    pass
            except Exception as e:                          # noqa: BLE001
                log.warning("manual_close_failed", symbol=sym, err=str(e))
                # Keep the intent so the next tick retries.
                remaining.append(it)
        try:
            intent_path.write_text(_json.dumps(remaining, indent=2,
                                                 default=str))
        except Exception:
            pass

    def _dynamic_symbols(self, cfg: Dict[str, Any]) -> List[str]:
        """Return a cached list of high-activity symbols from the
        SymbolScanner. Re-scans every 30 min (configurable). Filters:
          - news_mentions >= min_news_mentions
          - passes exclude_hard blocklist
          - max_dynamic_symbols cap
        """
        import time as _t
        import re as _re
        cache_ttl = 30 * 60   # 30 min
        now = _t.time()
        cache = getattr(self, "_dyn_universe_cache", None)
        if cache and (now - cache[0]) < cache_ttl:
            return cache[1]

        max_dyn = int(cfg.get("max_dynamic_symbols", 8))
        min_mentions = int(cfg.get("min_news_mentions", 3))
        exclude = set(cfg.get("exclude_hard", []) or [])
        base = set(self.s.universe)

        syms: List[str] = []
        try:
            # News-mentions via MultiProvider
            items = self.mp.news(None, limit=80) or []
            from .intelligence.symbol_scanner import _extract_tickers_from_text
            from collections import Counter as _C
            counter: _C = _C()
            for item in items:
                txt = f"{item.headline} {item.summary}"
                for sym in _extract_tickers_from_text(txt):
                    counter[sym] += 1
                for t in (item.tickers or []):
                    counter[t.upper()] += 2       # provider-native weight
            # Pick top candidates not already in base/exclude
            for sym, n in counter.most_common(40):
                if n < min_mentions:
                    break
                if sym in base or sym in exclude:
                    continue
                # Sanity-check format: 2-5 uppercase letters
                if not _re.match(r"^[A-Z]{2,5}$", sym):
                    continue
                syms.append(sym)
                if len(syms) >= max_dyn:
                    break
        except Exception as e:                              # noqa: BLE001
            log.info("dynamic_symbols_scan_err", err=str(e)[:120])
        # Cache + log
        self._dyn_universe_cache = (now, syms)
        if syms:
            log.info("dynamic_universe_refreshed",
                     n=len(syms), symbols=syms)
        return syms

    def _post_fade_advisory(self, adv) -> None:
        """Post a richly-formatted position-fade advisory to Discord.
        Includes LLM reasoning, chart signals cited by name, risk/reward
        remaining, alternative actions, and manual-override CTA."""
        try:
            from .intelligence.position_advisor import save_advisory
            aid = save_advisory(adv)
        except Exception:
            aid = ""
        urgency_icon = {"urgent": "🚨", "normal": "⚠️", "low": "💡"}.get(
            adv.urgency, "⚠️"
        )
        rec_icon = {"close": "🛑 CLOSE", "trim": "✂️ TRIM",
                     "hold": "✋ HOLD"}.get(adv.recommendation, "•")
        # Header
        lines = [
            f"{urgency_icon} **Position Fade Advisory · {adv.symbol}**",
        ]
        contract = (f"{adv.direction.upper()} ${adv.strike:g}"
                     if adv.strike else adv.direction.upper())
        if adv.expiry:
            contract += f" · exp {adv.expiry}"
        contract += f" · qty {adv.qty}"
        lines.append(f"_{contract}_")
        lines.append("")
        # P&L snapshot
        give_back_pct = (
            ((adv.peak_pnl_pct - adv.current_pnl_pct) /
              max(adv.peak_pnl_pct, 1e-9)) * 100
            if adv.peak_pnl_pct > 0 else 0
        )
        lines.append(
            f"**Peak:** +{adv.peak_pnl_pct*100:.2f}%  →  "
            f"**Now:** {adv.current_pnl_pct*100:+.2f}%  "
            f"(gave back **{give_back_pct:.0f}%**)"
        )
        lines.append(
            f"Entry ${adv.entry_price:.2f} → Now ${adv.current_price:.2f}"
        )
        lines.append("")
        # Recommendation header
        lines.append(
            f"**Recommendation:** {rec_icon} · *{adv.confidence}* confidence"
        )
        # LLM rationale (the "why")
        if adv.rationale:
            lines.append(f"> 💭 {adv.rationale}")
        # Chart signals — the specific features driving the call
        if adv.chart_signals:
            lines.append("")
            lines.append("**📊 Chart signals**")
            for sig in adv.chart_signals[:4]:
                lines.append(f"  · {sig}")
        elif adv.bars_summary:
            lines.append(f"_Chart: {adv.bars_summary}_")
        # Key levels
        if adv.key_levels:
            lines.append("")
            lines.append("**🎯 Key levels**")
            for kl in adv.key_levels[:4]:
                lines.append(f"  · {kl}")
        # Risk/reward remaining
        if adv.risk_reward_remaining:
            lines.append("")
            lines.append(f"**⚖ R:R if held** — {adv.risk_reward_remaining}")
        # Time context (DTE + theta)
        if adv.time_context:
            lines.append(f"**⏱ Time** — {adv.time_context}")
        # Alternative actions
        if adv.alternative_actions:
            lines.append("")
            lines.append("**🔀 Alternatives**")
            for alt in adv.alternative_actions[:3]:
                lines.append(f"  · {alt}")
        # Footer
        lines.append("")
        if adv.model:
            lines.append(f"_source: {adv.model}_")
        if aid:
            lines.append(
                f"_advisory `{aid}` · manual override: `!close {aid}` / "
                f"`!trim {aid}`_"
            )
        meta = {
            "Symbol":        adv.symbol,
            "Peak":          f"{adv.peak_pnl_pct*100:+.2f}%",
            "Now":           f"{adv.current_pnl_pct*100:+.2f}%",
            "Recommend":     adv.recommendation,
            "Urgency":       adv.urgency,
            "_advisory_id":  aid,
            "_footer":       "position_advisor",
        }
        try:
            self.notifier.notify(
                "\n".join(lines), level="info",
                title="position_advisor", meta=meta,
            )
        except Exception:
            pass

    def _evaluate_credit_spreads(self, open_pos, marks) -> None:
        """Group credit-spread legs by tag, evaluate each as a unit,
        submit a single close-combo if the exit criteria fire.

        Keeps exit decisions symmetric: a credit spread's two legs
        either both stay open or both close. No half-closes via per-leg
        logic.
        """
        from .exits.credit_spread_exits import (
            group_spread_positions, evaluate_spread, build_close_combo,
        )
        from datetime import datetime as _dt
        try:
            from zoneinfo import ZoneInfo
            now_et = _dt.now(tz=ZoneInfo("America/New_York"))
        except Exception:
            now_et = _dt.utcnow()

        groups = group_spread_positions(open_pos)
        if not groups:
            return

        # Build a {symbol → OptionContract} map from current chains so
        # the close combo can use live bid/ask for its limit price.
        contracts_by_symbol: Dict[str, Any] = {}
        underlyings = {p.underlying for legs in groups.values() for p in legs
                       if p.underlying}
        for under in underlyings:
            spot = self.data.latest_price(under)
            if not spot:
                continue
            try:
                # Pull a wide window so both the short and long legs land in it
                chain = self.chain_provider.chain(under, float(spot),
                                                   target_dte=5)
            except Exception:
                chain = []
            for c in chain or []:
                contracts_by_symbol[c.symbol] = c

        for tag, legs in groups.items():
            try:
                decision = evaluate_spread(
                    legs, marks, self.credit_spread_exits, now_et=now_et,
                )
                if decision is None or not decision.should_close:
                    continue
                combo = build_close_combo(decision, contracts_by_symbol)
                if combo is None:
                    log.warning("cs_close_build_failed tag=%s", tag)
                    continue
                log.info("cs_close_submit tag=%s reason=%s pnl=%.2f%%",
                          tag, decision.reason,
                          decision.net_pnl_pct_of_credit * 100)
                self.broker.submit_combo(combo)
                self.notifier.notify(
                    f"CLOSE credit spread ({tag}) — {decision.reason}",
                    title="exit", level=("success"
                         if decision.net_pnl_pct_of_credit > 0 else "warn"),
                    meta={
                        "strategy": "credit_spread_exit",
                        "tag": tag,
                        "reason": decision.reason,
                        "net_pnl_vs_credit": f"{decision.net_pnl_pct_of_credit:+.1%}",
                    },
                )
            except Exception as e:                         # noqa: BLE001
                log.warning("cs_eval_failed tag=%s err=%s", tag, e)

    def _refresh_catalysts(self) -> None:
        """Pull earnings/FDA events and hydrate the EconomicCalendar with
        per-symbol blackouts. Swallows errors (network, missing deps).

        Discord notification throttled to once per day — the catalyst
        list only changes slowly (earnings dates are set weeks ahead)
        so re-posting on every watchdog restart is pure noise.
        """
        try:
            events = self.catalyst_calendar.refresh(self.s.universe)
            n = self.catalyst_calendar.hydrate_econ_calendar(self.econ_calendar)
            log.info("catalysts_refreshed", events=len(events), blackouts=n)
            # Throttle: once per calendar day. _last_catalyst_notify_date
            # is initialized to None so first-ever post happens; after
            # that only when the date changes.
            from datetime import date as _date
            today = _date.today()
            last = getattr(self, "_last_catalyst_notify_date", None)
            if last == today:
                return
            self._last_catalyst_notify_date = today
            if events:
                # Group by symbol so the list stays scannable. With
                # structured notifier `meta` we get one field per event
                # — Discord embeds display up to 25 inline fields.
                # Previous version truncated to 8 in a one-line text
                # dump which made "23 catalysts" produce an unreadable
                # string.
                if bool(self.s.get("notifier.catalysts_notify", False)):
                    meta = {}
                    for e in events[:24]:     # Discord embed field cap
                        key = f"{e.symbol} {e.event_type}"
                        meta[key] = str(e.when)
                    if len(events) > 24:
                        meta["_footer"] = f"{len(events) - 24} more not shown"
                    self.notifier.notify(
                        f"{len(events)} upcoming catalysts in the next "
                        f"{int(self.s.get('catalysts.lookahead_days', 14))} days",
                        title="catalysts",
                        level="info",
                        meta=meta,
                    )
        except Exception as e:   # noqa: BLE001
            log.warning("catalyst_refresh_failed", err=str(e))

    def fast_loop(self) -> None:
        interval = float(self.s.get("exits.fast_thread_interval_sec", 5))
        kill_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "KILL",
        )
        # Heartbeat file — main_loop checks this and alerts if >90s
        # stale. Catches the silent-stall scenario that caused the
        # QQQ 654 call to sit unexited for 24 min (bot process alive,
        # fast_loop thread wedged or blocked on a network call).
        import time as _t
        _hb_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "fast_loop_heartbeat.txt",
        )
        def _write_heartbeat():
            try:
                with open(_hb_path, "w") as f:
                    f.write(str(_t.time()))
            except Exception:
                pass
        while not self._stop.is_set():
            _write_heartbeat()
            # Cooperative kill check — needs to be here too, not just main_loop.
            # At a 5s cadence we react within ~5s even if main_loop is blocked.
            if os.path.exists(kill_file):
                log.warning("kill_switch_seen_fast", path=kill_file)
                self._stop.set()
                break
            try:
                open_pos = list(self.broker.positions())
                # Process any manual close/trim intents from Discord BEFORE
                # the normal exit loop — so `!close SPY` closes within ~5s.
                try:
                    self._process_manual_close_intents(open_pos)
                except Exception as _ie:                    # noqa: BLE001
                    log.info("manual_close_intent_err", err=str(_ie)[:160])
                if not open_pos:
                    self._stop.wait(interval)
                    continue
                # Options positions need per-contract marks (from the chain),
                # NOT underlying spot. Using spot here would feed $250 into
                # an exit evaluator that's comparing to a $1.20 option entry,
                # triggering the hard-profit-cap instantly every tick.
                marks = self._build_mark_prices(open_pos)

                # -------- credit-spread exits (evaluated FIRST) --------
                # A credit spread has two legs that must unwind together.
                # Evaluate by spread tag BEFORE the per-position loop so
                # we don't half-close a spread via the per-leg logic.
                if self.credit_spread_exits is not None:
                    try:
                        self._evaluate_credit_spreads(open_pos, marks)
                    except Exception as e:                    # noqa: BLE001
                        log.warning("credit_spread_exit_failed", err=str(e))
                        try:
                            from .notify.issue_reporter import report_issue
                            report_issue(
                                scope="credit_spread_exits",
                                message=f"exit eval failed: {type(e).__name__}: {e}",
                                exc=e,
                            )
                        except Exception:
                            pass
                    # Refresh position list — some legs may have just
                    # been submitted for close. Stale positions would
                    # cause the per-leg loop below to issue a second
                    # close order.
                    open_pos = list(self.broker.positions())

                for pos in open_pos:
                    price = marks.get(
                        pos.symbol,
                        # fallback: option avg_price (zero-change), stock spot
                        pos.avg_price if pos.is_option
                        else (self.data.latest_price(pos.underlying or pos.symbol)
                              or pos.avg_price),
                    )
                    # Route to the right exit evaluator based on position sign:
                    # short-option (qty<0) → wheel exit (premium math)
                    # long (qty>0)          → standard FastExitEvaluator
                    if (self.wheel_exits is not None
                            and pos.is_option and pos.qty < 0):
                        d = self.wheel_exits.evaluate(pos, price)
                    else:
                        # Pass bars of the UNDERLYING so the momentum-exit
                        # check has trend/volume data. Cheap — data
                        # adapter caches per-symbol bars.
                        und = pos.underlying or pos.symbol
                        try:
                            bars_for_exit = self.data.get_bars(und, limit=40)
                        except Exception:
                            bars_for_exit = None
                        d = self.fast.evaluate(pos, price, bars=bars_for_exit)
                    # Green-to-red kill switch — DTE-aware thresholds.
                    # 0DTE: ANY fade from positive to negative = close.
                    # Short: was +1%+, now below -0.5% = close.
                    # Swing: was +3%+, now below -2% = close.
                    # Honors entry_grace_sec — no fade kill inside
                    # the grace window to avoid closing on spread wash.
                    import time as _t_g
                    _entry_age_g = _t_g.time() - float(pos.entry_ts or 0)
                    _grace_g = float(self.s.get(
                        "exits.entry_grace_sec", 60.0
                    ))
                    if (d is None or not d.should_close) and \
                            pos.is_option and pos.qty > 0 and \
                            _entry_age_g >= _grace_g:
                        peak_pnl = getattr(pos, "peak_pnl_pct", None) or 0.0
                        pnl_now = pos.unrealized_pnl_pct(price)
                        pos_dte = pos.dte() if hasattr(pos, "dte") else 7
                        # DTE-aware thresholds
                        if pos_dte == 0:
                            peak_min, now_max = 0.005, -0.001
                            dte_label = "0dte"
                        elif pos_dte >= 14:
                            peak_min, now_max = 0.03, -0.02
                            dte_label = "swing"
                        else:
                            peak_min, now_max = 0.01, -0.005
                            dte_label = "short"
                        if peak_pnl >= peak_min and pnl_now <= now_max:
                            from .exits.fast_exit import ExitDecision as _ED
                            d = _ED(
                                True,
                                (f"green_to_red_{dte_label}:peak={peak_pnl:+.2%}"
                                 f"_now={pnl_now:+.2%}_dte={pos_dte}"),
                                layer=0,
                            )
                        # Absolute drawdown kill — independent of peak
                        # tracking. Protects against the case where peak
                        # wasn't recorded (bot restart, old position).
                        #
                        # Grace period: don't fire within the first
                        # `entry_grace_sec` (60s default) because the
                        # bid/ask spread wash on fresh entries makes
                        # pnl look like -5-8% on the first tick even
                        # though the market hasn't moved. Without this
                        # grace period, the killswitch fires 6 seconds
                        # after every entry, never giving trades a
                        # chance to develop.
                        import time as _t
                        _entry_age = _t.time() - float(pos.entry_ts or 0)
                        _grace = float(self.s.get(
                            "exits.entry_grace_sec", 60.0
                        ))
                        if (d is None or not d.should_close) and \
                                _entry_age >= _grace:
                            if pos_dte == 0:
                                max_dd = -0.05
                            elif pos_dte >= 14:
                                max_dd = -0.08
                            else:
                                max_dd = -0.06
                            if pnl_now <= max_dd:
                                from .exits.fast_exit import (
                                    ExitDecision as _ED,
                                )
                                d = _ED(
                                    True,
                                    (f"absolute_drawdown_{dte_label}:"
                                     f"now={pnl_now:+.2%}<={max_dd:+.2%}_dte={pos_dte}"),
                                    layer=0,
                                )
                    # Position-fade advisor: LLM second opinion + auto-
                    # execute on 'urgent close' recommendations.
                    if (d is None or not d.should_close) and \
                            self.position_advisor is not None and \
                            pos.is_option and pos.qty > 0:
                        try:
                            pnl_pct_now = pos.unrealized_pnl_pct(price)
                            adv = self.position_advisor.maybe_advise(
                                pos, price, pnl_pct_now, bars=bars_for_exit,
                            )
                            if adv is not None:
                                self._post_fade_advisory(adv)
                                # AUTO-EXECUTE urgent close recommendations
                                # ONLY when the position had meaningful
                                # peak (≥ +5% for swing, ≥ +3% for short,
                                # ≥ +2% for 0DTE) AND has given back
                                # >=50%. Prevents LLM-closing a +1%
                                # winner on a −2% dip. Operator case:
                                # NVDA closed at peak +1.03% → −2.23%
                                # which is just noise, not a fade.
                                _peak = float(adv.peak_pnl_pct or 0)
                                _now = float(adv.current_pnl_pct or 0)
                                _dte_pos = pos.dte() if hasattr(pos, "dte") else 7
                                _min_peak = (0.02 if _dte_pos == 0
                                             else 0.05 if _dte_pos >= 14
                                             else 0.03)
                                _gave_back = ((_peak - _now) / max(_peak, 1e-9)
                                               if _peak > 0 else 0.0)
                                _llm_execute = (
                                    adv.recommendation == "close"
                                    and adv.urgency == "urgent"
                                    and _peak >= _min_peak
                                    and _gave_back >= 0.50
                                )
                                if _llm_execute:
                                    from .exits.fast_exit import (
                                        ExitDecision as _ED,
                                    )
                                    d = _ED(
                                        True,
                                        (f"llm_urgent_close:peak="
                                         f"{adv.peak_pnl_pct:+.2%}_now="
                                         f"{adv.current_pnl_pct:+.2%}"),
                                        layer=0,
                                    )
                                elif (adv.recommendation == "trim"
                                        and adv.urgency == "urgent"):
                                    # Trim half — close_qty set so fast
                                    # exit only closes 50%.
                                    from .exits.fast_exit import (
                                        ExitDecision as _ED,
                                    )
                                    d = _ED(
                                        True,
                                        (f"llm_urgent_trim:peak="
                                         f"{adv.peak_pnl_pct:+.2%}_now="
                                         f"{adv.current_pnl_pct:+.2%}"),
                                        layer=0,
                                        close_qty=max(
                                            1, abs(pos.qty) // 2,
                                        ),
                                    )
                        except Exception as _e:             # noqa: BLE001
                            log.info("position_advisor_err", err=str(_e)[:120])
                    if d and d.should_close:
                        side = Side.SELL if pos.qty > 0 else Side.BUY
                        qty_abs = abs(pos.qty)
                        entry_px = pos.avg_price
                        o = Order(symbol=pos.symbol, side=side, qty=qty_abs,
                                  is_option=pos.is_option, limit_price=price,
                                  tag=f"fast:{d.reason}")
                        fill = self.broker.submit(o)
                        # Realized P&L + percentage for the notification.
                        # For a long position closed at `price`:
                        #   P&L = (price - avg_price) * qty * multiplier
                        mult = pos.multiplier or (100 if pos.is_option else 1)
                        realized_usd = ((float(fill.price) if fill else float(price))
                                          - entry_px) * qty_abs * mult * (
                            1 if pos.qty > 0 else -1
                        )
                        pnl_pct = (((float(fill.price) if fill else float(price))
                                      - entry_px) / max(entry_px, 1e-9)) * (
                            100 if pos.qty > 0 else -100
                        )
                        log.info("fast_exit", symbol=pos.symbol,
                                  reason=d.reason, price=price,
                                  realized_usd=round(realized_usd, 2),
                                  pnl_pct=round(pnl_pct, 2))
                        # Saves tracker — log defensive exits for
                        # later re-check to prove the exit engine is
                        # paying off. No-op if reason isn't defensive.
                        try:
                            from .intelligence.saves_tracker import record_exit
                            record_exit(
                                symbol=pos.symbol,
                                underlying=pos.underlying or pos.symbol,
                                exit_reason=str(d.reason),
                                exit_price=float(fill.price) if fill else float(price),
                                qty=int(qty_abs),
                                peak_pnl_pct=float(getattr(pos, "peak_pnl_pct", 0) or 0),
                                exit_pnl_pct=float(pnl_pct) / 100.0,
                                dte=int(pos.dte() if hasattr(pos, "dte") else 7),
                            )
                        except Exception:
                            pass
                        # Discord: green if profit, red if loss.
                        underlying = pos.underlying or pos.symbol
                        right_str = (pos.right.value.upper() if pos.right
                                       else ("CALL" if pos.is_option else ""))
                        exit_px = float(fill.price) if fill else float(price)
                        # Decode the internal exit reason into a
                        # plain-English explanation.
                        reason_raw = str(d.reason)
                        reason_human = {
                            "fast_pt_hit": "Hit profit target (+35%) — full close (no scale-out path available)",
                            "fast_scale_out_at_pt": "Hit profit target — closed 50%, trailing the rest",
                            "fast_sl_hit": "Hit stop loss — cutting the trade to cap the loss",
                            "fast_hard_cap": "Hit hard profit cap (+150%) — full close",
                            "fast_trailing_stop": "Retraced >10% from peak — locking the remaining gain",
                            "fast_0dte_scalp_timeout": "0DTE held past scalp window — theta was going to eat it",
                            "momentum_reversal": "Underlying put in consecutive bars against us — trend died",
                            "volume_dry_up": "Volume fell below 50% baseline with no new high — interest faded",
                            "profit_lock": "Profit-lock trailing — retraced past adaptive give-back from peak, took the gain",
                            "profit_floor": "Hit the ratcheting floor — big peak, locked the tier minimum",
                            "support_break": "Underlying closed below 5-bar support — thesis cracked, took profit",
                            "resistance_break": "Underlying closed above 5-bar resistance — bearish thesis invalidated, took profit",
                            "chart_lower_highs": "Chart made consecutive lower highs — trend rolling over, took profit",
                            "chart_higher_lows": "Chart made consecutive higher lows — downtrend broken, took profit",
                            "vwap_break": "Underlying lost VWAP — institutional value line gone, took profit",
                            "vwap_break_up": "Underlying reclaimed VWAP — bearish thesis invalid, took profit",
                            "green_to_red_killswitch": "Was in profit and started going red — auto-closed to preserve any gain / stop the bleed",
                            "green_to_red_0dte": "0DTE went green → red — closed instantly (no recovery window on same-day expiry)",
                            "green_to_red_short": "Short-dated position went green → red — closed before the bleed deepens",
                            "green_to_red_swing": "Swing position: was +3%+, now -2%+ — thesis broken, cutting",
                            "profit_lock_0dte": "0DTE profit-lock — tight retrace detected, locked the gain fast",
                            "profit_lock_short": "Profit-lock — retraced past the floor, took the gain",
                            "profit_lock_swing": "Swing profit-lock — loose threshold let it breathe, then locked",
                            "llm_urgent_close": "LLM chart review said URGENT CLOSE — auto-executed without waiting for operator",
                            "llm_urgent_trim": "LLM said urgent trim — closed 50%, trailing the rest",
                            "zdte_snap_absolute": "0DTE at +40%+ — took it instantly (theta was about to erase it)",
                            "zdte_snap_close": "0DTE at +25% — grabbed the profit before theta ate it",
                            "zdte_snap_trim": "0DTE at +15% — trimmed half, let the rest run",
                            "absolute_drawdown_0dte": "0DTE down >3% from entry — hard cut (no peak memory needed)",
                            "absolute_drawdown_short": "Short-dated down >4% from entry — hard cut",
                            "absolute_drawdown_swing": "Swing down >5% from entry — hard cut",
                            "zdte_exhaustion": "0DTE in profit but chart shows NO upside signal — took the gain before theta ate it (no new high, flat volume, VWAP not expanding)",
                            "active_downside": "Chart is ACTIVELY going against us (VWAP break / red volume surge / lower high) — cut and look for a better re-entry",
                            "active_upside_vs_put": "Long put but chart is going UP — thesis invalidated, cut and reassess direction",
                        }
                        prefix = reason_raw.split(":")[0]
                        why = reason_human.get(prefix, reason_raw)
                        pnl_emoji = "🟢" if realized_usd > 0 else ("🔴" if realized_usd < 0 else "⚪")
                        description = (
                            f"**P&L**: `{pnl_pct:+.2f}%` = **${realized_usd:+,.2f}**  "
                            f"({qty_abs}× closed at ${exit_px:.2f}, entered at "
                            f"${entry_px:.2f})\n\n"
                            f"**Exit reason**: {why}"
                        )
                        self.notifier.notify(
                            description,
                            title=(f"{pnl_emoji} EXIT — {underlying} {right_str} "
                                     f"{pnl_pct:+.1f}% (${realized_usd:+,.0f})"),
                            level="success" if realized_usd > 0 else (
                                "error" if realized_usd < 0 else "info"
                            ),
                            meta={
                                "📊 Symbol": underlying,
                                "📊 Contract": f"{right_str} {pos.strike or '—'}",
                                "📊 Qty Closed": qty_abs,
                                "💰 Entry Price": f"${entry_px:.2f}/share",
                                "💰 Exit Price": f"${exit_px:.2f}/share",
                                "💰 Realized P&L": f"{pnl_pct:+.2f}% = ${realized_usd:+,.2f}",
                                "🔎 Trigger": prefix,
                                "🔎 Detail": reason_raw.split(":", 1)[1] if ":" in reason_raw else "—",
                                "_footer": f"OCC {pos.symbol}",
                            },
                        )
            except Exception as e:  # noqa: BLE001
                log.warning("fast_loop_error", err=str(e))
                # Push to Discord alerts. Throttled by (scope, message)
                # fingerprint inside report_issue, so a repeating error
                # in a tight loop sends one alert every 5 minutes.
                try:
                    from .notify.issue_reporter import report_issue
                    report_issue(
                        scope="fast_loop",
                        message=f"fast-exit thread: {type(e).__name__}: {e}",
                        exc=e,
                    )
                except Exception:
                    pass
            self._stop.wait(interval)

    def main_loop(self) -> None:
        interval = float(self.s.get("exits.main_loop_interval_sec", 180))
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kill_file = os.path.join(repo_root, "KILL")
        # Tests redirect heartbeat writes so they don't collide with a
        # real running watchdog on the same machine.
        _sandbox = os.getenv("TRADEBOT_SANDBOX_LOGS", "").strip()
        heartbeat_file = (os.path.join(_sandbox, "heartbeat.txt") if _sandbox
                           else os.path.join(repo_root, "logs", "heartbeat.txt"))
        os.makedirs(os.path.dirname(heartbeat_file), exist_ok=True)
        while not self._stop.is_set():
            # Heartbeat: the watchdog reads this and restarts us if it goes
            # stale (process alive but main loop wedged). Cost: one syscall
            # per tick.
            try:
                with open(heartbeat_file, "w", encoding="utf-8") as _hb:
                    _hb.write(datetime.now(tz=timezone.utc).isoformat())
            except Exception:
                pass
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
                    if self.strategy_mode == "wheel" and self.wheel_runner:
                        # Wheel bypasses the directional per-symbol tick.
                        # It has its own universe + cadence.
                        try:
                            self.wheel_runner.tick()
                        except Exception as e:              # noqa: BLE001
                            log.warning("wheel_runner_tick_failed", err=str(e))
                            try:
                                from .notify.issue_reporter import report_issue
                                report_issue(
                                    scope="wheel_runner.tick",
                                    message=f"wheel tick failed: {type(e).__name__}: {e}",
                                    exc=e,
                                )
                            except Exception:
                                pass
                    else:
                        # Build the effective universe per tick:
                        #   base universe (from settings.yaml)
                        #   + dynamic additions from SymbolScanner
                        #     (high-activity tickers from news/movers)
                        effective_universe = list(self.s.universe)
                        dyn_cfg = self.s.raw.get("dynamic_universe", {}) or {}
                        if dyn_cfg.get("enabled", False):
                            try:
                                dyn_syms = self._dynamic_symbols(dyn_cfg)
                                for s in dyn_syms:
                                    if s not in effective_universe:
                                        effective_universe.append(s)
                            except Exception as _de:         # noqa: BLE001
                                log.info("dynamic_universe_err",
                                          err=str(_de)[:120])
                        for symbol in effective_universe:
                            self._tick_symbol(symbol)
                    # Credit-spread runners tick AFTER directional/wheel.
                    # They have their own time-of-day gates + idempotency
                    # so running every main loop tick is safe.
                    for runner_name, runner in (
                        ("weekly_credit_spread", self.weekly_cs_runner),
                        ("zero_dte_credit_spread", self.zero_dte_cs_runner),
                    ):
                        if runner is None:
                            continue
                        try:
                            runner.tick()
                        except Exception as e:              # noqa: BLE001
                            log.warning("%s_tick_failed", runner_name, err=str(e))
                            try:
                                from .notify.issue_reporter import report_issue
                                report_issue(
                                    scope=f"{runner_name}.tick",
                                    message=f"{runner_name} tick failed: {type(e).__name__}: {e}",
                                    exc=e,
                                )
                            except Exception:
                                pass
                # Mark every open position at current market so equity reflects
                # live P&L (otherwise it's frozen at avg_price until a close).
                # Fail-soft — a bad chain fetch shouldn't stop the loop.
                try:
                    open_pos = self.broker.positions()
                    if open_pos:
                        self.broker.mark_to_market(
                            self._build_mark_prices(open_pos)
                        )
                except Exception as e:                        # noqa: BLE001
                    log.warning("mtm_tick_failed", err=str(e))
                    try:
                        from .notify.issue_reporter import report_issue
                        report_issue(
                            scope="mark_to_market",
                            message=f"MTM tick failed: {type(e).__name__}: {e}",
                            exc=e,
                        )
                    except Exception:
                        pass
            except Exception as e:  # noqa: BLE001
                log.error("main_loop_error", err=str(e))
                # title="error" routes to #tradebot-alerts per the
                # MultiChannelNotifier map. "tradebot" used to fall back
                # to default — this is the fix.
                try:
                    from .notify.issue_reporter import report_issue
                    report_issue(
                        scope="main_loop",
                        message=f"main_loop_error: {type(e).__name__}: {e}",
                        exc=e,
                    )
                except Exception:
                    # Belt-and-suspenders: if report_issue itself is
                    # unavailable, fall back to the old direct notifier.
                    self.notifier.notify(
                        f"main_loop_error: {e}",
                        level="error", title="error",
                    )
            self._stop.wait(interval)

    def _check_halt_conditions(self) -> None:
        if self._halted_today:
            return
        acct = self.broker.account()
        # Track peak equity for peak-to-trough DD
        if acct.equity > self._peak_equity:
            self._peak_equity = acct.equity

        # VIX regime gate (quant-advisor recommendation).
        # Three regimes:
        #   VIX < 15                     → low vol, trend strategies work → OK to trade
        #   VIX 15-25                    → normal, trade with care → OK
        #   VIX > regime_halt_vix        → crisis/fear, directional alpha compresses,
        #                                   correlations converge, bid-ask widens → STAND DOWN
        # When VIX/VIX3M > 1.0 (backwardation) it signals acute stress
        # ahead of the spot index — also a stand-down signal.
        try:
            regime_halt_vix = float(self.s.get("vix.regime_halt_vix", 25.0))
            backwardation_halt_enabled = bool(self.s.get(
                "vix.backwardation_halt_enabled", False))   # off by default
            vix_now = self.vix_probe.value()
            if vix_now > regime_halt_vix:
                self._halted_today = True
                log.warning("vix_regime_halt", vix=vix_now, threshold=regime_halt_vix)
                self.notifier.notify(
                    f"VIX regime halt — standing down for the day.",
                    level="error", title="HALT",
                    meta={
                        "vix_now": f"{vix_now:.2f}",
                        "threshold": f"{regime_halt_vix:.2f}",
                        "reason": "VIX above regime-halt threshold; directional alpha compresses and spreads widen in high-vol regimes",
                    },
                )
                return
            # Optional backwardation check (needs VIX3M feed)
            if backwardation_halt_enabled:
                vix3m = float(self.s.get("vix.vix3m_override", 0) or 0)
                if vix3m > 0 and vix_now / vix3m > 1.0:
                    self._halted_today = True
                    log.warning("vix_backwardation_halt",
                                 vix=vix_now, vix3m=vix3m,
                                 ratio=vix_now/vix3m)
                    self.notifier.notify(
                        f"VIX term structure in backwardation — crisis signal.",
                        level="error", title="HALT",
                        meta={
                            "vix": f"{vix_now:.2f}",
                            "vix3m": f"{vix3m:.2f}",
                            "ratio": f"{vix_now/vix3m:.3f}",
                            "reason": "VIX > VIX3M (backwardation) historically precedes sharp moves; stand down",
                        },
                    )
                    return
        except Exception as e:                               # noqa: BLE001
            log.warning("vix_regime_check_failed", err=str(e))
        cap = self.s.max_daily_loss_pct * acct.equity
        if acct.day_pnl <= -cap:
            self._halted_today = True
            log.warning("daily_loss_halt", day_pnl=acct.day_pnl, cap=cap)
            self.notifier.notify(
                f"Daily loss cap breached — no new entries for the rest of today.",
                level="error", title="HALT",
                meta={
                    "day_pnl": f"${acct.day_pnl:+,.2f}",
                    "equity": f"${acct.equity:,.2f}",
                    "cap": f"-${cap:,.2f}",
                    "dd_from_peak": f"{(1 - acct.equity/max(self._peak_equity, 1e-9)) * 100:.2f}%",
                },
            )
            return
        # Tiered drawdown guard (toggleable). When disabled via
        # account.drawdown_guard_enabled=false, portfolio-wide DD halt
        # and scale-down do NOT fire — per-trade stop losses
        # (exits.stop_loss_*_pct) are the only risk bound. Operator
        # preference: keep bot trading, let tight stops do the work.
        if not bool(self.s.raw.get("account", {}).get(
                "drawdown_guard_enabled", True)):
            return
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
            mark_prices = self._build_mark_prices(open_before)
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
        level = "success" if acct.day_pnl >= 0 else "error" if acct.day_pnl <= -abs(self.s.max_daily_loss_pct * acct.equity) else "warn"
        self.notifier.notify(
            f"End of session {today}. {len(open_before)} positions flattened.",
            title="daily",
            level=level,
            meta={
                "equity": f"${acct.equity:,.2f}",
                "day_pnl": f"${acct.day_pnl:+,.2f}",
                "day_pct": f"{(acct.day_pnl/max(acct.equity-acct.day_pnl, 1e-9))*100:+.2f}%",
                "total_pnl": f"${acct.total_pnl:+,.2f}",
                "flattened": len(open_before),
                "cash": f"${acct.cash:,.2f}",
            },
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
        # Periodic snapshot log so the LLM chat / research agent can
        # scrape current VIX + regime from tradebot.out even when no
        # ensemble decision fires. Throttled to once every 5 minutes
        # per process so it doesn't spam.
        try:
            import time as _time
            _last = getattr(self, "_regime_log_last", 0.0)
            if _time.time() - _last > 300.0:
                # Include breadth when we have it — otherwise the chat
                # LLM falls back to "no breadth available".
                breadth_val = None
                if hasattr(self, "breadth_probe") and self.breadth_probe:
                    try:
                        snap = self.breadth_probe.latest_snapshot()
                        # Handle both BreadthSnapshot dataclass + dict shape
                        if hasattr(snap, "score"):
                            breadth_val = snap.score
                        elif isinstance(snap, dict):
                            breadth_val = snap.get("score")
                        elif isinstance(snap, (int, float)):
                            breadth_val = float(snap)
                    except Exception as _be:                  # noqa: BLE001
                        log.info("breadth_snapshot_err", err=str(_be)[:120])
                        breadth_val = None
                log.info(
                    "market_state_snapshot",
                    regime=regime.value,
                    vix=round(float(vix_reading.value), 2),
                    breadth_score=(round(float(breadth_val), 3)
                                     if breadth_val is not None else None),
                )
                self._regime_log_last = _time.time()
        except Exception:
            pass

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

        # --- LLM brain review ---
        # Runs BEFORE the filter chain so if the brain vetoes (hard mode)
        # or scales confidence below threshold, we skip the rest of the
        # work. Gets a compact structured summary, not raw bars. Always
        # fail-open — an LLM outage or malformed JSON returns 1.0x so
        # the rule-based decision passes through unchanged.
        if self.llm_brain is not None:
            try:
                from .intelligence.llm_brain import (
                    CandidateDecision, ReviewContext,
                )
                from .core.signal_audit import log_emit
                breadth_snap = (self.breadth_probe.snapshot()
                                 if self.breadth_probe is not None else None)
                rsi_val = None
                try:
                    from .signals.credit_spread_runner import _rsi as _rsi_fn
                    closes = [b.close for b in bars]
                    rsi_val = _rsi_fn(closes, 14)
                except Exception:
                    pass
                move5 = None
                if len(bars) > 5 and bars[-6].close > 0:
                    move5 = (bars[-1].close - bars[-6].close) / bars[-6].close
                ctx_for_brain = ReviewContext(
                    spot=float(spot), vwap=float(vwap),
                    regime=(getattr(self, "_last_regime", None)),
                    vix=float(vix_now) if vix_now is not None else None,
                    breadth_score=(breadth_snap.score if breadth_snap else None),
                    breadth_is_risk_off=bool(breadth_snap.is_risk_off if breadth_snap else False),
                    rsi_14=rsi_val,
                    five_bar_move_pct=move5,
                    open_positions=len(self.broker.positions()),
                    position_on_symbol=sum(p.qty for p in self.broker.positions()
                                            if (p.underlying or "") == sig.symbol),
                    catalyst_in_24h=econ_blackout,
                    news_score=float(news_score),
                    contributing_signals=[sig.source],
                )
                candidate = CandidateDecision(
                    symbol=sig.symbol,
                    action=("enter_long" if sig.option_right and sig.option_right.value == "call"
                             else "enter_short"),
                    direction=("bullish" if (isinstance(sig.meta, dict)
                                and sig.meta.get("direction") == "bullish")
                                else "bearish"),
                    source=sig.source,
                    confidence=float(sig.confidence or 0.5),
                    rationale=sig.rationale[:120],
                )
                review = self.llm_brain.review(candidate, ctx_for_brain)
                # Audit every review so the operator can evaluate the
                # brain's impact over time (rules-only vs rules+brain).
                try:
                    log_emit(
                        source="llm_brain", symbol=sig.symbol,
                        emitted=(review.action != "veto"),
                        confidence=review.confidence_multiplier,
                        rationale=(review.reason or "")[:200],
                        meta={"action": review.action,
                              "latency_ms": review.latency_ms,
                              "from_cache": review.from_cache,
                              "model": review.model},
                    )
                except Exception:
                    pass
                if review.is_veto:
                    log.info("llm_brain_veto sym=%s reason=%s",
                              sig.symbol, review.reason[:80])
                    return
                # Soft mode: scale the signal's confidence
                old_conf = float(sig.confidence or 0.5)
                new_conf = max(0.0, min(1.0,
                                          old_conf * review.confidence_multiplier))
                if abs(new_conf - old_conf) > 0.001:
                    log.info("llm_brain_adjust sym=%s mult=%.2f %.2f->%.2f reason=%s",
                              sig.symbol, review.confidence_multiplier,
                              old_conf, new_conf, review.reason[:80])
                    sig.confidence = new_conf
            except Exception as e:
                log.warning("llm_brain_review_failed err=%s", e)

        # Pick the contract FIRST (before the filter chain) so filters
        # f11 (spread), f12 (open interest), f18 (scalp viability) can
        # inspect the actual contract. Previously we picked AFTER filters
        # ran, which meant contract-dependent filters were advisory-pass
        # for every trade.
        from .core.types import OptionContract, OptionRight
        from .data.options_chain import SyntheticOptionsChain
        meta_dir = sig.meta.get("direction") if isinstance(sig.meta, dict) else None
        if sig.option_right is not None:
            right = sig.option_right
        elif meta_dir == "bearish":
            right = OptionRight.PUT
        else:
            right = OptionRight.CALL

        # Weighted strategy-bucket allocation. Operator requested:
        # "I like the long strategy but also experiment with 0DTE, swing
        # so we have data." Each entry gets tagged with its bucket so
        # the nightly report can compare P&L per strategy.
        #
        # Default allocation (overridable via execution.strategy_buckets):
        #   0dte    -> DTE in [0, 1]         — same-day / overnight lottery
        #   short   -> DTE in [2, 5, 7]      — directional swing
        #   swing   -> DTE in [14, 21, 30]   — thesis has time to play out
        #
        # Runtime override honored via runtime_overrides.json key
        # 'strategy_bucket_weights' = {"0dte": 30, "short": 30, "swing": 40}.
        import random as _random
        buckets_cfg = (self.s.get("execution.strategy_buckets", None) or {
            "0dte":  {"dtes": [0, 1],          "weight": 20},
            "short": {"dtes": [2, 5, 7],       "weight": 30},
            "swing": {"dtes": [14, 21, 30],    "weight": 50},
        })
        try:
            from .core.runtime_overrides import get_override
            live_weights = get_override("strategy_bucket_weights", None)
        except Exception:
            live_weights = None
        if isinstance(live_weights, dict):
            for name, w in live_weights.items():
                if name in buckets_cfg:
                    buckets_cfg[name]["weight"] = int(w)
        # Weighted roll
        names = list(buckets_cfg.keys())
        weights = [max(0, int(buckets_cfg[n].get("weight", 0))) for n in names]
        if sum(weights) > 0:
            strategy_bucket = _random.choices(names, weights=weights, k=1)[0]
            dtes = buckets_cfg[strategy_bucket].get("dtes", [7])
            target_dte = int(_random.choice(dtes))
        else:
            # Fallback to legacy target_dte_pool / target_dte if buckets
            # are misconfigured (all weights 0).
            _dte_pool = self.s.get("execution.target_dte_pool", None) or []
            if _dte_pool:
                target_dte = int(_random.choice(_dte_pool))
            else:
                target_dte = int(self.s.get("execution.target_dte", 7))
            strategy_bucket = (
                "0dte" if target_dte == 0 else
                "short" if target_dte <= 7 else "swing"
            )
        chain = self.chain_provider.chain(sig.symbol, spot, target_dte=target_dte)
        min_oi = int(self.s.get("execution.min_open_interest", 100))
        min_tv = int(self.s.get("execution.min_today_option_volume", 50))
        contract = SyntheticOptionsChain.find_atm_liquid(
            chain, spot, right,
            min_oi=min_oi, min_today_volume=min_tv,
            max_strike_dist_pct=float(
                self.s.get("execution.max_strike_dist_pct", 0.05)
            ),
        )

        ectx = ExecutionContext(
            signal=sig, now=bars[-1].ts,
            account_equity=acct.equity, day_pnl=acct.day_pnl,
            open_positions_count=len(self.broker.positions()),
            current_bar_volume=bars[-1].volume,
            avg_bar_volume=sum(b.volume for b in bars[-20:]) / 20,
            opening_range_high=or_hi, opening_range_low=or_lo,
            spot=spot, vwap=vwap, vix=vix_now,
            is_etf=_is_etf(sig.symbol),
            econ_blackout=econ_blackout,
            news_score=news_score, news_label=news_label,
            news_rationale=news_rationale,
            recent_bars=bars[-20:],   # momentum filter needs last ~20 bars
            contract=contract,        # f11/f12/f18 need this
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
        # ---- Pick a real option from the chain -----------------------
        # Directional signal → map to CALL (bullish) or PUT (bearish).
        # For the ensemble path, `sig.side` is BUY for bullish entries
        # across the board (we always long the option, not short it).
        # The dominant_direction on the underlying decides right.
        # Contract already picked before the filter chain — skip duplicate pick.
        # Reject only when we have HARD evidence of an unusable contract.
        # Three concrete reject cases:
        #   (a) no contract, or zero bid/ask — stale / no market
        #   (b) partial OI/vol data that fails the floor when one side
        #       is non-zero (provider IS reporting, and it's genuinely low)
        #   (c) strike distance > max_strike_dist_pct — tier-3 fallback
        #       inside find_atm_liquid may return a far strike when
        #       neither liquid nor quote-only tiers found anything
        #       within the distance cap. For a tiny account this shows
        #       up as deep-ITM puts that cost $100+/share — Kelly can't
        #       afford them, everything sizes to zero, nothing trades.
        max_dist = float(self.s.get("execution.max_strike_dist_pct", 0.05))
        if contract is None or contract.bid <= 0 or contract.ask <= 0:
            log.info(
                "entry_skip_no_liquid_strike",
                symbol=sig.symbol, source=sig.source, right=right.value,
                picked_strike=(contract.strike if contract else None),
                picked_oi=(contract.open_interest if contract else None),
                picked_vol=(contract.today_volume if contract else None),
                reason="no_bid_ask",
            )
            return
        # Enforce distance cap universally (was only enforced inside the
        # liquid/quote-only tiers, not the tier-3 fallback).
        strike_dist = abs(contract.strike - spot) / max(spot, 1e-9)
        if strike_dist > max_dist:
            log.info(
                "entry_skip_strike_too_far",
                symbol=sig.symbol, source=sig.source, right=right.value,
                picked_strike=contract.strike, spot=round(spot, 2),
                dist_pct=round(strike_dist * 100, 2),
                max_pct=round(max_dist * 100, 2),
            )
            return
        partial = contract.open_interest > 0 or contract.today_volume > 0
        if partial and (contract.open_interest < min_oi
                         or contract.today_volume < min_tv):
            log.info(
                "entry_skip_no_liquid_strike",
                symbol=sig.symbol, source=sig.source, right=right.value,
                picked_strike=contract.strike,
                picked_oi=contract.open_interest,
                picked_vol=contract.today_volume,
                reason="below_min_liquidity",
            )
            return
        if not partial:
            log.info(
                "entry_liquidity_unknown",
                symbol=sig.symbol, picked_strike=contract.strike,
                bid=contract.bid, ask=contract.ask,
            )
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
        kelly_n = n
        # Apply vol scaling + drawdown guard multiplier AFTER Kelly sizing
        n = int(round(n * vol_mult * self._dd_size_multiplier))

        # Operator policy override: default 1 contract per entry per
        # ticker. Scale up to `max_qty_if_strong` only when ALL of the
        # scale-up conditions are met:
        #   - delta in tight sweet-spot [scalp_delta_scale_min, scale_max]
        #   - underlying 5-bar move >= scalp_scale_min_underlying_move
        #   - current bar volume >= scalp_scale_min_volume_multiple × avg
        # Otherwise: buy 1 contract, regardless of Kelly.
        # Runtime overrides — Discord autopanel "📦 Qty +N" buttons
        # write these live so the operator can scale up without editing
        # settings.yaml. Takes effect on the next entry.
        try:
            from .core.runtime_overrides import get_override
            default_qty = int(get_override(
                "default_qty_per_entry",
                self.s.get("execution.default_qty_per_entry", 1),
            ))
            max_strong = int(get_override(
                "max_qty_if_strong",
                self.s.get("execution.max_qty_if_strong", 3),
            ))
        except Exception:
            default_qty = int(self.s.get("execution.default_qty_per_entry", 1))
            max_strong = int(self.s.get("execution.max_qty_if_strong", 3))
        delta_lo = float(self.s.get("execution.scalp_delta_scale_min", 0.40))
        delta_hi = float(self.s.get("execution.scalp_delta_scale_max", 0.55))
        mv_min = float(self.s.get("execution.scalp_scale_min_underlying_move", 0.005))
        vol_mult_min = float(self.s.get("execution.scalp_scale_min_volume_multiple", 2.0))
        # Read delta stashed by f18 (abs value, 0.0 if IV unknown).
        delta_val = float(getattr(ectx, "contract_delta", 0.0) or 0.0)
        # 5-bar underlying move in signal direction:
        direction = (sig.meta.get("direction") or "").lower() if isinstance(sig.meta, dict) else ""
        move_pct = 0.0
        if len(bars) >= 5 and bars[-5].open > 0:
            move_pct = (bars[-1].close - bars[-5].open) / bars[-5].open
            if direction == "bearish":
                move_pct = -move_pct    # flip sign for bearish
        # Current bar volume multiple:
        avg_vol_20 = sum(b.volume for b in bars[-20:]) / 20 if len(bars) >= 20 else 0
        vol_ratio = (bars[-1].volume / avg_vol_20) if avg_vol_20 > 0 else 0
        strong_momentum = (
            delta_lo <= delta_val <= delta_hi
            and move_pct >= mv_min
            and vol_ratio >= vol_mult_min
        )
        if strong_momentum:
            n = min(n if n > 0 else default_qty, max_strong)
            qty_reason = (f"strong: delta={delta_val:.2f} "
                           f"move={move_pct:+.2%} vol={vol_ratio:.1f}x")
        else:
            n = default_qty
            qty_reason = (f"default_1: delta={delta_val:.2f} "
                           f"move={move_pct:+.2%} vol={vol_ratio:.1f}x")
        log.info("qty_policy", symbol=sig.symbol, qty=n, reason=qty_reason)

        if n <= 0:
            # Surface WHY. This is the silent-drop that was invisible: Kelly
            # returns 0 when the priors imply negative EV, vol_scale clips
            # near zero, or drawdown guard is fully engaged.
            log.info(
                "entry_skip_qty_zero",
                symbol=sig.symbol, source=sig.source,
                kelly_n=kelly_n, vol_mult=round(vol_mult, 3),
                dd_mult=round(self._dd_size_multiplier, 3),
                win_rate=self._win_rate, avg_win=self._avg_win,
                avg_loss=self._avg_loss,
            )
            return
        # Directional bet on the underlying is expressed as LONG options:
        # bullish → BUY CALL, bearish → BUY PUT. We don't sell premium here;
        # that's a separate strategy that would need explicit configuration.
        #
        # Limit pricing — don't just pay the ask. Starting at
        #   bid + 30%-of-spread
        # means we're still an aggressive buyer (more likely to fill
        # before market walks away) but save ~70% of the spread cost.
        # If the paper broker's fill simulator won't fill at this price,
        # we'll see it reject and can relax later. On real Alpaca paper
        # the simulator generally fills at-limit if the order sits, so
        # this works. Operator asked: "optimize bid/ask entry, don't
        # just enter at full ask."
        spread = max(0.0, contract.ask - contract.bid)
        # Runtime override honored — Discord !autopanel → Smarter Bids
        # can tighten this live without a restart. Lower = closer to
        # bid (better price, worse fill odds); higher = closer to ask.
        try:
            from .core.runtime_overrides import get_override
            aggressiveness = float(get_override(
                "entry_spread_pct",
                self.s.get("execution.entry_spread_pct", 0.15),
            ))
        except Exception:
            aggressiveness = float(self.s.get("execution.entry_spread_pct", 0.15))
        limit = round(contract.bid + spread * aggressiveness, 2)
        if limit <= 0:
            limit = round(contract.ask * 0.98, 2)   # fallback if bid=0
        # Rich entry rationale — one-line JSON-ish string captured in
        # the order tag (and thus the trade journal). Gives the
        # operator full "why did the bot do this" audit per trade for
        # model-tuning later. Stored compactly to fit within the tag
        # column; dashboard parses key=value pairs back out.
        rationale = (
            f"entry|src={sig.source}|sym={sig.symbol}|right={contract.right.value}"
            f"|strike={contract.strike}|dte={target_dte}|delta={delta_val:.2f}"
            f"|iv={contract.iv:.2f}|regime={regime or '?'}"
            f"|sig_score={round(float(sig.confidence or 0), 3)}"
            f"|5bar_move={move_pct:+.3%}|vol_ratio={vol_ratio:.1f}x"
            f"|spot={spot:.2f}|vwap={vwap:.2f}"
            f"|bid={contract.bid:.2f}|ask={contract.ask:.2f}"
            f"|qty_policy={qty_reason}|limit={limit:.2f}"
            f"|strategy={strategy_bucket}"
        )
        order = Order(symbol=contract.symbol, side=Side.BUY, qty=n,
                      is_option=True, limit_price=limit,
                      tag=rationale)
        v = self.validator.validate(order, contract, acct.buying_power,
                                    self.s.get("account.max_open_positions", 5))
        if not v.ok:
            log.info("order_reject", reason=v.reason, symbol=sig.symbol)
            return
        # Push the current quote + VIX into the slippage model so the fill
        # cost reflects real microstructure, not a fixed constant.
        self.broker.update_market_context(
            contract.symbol,
            MarketContext(
                bid=contract.bid, ask=contract.ask,
                bid_size=1000, ask_size=1000, vix=vix_now,
                recent_spread_pct=(contract.ask - contract.bid) / max(contract.ask, 0.01),
            ),
        )
        # Auto PT/SL at entry — CLAUDE.md hard rule. We price on the option,
        # not the underlying. Short-DTE → tighter targets (matches the
        # exit-engine defaults from config/settings.yaml).
        dte = (contract.expiry - date.today()).days if contract.expiry else 9999
        is_short_dte = dte <= 1
        pt_pct = float(self.s.get(
            "exits.profit_target_short_dte_pct" if is_short_dte
            else "exits.profit_target_multi_dte_pct",
            0.35 if is_short_dte else 0.50,
        ))
        sl_pct = float(self.s.get(
            "exits.stop_loss_short_dte_pct" if is_short_dte
            else "exits.stop_loss_multi_dte_pct",
            0.20 if is_short_dte else 0.30,
        ))
        entry_px = limit  # the order's limit; fill may differ slightly
        auto_pt = round(entry_px * (1 + pt_pct), 4)
        auto_sl = round(max(0.01, entry_px * (1 - sl_pct)), 4)
        fill = self.broker.submit(
            v.adjusted_order or order,
            contract=contract,
            auto_profit_target=auto_pt,
            auto_stop_loss=auto_sl,
        )
        if fill:
            log.info("fill", symbol=contract.symbol, underlying=sig.symbol,
                      right=contract.right.value, strike=contract.strike,
                      qty=n, price=fill.price, src=sig.source,
                      auto_pt=auto_pt, auto_sl=auto_sl)
            # Human-readable description at the top of the embed —
            # explains the trade in one paragraph before the structured
            # field breakdown. This is what the operator reads first.
            direction_word = "bullish" if contract.right.value == "call" else "bearish"
            side_word = contract.right.value.upper()
            description = (
                f"**Thesis**: {direction_word} on **{sig.symbol}** over the "
                f"next {target_dte} day{'s' if target_dte != 1 else ''}.  Bought "
                f"**{n}× {side_word}** at ${contract.strike} strike for "
                f"**${fill.price:.2f}/share** (total cost "
                f"**${n * fill.price * 100:,.2f}**).\n\n"
                f"**Why now**: {sig.source} signal fired at "
                f"{float(sig.confidence or 0):.2f} confidence in "
                f"`{regime or '?'}` regime.  Underlying moved "
                f"**{move_pct:+.2%}** over the last 5 bars with volume at "
                f"**{vol_ratio:.1f}× avg**.  Delta **{delta_val:.2f}** means "
                f"the option moves about **${delta_val*100:.0f}** for every "
                f"$1 move in {sig.symbol}.\n\n"
                f"**Exits auto-set**: profit target **${auto_pt}**, "
                f"stop loss **${auto_sl}**.  Scale-out locks 50% at PT, "
                f"the rest rides a trailing stop."
            )
            self.notifier.notify(
                description,
                title=f"🟢 ENTRY — {sig.symbol} {side_word} {contract.strike} ({target_dte}d)",
                level="success",
                meta={
                    # --- Position (row 1) -------------------------------
                    "📊 Symbol": sig.symbol,
                    "📊 Contract": f"{side_word} {contract.strike} · {target_dte}d",
                    "📊 Qty × Price": f"{n} × ${fill.price:.2f} = ${n*fill.price*100:,.0f}",
                    # --- Greeks (row 2) ---------------------------------
                    "🎯 Delta": (
                        f"{delta_val:.2f} (option moves "
                        f"≈${delta_val*100:.0f}/$1 stock move)"
                        if delta_val else "n/a"
                    ),
                    "🎯 IV": (
                        f"{contract.iv:.1%} annualized" if contract.iv else "n/a"
                    ),
                    "🎯 Bid/Ask": f"${contract.bid:.2f} / ${contract.ask:.2f}",
                    # --- Momentum (row 3) -------------------------------
                    "📈 5-bar Move": (
                        f"{move_pct:+.2%} in {direction_word} direction"
                    ),
                    "📈 Volume": f"{vol_ratio:.1f}× 20-bar average",
                    "📈 Regime": str(regime or "?"),
                    # --- Price context (row 4) --------------------------
                    "📍 Spot": f"${spot:.2f}",
                    "📍 VWAP": f"${vwap:.2f} ({((spot-vwap)/vwap*100):+.2f}% vs spot)",
                    "📍 Signal Score": f"{float(sig.confidence or 0):.2f}",
                    # --- Risk controls (row 5) --------------------------
                    "🛑 Profit Target": (
                        f"${auto_pt} "
                        f"(+{((auto_pt-fill.price)/fill.price*100):.0f}%)"
                    ),
                    "🛑 Stop Loss": (
                        f"${auto_sl} "
                        f"({((auto_sl-fill.price)/fill.price*100):.0f}%)"
                    ),
                    "🛑 Qty Policy": qty_reason.split(":")[0],  # "default_1" / "strong"
                    "_footer": (
                        f"{contract.symbol} · src={sig.source} · "
                        f"expires {contract.expiry.isoformat() if contract.expiry else '?'}"
                    ),
                },
            )

    def run(self) -> None:
        if self.s.live_trading:
            log.error("live_trading_blocked_in_main_of_default_build — remove guard explicitly")
            return
        configure_logging("INFO")

        # Startup ping to the notifier. Double-purpose:
        # (1) confirms the Discord/Slack webhook is actually wired (if
        #     the user never sees this message, their URL is wrong or
        #     env var didn't load).
        # (2) telemetry: each start is a restart event worth seeing.
        try:
            mode = "LIVE" if self.s.live_trading else "paper"
            if bool(self.s.get("notifier.startup_notify", False)):
                self.notifier.notify(
                    f"tradebot started — {mode}, broker={self.s.get('broker.name','paper')}",
                    level="info", title="startup",
                )
        except Exception:
            pass

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
            # Shutdown flatten — preserve positions across restart unless
            # explicitly enabled. Paper broker state is snapshotted to
            # broker_state.json and restored on next startup, so we DON'T
            # need to close positions just because the process exits.
            # Flattening on every restart was causing:
            #   1. Trades tagged 'eod_force_close' scattered through session
            #   2. Local ↔ Tradier desync (local closed, Tradier still open)
            #   3. Losses on spread cross for positions that would have been
            #      fine after restart
            # Set `broker.flatten_on_shutdown: true` ONLY for live-trading
            # where you want to end the session flat regardless.
            try:
                if bool(self.s.get("broker.flatten_on_shutdown", False)):
                    open_now = list(self.broker.positions())
                    if open_now:
                        log.warning("shutdown_flatten", n=len(open_now))
                        mark = self._build_mark_prices(open_now)
                        self.broker.flatten_all(mark_prices=mark,
                                                 tag="shutdown_flatten")
                else:
                    open_now = list(self.broker.positions())
                    if open_now:
                        log.info("shutdown_preserve_positions",
                                 n=len(open_now),
                                 note="positions saved via snapshot; "
                                       "will restore on next startup")
            except Exception as e:                    # noqa: BLE001
                log.error("shutdown_handler_error", err=str(e))


def main() -> None:
    s = load_settings()
    bot = TradeBot(s)
    bot.run()


if __name__ == "__main__":
    main()
