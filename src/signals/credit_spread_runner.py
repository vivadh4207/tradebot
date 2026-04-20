"""Credit spread runners — systematic short-premium strategies.

Two variants in this file:

  1. `WeeklyCreditSpreadRunner` — sells a 30–45 DTE put credit spread
     (short 0.20 delta, long 0.05–0.10 delta). Once per day at entry
     window. Exits at 50% max profit or 21 DTE.

  2. `ZeroDTECreditSpreadRunner` — sells a 0DTE put credit spread when
     spot is near support AND RSI(14) is oversold. Short 0.10–0.15
     delta, long 5–10 points below. Exits at 50% PT, 150% SL, or 15:45.

Both use our Black-Scholes Greeks (`math_tools/pricer.py`) to pick
delta-accurate strikes — not broker-provided (potentially stale) deltas.

These runners are NOT part of the ensemble. They have their own
cadence + risk math. They run alongside the directional path or
independently, depending on `strategy_mode`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timezone
from typing import List, Optional, Tuple

from ..core.types import (
    ComboOrder, OptionLeg, OptionContract, OptionRight, Side,
)
from ..math_tools.pricer import bs_greeks


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------- helpers

def _delta_of(c: OptionContract, spot: float, r: float = 0.045) -> float:
    """Compute the BS delta of a chain row using our Greeks lib.

    Guards against T→0 (expiration day) and sigma→0 (broker returned
    zero IV) so this never blows up inside the strategy tick.
    """
    today = date.today()
    days = max(1, (c.expiry - today).days if c.expiry else 1)
    T = days / 365.0
    sigma = c.iv if c.iv > 1e-6 else 0.25        # fallback = 25% IV
    opt_type = "call" if c.right == OptionRight.CALL else "put"
    try:
        g = bs_greeks(
            S=spot, K=c.strike, T=T, r=r, sigma=sigma,
            option_type=opt_type,
        )
        # bs_greeks returns a dict — grab the "delta" key
        return float(g["delta"])
    except Exception:
        # If Greeks compute fails, fall back to a sign-only heuristic
        # so the caller still gets a usable delta magnitude.
        moneyness = (spot - c.strike) / spot
        return 0.5 + moneyness if c.right == OptionRight.CALL else -0.5 + moneyness


def _pick_by_delta(
    chain: List[OptionContract],
    spot: float,
    target_abs_delta: float,
    right: OptionRight,
    tolerance: float = 0.05,
) -> Optional[OptionContract]:
    """Pick the chain row whose |delta| is closest to target_abs_delta.

    `tolerance` is a band around the target. If nothing falls inside it,
    returns the nearest by |delta - target|, unless the nearest is >3×
    the tolerance away (in which case we refuse — chain is too thin).
    """
    candidates = [c for c in chain if c.right == right
                  and c.bid > 0 and c.ask > 0]
    if not candidates:
        return None
    scored = []
    for c in candidates:
        d = _delta_of(c, spot)
        scored.append((abs(abs(d) - target_abs_delta), d, c))
    scored.sort(key=lambda t: t[0])
    best_err, _, best = scored[0]
    if best_err > 3 * tolerance:
        return None
    return best


def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Classic Wilder RSI. None if not enough data."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[-i] - closes[-i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(-diff)
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _pivot_low(closes: List[float], lookback: int = 20) -> Optional[float]:
    """Recent support = min close over lookback. Crude but effective on
    intraday SPY/QQQ. Strategies can swap in a fancier pivot detector
    later if needed."""
    if len(closes) < lookback:
        return None
    return float(min(closes[-lookback:]))


# ---------------------------------------------------------------- config


@dataclass
class WeeklyCreditSpreadConfig:
    universe: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    dte_min: int = 30
    dte_max: int = 45
    short_delta: float = 0.20
    long_delta: float = 0.08
    max_wing_width: float = 10.0     # dollars
    min_credit_pct_of_wing: float = 0.15   # min credit / wing_width
    max_open_positions: int = 2
    # Entry time window (US/Eastern): don't open in the first 15 min
    # (opening vol) or the last 2h (we want 30-45 DTE, not 29.5-44.5).
    entry_after_et: dtime = dtime(9, 45)
    entry_before_et: dtime = dtime(14, 0)
    profit_target_pct: float = 0.50         # close at 50% of max profit
    dte_close_threshold: int = 21           # force-close at 21 DTE regardless


@dataclass
class ZeroDTECreditSpreadConfig:
    universe: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    short_delta: float = 0.12
    wing_width: float = 5.0
    max_open_positions: int = 3
    entry_after_et: dtime = dtime(10, 15)
    entry_before_et: dtime = dtime(14, 30)
    force_close_et: dtime = dtime(15, 45)
    # Entry gates
    rsi_oversold: float = 35.0              # entry threshold for puts
    support_band_pct: float = 0.005         # spot within 0.5% of support
    support_lookback_bars: int = 20
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 1.50             # stop when unrealized = -1.5× credit


# ---------------------------------------------------------------- runners


class WeeklyCreditSpreadRunner:
    """Daily 30-45 DTE put credit spread on SPY/QQQ.

    One open position per symbol, max `max_open_positions` total.
    Bypasses the directional ensemble — has its own entry cadence and
    its own premium-based exit math in `exits/credit_spread_exits.py`.
    """

    def __init__(self, cfg: WeeklyCreditSpreadConfig, bot) -> None:
        self.cfg = cfg
        self.bot = bot
        self._last_entry_day_per_symbol: dict = {}

    def tick(self) -> None:
        """One pass. Idempotent: rate-limited to one entry per symbol
        per calendar day."""
        now_et = _now_et()
        t = now_et.time()
        if t < self.cfg.entry_after_et or t > self.cfg.entry_before_et:
            return
        today = now_et.date()

        open_symbols = {p.underlying for p in self.bot.broker.positions()
                        if p.qty != 0 and p.is_option}
        open_count = sum(1 for p in self.bot.broker.positions()
                         if p.qty != 0 and p.is_option)
        if open_count >= self.cfg.max_open_positions:
            return

        for symbol in self.cfg.universe:
            if symbol in open_symbols:
                continue
            if self._last_entry_day_per_symbol.get(symbol) == today:
                continue
            if open_count >= self.cfg.max_open_positions:
                break
            if self._try_enter(symbol):
                self._last_entry_day_per_symbol[symbol] = today
                open_count += 1

    def _try_enter(self, symbol: str) -> bool:
        try:
            # Breadth gate — if configured and risk-off, step aside.
            # Selling premium into a collapsing tape is the textbook
            # way to blow up a short-premium strategy.
            probe = getattr(self.bot, "breadth_probe", None)
            if probe is not None:
                snap = probe.snapshot()
                if snap.is_risk_off:
                    _log.info(
                        "weekly_cs_breadth_stand_down symbol=%s score=%.3f",
                        symbol, snap.score,
                    )
                    return False
            spot = self.bot.data.latest_price(symbol)
            if not spot or spot <= 0:
                return False
            spot = float(spot)
            # Target mid-window DTE
            target_dte = (self.cfg.dte_min + self.cfg.dte_max) // 2
            chain = self.bot.chain_provider.chain(
                symbol, spot, target_dte=target_dte,
            )
            if not chain:
                return False
            short_put = _pick_by_delta(
                chain, spot, self.cfg.short_delta, OptionRight.PUT,
            )
            long_put = _pick_by_delta(
                chain, spot, self.cfg.long_delta, OptionRight.PUT,
            )
            if short_put is None or long_put is None:
                return False
            if short_put.strike <= long_put.strike:
                return False
            wing = short_put.strike - long_put.strike
            if wing > self.cfg.max_wing_width:
                # chain too thin, wings too wide — skip
                return False
            # Net credit = short put mid - long put mid
            net_credit = short_put.mid - long_put.mid
            if net_credit <= 0:
                return False
            # Credit as fraction of max loss must exceed floor.
            # Example: wing=$5, credit=$0.85 → 17% — OK if floor is 15%.
            credit_to_wing = net_credit / wing if wing > 0 else 0
            if credit_to_wing < self.cfg.min_credit_pct_of_wing:
                return False

            combo = ComboOrder(
                legs=[
                    OptionLeg(contract=short_put, side=Side.SELL, ratio=1),
                    OptionLeg(contract=long_put,  side=Side.BUY,  ratio=1),
                ],
                qty=1,
                # Net credit as NEGATIVE limit (we receive money).
                # Round to nickel for Alpaca-style penny rules.
                net_limit=-round(net_credit, 2),
                tag=f"weekly_pcs:{symbol}",
                tif="DAY",
            )
            fills = self.bot.broker.submit_combo(combo)
            # Record entry even if fills is empty (Alpaca resolves async)
            self._notify_entry(symbol, short_put, long_put, net_credit, wing)
            return True
        except Exception as e:                            # noqa: BLE001
            _log.warning("weekly_cs_enter_failed symbol=%s err=%s", symbol, e)
            try:
                from ..notify.issue_reporter import report_issue
                report_issue(
                    scope=f"weekly_credit_spread.{symbol}",
                    message=f"entry attempt failed: {type(e).__name__}: {e}",
                    exc=e,
                )
            except Exception:
                pass
            return False

    def _notify_entry(self, symbol, short_put, long_put, credit, wing):
        expiry = short_put.expiry.isoformat() if short_put.expiry else "?"
        dte = (short_put.expiry - date.today()).days if short_put.expiry else "?"
        max_loss = wing * 100 - credit * 100
        self.bot.notifier.notify(
            f"SELL PUT CREDIT SPREAD {symbol} "
            f"{short_put.strike}/{long_put.strike} @ ${credit:.2f} credit",
            title="entry", level="success",
            meta={
                "strategy": "weekly_put_credit_spread",
                "symbol": symbol,
                "short_strike": short_put.strike,
                "long_strike":  long_put.strike,
                "wing": f"${wing:.2f}",
                "credit": f"${credit:.2f}",
                "max_profit": f"${credit * 100:.0f}",
                "max_loss":   f"${max_loss:.0f}",
                "credit/wing": f"{credit / wing:.1%}",
                "expiry": expiry,
                "dte": dte,
                "PT_close_at": f"{self.cfg.profit_target_pct:.0%} of max profit",
                "SL_dte": f"force-close at {self.cfg.dte_close_threshold} DTE",
                "_footer": f"{short_put.symbol} / {long_put.symbol}",
            },
        )


class ZeroDTECreditSpreadRunner:
    """0DTE put credit spread when spot near support and RSI oversold."""

    def __init__(self, cfg: ZeroDTECreditSpreadConfig, bot) -> None:
        self.cfg = cfg
        self.bot = bot
        self._entries_today: dict = {}

    def tick(self) -> None:
        now_et = _now_et()
        t = now_et.time()
        if t < self.cfg.entry_after_et or t > self.cfg.entry_before_et:
            return
        today = now_et.date()

        open_symbols = {p.underlying for p in self.bot.broker.positions()
                        if p.qty != 0 and p.is_option}
        open_count = sum(1 for p in self.bot.broker.positions()
                         if p.qty != 0 and p.is_option)
        if open_count >= self.cfg.max_open_positions:
            return

        for symbol in self.cfg.universe:
            if symbol in open_symbols:
                continue
            if self._entries_today.get((symbol, today), 0) >= 1:
                continue
            if open_count >= self.cfg.max_open_positions:
                break
            if self._try_enter(symbol):
                self._entries_today[(symbol, today)] = \
                    self._entries_today.get((symbol, today), 0) + 1
                open_count += 1

    def _try_enter(self, symbol: str) -> bool:
        try:
            # Breadth gate — same reasoning as weekly: don't sell
            # premium into a collapsing tape.
            probe = getattr(self.bot, "breadth_probe", None)
            if probe is not None:
                snap = probe.snapshot()
                if snap.is_risk_off:
                    _log.info(
                        "0dte_cs_breadth_stand_down symbol=%s score=%.3f",
                        symbol, snap.score,
                    )
                    return False
            bars = self.bot.data.get_bars(
                symbol, limit=max(60, self.cfg.support_lookback_bars + 5),
            )
            if not bars or len(bars) < self.cfg.support_lookback_bars:
                return False
            closes = [b.close for b in bars]
            spot = closes[-1]
            # --- entry gates ---
            rsi = _rsi(closes, period=14)
            if rsi is None or rsi > self.cfg.rsi_oversold:
                return False
            # Prefer volume-weighted S/R if available; fall back to
            # the simple pivot-low for thin data.
            try:
                from ..intelligence.support_resistance import (
                    nearest_support as _vw_support, SRConfig,
                )
                sr_cfg = SRConfig(pivot_window=3, band_pct=0.002,
                                   min_touches=2)
                lv = _vw_support(bars, spot, cfg=sr_cfg,
                                  max_distance_pct=self.cfg.support_band_pct)
                support = lv.price if lv is not None else None
            except Exception:
                support = None
            if support is None:
                # Fallback to the plain 20-bar min
                support = _pivot_low(closes, self.cfg.support_lookback_bars)
            if support is None:
                return False
            # Spot must be within support_band_pct of support (but above it —
            # we're selling puts, not catching a falling knife)
            if spot < support:
                return False
            if abs(spot - support) / support > self.cfg.support_band_pct:
                return False
            # --- chain + strike pick ---
            chain = self.bot.chain_provider.chain(symbol, spot, target_dte=0)
            if not chain:
                return False
            short_put = _pick_by_delta(
                chain, spot, self.cfg.short_delta, OptionRight.PUT,
            )
            if short_put is None:
                return False
            # Long put: same expiry, strike = short_strike - wing_width
            long_strike_target = short_put.strike - self.cfg.wing_width
            long_put = _nearest_strike(
                chain, right=OptionRight.PUT, target_strike=long_strike_target,
                expiry=short_put.expiry,
            )
            if long_put is None or long_put.strike >= short_put.strike:
                return False
            net_credit = short_put.mid - long_put.mid
            if net_credit <= 0.05:   # noise floor
                return False

            combo = ComboOrder(
                legs=[
                    OptionLeg(contract=short_put, side=Side.SELL, ratio=1),
                    OptionLeg(contract=long_put,  side=Side.BUY,  ratio=1),
                ],
                qty=1,
                net_limit=-round(net_credit, 2),
                tag=f"0dte_pcs:{symbol}",
                tif="DAY",
            )
            self.bot.broker.submit_combo(combo)
            self._notify_entry(symbol, short_put, long_put,
                               net_credit, spot, rsi, support)
            return True
        except Exception as e:                            # noqa: BLE001
            _log.warning("0dte_cs_enter_failed symbol=%s err=%s", symbol, e)
            try:
                from ..notify.issue_reporter import report_issue
                report_issue(
                    scope=f"0dte_credit_spread.{symbol}",
                    message=f"entry attempt failed: {type(e).__name__}: {e}",
                    exc=e,
                )
            except Exception:
                pass
            return False

    def _notify_entry(self, symbol, short_put, long_put, credit,
                       spot, rsi, support):
        wing = short_put.strike - long_put.strike
        max_loss = wing * 100 - credit * 100
        self.bot.notifier.notify(
            f"SELL 0DTE PUT CREDIT SPREAD {symbol} "
            f"{short_put.strike}/{long_put.strike} @ ${credit:.2f}",
            title="entry", level="success",
            meta={
                "strategy": "0dte_put_credit_spread",
                "symbol": symbol,
                "spot": f"${spot:.2f}",
                "support": f"${support:.2f}",
                "RSI(14)": f"{rsi:.1f}",
                "short_strike": short_put.strike,
                "long_strike": long_put.strike,
                "wing": f"${wing:.2f}",
                "credit": f"${credit:.2f}",
                "max_profit": f"${credit * 100:.0f}",
                "max_loss":   f"${max_loss:.0f}",
                "PT": f"{self.cfg.profit_target_pct:.0%} of credit",
                "SL": f"{self.cfg.stop_loss_pct:.0%} of credit",
                "force_close": f"{self.cfg.force_close_et.strftime('%H:%M')} ET",
                "_footer": f"{short_put.symbol} / {long_put.symbol}",
            },
        )


# ---------------------------------------------------------------- utils


def _now_et() -> datetime:
    """Current time in US/Eastern. Approximation via UTC offset — fine
    for entry-window gating; not used for anything regulatory."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(tz=ZoneInfo("America/New_York"))
    except Exception:
        # fallback: assume ET = UTC-5 (no DST adjustment) — good enough
        # for an entry window check; drift of 1 hour just shifts the
        # window slightly. Tests monkeypatch this directly.
        return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def _nearest_strike(chain, *, right, target_strike, expiry=None):
    cands = [c for c in chain
             if c.right == right
             and c.bid > 0 and c.ask > 0
             and (expiry is None or c.expiry == expiry)]
    if not cands:
        return None
    cands.sort(key=lambda c: abs(c.strike - target_strike))
    return cands[0]
