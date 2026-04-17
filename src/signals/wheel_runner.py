"""Wheel runner — bypasses the ensemble and directional path entirely.

WHY a separate runner: the wheel is a premium-harvesting strategy, not a
directional bet. It has nothing to do with "is the market going up?"
signals like momentum or ORB. It wants to SELL a cash-secured put when:
  - Option has enough premium to make selling worth it
  - Underlying is liquid (SPY/QQQ only for MVP)
  - No existing wheel position on that symbol
  - Cash reserve is sufficient to cover assignment

Mixing it into the ensemble would force the wheel to compete with
momentum/ORB signals that trade on a completely different thesis.
Cleaner to run it as its own scheduled sweep.

This runner is called once per main-loop tick when strategy_mode=wheel.
It walks the wheel universe, picks a target put per symbol, and submits
a SELL-to-open order if all conditions are met.

Exit logic is in `src/exits/wheel_exits.py` — it fires from the
main_loop and fast_loop just like the directional exit engine, but uses
premium-based P&L math instead of directional price math.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

from ..core.types import Order, OptionContract, OptionRight, Side, Position
from ..data.options_chain import SyntheticOptionsChain


_log = logging.getLogger(__name__)


@dataclass
class WheelRunnerConfig:
    universe: List[str]
    target_dte: int = 35
    # Delta target — 30-delta put is ~3% OTM. We approximate by strike
    # distance since we don't have a greeks API in the options chain.
    target_delta: float = 0.30
    # Minimum premium required (as % of spot) to bother selling.
    # 0.40% of $700 SPY = $2.80 min premium. Below that the trade is
    # just transaction-cost collection.
    min_premium_pct: float = 0.004
    max_open_positions: int = 2
    # Cash reserve — always keep this fraction of account cash free
    # so a sudden assignment doesn't drain margin.
    min_cash_reserve_pct: float = 0.10


class WheelRunner:
    """Sweeps the wheel universe on each main-loop tick, submits CSPs."""

    def __init__(self, cfg: WheelRunnerConfig, bot) -> None:
        self.cfg = cfg
        self.bot = bot          # ref to TradeBot (broker, journal, notifier)

    def tick(self) -> None:
        """One pass over the wheel universe. Idempotent: skips symbols
        where we already hold a short put."""
        acct = self.bot.broker.account()
        existing = {p.underlying for p in self.bot.broker.positions()
                      if p.qty < 0 and p.is_option}
        open_count = sum(1 for p in self.bot.broker.positions()
                          if p.qty < 0 and p.is_option)
        if open_count >= self.cfg.max_open_positions:
            return
        for symbol in self.cfg.universe:
            if symbol in existing:
                continue   # already short a put on this name
            if open_count >= self.cfg.max_open_positions:
                break
            if self._try_enter_one(symbol, acct):
                open_count += 1

    def _try_enter_one(self, symbol: str, acct) -> bool:
        """Attempt to open one short put on `symbol`. Returns True if
        a fill happened. Swallows + logs all non-fatal errors."""
        try:
            spot = self.bot.data.latest_price(symbol)
            if not spot or spot <= 0:
                bars = self.bot.data.get_bars(symbol, limit=3)
                if bars:
                    spot = bars[-1].close
            if not spot or spot <= 0:
                return False
            chain = self.bot.chain_provider.chain(
                symbol, float(spot), target_dte=self.cfg.target_dte,
            )
            pick = self._pick_put(chain, float(spot))
            if pick is None:
                return False
            mid = pick.mid if pick.mid > 0 else (pick.bid + pick.ask) / 2
            if mid <= 0:
                return False
            prem_pct = mid / spot
            if prem_pct < self.cfg.min_premium_pct:
                _log.info(
                    "wheel_skip_low_premium symbol=%s strike=%s prem=$%.2f (%.3f%% < %.3f%%)",
                    symbol, pick.strike, mid, prem_pct * 100,
                    self.cfg.min_premium_pct * 100,
                )
                return False
            # Cash-secured sizing: strike × 100 is the collateral required
            # per contract. Stay under our reserved-cash rule.
            required_per_contract = pick.strike * 100.0
            reserve = acct.equity * self.cfg.min_cash_reserve_pct
            deployable = max(0.0, acct.cash - reserve)
            max_contracts = int(deployable // required_per_contract)
            if max_contracts < 1:
                _log.info(
                    "wheel_skip_insufficient_cash symbol=%s need=$%.0f "
                    "deployable=$%.0f",
                    symbol, required_per_contract, deployable,
                )
                return False
            # MVP: always 1 contract per entry. Simpler to paper-test.
            qty = 1
            # Sell-to-open at mid + small concession to bid (we want to
            # actually fill, not sit on the order).
            spread = max(0.0, pick.ask - pick.bid)
            limit = round(max(0.01, pick.bid + 0.30 * spread), 2)
            order = Order(
                symbol=pick.symbol, side=Side.SELL, qty=qty,
                is_option=True, limit_price=limit,
                tif="DAY",
                tag=f"wheel_csp:{symbol}",
            )
            # Validate + submit. Wheel bypasses the 14-filter directional
            # chain because those filters were built for long-option
            # entries. Wheel has its own sanity checks above.
            # Entry-side PT/SL are computed on the PREMIUM we collect,
            # not on the option price:
            #   PT: close at 50% of premium captured
            #   SL: close if option doubles (lose 1× premium)
            credit_per_contract = mid * 100  # what we receive
            self.bot.broker.update_market_context(
                pick.symbol,
                _market_context(pick, self.bot.vix_probe.value()),
            )
            fill = self.bot.broker.submit(
                order,
                contract=pick,
                auto_profit_target=round(limit * 0.50, 4),   # option price halving
                auto_stop_loss=round(limit * 2.00, 4),       # option price doubling
            )
            if fill is None:
                return False
            # Discord + log. Use notifier meta so the entry card shows
            # the credit captured (the whole point of the strategy).
            dte = (pick.expiry - date.today()).days if pick.expiry else None
            _log.info(
                "wheel_csp_fill symbol=%s strike=%s expiry=%s dte=%s "
                "credit=$%.2f",
                symbol, pick.strike, pick.expiry, dte, credit_per_contract,
            )
            self.bot.notifier.notify(
                f"SELL-TO-OPEN {qty} × PUT {symbol} {pick.strike} "
                f"@ ${fill.price:.2f} → collected ${credit_per_contract:.0f} "
                f"credit ({dte}d)",
                title="entry", level="success",
                meta={
                    "strategy": "wheel-csp",
                    "symbol": symbol,
                    "strike": pick.strike,
                    "expiry": pick.expiry.isoformat() if pick.expiry else "—",
                    "dte": dte,
                    "qty": qty,
                    "sell_px": round(float(fill.price), 4),
                    "credit_usd": f"${credit_per_contract:.2f}",
                    "cash_secured": f"${required_per_contract:.0f}",
                    "PT_close_at": f"${round(limit*0.50, 2)} (50% capture)",
                    "SL_close_at": f"${round(limit*2.00, 2)} (1× premium loss)",
                    "_footer": f"OCC {pick.symbol}",
                },
            )
            return True
        except Exception as e:                               # noqa: BLE001
            _log.warning("wheel_try_enter_failed symbol=%s err=%s", symbol, e)
            return False

    def _pick_put(self, chain, spot: float) -> Optional[OptionContract]:
        """Pick the put closest to the target delta (approximated by
        strike distance: 30-delta ≈ 3% OTM)."""
        target_strike = spot * (1 - self.cfg.target_delta * 0.10)  # 0.30 delta ≈ 3% OTM
        puts = [c for c in chain if c.right == OptionRight.PUT
                 and c.strike < spot       # must be OTM
                 and c.bid > 0 and c.ask > 0]
        if not puts:
            return None
        puts.sort(key=lambda c: abs(c.strike - target_strike))
        return puts[0]


def _market_context(contract, vix):
    """Build a MarketContext for the slippage model from the chain row."""
    from ..brokers.slippage_model import MarketContext
    return MarketContext(
        bid=contract.bid, ask=contract.ask,
        bid_size=1000, ask_size=1000, vix=vix,
        recent_spread_pct=(contract.ask - contract.bid) / max(contract.ask, 0.01),
    )
