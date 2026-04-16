# tradebot

A modular, paper-trading-first options/equities trading framework for US markets.

## Honest expectations

- **No guarantees.** No system delivers "guaranteed profit", 90-95% win rates, or 20-30% returns in 5 minutes. Anything promising that is a scam. This bot is a serious framework; whether it has edge is an empirical question answered by the backtest and paper-trading results.
- **Paper first, always.** Live trading is disabled by default. You must explicitly flip a config flag and point at a real broker after 30-90 days of paper results that show positive risk-adjusted returns net of fees and slippage.
- **Robinhood is not supported.** Robinhood has no official trading API and automated trading violates their ToS. We use Alpaca (official, commission-free, built-in paper trading). Other broker adapters (IBKR, Tradier) can be added behind the `BrokerAdapter` interface.
- **Retail-scale, not Citadel-scale.** This runs on a laptop, trades liquid symbols, and uses broker-grade latency. That's fine. It is not HFT and cannot pretend to be.

## Architecture

```
MarketData → QuoteFilter → PricingEngine (BS, Greeks, IV solver)
                            │
Intelligence (VIX, breadth, gamma, news, econ calendar)
                            │
Signal stack (momentum, ORB, VWAP, VRP, master composite)
                            │
14-filter execution chain → Position sizer (Hybrid Kelly+VIX)
                            │
Order validator → Broker adapter (Alpaca / Paper)
                            │
6-layer exit engine + 5-sec fast exit thread
```

## Layout

```
src/
  core/            # types, clock, logger
  data/            # universe, bars, options chain
  math_tools/      # BS, Greeks, IV solver, SVI, HAR-RV, calculator
  brokers/         # base adapter, Alpaca, paper sim
  signals/         # momentum, ORB, VWAP, VRP, master stack, claude_ai
  intelligence/    # VIX, breadth, gamma/GEX, news, econ calendar, MI+Edge
  risk/            # 14-filter chain, position sizer, order validator, portfolio risk
  exits/           # 6-layer engine, fast exit, tagged profiles, momentum boost
  backtest/        # replay, fill sim, metrics, walk-forward
scripts/
  run_paper.py
  run_backtest.py
tests/
config/settings.yaml
.env.example
```

## Quickstart (paper)

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY
python scripts/run_backtest.py       # synthetic-data dry run
python scripts/run_paper.py          # live market data, paper fills
```

## The hard rules (non-negotiable)

1. `LIVE_TRADING=false` by default; must be explicitly set to `true`, and only for a broker adapter that passed 30+ days of paper.
2. Daily max loss: 2% of account. Breaches halt new entries for the rest of the day.
3. Max open positions: 5.
4. Session window: 9:30 ET – 3:45 ET. No new entries after 15:30 ET. EOD force-close at 15:45 ET.
5. VIX > 40 → halt all new entries. 12 < VIX ≤ 40 allowed under normal filters. VIX < 12 → no 0DTE longs.
6. IV rank gates: < 30% blocks premium selling; > 70% blocks premium buying.
7. Every order goes through the 14-filter chain + order validator + portfolio risk check. No exceptions.
8. Every entry sets auto profit target and stop loss *at entry*. Exits are event-driven, not hope-driven.

## References

Algorithms and risk patterns are drawn from the uploaded playbook: BS/Greeks/IV, VRP, SVI, HAR-RV, Hybrid Kelly+VIX, PortfolioRiskManager, QuoteValidator, smart limit placement, walk-forward backtest. See `docs/playbook_mapping.md` for where each playbook section maps in code.

**Disclaimer:** None of this is financial advice. Options trading carries substantial risk of loss. Paper-trade everything first.
