# Historical backtesting — free data

Run the full bot pipeline against real historical market bars instead of
synthetic GBM noise, using data sources you already have or can get for
free.

## Data sources (free, stacked by preference)

1. **Alpaca historical bars** — best quality, uses your existing
   `ALPACA_API_KEY_ID`/`_SECRET`. Roughly 5 years of minute data for US
   equities. Activates automatically when the keys are present.
2. **yfinance** — completely free, no API key, no signup. Good for up
   to 2 years of daily data and 7 days of 1-minute data. Falls back to
   5-minute bars when 1-minute isn't available.
3. **Synthetic** — the pure-GBM generator. Always works, useful for
   pipeline smoke-tests.

All fetches cache to `data_cache/hist_<SYMBOL>_<TF>_<hash>.json` so
re-running the same backtest is instant after the first pull.

## Usage

```bash
# synthetic (default)
python scripts/run_backtest.py

# real historical: last 30 days of 1-minute bars
python scripts/run_backtest.py --data historical --days 30

# longer window, 5-minute bars, more bars per symbol
python scripts/run_backtest.py --data historical --days 90 --timeframe-min 5 --total-bars 1500
```

Fills, closed trades, and the equity curve land in whichever journal
backend is configured — SQLite by default, CockroachDB if flipped — so
you can inspect the run in the dashboard the same way you would a paper
session.

## Gotchas

- **yfinance 1-minute quota:** Yahoo caps 1-minute history at ~7 days.
  For anything longer, the adapter silently switches to 5-minute bars.
- **Alpaca free-tier minute data:** limited to IEX feed, slightly sparser
  than SIP. Good enough for backtesting.
- **Dividends / splits:** the adapter currently does NOT adjust for
  corporate actions. For a 10-symbol large-cap universe over a 30-day
  window this is negligible, but do not rely on this for multi-year
  backtests without plumbing through adjusted closes.
- **Slippage model:** the default 2 bps is applied in `PaperBroker`.
  Realistic for liquid names during RTH; too optimistic for thin names
  or after-hours.

## Interpreting the output

A synthetic run on pure random walk with a positive-edge prior produces
negative Sharpe (the harness being honest — no free money on noise).
A historical run on real bars with the same strategy reveals whether
there's actual edge.

The right sequence:

1. Run `--data synthetic` → confirm pipeline + no crashes.
2. Run `--data historical --days 90` → measure the strategy's Sharpe,
   max drawdown, and win rate on real bars for a recent quarter.
3. Compare to a plain buy-and-hold on SPY for the same window — if the
   bot isn't beating buy-and-hold on a risk-adjusted basis, there's no
   reason to run it live.
4. Iterate. Adjust signal or filter thresholds, re-run, re-measure.

## Important

Historical results DO NOT guarantee future performance. Backtest Sharpe
is a necessary condition for going live, not a sufficient one. The paper
period (30+ days of run_paper.py against live Alpaca data) is the real
test — backtests lack microstructure effects that matter a lot for a
minute-timescale bot.
