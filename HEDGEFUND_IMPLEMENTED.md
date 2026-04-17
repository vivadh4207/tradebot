# Hedge-fund roadmap — implementation report

**Test count:** 145/145 passing (+13 new)

## Tier 1 — highest Sharpe impact (all 5 done)

### 1. LOB-aware stochastic slippage model
File: `src/brokers/slippage_model.py`
`StochasticCostModel(quote, order, vix)` → `FillCost(executed_price, slippage_bps, components)`.
Combines half-spread + √(size/displayed_size) impact + VIX stress + queue noise. `PaperBroker` accepts an optional `slippage_model` param; falls back to the legacy fixed-bps model when not provided. Wire live quotes in via `broker.update_market_context(symbol, ctx)`.
Also exported: `LinearCostModel` (the legacy fixed-bps behavior as an explicit class).

### 2. Joint Kelly with correlation
File: `src/risk/joint_kelly.py`
`joint_kelly(symbols, expected_returns, cov_matrix)` solves f* = Σ⁻¹·μ with fractional Kelly + hard cap + Tikhonov regularization. `rolling_covariance(returns_by_symbol)` computes the sample covariance for a fixed lookback. Report includes `diagonal_only` and `correlation_penalty` for comparison vs. per-trade Kelly.

### 3. Realized-vol-scaled sizing
File: `src/risk/vol_scaling.py`
`vol_scale(bars, target_annual_vol=0.20)` returns a multiplier that normalizes exposure to a target vol per trade. NVDA (vol 40%) gets scaled down, KO (vol 15%) scaled up. Clipped `[0.25, 2.0]`.

### 4. Drawdown-based leverage reduction
File: `src/risk/drawdown_guard.py`
Tiered guard (DDs of 5%, 8%, 12%) that maps current equity + peak equity to a size multiplier in `{1.0, 0.75, 0.5, 0.0}`. The 12% tier **halts** new entries.

### 5. Monte Carlo VaR / CVaR
File: `src/risk/monte_carlo_var.py`
`monte_carlo_var(positions, spots, vols, horizon_days=1, n_paths=10_000)` simulates GBM paths per underlying, re-prices option positions at terminal spots via BS, returns 95%/99% VaR and CVaR + per-position breakdown. **CVaR is the number your risk committee actually cares about** — it measures what you lose in the tail, not just where the tail starts.

## Tier 2 — quality improvements

### 7. Strategy orthogonalization
Script: `scripts/orthogonalize_signals.py`
Reads last N days of `ensemble_decisions`, pivots contributors, computes Pearson correlation and Jaccard similarity between every pair of (source, direction). Flags pairs with |r| > 0.6 — signals that are not independent evidence.

### 8. HMM regime classifier
File: `src/intelligence/regime_hmm.py`
Pure-numpy 2-state Gaussian HMM (no `hmmlearn` dependency). Baum-Welch EM fit → Viterbi decoding. Returns the most likely state (low_vol / high_vol) for the most recent observation. Produces smoother transitions than the rule-based classifier. Drop-in augment to `RegimeClassifier`.

### 9. TWAP order slicer
File: `src/brokers/slicer.py`
`TWAPSlicer(broker, slices, interval_sec).submit(parent)` breaks a parent order into N child slices submitted at regular intervals. Blocking or background-thread mode.

### 10. Feature drift monitor
File: `src/ml/feature_drift.py`
`check_drift(train_matrix, live_matrix, feature_names)` runs KS-test per column. Warns at ks=0.08, alerts at ks=0.15. Uses `scipy.stats.ks_2samp` when available, falls back to manual empirical-CDF computation otherwise.

### 6. Local vol (Dupire on SVI)
File: `src/math_tools/local_vol.py`
`dupire_local_vol(spot, strike, T, r, q, svi_params_T)` — computes local vol from numerical derivatives of the SVI total-variance surface. Requires a fitted SVI slice per expiry (use `fit_svi_slice` from `svi.py`). **Honest limitations documented in the module docstring**: this is a retail-grade approximation; a production MM desk would use SLV or Markov-functional calibration.

## Tier 3 — infrastructure maturity

### 11. Research vs production separation
Folder: `research/` with `README.md` + graduation checklist.
`CLAUDE.md` updated: **"production code must NEVER import from `research/`".**

### 12. Backtest run registry
File: `src/backtest/run_registry.py`
Every backtest run writes a JSONL record to `logs/backtest_runs.jsonl` capturing git SHA (first 12 chars), SHA256 of settings.yaml contents, seed, data source, window, and final metrics. Reproduce any result exactly by checking out the commit + restoring the config with the matching hash.

### 13. Pydantic config schema
File: `src/core/config_schema.py`
`TradebotSettings` Pydantic model with type + bounds checking on every key in settings.yaml. `load_settings()` now validates at startup; bounds violations are **fatal** (raises `ValueError`). Example: a typo like `kelly_fraction_cap: 2.5` (should be 0.25) now fails loudly at startup instead of running and destroying capital.
Graceful fallback to manual bounds-check when `pydantic` is not installed.

### 15. Structured P&L attribution
File: `src/analytics/pnl_attribution.py`
`attribute_pnl(pos, S_t0, S_t1, sigma_t0, sigma_t1, T_t0, T_t1)` decomposes a position's P&L between two time points into:
  - delta P&L = Δ × ΔS
  - gamma P&L = 0.5 × Γ × ΔS²
  - vega P&L = Vega × Δσ
  - theta P&L = Θ × Δt
  - residual (everything else)

On a BS-consistent desk, residual should be < 20% of |total|. Larger residuals signal model error (skew, early exercise, etc.).
Also provides `attribute_book(positions, snapshots)` for bulk portfolio attribution.

## Tier 4 — intentionally NOT implemented

16-20 (stat arb, colo latency, cross-venue SOR, options market-making, feature store). The roadmap explicitly called these "only worthwhile at institutional scale." For retail / prosumer AUM, they're negative ROI.

## What's deferred to real operator work

Can't be automated purely in code — requires your ongoing observation:

- **Real slippage calibration**: after 30 days of paper fills, compute realized slippage from `fills` table and tune `StochasticCostModel` constants to match.
- **Correlation matrix refresh**: `rolling_covariance()` is available but you must call it on a cron schedule that refits every N days.
- **HMM vs rule-based A/B**: both classifiers exist; run both in parallel and compare regime win rates to pick the better one.
- **Local vol usage**: a fitted SVI surface per expiry needs to be maintained (use `fit_svi_slice()` on live Alpaca chain data daily).
- **Feature drift cron**: `scripts/monitor_feature_drift.py` would complete the loop; wire into your weekly cron.

## New files (this batch)

```
src/brokers/slippage_model.py            # stochastic cost model
src/brokers/slicer.py                     # TWAP order slicer
src/risk/joint_kelly.py                   # correlation-aware sizing
src/risk/vol_scaling.py                   # realized-vol-scaled sizing
src/risk/drawdown_guard.py                # DD-based leverage reduction
src/risk/monte_carlo_var.py               # 10k-path VaR/CVaR
src/intelligence/regime_hmm.py            # pure-numpy 2-state HMM
src/ml/feature_drift.py                   # KS-test feature drift
src/math_tools/local_vol.py               # Dupire on SVI
src/analytics/__init__.py                 # new subpackage
src/analytics/pnl_attribution.py          # Greek decomposition
src/backtest/run_registry.py              # reproducibility
src/core/config_schema.py                 # pydantic validation
scripts/orthogonalize_signals.py          # signal correlation report
research/README.md                        # production/research rule
tests/test_hedgefund_batch.py             # 13 regression tests
```

## Verification

```
145 passed in 3.35s
```

All 13 new tests specifically lock down the hedge-fund features:
slippage grows with size & VIX; joint Kelly penalizes correlation; vol
scaling moves the right direction; drawdown guard hits every tier; VaR
has correct ordering (CVaR ≥ VaR, 99% ≥ 95%); HMM finds the volatile
tail; TWAP slicer splits correctly; feature drift catches a shifted
distribution; run registry round-trips; Pydantic rejects out-of-range;
PnL attribution residual is small for BS-consistent moves.

## Push

```bash
cd ~/Documents/Claude/Projects/tradebot && git pull --rebase
git add -A
git commit -m "Hedge-fund roadmap: Tier 1-3 implementation (14 modules, 13 new tests, 145/145 green)"
git push
```

## One honest caveat

The code is in. **The work that remains is running it for 30+ days and actually reading the output.** Tier 1 items are useful only once you're:

- Feeding live quotes into the slippage model so it calibrates to your realized costs
- Running the joint-Kelly sizer with real rolling covariance from at least 30 days of your realized returns
- Getting regular VaR reports and comparing realized worst-days to predicted tail
- Checking `orthogonalize_signals` output weekly and dropping redundant signals
- Triggering `check_drift` on your LSTM features weekly

The infrastructure now matches what a real quant desk has. The constraint on actual performance is still what it was before: **measuring, observing, and iterating on real data**. Don't let the feature count make you complacent — *edge detection*, not feature count, is what separates a profitable bot from an expensive learning project.
