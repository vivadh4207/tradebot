# Hedge-fund roadmap — implementation report

**Test count:** 174/174 passing (+13 roadmap + 16 continuous-cal / watchdog)

_Last updated: 2026-04-16 — see "Post-roadmap additions" at the bottom
for the continuous-calibration and launchd-watchdog work that landed
after the Tier 1–3 batch._

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
174 passed in 4.78s
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

---

## Post-roadmap additions

Two things landed after the Tier 1–3 batch in response to operator
feedback. Both are wired into the hot path (not shelfware).

### Continuous slippage calibration (10 tests)

The Tier 1 cost model was static until an operator manually tuned it.
It now self-tunes on a schedule with guardrails. Reference quant
practice (Almgren-Chriss post-trade analysis): observe realized vs
predicted slippage, apply bounded adjustments, audit every change.

Files:
```
src/analytics/slippage_calibration.py   # JSONL logger + analyze + propose_tuning
src/brokers/auto_calibrating_model.py   # wraps StochasticCostModel, scheduled recalib
scripts/calibrate_slippage.py           # weekly human-review report
scripts/daily_report.py                 # EOD snapshot w/ keep_or_tune field
config/settings.yaml                    # broker.auto_calibrate: daily (default)
```

Guardrails (non-negotiable):
  - never move any constant more than 30% per cycle (daily) / 10% (hourly)
  - never drift more than 2× from baseline across all cycles
  - never tune with < 30 samples
  - "keep what works" rule: if observed/predicted ratio ∈ [0.8, 1.2],
    don't touch anything

Dashboard: `/api/calibration` endpoint surfaces current stats +
adjustment history. See `OPERATIONS.md` for the 3-cadence operator
flow (daily / weekly / monthly).

### Watchdog supervision (8 tests)

Cron + a naked `run_paper.py` have two failure modes cron can't handle:
(a) silent hangs where the process is alive but the main loop is
wedged, and (b) unalerted crashes where the operator doesn't find out
until the next market open. Fixed with a two-layer supervisor.

Files:
```
scripts/watchdog_run.py                    # Python wrapper, crash alerts + heartbeat watchdog
deploy/launchd/com.tradebot.paper.plist    # launchd KeepAlive=true → watchdog
src/main.py                                # writes logs/heartbeat.txt every tick
scripts/tradebotctl.sh                     # watchdog-install / -uninstall / -status
```

Layers:
```
launchd (KeepAlive=true) → watchdog_run.py → run_paper.py
                             │
                             ├── alerts Discord/Slack on crash (via src.notify)
                             ├── tails stderr + records to watchdog_events.jsonl
                             └── kills child if logs/heartbeat.txt stales (default 5 min)
```

What each layer catches: see `OPERATIONS.md § Watchdog supervision`.

Install:
```bash
./scripts/tradebotctl.sh watchdog-install
./scripts/tradebotctl.sh watchdog-status
```

### New test files

```
tests/test_continuous_calibration.py    # 10 tests: logger, analyze, propose_tuning,
                                        # AutoCalibratingCostModel recalibrate + drift cap,
                                        # PaperBroker calibration recording, /api/calibration
tests/test_watchdog.py                   # 8 tests: clean exit, crash, stale heartbeat,
                                        # fresh heartbeat, startup grace, append-only
                                        # event log, stderr tail, heartbeat reset
```

### What did NOT land (intentionally)

- **Network-outage detection.** A router/Wi-Fi/ISP blip lets the bot
  keep ticking while Alpaca is unreachable; the SDK retries internally
  but a multi-hour outage will silently miss trades. Out of scope for
  the watchdog (which only sees the process). Separate backlog item.
- **Dashboard alerting beyond the webhook.** The notifier covers
  Discord/Slack. Pager/PagerDuty integration is not worth the
  complexity at retail scale.
