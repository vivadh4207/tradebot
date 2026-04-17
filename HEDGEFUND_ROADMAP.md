# Hedge-fund-level roadmap

What separates a *competent retail algo* from what a systematic quant desk
at Citadel / Jane Street / SIG / Squarepoint actually runs. Reality check
first, then the prioritized elevation list.

## Reality check

**What you have today** (after the audit + enhancements batch):
- Multi-signal ensemble with regime-aware weighting and auto-tuner
- Portfolio risk with Greek limits and notional check
- 14-filter execution chain (fully audited and thread-safe)
- Kelly-hybrid sizing with regime multipliers
- Walk-forward backtest with look-ahead-free fills
- Journal + CockroachDB persistence + crash-recovery snapshots
- LSTM price-direction signal with calibration and A/B framework
- Real news filter (local LLM capable)
- Catalyst calendar (earnings + FDA)
- Live VIX feed + dividend yield per symbol
- Put-call parity sanity check
- Executive dashboard with health + regime + attribution + ML calibration
- 132 unit tests

**What real quant desks have that this doesn't:**

| Capability | Desk standard | We have | Gap |
|---|---|---|---|
| Pricing model | Local vol / SABR / Heston | BS + binomial option | model risk on ITM puts, skew |
| Transaction cost model | LOB-aware, size-dependent | fixed 2 bps slippage | can overstate Sharpe 0.3-0.8 on backtests |
| Risk limits | per-factor + stress-test driven | per-Greek floor/ceiling | no dynamic risk |
| Hedging | adaptive delta-gamma | auto profit/stop only | no gamma hedge |
| Sizing across book | Kelly + covariance matrix | per-trade Kelly | ignores correlation |
| Regime detection | HMM / Kalman / structural breaks | rule-based on VIX + autocorr | noisy labels |
| Execution | TWAP/VWAP/POV slicing | single-shot limit | slippage on large orders |
| Research/production | strict separation, reproducible | flat | can't rerun experiments cleanly |
| Feature drift monitoring | KS-test / population stability | none | model silently degrades |
| VaR / CVaR | Monte Carlo (10k paths) | 5-point stress | doesn't capture tails |
| Latency | microseconds colocated | 50-150 ms home internet | fine for 3-min cycle, fatal for HFT |
| Capital | $50M – $50B AUM | whatever you have | determines what strategies work |

You can close most of the capability gap. You cannot close the capital /
latency gaps without institutional infrastructure, and that's fine — a
solid retail+ bot doesn't need them to make money.

## Prioritized elevation list

Ranked by **impact on risk-adjusted returns**, not by technical difficulty.
Each item has a concrete fix path + expected Sharpe / DD improvement based
on what the literature says about similar retail → semi-pro moves.

### Tier 1 — biggest win-rate / Sharpe gains (do these first)

**1. Proper transaction cost model (LOB-aware slippage)** — HIGHEST IMPACT
Replace the fixed 2-bps slippage with a model that:
  - Scales slippage with `order_qty / displayed_size` (larger orders move the book)
  - Widens under stress (slippage ∝ VIX and ∝ observed bid-ask spread)
  - Adds a half-spread + quarter-tick for market-crossing orders
Real backtest Sharpe drops 0.2-0.5 points when you do this honestly. A bot
that shows Sharpe 1.5 on fixed-bps often shows Sharpe 0.9-1.1 on
realistic cost modeling. Better to discover that on a backtest than live.
Fix: `src/brokers/slippage_model.py` — `StochasticCostModel(quote, order, vix)`
returning realized fill px. ~150 lines.

**2. Kelly with correlation (position-level, not trade-level)**
Current: each trade sized independently. Problem: 3 bullish-SPY trades
across SPY / QQQ / IWM are highly correlated → you've implicitly levered
up 3× on one bet. Fix: maintain a rolling 20-day return covariance
matrix; compute joint Kelly using `kelly_multi = inverse(Sigma) @ expected_returns`.
This is what risk-parity desks do.
Fix: `src/risk/joint_kelly.py` + wire into `PositionSizer`. ~200 lines.
Expected impact: max drawdown cut ~25% during correlated regimes.

**3. Realized-vol-scaled position sizing**
Current sizer is regime-aware but not vol-aware at the symbol level.
A $100 risk budget on NVDA (realized vol 40%) and on KO (vol 15%) is NOT
equivalent exposure. Fix: size `qty = risk_budget / (vol × spot × multiplier)`.
Fix: `PositionSizer._per_symbol_vol_scale()`. ~60 lines.
Expected impact: Sharpe up 0.1-0.3 depending on vol dispersion.

**4. Drawdown-based leverage reduction**
Weekly/monthly max DD triggers a scale-down. If MTD drawdown > 5%,
multiply all position sizes by 0.5; if > 10%, go to paper-only for 48h.
Forces you to let a losing strategy cool off instead of doubling down.
Fix: `src/risk/drawdown_guard.py`. ~100 lines.
Expected impact: lowers tail-risk exit probability; keeps capital intact
during model failures.

**5. Monte Carlo VaR / CVaR**
Current stress test hits 5 scenarios. Quant desks run 10k simulated
paths per position each day and report 95% / 99% VaR and CVaR. Catches
tail risk the 5-scenario stress misses. Add a nightly cron that
simulates 10k paths using the fitted SVI + HAR-RV.
Fix: `src/risk/monte_carlo_var.py` + `scripts/daily_var.py`. ~250 lines.
Expected impact: quantifies actual tail exposure; becomes your go/no-go
signal for increasing live capital.

### Tier 2 — structural / quality improvements

**6. Local-vol or SABR pricing**
Replace BS for options further than 2% OTM. Dupire local-vol is the
standard desk implementation. Worth 1-3 bps/trade on an edge-case basis.
Fix: `src/math_tools/local_vol.py` — fit local vol surface from SVI
parameters. ~300 lines.

**7. Strategy-level orthogonalization**
Check each pair of signals for correlation of their emitted directions.
If VRP and Wheel both fire bullish 80% of the time, they're not two
strategies, they're one. Replace with a single weighted signal.
Fix: `scripts/orthogonalize_signals.py` — weekly report. ~100 lines.

**8. HMM regime classifier**
Replace the rule-based `RegimeClassifier` with a Hidden Markov Model
trained on historical regime labels. Produces smoother transitions and
captures the "structural break" regime the current one labels as noise.
Fix: `src/intelligence/regime_hmm.py` using `hmmlearn`. ~200 lines.

**9. Order-slicing (TWAP / VWAP / POV)**
Any order > 10 contracts should be sliced over N minutes, not submitted
in one shot. Reduces market impact. Relevant once you're trading larger
size; today's micro account doesn't need it.
Fix: `src/brokers/slicer.py`. ~150 lines.

**10. Feature drift monitoring**
KS-test each LSTM input feature's distribution vs training distribution
weekly. If any feature drifts > 2σ, alert — model is seeing data it
wasn't trained on.
Fix: `scripts/feature_drift.py` → journal + Discord alert. ~120 lines.

### Tier 3 — infrastructure maturity

**11. Research-production separation**
Add `research/` (notebooks, scratch, experimental signals) vs
`src/` (production, tested, shippable). CI rule: nothing in `research/`
can be imported from `src/`.
Fix: `research/` folder + `.github/workflows/ci.yml` guard. ~30 lines.

**12. Reproducibility: every backtest run stamped**
Save (git commit SHA, config hash, bars hash, seed, final Sharpe) to a
`runs` table. Allow re-running any historical result exactly.
Fix: `src/backtest/run_registry.py`. ~80 lines.

**13. Config validation at startup**
Pydantic schema for `settings.yaml`. Type-check every value, bound-check
ranges, fail fast. Currently a typo in `kelly_fraction_cap: 2.5` (should
be 0.25) would run and blow up capital.
Fix: `src/core/config_schema.py`. ~150 lines.

**14. Integration tests against a recorded Alpaca session**
Record 1 day of real Alpaca data (quotes + bars + options chain) into
fixtures. Replay against the bot end-to-end. Catches bugs that unit
tests miss.
Fix: `tests/integration/recorded_session/`. ~300 lines.

**15. Structured P&L attribution per tick**
Currently P&L is a scalar. Quant desks decompose it into:
  - delta P&L = delta × spot_move
  - gamma P&L = 0.5 × gamma × spot_move²
  - vega P&L = vega × vol_move
  - theta P&L = theta × dt
  - residual (slippage, skew, model error)
Fix: `src/analytics/pnl_attribution.py`. ~200 lines.

### Tier 4 — advanced (only worthwhile at scale)

**16. Stat arb / pairs trading** — when you have AUM that single-name momentum can't absorb.
**17. Latency infrastructure** — only matters if you're racing other algos.
**18. Cross-venue smart order routing** — Alpaca → IBKR → Tradier based on best price.
**19. Options market-making** — provide liquidity rather than take it. Orders of magnitude more complex.
**20. ML feature pipeline** (Feast / Tecton) — production feature store.

## What I'd actually do next (my order)

If you want to get this to *genuinely* quant-desk grade over the next
month, the ROI-optimized sequence:

**Week 1 — Better cost modeling (Tier 1 #1)**
Re-run your backtest with realistic slippage. If Sharpe stays above 1.0
after that, keep going. If it collapses to < 0.5, **stop and investigate
your strategy** — the edge was an artifact. This is the single most
important honesty check.

**Week 2 — Kelly with correlation + vol-scaled sizing (Tier 1 #2, #3)**
Meaningfully reduces drawdown without touching strategy logic.

**Week 3 — Monte Carlo VaR + drawdown guard (Tier 1 #4, #5)**
Now you have production-grade risk management, not just filters.

**Week 4 — Feature drift monitoring + HMM regime (Tier 2 #8, #10)**
Now the ML side is self-monitoring and regime labels are less noisy.

After that — you're actually doing what a systematic equity vol desk
does, at a retail scale. The only things left that institutions have and
you don't are:
1. Capital (not a code problem)
2. Latency (colo is $5-50k/mo; not worth it for our cadence)
3. Proprietary data feeds (not currently necessary)
4. Counterparty relationships for block trading (same, irrelevant for <$500k AUM)

## Pragmatic framing

You don't become Citadel by writing more code. At some point the next
improvement in the bot is worth less than the next improvement in:
- Understanding *when* the strategies work and *why*
- Running paper long enough (60+ days) to get statistically stable priors
- Resisting the urge to add more signals before measuring whether
  existing ones have edge

The infrastructure in this repo is now good enough that the binding
constraint on performance is **your edge detection**, not your code.
Spend the next 30 days in paper, not in the editor.

That said — if you want to work through Tier 1 items, I'm here.
