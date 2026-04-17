# Regime-Aware Ensemble

## What it does

Collects all the raw signals emitted by every enabled strategy on a given
tick, classifies the current **market regime**, multiplies each signal's
confidence by a regime-specific weight, and emits AT MOST ONE consolidated
"ensemble" signal per symbol per tick. That one signal is what goes
through the 14-filter execution chain.

Before this: three strategies firing on SPY in the same tick would
independently try to enter three positions. Now: they combine into one
weighted decision, and the 14-filter chain sees exactly one candidate.

## The six regimes

Classified by time-of-day + volatility + price-return autocorrelation.

| Regime | When | Favored signals |
|---|---|---|
| `OPENING` | 09:30 – 10:30 ET | **ORB**, gap plays |
| `CLOSING` | 15:00 – 15:30 ET | Effectively throttled — EOD sweep owns this |
| `TREND_LOWVOL` | mid-day, VIX < 25, rets auto-corr > 0.15 | **Momentum**, **LSTM** |
| `TREND_HIGHVOL` | mid-day, VIX ≥ 25, rets auto-corr > 0.15 | Momentum (reduced size) |
| `RANGE_LOWVOL` | mid-day, VIX < 25, rets auto-corr ≤ 0.15 | **VWAP reversion**, ORB |
| `RANGE_HIGHVOL` | mid-day, VIX ≥ 25, rets auto-corr ≤ 0.15 | **VRP**, **Wheel** (premium harvest) |

Labels are deliberately coarse — stable over 10-30 minute windows, not
per-bar flicker.

## How weighting works

For each raw signal the coordinator computes:

```
weighted_confidence = raw_confidence × weights[regime][source]
```

Then groups by `meta["direction"]` (`bullish` / `bearish` / `premium_harvest`),
sums within each group, and decides:

- **Emit** if the dominant direction's weighted score ≥
  `min_weighted_confidence` (default 0.70) AND
  `dominant / opposing ≥ dominance_ratio` (default 1.5).
- **Block** otherwise, with a reason string like `below_threshold:0.52<0.70`
  or `conflict:0.91/0.84=1.08<1.50`.

The emitted signal inherits `side`, `strike`, `expiry`, and `entry_tag`
from the strongest contributor in the winning direction. Its `source` is
`"ensemble"` and its `meta` includes the full list of contributor sources
and the regime label for downstream audit.

## Default weights

In [`config/settings.yaml`](../config/settings.yaml):

```yaml
ensemble:
  enabled: true
  min_weighted_confidence: 0.70
  dominance_ratio: 1.5
  weights:
    trend_lowvol:  {momentum: 1.30, orb: 0.80, vwap_reversion: 0.60, vrp: 0.70, wheel: 0.70, lstm: 1.20, claude_ai: 1.10}
    trend_highvol: {momentum: 1.10, orb: 0.70, vwap_reversion: 0.50, vrp: 0.80, wheel: 0.60, lstm: 0.90, claude_ai: 0.90}
    range_lowvol:  {momentum: 0.60, orb: 1.00, vwap_reversion: 1.30, vrp: 1.00, wheel: 1.00, lstm: 0.80, claude_ai: 0.90}
    range_highvol: {momentum: 0.50, orb: 0.90, vwap_reversion: 1.10, vrp: 1.40, wheel: 1.30, lstm: 0.70, claude_ai: 0.80}
    opening:       {momentum: 0.80, orb: 1.50, vwap_reversion: 0.80, vrp: 0.60, wheel: 0.60, lstm: 0.80, claude_ai: 0.90}
    closing:       {momentum: 0.30, orb: 0.30, vwap_reversion: 0.30, vrp: 0.20, wheel: 0.20, lstm: 0.30, claude_ai: 0.30}
```

Unlisted regime/source combos default to `1.0` (neutral). Set any weight
to `0.0` to forbid that signal in that regime entirely.

## Observability

Every tick-per-symbol that had at least one raw signal is written to
`ensemble_decisions` in the journal, whether or not the coordinator
ultimately emitted. Fields include:

- `regime` — which of the six was active
- `emitted` — did we emit or block
- `dominant_direction` + `dominant_score` + `opposing_score`
- `reason` — terse human-readable: `emit:bullish:1.23`, `below_threshold:0.52<0.70`, `conflict:0.91/0.84=1.08<1.50`, ...
- `contributors` — JSON blob of `[{source, direction, raw, weight}, ...]`

Runtime logs also tag every decision:

```
ensemble_emit symbol=SPY regime=trend_lowvol direction=bullish score=1.23
  contributors=['momentum', 'lstm']

ensemble_skip symbol=NVDA regime=range_highvol reason=below_threshold:0.52<0.70
```

## Running the analyzer

```bash
./scripts/tradebotctl.sh analyze-ensemble --days 14
```

Sample output:

```
Ensemble decisions: n=3421  lookback=14d  backend=cockroach

REGIME                n   emit   emit%   avg_n_inputs
--------------------------------------------------------
opening            712    248   34.8%           1.84
range_highvol      183    110   60.1%           2.02
range_lowvol      1456    412   28.3%           1.67
trend_highvol      121     89   73.6%           2.28
trend_lowvol       894    501   56.0%           1.95
closing             55      4    7.3%           1.60

Blocking reasons (when not emitted):
  below_threshold              1824
  conflict                      208
  no_directed_signals            32

Contributor frequency by regime:
  opening           orb=612, vwap_reversion=203, momentum=150, lstm=88
  range_highvol     vrp=174, wheel=91, vwap_reversion=110, momentum=77
  range_lowvol      vwap_reversion=820, orb=410, momentum=241, lstm=121
  trend_lowvol      momentum=640, lstm=515, orb=102, vwap_reversion=56
  ...

Emitted ensemble decisions → later closed trade within 60 min:
  matched trades n=1110  win_rate=0.538  mean_pnl_pct=+0.0034
  per-regime win rate:
    opening         n= 248  win_rate=0.504  mean=+0.0021
    range_highvol   n= 110  win_rate=0.627  mean=+0.0078
    range_lowvol    n= 412  win_rate=0.519  mean=+0.0019
    trend_highvol   n=  89  win_rate=0.573  mean=+0.0041
    trend_lowvol    n= 501  win_rate=0.562  mean=+0.0045
    closing         n=   4  win_rate=0.500  mean=-0.0008
```

What to look for:

1. **`emit%` varies across regimes.** Good — that's the system being
   opinionated. If every regime has the same emit rate, weights aren't
   doing any work.
2. **`win_rate` varies across regimes.** Even better — that's evidence
   the regime label is predictive. If `range_highvol` has 62% win rate
   and `trend_highvol` has 48%, crank the `range_highvol` weights up
   and the `trend_highvol` weights down.
3. **`range_lowvol` + `trend_lowvol` together should be the bulk of
   emissions** since they're the majority of mid-day time. If opening /
   closing dominates your emit volume, your daily loss profile is
   front/back-loaded and the coordinator is doing the opposite of
   what you want.

## Tuning loop

1. Run for 2-4 weeks of paper with default weights.
2. `analyze-ensemble --days 14` → identify worst regime by win rate.
3. Nudge that regime's weights in `config/settings.yaml`:
   - If the WORST win rate belongs to a regime where `momentum` shows up
     as a top contributor, lower `momentum` in that regime to 0.5 or 0.0.
   - If the BEST win rate belongs to a regime where `vrp` + `wheel` are
     top contributors, raise them further to 1.6, 1.5.
4. Repeat next week. Don't touch weights more than once a week — you
   need multiple-session data to distinguish signal from noise.

## Emergency off-switch

```yaml
ensemble:
  enabled: false
```

With `enabled: false` the bot falls back to the legacy per-signal flow
(each strategy tries its own entry). Useful if you want to A/B the
ensemble itself against the pre-ensemble baseline over the same window.

Or at runtime: `touch KILL` for cooperative shutdown, then flip the
setting and restart.

## Live VIX feed

The regime classifier used to see a hardcoded `vix=15.0`. It now reads a
live `VixProbe` with a 60-second cache:

```yaml
vix:
  prefer: auto                   # auto | alpaca | yfinance | fallback
  cache_seconds: 60
  fallback: 15.0                 # used only when both data sources fail
```

`prefer: auto` tries Alpaca first (uses your existing credentials), falls
back to yfinance (`^VIX`, no API key), then to the static fallback. The
cache prevents hammering either source — typical cadence is ~6 fetches
per trading hour.

## Regime-aware position sizing

After Kelly-hybrid sizing computes a contract count, we multiply by a
regime-specific multiplier so size grows in regimes with proven edge and
shrinks in weak ones:

```yaml
sizing:
  regime_multipliers:
    trend_lowvol: 1.20
    trend_highvol: 0.90
    range_lowvol: 1.00
    range_highvol: 1.10
    opening: 0.80
    closing: 0.30
```

Values are clipped to `[0, 2]`. Set to `0.0` to forbid entries in that
regime entirely. The sizer applies the multiplier only when the ensemble
stamps the signal's `meta["regime"]`, so legacy per-signal mode (ensemble
disabled) is unaffected.

## Auto-tuner — nightly proposer

`propose_weights.py` reads the last 30 days of ensemble decisions + closed
trades, joins them on approximate time (decision → trade within 60 min
of open), measures realized win rate per `(regime, contributor)`, and
proposes a nudge:

```
nudge = old_weight × (1 + 0.30 × (win_rate - 0.5) × conf_scale)
```

Where `conf_scale` ranges from `0` (10 matched trades — min) to `1.0`
(50+ matched trades — full credit). Proposals are clipped to `[0.3, 1.8]`
per cell.

```bash
./scripts/tradebotctl.sh propose-weights --days 30
./scripts/tradebotctl.sh propose-weights --days 30 --json   # machine-readable
```

Sample output:

```
REGIME             SOURCE                 N      WR    OLD    NEW      Δ
----------------------------------------------------------------------------
range_highvol      vrp                   54   0.607   1.40   1.45  +0.045
trend_highvol      momentum              41   0.415   1.10   1.00  -0.101
trend_lowvol       lstm                  73   0.562   1.20   1.23  +0.027

Proposed ensemble.weights block (review before pasting):
ensemble:
  weights:
    trend_lowvol: {momentum: 1.30, lstm: 1.23, ...}
    ...
```

**Critical: this script never auto-applies changes.** You review the
deltas, decide if they make sense, and paste into `config/settings.yaml`
yourself. Cron runs it weekly (Sunday 21:00 ET) and writes the output to
`logs/propose_weights.YYYYMMDD.log` for Monday-morning review.

Three situations where you should IGNORE the proposal:

1. **Sample size < 50.** `conf_scale` damps small-sample noise but
   doesn't eliminate it. A (regime, source) with only 11 matched trades
   gives you a very noisy win rate estimate.
2. **The whole market regime shifted this week.** If the SPY had one
   crazy day that dominated your trade mix, don't let that warp the
   next month's weights.
3. **The nudge contradicts your prior.** If you strongly believe VRP
   works in high-vol regimes but 30 days of data shows the opposite,
   investigate WHY before trusting the proposal. Often it's a data bug,
   not a regime change.
