# LSTM price-direction signal

## Honest framing first

An LSTM trained on OHLCV bars alone is **a weak predictor**. On liquid
US large-caps it will typically produce a validation accuracy in the
35–45% range for a 3-class (bearish / neutral / bullish) label. That's
above the 33% random baseline — enough to be useful as an *ensemble
member*, not as a standalone entry trigger.

This signal lives alongside `momentum`, `vwap_reversion`, and `orb` in
the signal bus. The 14-filter execution chain is what converts a signal
into an order. If the LSTM says "bullish" and the news filter says
"blocked by downgrade", we don't trade — as designed.

**Do not flip `LIVE_TRADING=true` on the basis of a good backtest of
this signal alone.** Measure it in paper for 30+ days first.

## What it predicts

Given the last `seq_len` bars (default 30), predict the class of the
forward return over the next `horizon` bars (default 5):

- `bearish`  return ≤ `down_thr` (default −0.15%)
- `neutral`  in between
- `bullish`  return ≥ `up_thr` (default +0.15%)

At inference time a signal is emitted only if the top class is bullish
or bearish AND the softmax probability exceeds `min_confidence`
(default 0.55).

## Input features (7)

Computed causally per bar:

| feature | purpose |
|---|---|
| `log_return` | bar-over-bar returns |
| `log_range` | log((high−low)/close) — volatility proxy |
| `log_volume_ratio` | current volume vs. 20-bar rolling mean |
| `vwap_dev` | (close − vwap) / vwap |
| `rsi_14` | classic RSI, centered to [−1, 1] |
| `minute_sin` / `minute_cos` | minute of day, cyclically encoded |

All features are normalized per-column (z-score) using statistics
computed on the training set. The normalization is baked into the
checkpoint (`CheckpointMeta.stats`), so inference can't accidentally
use a different distribution.

## Model

- 2-layer LSTM, hidden=64, dropout=0.2
- LayerNorm on the final timestep
- 2-layer MLP head → 3-class logits
- ~37k parameters. Fits a 30-bar × 7-feature window in <1 ms on the Orin GPU.

## Training

### On the Jetson (recommended)

```bash
# One-time: PyTorch for JetPack 6 / CUDA 12
bash deploy/jetson/scripts/install_pytorch.sh .

# Quick training run on 6 months of 5-min bars
bash deploy/jetson/scripts/train_lstm.sh .

# Or the full CLI with overrides
python scripts/train_lstm.py \
  --days 365 --timeframe-min 5 \
  --seq-len 30 --horizon 5 \
  --hidden-size 64 --num-layers 2 \
  --epochs 40 --batch-size 256 \
  --out checkpoints/lstm_best.pt
```

GPU training on the Orin: ~1 minute per epoch for a typical 20k-sample
training set. Total training time for a 40-epoch run: 30–45 minutes
including early-stopping overhead.

### Elsewhere (Mac / VPS)

Install CPU torch (`pip install torch`) and run the same command.
Training will work — just slower. 5-10x slower on an Apple-silicon Mac,
20-30x slower on a small VPS.

### Walk-forward retraining

Drop this into cron (already built for priors + walk-forward):

```
# Sundays at 23:00 ET — retrain on the last year of bars
0 23 * * 0    $CTL train-lstm --days 365 --timeframe-min 5 --epochs 30
```

The bot rereads the checkpoint on restart, so pair it with a
`systemctl restart tradebot` in a Monday-morning cron if you want the
fresh weights live immediately.

## Activation

Set these in `config/settings.yaml` (defaults shown):

```yaml
ml:
  lstm_enabled: true
  lstm_checkpoint: checkpoints/lstm_best.pt
  lstm_min_confidence: 0.55
```

Or env override: `LSTM_MODEL_PATH=/absolute/path/to/lstm_best.pt`.

The bot registers the signal automatically when the checkpoint loads.
If it fails to load (file missing, torch missing, architecture
mismatch), the signal silently no-ops and the other strategies continue
normally. Check `logs/tradebot.out` for a `lstm_signal_disabled` or
`lstm_signal_ready` line after startup.

## How to know it's working

Three independent pieces of evidence, each cheap to produce.

### 1. Training val_acc above random

After `python scripts/train_lstm.py`: look at the printed `val_acc`. Above
0.33 (random) is required; 0.38+ is promising. If you see 0.5+, suspect
a label leak in your features.

### 2. A/B backtest: LSTM ON vs OFF

```bash
./scripts/tradebotctl.sh compare-lstm --data historical --days 60
```

Runs the same backtest window twice and reports side-by-side Sharpe,
max-DD, win-rate, and total PnL. The script emits a terse verdict:

```
verdict: LSTM helps — better Sharpe, comparable DD
caveat : one backtest window is not proof. Re-run across
         multiple periods before drawing conclusions.
```

### 3. Live calibration from paper-trade predictions

Every LSTM inference is logged to the `ml_predictions` journal table
(schema: timestamp, symbol, predicted class + confidence, horizon,
entry price). A cron job resolves each row once the horizon has
passed by pulling the forward close price:

```bash
# manual one-shot
./scripts/tradebotctl.sh resolve-ml
./scripts/tradebotctl.sh analyze-ml --days 14
```

Example output:

```
Resolved predictions: n=1834  model=lstm-v1  lookback=14d  min_conf=0.0
Overall accuracy: 0.4102  (baseline = 0.3333)
  → ABOVE random by a healthy margin

Confusion matrix (rows=TRUE, cols=PRED):
           bearish   neutral   bullish     sum
bearish        185       322       108     615
neutral        112       441       117     670
bullish         86       330       133     549

Per-class metrics:
           precision     recall         f1     n_pred
bearish       0.4832     0.3008     0.3710        383
neutral       0.4037     0.6582     0.5004       1093
bullish       0.3714     0.2422     0.2931        358

Calibration (confidence bin -> empirical accuracy):
  [0.33-0.45)  n=  42  acc=0.369  Δ=-0.021     well-calibrated
  [0.45-0.55)  n= 890  acc=0.452  Δ=-0.048     well-calibrated
  [0.55-0.65)  n= 607  acc=0.521  Δ=-0.079  OVERCONFIDENT
  [0.65-0.75)  n= 218  acc=0.673  Δ=-0.027     well-calibrated
  [0.75-0.85)  n=  63  acc=0.762  Δ=-0.038     well-calibrated
  [0.85-1.01)  n=  14  acc=0.857  Δ=-0.073  OVERCONFIDENT

Directional-only (predictions that the bot would have TRADED): n=741
  exact hit rate  : 0.4292
  wrong side rate : 0.2619  (bullish→bearish or bearish→bullish)
  neutral outcome : 0.3090
```

Reading this: the `directional-only hit rate` (0.43) is what matters —
that's the bot's real success rate when the LSTM actually triggers. If
`wrong side rate` is above 0.35, the model is a liability.

### 4. Cron-automated

Already in `deploy/cron/crontab.example`:

- Every 15 min during market hours → `resolve-ml` (backfills forward labels)
- 19:00 ET Mon-Fri → `analyze-ml --days 14` (calibration report to logs)
- Sunday 23:00 ET → `train-lstm --days 365` (fresh checkpoint for the week)

### Alert thresholds worth paying attention to

| Metric | Yellow flag | Red flag |
|---|---|---|
| Overall accuracy | < 0.37 | < 0.34 |
| Directional hit rate | < 0.40 | < 0.35 |
| Wrong side rate | > 0.30 | > 0.35 |
| Calibration Δ (in trading range 0.55-0.85) | > 0.07 overconfident | > 0.12 |
| Weekly A/B Sharpe delta | < +0.05 | < 0.00 |

Red flags = pull the checkpoint. Go back to training with more data,
tighter regularization, or different features. Don't override your
better judgment just because the model is "advanced".

## When to NOT enable it

- First week of paper. Let the base strategies run clean so you have a
  baseline to compare against.
- Right after retraining. Always do one backtest pass before pushing a
  new checkpoint into the live paper bot.
- If training val_acc is below 0.35 — the model didn't learn enough to
  beat random; using it will just add noise.
