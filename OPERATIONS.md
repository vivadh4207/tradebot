# Operations Runbook — continuous calibration edition

**Philosophy: Keep what works. Tune what doesn't. Every day, every week,
every 30 days — measuring and adjusting at three cadences
simultaneously.**

There is no "wait 30 days" gate. The system calibrates from day 1. You
observe at the same cadence it does.

---

## The three loops

| Loop | Cadence | What it tunes | Human review |
|---|---|---|---|
| **Auto-calibration (inside the bot)** | Hourly or daily (config) | Slippage model constants | No — runs automatically with small steps |
| **Weekly review** | Sunday evening | Ensemble weights, signal orthogonality, feature drift, LSTM retrain | Yes — you read reports + decide |
| **Monthly tuning** | Every ~30 days | Measured priors, universe composition, regime classifier choice (rule vs HMM), going-live gate | Yes — decision-level |

The bot's in-process calibrator moves the slippage model's constants
**within strict guardrails** (max 30% per cycle, max 2× drift from
baseline). It never overrides the weekly or monthly human-in-the-loop
review — those stay yours.

---

## Phase 0 — Today (30 minutes)

```bash
cd ~/Documents/Claude/Projects/tradebot
source .venv/bin/activate
pip install -r requirements.txt           # httpx + pydantic if missing
pytest -q --tb=short                      # must be green

python scripts/test_db_connection.py      # Cockroach OK
python scripts/run_backtest.py            # synthetic; registers the run

python scripts/run_dashboard.py           # in another tab → http://127.0.0.1:8000
```

Verify the dashboard shows every panel: health, open positions, regime,
attribution, LSTM, catalysts, ensemble, **calibration** (new),
**daily report** (new), logs.

---

## Phase 1 — Day 1 (even before you have 30 days of data)

### Morning

```bash
./scripts/tradebotctl.sh catalysts        # refresh earnings blackouts
./scripts/tradebotctl.sh start            # begin paper trading
./scripts/tradebotctl.sh logs             # tail in second tab
```

What's happening **inside the bot from the first tick**:
- Every fill writes a calibration row to `logs/slippage_calibration.jsonl`.
- Every 24 hours (configurable to hourly), the auto-calibrator reads the
  last 24h of fills and adjusts model constants within the 30% step cap.
- Every adjustment is logged to `logs/calibration_history.jsonl`.
- Discord/Slack gets a note if the observed/predicted ratio leaves the
  calibrated band (< 0.5× or > 1.5×).

### After close (4:00 PM ET)

```bash
./scripts/tradebotctl.sh stop             # clean EOD flatten + flush
./scripts/tradebotctl.sh daily-report     # today's snapshot
./scripts/tradebotctl.sh calibrate --days 1   # even day-1 calibration
```

Read the output. Pay attention to `keep_or_tune` on the slippage line:

- **keep**: model ratio is in `[0.8, 1.2]`. Don't touch anything.
- **tune_up**: observed slippage > predicted. Auto-calibrator will bump
  constants up on the next cycle.
- **tune_down**: observed < predicted. Auto-calibrator will back off.
- **insufficient**: fewer than 30 fills. Too little data to tune; wait.

### Day-1 acceptance criteria (what "working" looks like on day 1)

- Dashboard Health strip shows `live trading: paper`, log growing.
- No `main_loop_error` / `HALT` notifications in the log.
- At least 10 `ensemble_emit` OR `ensemble_skip` lines (system is making decisions).
- Journal has rows: `broker_state.json` shows non-zero `day_pnl` OR snapshot saved.

**If any of those fail, STOP.** Debug before continuing.

---

## Phase 2 — Week 1 (manual runs, daily cadence)

Run **every trading morning and evening** the same way as day 1:

```bash
# morning
./scripts/tradebotctl.sh catalysts
./scripts/tradebotctl.sh start

# evening
./scripts/tradebotctl.sh stop
./scripts/tradebotctl.sh daily-report
./scripts/tradebotctl.sh calibrate --days $(( $(date +%u) ))
# ^ passes days-so-far-this-week; e.g. on Wed runs --days 3
```

### Questions to ask yourself at EOD every day

1. **Is the model still calibrated?** Dashboard → Calibration panel →
   "ratio" column. If it's drifted outside `[0.8, 1.2]` and the
   auto-calibrator's step caps haven't brought it back within 2 days,
   investigate manually (possibly dominant regime change).

2. **Did we emit when we should have?** Dashboard → Recent ensemble
   decisions → count `emit` vs `block`. Too many `block`? Regime
   weights might need review. Too many `emit` that lost money? Same
   thing, different direction.

3. **Are priors moving?** Sunday's `compute_priors --days 7` shows
   win_rate / avg_win / avg_loss. Compare to last Sunday. Trending
   positive?

4. **Any silent exception?** Check `grep -E "warning|error" logs/tradebot.out | tail -20`.

---

## Phase 3 — End of Week 1 (install cron + first weekly review)

On Sunday evening, after running manually all week:

```bash
# 1. Install cron
sed -i '' 's|/ABSOLUTE/PATH/TO/tradebot|/Users/vivekadhikari/Documents/Claude/Projects/tradebot|g' \
  deploy/cron/crontab.example
crontab deploy/cron/crontab.example
crontab -l                                # verify
```

Grant cron full-disk access (Mac-only, once): **System Settings →
Privacy & Security → Full Disk Access → `+` → Cmd+Shift+G →
`/usr/sbin/cron` → toggle ON**.

### First weekly review (30 minutes)

```bash
# Check weekly outputs — cron will run these automatically going forward
./scripts/tradebotctl.sh calibrate --days 7         # slippage
./scripts/tradebotctl.sh analyze-ensemble --days 7  # per-regime win rates
./scripts/tradebotctl.sh orthogonalize --days 7     # signal independence
./scripts/tradebotctl.sh propose-weights --days 7   # weight nudges
./scripts/tradebotctl.sh drift --days 7             # LSTM drift
./scripts/tradebotctl.sh walkforward                # refit priors windowed
```

For each report, apply the **KEEP WHAT WORKS rule**:

| Report shows | Action |
|---|---|
| `keep_or_tune: keep` (calibration) | Do nothing |
| `keep_or_tune: tune_*` (calibration) | Let auto-cal handle it; only intervene if it's been stuck for 2+ weekends |
| `TRADABLE yes` in walk-forward | Strategy has edge; keep going |
| `TRADABLE no` everywhere | No measured edge this week; wait — do NOT add more signals |
| Propose-weights suggests shifts you UNDERSTAND | Paste into `settings.yaml`, `tradebotctl restart` |
| Propose-weights suggests shifts you DON'T understand | Skip the suggestion. Revisit next week with more data. |
| Orthogonalize finds `|r| > 0.6` pair | Drop the weaker signal's weight to 0 in regime where they overlap |
| Drift alert | Retrain LSTM NOW: `tradebotctl train-lstm` |

---

## Phase 4 — Weekly cadence (automated by cron)

Cron now runs (ET):

| When | Job | Philosophy |
|---|---|---|
| 07:00 M-F | Catalyst refresh | blackout discovery |
| 08:30 M-F | Monte Carlo VaR | pre-market tail check |
| 09:25 M-F | Bot start | paper trading |
| 15:45 internally | EOD flatten (by bot) | no overnight carry |
| 16:05 M-F | Bot stop | clean shutdown |
| **16:30 M-F** | **Daily report** | **today's trades + calibration** |
| **17:00 M-F** | **Weekly calibration (rolling 7d)** | **human-review calibration** |
| 19:00 M-F | LSTM calibration | if checkpoint exists |
| 19:15 M-F | Ensemble analysis | per-regime win rates |
| 22:00 M-F | Walk-forward | refit priors |
| 20:00 Sun | Quarterly walk-forward | longer window |
| 21:00 Sun | Propose weights | human-review |
| 21:30 Sun | Orthogonalize | correlation check |
| 22:30 Sun | Feature drift | KS-test |
| 23:00 Sun | Retrain LSTM | weekly model refresh |

Your job: **15 minutes Monday morning** to read last weekend's outputs
(Cmd+F in each `.log` for "TRADABLE", "keep_or_tune", "alert").

---

## Phase 5 — End of Day 30: first big recalibration

By day 30 you have:
- ~6000-20000 fill-level calibration rows
- ~20-50 closed trades (or more, depending on signal hit rate)
- 4 weekly reports to diff against each other
- LSTM checkpoint (probably retrained 4 times)

Run the 30-day recalibration pass:

```bash
./scripts/tradebotctl.sh calibrate --days 30     # slippage, full month
./scripts/tradebotctl.sh propose-weights --days 30 --min-trades 30
./scripts/tradebotctl.sh walkforward --train-days 252 --test-days 30
./scripts/tradebotctl.sh compare-lstm --data historical --days 30
./scripts/tradebotctl.sh analyze-ml --days 30
```

Decision points after reading all five:

1. **Is slippage calibrated?** (ratio ∈ `[0.8, 1.2]` stable across 4
   weekly samples). If yes → the model is self-sustaining; stop
   worrying about it.
2. **Does the strategy have measured edge?** `propose-weights --days 30`
   should show `EV per trade > 0`. If negative, pause and rethink.
3. **Is LSTM pulling its weight?** `compare-lstm` shows Sharpe delta.
   If negative, set `ml.lstm_enabled: false` in settings.
4. **Is the regime classifier picking the right labels?** Per-regime
   win rate should show meaningful spread. If all regimes are within
   5% of each other, the classifier isn't useful yet.

---

## Going-live decision framework (no calendar gate)

Don't wait 30 days arbitrarily — wait until **all five conditions** are
measured true, however many days that takes:

1. **Slippage calibrated** — `keep_or_tune: keep` in 4 of the last 4
   weekly calibration reports.
2. **Positive EV per trade** — `compute-priors --days 30 --min-trades 50`
   returns positive EV.
3. **Sharpe > 1.0 with stochastic slippage** — `run_backtest --data
   historical --days 90` with `cost_model: stochastic`.
4. **No drift alerts** — in last 2 weeks of `drift` output.
5. **Orthogonal signals** — ≤1 high-correlation (|r| > 0.6) pair.

If all five hold for 2 consecutive weekly reviews → you're eligible.
Migrate to a NJ VPS, start at 10% capital, scale weekly.

If even one fails, you're not ready. No excuses, no calendar reprieve.

---

## Tuning philosophy (keep-what-works)

Every calibration decision — auto or manual — follows these rules:

1. **If the ratio is in `[0.8, 1.2]`, do nothing.** The model is
   calibrated enough. Tuning a calibrated model on noise makes it
   worse, not better.

2. **Move in small steps.** Auto-cal caps at 10% / hour or 30% / day.
   Manual weekly review should not exceed those caps either.

3. **Cap drift from baseline at 2×.** A constant that was `0.25` at
   baseline can never go above `0.5` or below `0.125`. Prevents
   runaway calibration when data is weird.

4. **Always have a rollback.** `logs/calibration_history.jsonl` stores
   every change. If you need to revert, paste the previous constants
   back into `settings.yaml`.

5. **Observe at 3 cadences simultaneously.**
   - Daily: read the daily report — are trades firing, calibration OK?
   - Weekly: read 6 reports Sunday night — are weights, signals, drift healthy?
   - Monthly (30d): recalibrate priors + decide go-live readiness.

6. **Never add features because a number looks bad.** Debug FIRST. Add
   features only when you've measured what's broken and know no
   existing feature can fix it.

---

## What you do every single day (15-minute minimum)

| Time (ET) | Action |
|---|---|
| 08:30 | `cat logs/var_report.json | python -m json.tool | head` — is 95% VaR < 5% of cash? |
| 15:55 | Glance at dashboard → Open Positions — will EOD flatten close these cleanly? |
| 16:30 | Open `/api/daily_report` on dashboard — today's win rate, EV, calibration |
| 17:00 | Skim `logs/calibrate.<today>.log` — did the auto-cal shift anything material? |
| 21:00 | (optional) Tail `logs/tradebot.out` for any `error`/`HALT` lines you missed |

**Sunday evening only**: 20-minute weekly review. Everything else runs
itself.

---

## Emergency procedures (unchanged from prior runbook)

```bash
# Cooperative stop (≤5s)
touch KILL

# Hard stop
./scripts/tradebotctl.sh stop

# Reconcile after crash
cat logs/broker_state.json | python -m json.tool | grep -A2 positions
# Cross-reference with Alpaca's web UI.
```

Rollback a bad auto-calibration:

```bash
# See what changed
tail -5 logs/calibration_history.jsonl | python -m json.tool

# Copy the "old" values from the most recent entry, paste into settings.yaml
# under `broker:` block, then:
./scripts/tradebotctl.sh restart
```

---

## The honest core message

The infrastructure is now truly continuous. It calibrates from the first
fill. It doesn't need you to wait 30 days before acting.

But **it can only calibrate what it can observe**. Some of the most
important questions — "is there real edge?", "does the LSTM help?",
"which regime matters most?" — only become answerable after
**statistically meaningful** sample sizes, which for most retail
strategies means 30-100 closed trades. That's a calendar minimum of a
few weeks, not because of a rule, but because of statistics.

So: run from day 1. Measure continuously. Tune with the keep-what-works
rule. Accept that your first weeks of numbers are noisy and don't
over-trust them. Move capital up only when the evidence is loud and
consistent across multiple cadences.
