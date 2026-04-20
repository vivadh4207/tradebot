"""Calibration + accuracy report for resolved ML predictions.

Reads `ml_predictions` rows where `resolved_at IS NOT NULL` and prints:
  - overall accuracy vs. 1/3 random baseline
  - per-class precision / recall / F1
  - confidence-bucketed calibration (confidence bin -> empirical accuracy)
  - confusion matrix
  - directional-only summary (bullish vs bearish, ignoring neutral) —
    this is what matters for the bot because neutral doesn't trade.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.core.config import load_settings
from src.storage.journal import build_journal
from src.notify.issue_reporter import alert_on_crash


CLASSES = ["bearish", "neutral", "bullish"]


@alert_on_crash("analyze_ml_predictions", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="only include predictions at or above this confidence")
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    backend = args.backend or s.get("storage.backend", "sqlite")
    j = build_journal(
        backend=backend,
        sqlite_path=s.get("storage.sqlite_path", str(ROOT / "logs" / "tradebot.sqlite")),
        dsn_env_var=s.get("storage.cockroach_dsn_env", "COCKROACH_DSN"),
    )
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=args.days)
        rows = j.resolved_ml_predictions(model=args.model, since=since)
    finally:
        j.close()

    rows = [r for r in rows if r.confidence >= args.min_confidence]
    n = len(rows)
    if n == 0:
        print(f"No resolved predictions in the last {args.days}d. "
              f"Has `resolve_ml_predictions.py` been run?")
        return 1

    print(f"Resolved predictions: n={n}  model={args.model or 'ALL'}  "
          f"lookback={args.days}d  min_conf={args.min_confidence}")

    # overall
    correct = sum(1 for r in rows if r.pred_class == r.true_class)
    acc = correct / n
    print(f"Overall accuracy: {acc:.4f}  (baseline = 0.3333)")
    if acc > 0.40:
        tone = "ABOVE random by a healthy margin"
    elif acc > 0.36:
        tone = "modest edge over random"
    else:
        tone = "~random — model is not useful in this window"
    print(f"  → {tone}")

    # confusion matrix (rows=true, cols=pred)
    conf = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for r in rows:
        conf[r.true_class][r.pred_class] += 1

    print(f"\nConfusion matrix (rows=TRUE, cols=PRED):")
    print(f"{'':10s}" + "".join(f"{c:>10s}" for c in CLASSES) + f"{'sum':>8s}")
    for i, name in enumerate(CLASSES):
        row = conf[i]
        rs = sum(row)
        print(f"{name:10s}" + "".join(f"{v:>10d}" for v in row) + f"{rs:>8d}")

    # per-class precision / recall / F1
    print(f"\nPer-class metrics:")
    print(f"{'':10s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'n_pred':>10s}")
    for i, name in enumerate(CLASSES):
        tp = conf[i][i]
        fp = sum(conf[k][i] for k in range(3) if k != i)
        fn = sum(conf[i][k] for k in range(3) if k != i)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        n_pred = tp + fp
        print(f"{name:10s} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {n_pred:>10d}")

    # calibration: confidence buckets
    print(f"\nCalibration (confidence bin -> empirical accuracy):")
    bins = [(0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75),
             (0.75, 0.85), (0.85, 1.01)]
    for lo, hi in bins:
        subset = [r for r in rows if lo <= r.confidence < hi]
        if not subset:
            print(f"  [{lo:.2f}-{hi:.2f})  n=0")
            continue
        acc_b = sum(1 for r in subset if r.pred_class == r.true_class) / len(subset)
        mid = (lo + hi) / 2
        # perfectly calibrated: acc ≈ mid
        delta = acc_b - mid
        flag = "   well-calibrated" if abs(delta) < 0.05 else \
               (" OVERCONFIDENT" if delta < -0.05 else "    underconfident")
        print(f"  [{lo:.2f}-{hi:.2f})  n={len(subset):4d}  acc={acc_b:.3f}  "
              f"Δ={delta:+.3f}  {flag}")

    # directional-only (excluding neutral predictions — the ones that actually trade)
    directional = [r for r in rows if r.pred_class != 1]
    if directional:
        n_d = len(directional)
        correct_d = sum(1 for r in directional if r.pred_class == r.true_class)
        # treat "bearish predicted & true is bearish" as hit, otherwise miss
        # also count near-misses (predicted bullish but true neutral — no loss)
        wrong_side = sum(1 for r in directional
                         if (r.pred_class == 2 and r.true_class == 0) or
                            (r.pred_class == 0 and r.true_class == 2))
        print(f"\nDirectional-only (predictions that the bot would have TRADED): "
              f"n={n_d}")
        print(f"  exact hit rate  : {correct_d / n_d:.4f}")
        print(f"  wrong side rate : {wrong_side / n_d:.4f}  "
              f"(bullish→bearish or bearish→bullish)")
        print(f"  neutral outcome : {(n_d - correct_d - wrong_side) / n_d:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
