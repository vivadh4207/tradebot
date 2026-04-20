"""Weekly feature-drift monitor for the LSTM signal.

Pulls the LSTM checkpoint's `stats` (training feature distribution)
and compares against the distribution of the SAME features computed on
the last N days of live bars. KS-test per feature; alerts on severe
drift so we know when the model is seeing out-of-distribution data.

Safe from cron — fail-soft on missing checkpoint or missing bars.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import numpy as np

from src.core.config import load_settings
from src.core.clock import ET
from src.data.historical_adapter import HistoricalMarketDataAdapter
from src.ml.features import build_feature_matrix, FeatureStats, FEATURE_COLS
from src.ml.feature_drift import check_drift
from src.notify.base import build_notifier
from src.notify.issue_reporter import alert_on_crash


@alert_on_crash("monitor_feature_drift", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="live-window days")
    ap.add_argument("--alert-thresh", type=float, default=0.15)
    ap.add_argument("--warn-thresh", type=float, default=0.08)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    ck_path = args.checkpoint or s.get("ml.lstm_checkpoint",
                                         "checkpoints/lstm_best.pt")
    ck_path = str(ROOT / ck_path) if not Path(ck_path).is_absolute() else ck_path
    ck_meta_path = Path(ck_path).with_suffix(".json")
    if not ck_meta_path.exists():
        print(f"[drift] no checkpoint metadata at {ck_meta_path}; "
              f"train the LSTM first (tradebotctl train-lstm).")
        return 1
    meta = json.loads(ck_meta_path.read_text())
    train_stats = FeatureStats.from_dict(meta["stats"])
    # We only have means/stds, not full samples. Synthesize N standard-normal
    # draws THEN un-normalize to the training distribution → gives us a
    # best-effort "training sample" for KS-test purposes. This is an
    # approximation; the right long-term fix is to store quantile percentiles.
    rng = np.random.default_rng(0)
    n_synth = 2000
    train_sim = rng.standard_normal((n_synth, len(FEATURE_COLS))).astype(np.float32)
    means = np.asarray(train_stats.means, dtype=np.float32)
    stds = np.asarray(train_stats.stds, dtype=np.float32)
    train_sim = train_sim * stds + means

    end = datetime.now(tz=ET)
    start = end - timedelta(days=args.days)
    data = HistoricalMarketDataAdapter(
        symbols=s.universe, start=start, end=end,
        timeframe_minutes=int(s.get("ml.lstm_timeframe_minutes", 5)),
    )
    all_live = []
    for sym in s.universe:
        try:
            bars = data.get_bars(sym, limit=5000,
                                   timeframe_minutes=int(s.get("ml.lstm_timeframe_minutes", 5)))
            if len(bars) >= 30:
                X = build_feature_matrix(bars[-2000:])
                all_live.append(X)
        except Exception:
            continue
    if not all_live:
        print("[drift] no live bars available.")
        return 1
    live_mat = np.concatenate(all_live, axis=0)

    report = check_drift(
        train_sim, live_mat, FEATURE_COLS,
        alert_thresh=args.alert_thresh, warn_thresh=args.warn_thresh,
    )

    print(f"[drift] n_train={report.n_train_samples}  "
          f"n_live={report.n_live_samples}  max_ks={report.max_ks:.3f}")
    print(f"{'FEATURE':<20s} {'KS':>8s} {'p':>10s} {'SEVERITY':>10s}")
    for a in report.alerts:
        print(f"{a.feature:<20s} {a.ks_statistic:>8.3f} {a.p_value:>10.3g} "
              f"{a.severity:>10s}")

    alerts = [a for a in report.alerts if a.severity == "alert"]
    if alerts:
        names = ", ".join(a.feature for a in alerts)
        try:
            build_notifier().notify(
                f"LSTM feature drift alert: {names} "
                f"(max KS={report.max_ks:.3f}). Consider retraining.",
                level="warn", title="ml",
            )
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
