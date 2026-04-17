"""Weekly slippage calibration report + tuning proposal.

Reads the last N days of fills from `logs/slippage_calibration.jsonl`,
prints stats, and proposes new StochasticCostModel constants.

This is the HUMAN-REVIEW version. The bot's auto-calibrator runs on an
hourly/daily schedule with small steps; this is the weekly deeper pass
where a human reads the output and decides if any manual override is
warranted.

Philosophy: KEEP WHAT WORKS, TUNE WHAT DOESN'T.
  - Ratio in [0.8, 1.2] → model is calibrated. Don't touch.
  - Ratio > 1.2 → under-predicting. Bump up.
  - Ratio < 0.8 → over-predicting. Back off.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.analytics.slippage_calibration import (
    load_recent, analyze, propose_tuning,
)
from src.core.config import load_settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--path", default=None,
                    help="calibration log path (default: logs/slippage_calibration.jsonl)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    path = args.path or str(ROOT / "logs" / "slippage_calibration.jsonl")
    rows = load_recent(path, days=args.days)
    if not rows:
        print(f"No fills in the last {args.days}d at {path}. "
              f"Run the bot first, then re-run this script.")
        return 1
    stats = analyze(rows)
    if stats is None:
        print("Failed to analyze.")
        return 1

    # Current constants from settings (falls back to defaults if missing)
    cur = {
        "half_spread_mult": float(s.get("broker.half_spread_mult", 1.0)),
        "size_impact_coef": float(s.get("broker.size_impact_coef", 0.25)),
        "vix_impact_coef": float(s.get("broker.vix_impact_coef", 0.015)),
        "slip_noise_bps": float(s.get("broker.slip_noise_bps", 0.5)),
        "slip_floor_bps": float(s.get("broker.slip_floor_bps", 0.5)),
    }
    proposal = propose_tuning(stats, current=cur)

    if args.json:
        import dataclasses
        print(json.dumps({
            "stats": dataclasses.asdict(stats),
            "proposal": {
                "current": proposal.current,
                "proposed": proposal.proposed,
                "rationale": proposal.rationale,
            },
        }, indent=2))
        return 0

    print(f"Slippage calibration — last {args.days}d — n={stats.n} "
          f"({stats.days_covered:.1f}d of data)")
    print(f"  mean predicted bps   : {stats.mean_predicted:+.3f}")
    print(f"  mean observed bps    : {stats.mean_observed:+.3f}")
    print(f"  median observed bps  : {stats.median_observed:+.3f}")
    print(f"  p95 observed bps     : {stats.p95_observed:+.3f}")
    print(f"  p99 observed bps     : {stats.p99_observed:+.3f}")
    print(f"  mean_observed/pred   : {stats.mean_ratio:.3f}")
    print()
    print("Per-component mean (bps):")
    for k, v in stats.per_component_mean.items():
        print(f"  {k:<22s} {v:+.3f}")
    print()
    print("Per-symbol mean observed bps:")
    for k, v in sorted(stats.per_symbol_mean.items(), key=lambda kv: -abs(kv[1]))[:12]:
        print(f"  {k:<8s} {v:+.3f}")
    print()
    print("Tuning proposal:")
    for note in proposal.rationale:
        print(f"  - {note}")
    print()
    any_change = False
    for k in proposal.proposed:
        old = proposal.current[k]
        new = proposal.proposed[k]
        if abs(new - old) > 1e-6:
            any_change = True
            print(f"  {k:<22s} {old:>8.3f} → {new:<8.3f}  Δ={new - old:+.3f}")
    if not any_change:
        print("  (no changes — model is calibrated, keep current constants)")
    print()
    print("If you accept the proposal, paste this YAML into config/settings.yaml:")
    print("broker:")
    for k, v in proposal.proposed.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
