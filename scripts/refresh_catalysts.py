"""Refresh earnings + FDA catalyst calendar and print upcoming blackouts.

Safe to run from cron daily — e.g. 07:00 ET to capture new earnings announcements.
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

from src.core.config import load_settings
from src.intelligence.econ_calendar import EconomicCalendar
from src.intelligence.catalyst_calendar import build_default_catalyst_calendar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    cal = build_default_catalyst_calendar(
        static_yaml_path=s.get("catalysts.static_yaml", "config/catalysts.yaml"),
        lookahead_days=args.days,
    )
    events = cal.refresh(s.universe)
    econ = EconomicCalendar()
    n = cal.hydrate_econ_calendar(econ)

    if args.json:
        print(json.dumps({
            "n_events": len(events),
            "blackouts": econ.summary(),
            "events": [
                {"symbol": e.symbol, "type": e.event_type,
                 "when": e.when.isoformat(), "timing": e.timing,
                 "details": e.details} for e in events
            ],
        }, indent=2))
        return 0

    print(f"Loaded {len(events)} upcoming catalysts for {len(s.universe)} symbols:")
    print(f"{'SYMBOL':8s} {'DATE':12s} {'TYPE':10s} {'TIMING':8s} DETAILS")
    for e in sorted(events, key=lambda x: (x.when, x.symbol)):
        print(f"{e.symbol:8s} {e.when.isoformat():12s} {e.event_type:10s} "
              f"{e.timing:8s} {e.details}")
    if not events:
        print("  (none — feeds empty or no events in window)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
