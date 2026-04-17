"""Pairwise correlation of signal emissions — are two strategies
actually independent, or are they the same bet in different hats?

Reads the last N days of ensemble_decisions, pivots contributor sources
vs. direction into a binary matrix (1 if signal fired that tick, 0 if
not), and computes the Jaccard similarity and Pearson correlation of
every pair.

High correlation between two sources means adding them as independent
signals overcounts evidence. Consider merging or dropping one.

Usage:
  python scripts/orthogonalize_signals.py --days 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import numpy as np

from src.core.config import load_settings
from src.storage.journal import build_journal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--min-fires", type=int, default=20,
                    help="ignore sources with fewer firings than this")
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    j = build_journal(
        backend=s.get("storage.backend", "sqlite"),
        sqlite_path=s.get("storage.sqlite_path", str(ROOT / "logs" / "tradebot.sqlite")),
        dsn_env_var=s.get("storage.cockroach_dsn_env", "COCKROACH_DSN"),
    )
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=args.days)
        decisions = j.ensemble_decisions(since=since, limit=200_000)
    finally:
        j.close()

    if not decisions:
        print("No ensemble decisions. Run the bot first.")
        return 1

    # Each (ts, symbol) is a decision "event". For each event, which
    # (source, direction) pairs fired?
    events: List[Dict[str, int]] = []
    sources_seen = set()
    for d in decisions:
        if not d.contributors:
            continue
        try:
            contribs = json.loads(d.contributors)
        except Exception:
            continue
        row = {}
        for c in contribs:
            key = f"{c.get('source','?')}:{c.get('direction','?')}"
            row[key] = 1
            sources_seen.add(key)
        events.append(row)

    # Build binary matrix
    sources = sorted(sources_seen)
    fires = {s: sum(e.get(s, 0) for e in events) for s in sources}
    sources = [s for s in sources if fires[s] >= args.min_fires]
    if len(sources) < 2:
        print(f"Need at least 2 sources with >= {args.min_fires} firings. "
              f"Got: {fires}")
        return 0
    n_events = len(events)
    mat = np.zeros((n_events, len(sources)), dtype=np.float32)
    for i, e in enumerate(events):
        for j, s in enumerate(sources):
            mat[i, j] = e.get(s, 0.0)

    print(f"Signal orthogonalization report — days={args.days}  n_events={n_events}")
    print(f"Sources (with fire counts): "
          f"{', '.join(s + '=' + str(fires[s]) for s in sources)}\n")

    # Pearson correlation
    corr = np.corrcoef(mat, rowvar=False)
    print("Pearson correlation matrix:")
    header = "                  " + "  ".join(f"{s:>14.14s}" for s in sources)
    print(header)
    for i, s in enumerate(sources):
        row = "  ".join(f"{corr[i, j]:>14.3f}" for j in range(len(sources)))
        print(f"{s:>18.18s}  {row}")

    # Jaccard similarity (intersection over union of fire-events)
    print("\nJaccard similarity (fire-event sets):")
    print(header)
    for i, s in enumerate(sources):
        row = []
        for k in range(len(sources)):
            a = mat[:, i].astype(bool)
            b = mat[:, k].astype(bool)
            inter = int(np.sum(a & b))
            uni = int(np.sum(a | b))
            jac = inter / uni if uni > 0 else 0.0
            row.append(f"{jac:>14.3f}")
        print(f"{sources[i]:>18.18s}  {'  '.join(row)}")

    # Flag high-correlation pairs
    print("\nHigh-correlation pairs (|r| > 0.6):")
    found = False
    for i in range(len(sources)):
        for k in range(i + 1, len(sources)):
            if abs(corr[i, k]) > 0.6:
                found = True
                print(f"  {sources[i]} ↔ {sources[k]}: r={corr[i,k]:+.3f}")
                print(f"    → these two signals are NOT independent evidence; "
                      f"consider dropping one or merging.")
    if not found:
        print("  (none — signals appear reasonably orthogonal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
