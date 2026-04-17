"""Post-hoc analysis of ensemble decisions.

Reports, over a lookback window:
  - Decision volume by regime (how often each regime fires)
  - Emit rate per regime (how often the coordinator actually acts)
  - Most common blocking reasons (below_threshold / conflict)
  - Contributor frequency per regime (which signals show up where)
  - Link to downstream trades: when the ensemble emits, does it win?

Requires the journal to have `ensemble_decisions` + `trades` populated
(i.e. the bot has run for at least a session with the ensemble wired in).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--days", type=int, default=14)
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
        decisions = j.ensemble_decisions(since=since)
        trades = j.closed_trades(since=since)
    finally:
        j.close()

    n = len(decisions)
    if n == 0:
        print(f"No ensemble decisions in the last {args.days}d.")
        return 1

    print(f"Ensemble decisions: n={n}  lookback={args.days}d  backend={backend}")

    # --- decision volume by regime ---
    by_regime = defaultdict(list)
    for d in decisions:
        by_regime[d.regime].append(d)

    print(f"\n{'REGIME':<18s} {'n':>6s} {'emit':>6s} {'emit%':>7s} {'avg_n_inputs':>14s}")
    print("-" * 56)
    for regime, rows in sorted(by_regime.items()):
        n_r = len(rows)
        emits = sum(1 for r in rows if r.emitted)
        avg_inp = sum(r.n_inputs for r in rows) / max(1, n_r)
        pct = emits / n_r if n_r else 0.0
        print(f"{regime:<18s} {n_r:>6d} {emits:>6d} {pct:>6.1%}  {avg_inp:>14.2f}")

    # --- block reasons ---
    block_reasons = Counter(
        (d.reason.split(":", 1)[0] if d.reason else "") for d in decisions if not d.emitted
    )
    if block_reasons:
        print("\nBlocking reasons (when not emitted):")
        for k, v in block_reasons.most_common():
            print(f"  {k:<28s} {v:>6d}")

    # --- contributor frequency per regime ---
    contrib_by_regime: dict = defaultdict(Counter)
    for d in decisions:
        if not d.contributors:
            continue
        try:
            items = json.loads(d.contributors)
        except Exception:
            continue
        for c in items:
            contrib_by_regime[d.regime][c.get("source", "?")] += 1
    print("\nContributor frequency by regime:")
    for regime, counter in sorted(contrib_by_regime.items()):
        sep = ", ".join(f"{src}={n}" for src, n in counter.most_common())
        print(f"  {regime:<18s} {sep}")

    # --- link to trades by approximate time ---
    print("\nEmitted ensemble decisions → later closed trade within 60 min:")
    emitted = [d for d in decisions if d.emitted]
    tr_by_symbol = defaultdict(list)
    for t in trades:
        if t.closed_at:
            tr_by_symbol[t.symbol].append(t)
    matched_pnl_pct = []
    for d in emitted:
        ts = d.ts if d.ts.tzinfo else d.ts.replace(tzinfo=timezone.utc)
        window_close = ts + timedelta(minutes=60)
        for t in tr_by_symbol.get(d.symbol, []):
            ot = t.opened_at if t.opened_at else ts
            if ot.tzinfo is None:
                ot = ot.replace(tzinfo=timezone.utc)
            if ts <= ot <= window_close:
                if t.pnl_pct is not None:
                    matched_pnl_pct.append((d.regime, d.dominant_direction, t.pnl_pct))
                break
    if matched_pnl_pct:
        overall = [p for _, _, p in matched_pnl_pct]
        wins = [p for p in overall if p > 0]
        print(f"  matched trades n={len(overall)}  "
              f"win_rate={len(wins) / len(overall):.3f}  "
              f"mean_pnl_pct={sum(overall) / len(overall):+.4f}")
        by_r = defaultdict(list)
        for regime, _, p in matched_pnl_pct:
            by_r[regime].append(p)
        print("  per-regime win rate:")
        for r, pnls in sorted(by_r.items()):
            w = sum(1 for p in pnls if p > 0) / len(pnls)
            print(f"    {r:<18s} n={len(pnls):>4d}  win_rate={w:.3f}  "
                  f"mean={sum(pnls) / len(pnls):+.4f}")
    else:
        print("  (no matched trades yet — still early in paper)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
