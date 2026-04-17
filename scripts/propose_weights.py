"""Propose ensemble weight nudges from observed journal data.

Reads the last N days of `ensemble_decisions` and joins to `trades`
(opened within 60 min of the decision) to measure realized win rate
per regime × contributor. Compares that to the currently-configured
weight and emits a YAML patch you can inspect, adjust, and paste back
into `config/settings.yaml`.

This script NEVER auto-applies changes. It's a proposer, not a tuner.
Keep a human in the loop on weight drift.

Scoring:
  expected_win_rate = 0.5     (baseline)
  realized_win_rate = wins_when_contributor_in_winning_direction / n_matched
  edge = realized - 0.5
  nudge = current_weight × (1 + 0.3 × edge × conf_scale)
    where conf_scale in [0, 1] based on sample size (full credit at >= 50 trades).

Rules:
  - Never propose a weight outside [0.3, 1.8].
  - Require >= 10 matched trades per (regime, source) to even consider it.
  - Print OLD → NEW, delta, rationale per cell.
  - Emit the full patched block at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.core.config import load_settings
from src.storage.journal import build_journal
from src.signals.ensemble import DEFAULT_WEIGHTS


WEIGHT_FLOOR = 0.30
WEIGHT_CEIL = 1.80
MIN_MATCHED_TRADES = 10
FULL_CREDIT_N = 50
LEARNING_RATE = 0.30


def _current_weights(s) -> Dict[str, Dict[str, float]]:
    raw = s.get("ensemble.weights") or {}
    if raw:
        return {str(k): {str(sn): float(w) for sn, w in v.items()} for k, v in raw.items()}
    # fall back to in-code defaults
    return {r.value: dict(w) for r, w in DEFAULT_WEIGHTS.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--min-matched", type=int, default=MIN_MATCHED_TRADES)
    ap.add_argument("--json", action="store_true")
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

    if not decisions:
        print(f"No ensemble decisions in the last {args.days}d. "
              "Let the bot run longer before proposing changes.")
        return 1

    # match decision → trade within 60 min, same symbol
    tr_by_symbol = defaultdict(list)
    for t in trades:
        if t.opened_at is not None:
            tr_by_symbol[t.symbol].append(t)
    for k in tr_by_symbol:
        tr_by_symbol[k].sort(key=lambda t: t.opened_at)

    # {(regime, source): {n_matched, wins}}
    stat: Dict = defaultdict(lambda: {"n": 0, "wins": 0})

    for d in decisions:
        if not d.emitted:
            continue
        if not d.contributors:
            continue
        try:
            contribs = json.loads(d.contributors)
        except Exception:
            continue
        # Only credit contributors that were IN the winning direction.
        winning = [c for c in contribs
                    if str(c.get("direction")) == str(d.dominant_direction)]
        if not winning:
            continue
        # Find a matching trade
        ts = d.ts if d.ts.tzinfo else d.ts.replace(tzinfo=timezone.utc)
        window = ts + timedelta(minutes=60)
        matched = None
        for t in tr_by_symbol.get(d.symbol, []):
            ot = t.opened_at
            if ot.tzinfo is None:
                ot = ot.replace(tzinfo=timezone.utc)
            if ts <= ot <= window:
                matched = t
                break
            if ot > window:
                break
        if matched is None:
            continue
        is_win = (matched.pnl or 0) > 0
        for c in winning:
            src = c.get("source", "?")
            key = (d.regime, src)
            stat[key]["n"] += 1
            if is_win:
                stat[key]["wins"] += 1

    current = _current_weights(s)
    proposed = {r: dict(w) for r, w in current.items()}
    rows = []
    for (regime, src), s_ in sorted(stat.items()):
        if s_["n"] < args.min_matched:
            continue
        wr = s_["wins"] / s_["n"]
        old = current.get(regime, {}).get(src, 1.0)
        edge = wr - 0.5
        conf_scale = min(1.0, s_["n"] / FULL_CREDIT_N)
        nudge = old * (1.0 + LEARNING_RATE * edge * conf_scale)
        new = max(WEIGHT_FLOOR, min(WEIGHT_CEIL, round(nudge, 3)))
        delta = new - old
        if regime not in proposed:
            proposed[regime] = {}
        proposed[regime][src] = new
        rows.append((regime, src, s_["n"], wr, old, new, delta))

    if not rows:
        print(f"No (regime, contributor) had >= {args.min_matched} matched trades. "
              "Keep current weights.")
        return 0

    if args.json:
        print(json.dumps({"proposed": proposed, "rows": [
            {"regime": r, "source": s_, "n": n, "win_rate": round(wr, 4),
             "old": o, "new": n2, "delta": round(d_, 3)}
            for r, s_, n, wr, o, n2, d_ in rows
        ]}, indent=2))
        return 0

    print(f"Proposed weight nudges (lookback={args.days}d, "
          f"min_matched={args.min_matched}):")
    print(f"{'REGIME':<18s} {'SOURCE':<18s} {'N':>5s} {'WR':>7s} "
          f"{'OLD':>6s} {'NEW':>6s} {'Δ':>7s}")
    print("-" * 76)
    for r, src, n, wr, old, new, d_ in rows:
        print(f"{r:<18s} {src:<18s} {n:>5d} {wr:>7.3f} "
              f"{old:>6.2f} {new:>6.2f} {d_:>+7.3f}")

    print("\nProposed ensemble.weights block (review before pasting):")
    print("ensemble:")
    print("  weights:")
    for regime in sorted(proposed):
        entries = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(proposed[regime].items()))
        print(f"    {regime}: {{{entries}}}")
    print("\nThis script does NOT auto-apply. Paste into config/settings.yaml")
    print("after you've eyeballed the deltas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
