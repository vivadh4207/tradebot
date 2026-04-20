"""Resolve unresolved ML predictions by looking up forward prices.

For each prediction row where `resolved_at IS NULL` AND now >= ts + horizon,
pull the close price at ts + horizon minutes and compute the true class.

Safe to run from cron every few minutes during market hours; it's a no-op
when there's nothing to resolve.
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
from src.core.clock import ET
from src.data.historical_adapter import HistoricalMarketDataAdapter
from src.storage.journal import build_journal
from src.notify.issue_reporter import alert_on_crash


def _classify(fwd_ret: float, up_thr: float, down_thr: float) -> int:
    if fwd_ret > up_thr:
        return 2  # bullish
    if fwd_ret < down_thr:
        return 0  # bearish
    return 1      # neutral


@alert_on_crash("resolve_ml_predictions", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None,
                    help="sqlite|cockroach — default from settings.yaml")
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--lookback-days", type=int, default=14,
                    help="only look at predictions from the last N days")
    ap.add_argument("--tolerance-minutes", type=int, default=5,
                    help="how far off the exact forward timestamp to accept")
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    backend = args.backend or s.get("storage.backend", "sqlite")
    j = build_journal(
        backend=backend,
        sqlite_path=s.get("storage.sqlite_path", str(ROOT / "logs" / "tradebot.sqlite")),
        dsn_env_var=s.get("storage.cockroach_dsn_env", "COCKROACH_DSN"),
    )

    try:
        now = datetime.now(tz=timezone.utc)
        cutoff_old = now - timedelta(days=args.lookback_days)
        unresolved = j.unresolved_ml_predictions(older_than=now, limit=args.max)
        unresolved = [p for p in unresolved if p.ts >= cutoff_old]

        if not unresolved:
            print("[=] No unresolved predictions in window. Nothing to do.")
            return 0

        # Group by symbol for batched bar fetches.
        by_symbol: dict[str, list] = {}
        for p in unresolved:
            if p.entry_price is None or p.horizon_minutes <= 0:
                continue
            # skip predictions whose horizon hasn't elapsed yet
            if p.ts + timedelta(minutes=p.horizon_minutes + args.tolerance_minutes) > now:
                continue
            by_symbol.setdefault(p.symbol, []).append(p)

        if not by_symbol:
            print("[=] No predictions have their horizon elapsed yet.")
            return 0

        # Build one historical adapter window per symbol covering all
        # predictions for that symbol + their horizons.
        resolved = 0
        failed = 0
        for sym, preds in by_symbol.items():
            preds.sort(key=lambda p: p.ts)
            start = preds[0].ts.astimezone(ET)
            end = max(p.ts + timedelta(minutes=p.horizon_minutes + args.tolerance_minutes)
                      for p in preds).astimezone(ET)
            # pad a few bars on each side
            start = start - timedelta(minutes=30)
            end = end + timedelta(minutes=30)
            tf = 1   # always 1-min bars for best resolution
            data = HistoricalMarketDataAdapter(
                symbols=[sym], start=start, end=end, timeframe_minutes=tf,
            )
            bars = data.get_bars(sym, limit=100_000, timeframe_minutes=tf)
            if not bars:
                print(f"[!] No bars for {sym} in {start}..{end}; skipping {len(preds)} preds.")
                failed += len(preds)
                continue

            for p in preds:
                target_ts = p.ts + timedelta(minutes=p.horizon_minutes)
                # find the bar closest (at or after) target_ts within tolerance
                tol = timedelta(minutes=args.tolerance_minutes)
                match = None
                for b in bars:
                    bts = b.ts if b.ts.tzinfo else b.ts.replace(tzinfo=timezone.utc)
                    if bts >= target_ts - timedelta(minutes=1):
                        if bts <= target_ts + tol:
                            match = b
                        break
                if match is None:
                    failed += 1
                    continue
                fwd_return = (match.close - p.entry_price) / p.entry_price
                true_cls = _classify(fwd_return, p.up_thr, p.down_thr)
                j.resolve_ml_prediction(p.id, float(fwd_return), int(true_cls))
                resolved += 1

        print(f"[=] Resolved {resolved}, skipped {failed} (no matching forward bar).")
        return 0
    finally:
        j.close()


if __name__ == "__main__":
    raise SystemExit(main())
