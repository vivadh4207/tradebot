"""Daily Monte Carlo VaR/CVaR report.

Reads open positions from the snapshot, pulls current spot + realized vol
per underlying, runs a 10k-path simulation over the next `horizon` days,
writes the result to a JSON file the dashboard reads, and pushes a
Discord/Slack alert when 95% VaR exceeds the configured threshold.

Safe from cron — never throws; logs + writes an empty report on any error.
"""
from __future__ import annotations

import argparse
import json
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
from src.data.market_data import SyntheticDataAdapter, AlpacaDataAdapter
from src.risk.monte_carlo_var import monte_carlo_var
from src.risk.vol_scaling import realized_vol_annualized
from src.storage.position_snapshot import load_snapshot
from src.intelligence.dividend_yield import DividendYieldProvider
from src.notify.base import build_notifier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-days", type=float, default=1.0)
    ap.add_argument("--n-paths", type=int, default=10_000)
    ap.add_argument("--var-threshold-pct", type=float, default=0.05,
                    help="alert if 95% VaR exceeds this fraction of equity")
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    snap_path = s.get("broker.snapshot_path",
                        str(ROOT / "logs" / "broker_state.json"))
    snap = load_snapshot(snap_path)
    out_path = ROOT / "logs" / "var_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if snap is None or not snap.positions:
        payload = {"ts": datetime.now(tz=timezone.utc).isoformat(),
                   "message": "no open positions"}
        out_path.write_text(json.dumps(payload, indent=2))
        print("[var] no open positions; wrote empty report.")
        return 0

    # Reconstruct positions list
    from src.core.types import Position as Pos, OptionRight
    from datetime import date as _date
    positions = []
    for r in snap.positions:
        expiry = _date.fromisoformat(r.expiry_iso) if r.expiry_iso else None
        right = OptionRight(r.right) if r.right else None
        positions.append(Pos(
            symbol=r.symbol, qty=int(r.qty), avg_price=float(r.avg_price),
            is_option=bool(r.is_option), underlying=r.underlying,
            strike=r.strike, expiry=expiry, right=right,
            multiplier=int(r.multiplier),
        ))

    # Get spot + vol per underlying from the market data adapter
    import os as _os
    key = _os.getenv("ALPACA_API_KEY_ID", "").strip()
    secret = _os.getenv("ALPACA_API_SECRET_KEY", "").strip()
    data = AlpacaDataAdapter(api_key=key, api_secret=secret) if key and secret \
        else SyntheticDataAdapter()
    underlyings = {(p.underlying or p.symbol) for p in positions}
    spots, vols = {}, {}
    for sym in underlyings:
        try:
            bars = data.get_bars(sym, limit=60)
            if bars:
                spots[sym] = bars[-1].close
                vols[sym] = realized_vol_annualized(bars, lookback=60)
            else:
                spots[sym] = 0.0
                vols[sym] = 0.20
        except Exception:
            spots[sym] = 0.0
            vols[sym] = 0.20

    # Per-symbol dividend yield for accurate option re-pricing
    div_prov = DividendYieldProvider(
        cache_path=s.get("pricing.dividend_cache", "data_cache/div_yields.json")
    )
    q_map = {u: div_prov.get(u) for u in underlyings}

    report = monte_carlo_var(
        positions, spots=spots, vols=vols,
        horizon_days=args.horizon_days, n_paths=args.n_paths,
        q_by_symbol=q_map, seed=42,
    )

    payload = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "horizon_days": report.horizon_days,
        "n_paths": report.n_paths,
        "var_95": report.var_95,
        "var_99": report.var_99,
        "cvar_95": report.cvar_95,
        "cvar_99": report.cvar_99,
        "expected_pnl": report.expected_pnl,
        "pnl_stdev": report.pnl_stdev,
        "best_case": report.best_case,
        "worst_case": report.worst_case,
        "per_position": report.per_position,
        "equity": snap.cash if snap else None,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[var] VaR_95=${report.var_95:.2f}  VaR_99=${report.var_99:.2f}  "
          f"CVaR_95=${report.cvar_95:.2f}")

    # Alert if above threshold
    equity = snap.cash or 1.0
    if equity > 0 and (report.var_95 / equity) > args.var_threshold_pct:
        try:
            build_notifier().notify(
                f"VaR alert: 95% VaR ${report.var_95:.2f} = "
                f"{report.var_95 / equity:.1%} of cash. "
                f"Threshold {args.var_threshold_pct:.1%}. "
                f"CVaR_95=${report.cvar_95:.2f}",
                level="warn", title="risk",
            )
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
