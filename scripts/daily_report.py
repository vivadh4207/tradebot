"""Daily EOD snapshot.

Aggregates everything that happened today: trades, ensemble decisions,
slippage calibration stats, P&L by entry tag, measured-priors delta
vs. yesterday. Writes a single JSON file the dashboard reads + stdout
summary for the cron log.

The key metric to watch: `keep_or_tune` block — does today's slippage
match the model's prediction? Answer per the KEEP-WHAT-WORKS rule.
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
from src.analytics.slippage_calibration import load_recent, analyze
from src.notify.issue_reporter import alert_on_crash


@alert_on_crash("daily_report", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1,
                    help="lookback for 'today' (1) or custom window")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    out_path = Path(args.out or ROOT / "logs" / "daily_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    j = build_journal(
        backend=s.get("storage.backend", "sqlite"),
        sqlite_path=s.get("storage.sqlite_path", str(ROOT / "logs" / "tradebot.sqlite")),
        dsn_env_var=s.get("storage.cockroach_dsn_env", "COCKROACH_DSN"),
    )
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(days=args.days)
        trades = j.closed_trades(since=since)
        ens = j.ensemble_decisions(since=since, limit=20_000)
    finally:
        j.close()

    # Trade roll-up
    n = len(trades)
    wins = [t for t in trades if (t.pnl or 0) > 0]
    losses = [t for t in trades if (t.pnl or 0) < 0]
    win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0.0
    avg_win = sum((t.pnl_pct or 0) for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(abs(t.pnl_pct or 0) for t in losses) / len(losses) if losses else 0.0
    total_pnl = sum((t.pnl or 0) for t in trades)

    # Attribution
    by_tag: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"n": 0, "wins": 0, "pnl": 0.0}
    )
    for t in trades:
        b = by_tag[t.entry_tag or "(none)"]
        b["n"] += 1
        if (t.pnl or 0) > 0:
            b["wins"] += 1
        b["pnl"] += float(t.pnl or 0)

    # Ensemble roll-up
    ens_emit = sum(1 for d in ens if d.emitted)
    regime_mix: Dict[str, int] = defaultdict(int)
    for d in ens:
        regime_mix[d.regime] += 1

    # Slippage calibration (last 24h)
    cal_rows = load_recent(
        s.get("broker.calibration_path", "logs/slippage_calibration.jsonl"),
        days=args.days,
    )
    cal_stats = analyze(cal_rows) if cal_rows else None

    # KEEP WHAT WORKS: is the cost model calibrated?
    keep_or_tune = "unknown"
    if cal_stats is not None and cal_stats.n >= 30:
        r = cal_stats.mean_ratio
        if 0.8 <= r <= 1.2:
            keep_or_tune = "keep"
        elif r > 1.2:
            keep_or_tune = "tune_up"
        elif 0 < r < 0.8:
            keep_or_tune = "tune_down"

    payload: Dict[str, Any] = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "window_days": args.days,
        "trades": {
            "n": n,
            "wins": len(wins), "losses": len(losses),
            "win_rate": round(win_rate, 4),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "total_pnl": round(total_pnl, 2),
            "ev_per_trade_pct": round(
                win_rate * avg_win - (1 - win_rate) * avg_loss, 5
            ),
        },
        "attribution": {
            tag: {
                "n": b["n"],
                "win_rate": round(b["wins"] / b["n"], 4) if b["n"] else 0.0,
                "total_pnl": round(b["pnl"], 2),
            }
            for tag, b in by_tag.items()
        },
        "ensemble": {
            "n_decisions": len(ens),
            "n_emitted": ens_emit,
            "emit_rate": round(ens_emit / len(ens), 4) if ens else 0.0,
            "regime_mix": dict(regime_mix),
        },
        "slippage": (
            {
                "n_fills": cal_stats.n,
                "mean_predicted_bps": round(cal_stats.mean_predicted, 3),
                "mean_observed_bps": round(cal_stats.mean_observed, 3),
                "p95_observed_bps": round(cal_stats.p95_observed, 3),
                "ratio": round(cal_stats.mean_ratio, 3),
                "keep_or_tune": keep_or_tune,
            }
            if cal_stats
            else {"n_fills": 0, "keep_or_tune": "no_data"}
        ),
    }

    out_path.write_text(json.dumps(payload, indent=2))

    # Stdout summary
    print(f"=== Daily report ({args.days}d) ===")
    print(f"Trades: n={n}  wr={win_rate:.2%}  avg_win={avg_win:.2%}  "
          f"avg_loss={avg_loss:.2%}  total_pnl=${total_pnl:.2f}  "
          f"EV/trade={payload['trades']['ev_per_trade_pct']:+.4f}")
    print(f"Ensemble: {len(ens)} decisions, {ens_emit} emitted "
          f"(emit_rate={payload['ensemble']['emit_rate']:.1%})")
    if cal_stats:
        print(f"Slippage: n={cal_stats.n}  predicted={cal_stats.mean_predicted:+.2f}bps  "
              f"observed={cal_stats.mean_observed:+.2f}bps  ratio={cal_stats.mean_ratio:.2f}  "
              f"→ {keep_or_tune}")
    else:
        print("Slippage: no data (need at least 30 fills)")
    print(f"Written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
