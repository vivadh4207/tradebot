"""A/B backtest: LSTM signal ON vs OFF on the same data.

Runs two BacktestSimulator passes — one with the LSTM signal enabled, one
without — and prints side-by-side deltas. Uses deterministic seeds so the
two runs see identical market data.

Examples:
  python scripts/compare_lstm.py --data historical --days 60
  python scripts/compare_lstm.py --data synthetic --total-bars 300
"""
from __future__ import annotations

import argparse
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

from src.core.config import load_settings
from src.core.clock import ET
from src.data.market_data import SyntheticDataAdapter
from src.data.historical_adapter import HistoricalMarketDataAdapter
from src.backtest.simulator import BacktestSimulator, SimConfig
from src.backtest.metrics import performance_report
from src.notify.issue_reporter import alert_on_crash


def run(settings, data, seed_label: str, disable=()) -> dict:
    sim = BacktestSimulator(settings, data, SimConfig(
        starting_equity=settings["account"]["paper_starting_equity"],
        disable_signals=tuple(disable),
        verbose=False,
    ))
    result = sim.run(settings["universe"], total_bars=300)
    eq = result["equity_curve"] or [sim.broker.account().equity]
    report = performance_report(eq, [], days_traded=len(eq))
    return {
        "label": seed_label,
        "final_equity": round(result["final_equity"], 2),
        "total_pnl": round(result["total_pnl"], 2),
        "metrics": report.to_dict(),
        "n_strategies": len(sim.strategies),
    }


@alert_on_crash("compare_lstm", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "historical"], default="synthetic")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--total-bars", type=int, default=300)
    ap.add_argument("--timeframe-min", type=int, default=1)
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")

    if args.data == "historical":
        end = datetime.now(tz=ET)
        start = end - timedelta(days=args.days)
        data = HistoricalMarketDataAdapter(
            symbols=s.universe, start=start, end=end,
            timeframe_minutes=args.timeframe_min,
        )
        print(f"[data] historical: {args.days}d @ {args.timeframe_min}m")
    else:
        data = SyntheticDataAdapter(seed=42)
        print("[data] synthetic GBM (seed=42)")

    print(f"[universe] {len(s.universe)} symbols")
    print(f"[run A] LSTM OFF")
    a = run(s.raw, data, "lstm_off", disable=("lstm",))
    print(f"[run B] LSTM ON")
    b = run(s.raw, data, "lstm_on", disable=())

    def row(k, av, bv, fmt="{:>12}", pct=False):
        delta = bv - av
        arr = "→"
        if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
            arr = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        av_s = (f"{av*100:.2f}%" if pct else fmt.format(av))
        bv_s = (f"{bv*100:.2f}%" if pct else fmt.format(bv))
        d_s = (f"{delta*100:+.2f}pp" if pct else fmt.format(delta)) if isinstance(bv, (int, float)) else ""
        print(f"  {k:>24s}  {av_s:>14s}  {bv_s:>14s}  {arr}  {d_s}")

    print()
    print(f"  {'':>24s}  {'LSTM OFF':>14s}  {'LSTM ON':>14s}     Δ")
    print("  " + "-" * 68)
    row("strategies loaded", a["n_strategies"], b["n_strategies"], "{:>12d}")
    row("final equity", a["final_equity"], b["final_equity"])
    row("total pnl", a["total_pnl"], b["total_pnl"])
    ma, mb = a["metrics"], b["metrics"]
    row("n_trades", ma["n_trades"], mb["n_trades"], "{:>12d}")
    row("total return %", ma["total_return_pct"], mb["total_return_pct"])
    row("sharpe", ma["sharpe"], mb["sharpe"])
    row("max drawdown %", ma["max_drawdown_pct"], mb["max_drawdown_pct"])
    row("win rate", ma["win_rate"], mb["win_rate"], pct=True)
    row("tail ratio (95/5)", ma["tail_ratio_95_5"], mb["tail_ratio_95_5"])

    # Verdict
    verdict = "inconclusive"
    if mb["sharpe"] > ma["sharpe"] + 0.1 and mb["max_drawdown_pct"] >= ma["max_drawdown_pct"] - 2:
        verdict = "LSTM helps — better Sharpe, comparable DD"
    elif mb["sharpe"] < ma["sharpe"] - 0.1:
        verdict = "LSTM hurts — lower Sharpe"
    elif abs(mb["sharpe"] - ma["sharpe"]) < 0.1:
        verdict = "LSTM ~neutral — small Sharpe delta"
    print(f"\n  verdict: {verdict}")
    print("  caveat : one backtest window is not proof. Re-run across")
    print("           multiple periods before drawing conclusions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
