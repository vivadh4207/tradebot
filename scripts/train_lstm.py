"""Train the LSTM price-direction model on historical bars.

Uses:
  - src.data.historical_adapter.HistoricalMarketDataAdapter (Alpaca → yfinance → synthetic)
  - src.ml.dataset / src.ml.trainer
  - Saves best checkpoint to checkpoints/lstm_best.pt (+ .json sidecar)

Example:
  python scripts/train_lstm.py --days 365 --timeframe-min 5 --epochs 40
  python scripts/train_lstm.py --days 60  --timeframe-min 1 --epochs 20 \\
      --symbols SPY,QQQ,AAPL,NVDA --out checkpoints/lstm_5m.pt
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

from src.core.config import load_settings
from src.core.clock import ET
from src.data.historical_adapter import HistoricalMarketDataAdapter
from src.ml.dataset import build_dataset, time_ordered_split
from src.ml.checkpoint import CheckpointMeta
from src.notify.issue_reporter import alert_on_crash


@alert_on_crash("train_lstm", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None,
                    help="comma-separated override, else config/settings.yaml universe")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--timeframe-min", type=int, default=5,
                    help="5 is a good default: ~390/5 = 78 bars per day * 252 = 19k/year")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=5,
                    help="how many bars ahead to predict")
    ap.add_argument("--up-thr", type=float, default=0.0015)
    ap.add_argument("--down-thr", type=float, default=-0.0015)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "lstm_best.pt"))
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    symbols = (args.symbols.split(",") if args.symbols else s.universe)
    print(f"[*] Training on {len(symbols)} symbols over {args.days}d "
          f"@ {args.timeframe_min}m bars")

    end = datetime.now(tz=ET)
    start = end - timedelta(days=args.days)
    data = HistoricalMarketDataAdapter(
        symbols=symbols, start=start, end=end,
        timeframe_minutes=args.timeframe_min,
    )
    bars_by_symbol = {sym: data.get_bars(sym, limit=100_000,
                                           timeframe_minutes=args.timeframe_min)
                       for sym in symbols}
    total_bars = sum(len(v) for v in bars_by_symbol.values())
    print(f"[*] Loaded {total_bars} total bars "
          f"({total_bars // max(1, len(symbols))} avg / symbol)")

    X, y, stats, per_symbol = build_dataset(
        bars_by_symbol,
        seq_len=args.seq_len, horizon=args.horizon,
        up_thr=args.up_thr, down_thr=args.down_thr,
    )
    if X.shape[0] == 0:
        print("[!] No usable sequences — check data coverage and thresholds.")
        return 2

    X_tr, y_tr, X_va, y_va = time_ordered_split(X, y, val_frac=args.val_frac)
    cls_counts = [int((y_tr == c).sum()) for c in range(3)]
    print(f"[*] Train={len(X_tr)}  Val={len(X_va)}  train class counts: "
          f"bearish={cls_counts[0]} neutral={cls_counts[1]} bullish={cls_counts[2]}")

    from src.ml.trainer import train
    meta = CheckpointMeta(
        seq_len=args.seq_len, horizon=args.horizon, input_size=X.shape[-1],
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, num_classes=3,
        up_thr=args.up_thr, down_thr=args.down_thr,
        stats=stats.to_dict(),
        trained_symbols=[d.symbol for d in per_symbol],
        train_bar_count=total_bars,
        notes=f"{args.days}d @ {args.timeframe_min}m",
    )
    result = train(
        X_tr, y_tr, X_va, y_va, meta,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        checkpoint_path=args.out,
    )
    print(f"\n[=] Best val_loss={result.best_val_loss:.4f}  "
          f"val_acc={result.best_val_accuracy:.4f}  epoch={result.best_epoch}")
    print(f"[=] Saved: {args.out}")
    print(f"[=] Metadata: {Path(args.out).with_suffix('.json')}")
    print("\nNote: val_acc meaningfully above 1/3 (random) is promising, "
          "but backtest the signal before trusting it in paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
