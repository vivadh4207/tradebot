"""Reconcile local paper broker state FROM Tradier (make Tradier truth).

When local paper broker and Tradier diverge (bug causes phantom fills,
restart cascades, etc.), use this script to wipe local state and
rebuild it from what Tradier actually reports.

After running:
  - Local paper positions match Tradier exactly
  - Local cash = Tradier cash
  - day_pnl recomputed from scratch based on Tradier fills
  - Journal: does NOT rewrite history (journal is append-only audit log);
    it just updates the live state that the dashboard reads.

Use when:
  - Dashboard day_pnl disagrees with Tradier dashboard
  - Positions show as orphaned (on one side but not the other)
  - After a restart where local state got corrupted

Does NOT execute any new orders. Read-only from Tradier + local write.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main() -> int:
    from src.brokers.tradier_adapter import build_tradier_broker
    tb = build_tradier_broker()
    if tb is None:
        print("[!] Tradier not configured — need TRADIER_TOKEN in .env")
        return 2

    # Pull Tradier state
    try:
        tr_positions = list(tb.positions())
        acct = tb.account()
        tr_cash = float(getattr(acct, "cash", 0) or 0)
        tr_equity = float(getattr(acct, "equity", 0) or 0)
        tr_day_pnl = float(getattr(acct, "day_pnl", 0) or 0)
    except Exception as e:
        print(f"[!] Tradier query failed: {e}")
        return 3

    print(f"=== Tradier state ===")
    print(f"  cash:      ${tr_cash:,.2f}")
    print(f"  equity:    ${tr_equity:,.2f}")
    print(f"  day_pnl:   ${tr_day_pnl:+,.2f}")
    print(f"  positions: {len(tr_positions)}")
    for p in tr_positions:
        print(f"    · {p.symbol}  qty={p.qty}  avg=${p.avg_price:.2f}")

    # Build new local snapshot matching Tradier
    snap_path = ROOT / "logs" / "broker_state.json"
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    local_positions = []
    for p in tr_positions:
        exp_iso = None
        if p.expiry:
            try:
                exp_iso = p.expiry.isoformat()
            except Exception:
                pass
        local_positions.append({
            "symbol": p.symbol, "qty": int(p.qty),
            "avg_price": float(p.avg_price),
            "is_option": bool(p.is_option),
            "underlying": p.underlying,
            "strike": p.strike,
            "expiry_iso": exp_iso,
            "right": p.right.value if p.right else None,
            "multiplier": int(p.multiplier),
            "entry_ts": time.time(),
            "entry_tag": "reconciled_from_tradier",
            "auto_profit_target": None,
            "auto_stop_loss": None,
            "consecutive_holds": 0,
            "peak_price": None,
            "peak_pnl_pct": None,
            "scaled_out": False,
        })

    new_snap = {
        "version": 1,
        "saved_at": now_iso,
        "cash": tr_cash,
        "day_pnl": tr_day_pnl,
        "total_pnl": 0.0,          # journal rebuilds this from trades
        "positions": local_positions,
    }

    # Backup existing snapshot
    backup_path = snap_path.with_suffix(
        f".json.bak.{int(time.time())}"
    )
    if snap_path.exists():
        try:
            backup_path.write_text(snap_path.read_text())
            print(f"\n[ok] Old snapshot backed up → {backup_path.name}")
        except Exception as e:
            print(f"[!] backup failed: {e}")

    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(new_snap, indent=2, default=str))
    print(f"[ok] New snapshot written → {snap_path}")
    print()
    print("Restart the paper bot to load the reconciled state:")
    print("    bash scripts/tradebotctl.sh restart")
    print()
    print("Then refresh the dashboard. The top bar should now show "
          f"day_pnl ${tr_day_pnl:+.2f} matching Tradier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
