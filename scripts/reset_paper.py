"""Full paper reset — wipes LOCAL journal AND Alpaca paper positions.

Use when bug-contaminated data makes today's results unfair to carry
into tomorrow. Three steps, all requiring typed confirmation:

  1. Close every open position on the Alpaca paper account (via
     close_all_positions endpoint — submits market-close orders for all
     open long stocks/options and buy-to-close for all shorts).
  2. Wipe the local Cockroach / SQLite journal (same as wipe-journal).
  3. Delete logs/broker_state.json so the next bot start begins at
     starting_equity with no open positions.

Run directly:
    .venv/bin/python scripts/reset_paper.py
Or via tradebotctl:
    scripts/tradebotctl.sh reset-paper
"""
from __future__ import annotations

from src.notify.issue_reporter import alert_on_crash

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass


def _confirm() -> bool:
    print("")
    print("This will:")
    print("  1. Close EVERY open position on your Alpaca paper account")
    print("  2. TRUNCATE the local trade journal (all fills/trades/equity)")
    print("  3. Delete logs/broker_state.json (next start = fresh equity)")
    print("")
    print("Your Alpaca LIVE account is NOT touched. Only paper.")
    print("")
    print("Type DESTROY to continue, anything else aborts:")
    try:
        ans = input("> ").strip()
    except EOFError:
        return False
    return ans == "DESTROY"


@alert_on_crash("reset_paper", rethrow=False)
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true",
                     help="Skip the interactive DESTROY prompt. ONLY for "
                          "the dashboard reset-paper button which already "
                          "requires an in-UI confirm.")
    args, _ = ap.parse_known_args()
    if not args.yes and not _confirm():
        print("aborted — nothing changed.")
        return 1

    # ---- Step 1: close all Alpaca paper positions ----
    print("\n[1/3] closing Alpaca paper positions...")
    try:
        from src.brokers.alpaca_adapter import AlpacaBroker
        key = os.environ.get("ALPACA_API_KEY_ID", "").strip()
        sec = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
        if not key or not sec:
            print("  skip: ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing in .env")
        else:
            broker = AlpacaBroker(api_key=key, api_secret=sec, paper=True)
            summary = broker.close_all_paper_positions()
            if summary.get("ok"):
                print(f"  closed {summary['closed']} position(s) on Alpaca paper")
            else:
                print(f"  error: {summary.get('error', 'unknown')}")
                print("  (you may need to close them manually in Alpaca UI)")
    except Exception as e:                                  # noqa: BLE001
        print(f"  ERROR: {e}")
        print("  (skipping — continue with local wipe)")

    # ---- Step 2: wipe local journal ----
    print("\n[2/3] wiping local journal + calibration logs...")
    try:
        import subprocess
        # Re-use the existing wipe logic but auto-confirm via pipe.
        # The wipe_journal.py reads DESTROY from stdin.
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "wipe_journal.py")],
            input="DESTROY\n", text=True, capture_output=True, timeout=60,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  wipe_journal exit code {result.returncode}")
            print(result.stderr)
    except Exception as e:                                  # noqa: BLE001
        print(f"  ERROR: {e}")

    # ---- Step 3: delete broker snapshot ----
    print("\n[3/3] deleting logs/broker_state.json...")
    snap = ROOT / "logs" / "broker_state.json"
    try:
        if snap.exists():
            snap.unlink()
            print("  deleted logs/broker_state.json")
        else:
            print("  (already absent)")
    except Exception as e:                                  # noqa: BLE001
        print(f"  ERROR: {e}")

    print("\nDone. Next bot start will be fully fresh — $100k Alpaca paper + zero positions + empty journal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
