"""Build a concise digest of the last N minutes of tradebot.out and
post it to Discord via the existing MultiChannelNotifier.

Runs via systemd timer (Linux / Jetson) or launchd (Mac). Idempotent —
can be invoked on demand from Discord as `!summary`.

Output is one short message intended for passive monitoring, not
detailed post-mortem.

Usage:
  python3 scripts/post_log_summary.py            # 60-min window
  python3 scripts/post_log_summary.py --window 30
  python3 scripts/post_log_summary.py --no-post  # print, skip Discord
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.notify.base import build_notifier
from src.notify.issue_reporter import alert_on_crash
from src.reports.log_digest import build_digest


@alert_on_crash("log_summary", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=60,
                     help="Minutes of log history to summarize (default 60).")
    ap.add_argument("--log", type=str, default="",
                     help="Path to tradebot.out (default: logs/tradebot.out).")
    ap.add_argument("--no-post", action="store_true",
                     help="Print the digest to stdout, skip Discord.")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else (ROOT / "logs" / "tradebot.out")

    digest = build_digest(log_path, window_minutes=args.window)
    body = digest.to_markdown()

    if args.no_post:
        print(body)
        return 0

    # Pick level from the error count so the embed coloring reflects
    # health at a glance.
    if digest.n_errors > 0:
        level = "error"
    elif (digest.n_warnings >= 5 or digest.shutdown_signals > 0
          or digest.alpaca_network_errors >= 3):
        level = "warn"
    else:
        level = "info"

    meta = {
        "Window":       f"{args.window} min",
        "Lines parsed": digest.n_parsed,
        "Entries":      digest.entries_fired,
        "Exits":        digest.exits_fired,
        "Skips":        sum(digest.skips_by_reason.values()),
        "Errors":       digest.n_errors,
        "Warnings":     digest.n_warnings,
        "_footer":      f"digest · window={args.window}m",
    }

    build_notifier().notify(
        body,
        level=level,
        title="summary",            # → #tradebot-reports channel if mapped
        meta=meta,
    )
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
