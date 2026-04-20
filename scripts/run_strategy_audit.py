"""Trigger a strategy audit — 70B LLM reads current settings + recent
journal + market state, produces a scored report, appends to
logs/strategy_audit.jsonl.

Can be run:
  - On demand via tradebotctl.sh or the dashboard button
  - Nightly from cron:  0 21 * * 1-5  $TRADEBOT_PY $TRADEBOT/scripts/run_strategy_audit.py

Never in the live main-loop — 70B inference is far too slow (~3-6 tok/sec
on Jetson Orin AGX 64).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.core.config import load_settings
from src.notify.base import build_notifier
from src.notify.issue_reporter import alert_on_crash
from src.intelligence.strategy_auditor import build_auditor_from_settings
from src.storage.journal import build_journal


@alert_on_crash("strategy_audit", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-discord", action="store_true",
                     help="Print the report; skip Discord post.")
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    auditor = build_auditor_from_settings(s)
    if auditor is None:
        print("[!] auditor disabled or model path missing. Set "
              "LLM_AUDITOR_MODEL_PATH or strategy_auditor.model_path.")
        return 1

    # Open the journal read-only for the snapshot
    try:
        j = build_journal(sqlite_path=s.get("storage.sqlite_path",
                                               "logs/tradebot.sqlite"))
    except Exception as e:
        print(f"[!] could not open journal: {e}")
        j = None

    # Political news provider — pulls X/RSS/Alpaca if configured.
    # Dormant when political_news.enabled=false; safe either way.
    try:
        from src.intelligence.political_news import build_political_news_provider
        pol = build_political_news_provider(s)
    except Exception as e:
        print(f"[warn] political news provider failed to init: {e}")
        pol = None

    try:
        report = auditor.audit(s, journal=j, political_news=pol)
    finally:
        if j is not None:
            try:
                j.close()
            except Exception:
                pass

    if report is None:
        print("[!] audit failed — see logs.")
        return 2

    log_path = auditor.append(report)
    # Echo to stdout
    print(report.to_jsonl())
    print(f"\nappended to {log_path}")

    # Discord
    if not args.no_discord:
        meta = {
            "Overall health": f"{report.overall_health}/100",
            "Summary": (report.summary or "")[:300],
            "Issues (count)": len(report.issues),
            "Top issue": (f"{report.issues[0].severity.upper()} · "
                            f"{report.issues[0].area}: {report.issues[0].detail[:200]}"
                            if report.issues else "—"),
            "Model": report.model,
            "Latency": f"{report.latency_sec:.1f}s",
            "_footer": f"audit · {report.ts}",
        }
        level = ("success" if report.overall_health >= 80
                  else "warn" if report.overall_health >= 50
                  else "error")
        build_notifier().notify(
            f"Strategy audit — {report.overall_health}/100: "
            f"{(report.summary or '')[:160]}",
            level=level, title="edge_report",  # → #tradebot-reason channel
            meta=meta,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
