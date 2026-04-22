"""Options-research runner — fires the agent, posts the report.

Invocation:
  python3 scripts/run_options_research.py                 # default SPY+QQQ
  python3 scripts/run_options_research.py --symbols SPY
  python3 scripts/run_options_research.py --no-discord    # print, skip post
  python3 scripts/run_options_research.py --model llama3.3 --window 30

Scheduled via launchd / systemd timer to fire every 30 min during
session. On-demand from Discord via `!research [SYMBOL]` which calls
the agent directly rather than re-forking Python.
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
from src.data.multi_provider import MultiProvider
from src.intelligence.options_research import OptionsResearchAgent


@alert_on_crash("options_research", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ",
                     help="Comma-separated underlyings (default: SPY,QQQ)")
    ap.add_argument("--dynamic", action="store_true",
                     help="Also scan news + movers for additional tickers "
                          "via SymbolScanner, append top-N to --symbols.")
    ap.add_argument("--max-dynamic", type=int, default=4,
                     help="Max extra symbols from dynamic scan (default 4).")
    ap.add_argument("--model", default=None,
                     help="Override LLM model tag for this run.")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--timeout-sec", type=float, default=240.0)
    ap.add_argument("--no-discord", action="store_true",
                     help="Print to stdout, skip Discord post.")
    args = ap.parse_args()

    underlyings = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not underlyings:
        print("[!] no symbols supplied.", file=sys.stderr)
        return 2

    mp = MultiProvider.from_env()

    # Dynamic scan — append top-N trending tickers to the base list.
    if args.dynamic:
        try:
            from src.intelligence.symbol_scanner import SymbolScanner
            scanner = SymbolScanner(mp, base_universe=underlyings,
                                      max_dynamic=int(args.max_dynamic))
            underlyings = scanner.scan()
            print(f"[dynamic] scanning underlyings: {underlyings}")
        except Exception as e:
            print(f"[dynamic] scan failed, using static list: {e}")

    agent = OptionsResearchAgent(
        mp,
        model_name=args.model,
        max_tokens=int(args.max_tokens),
        timeout_sec=float(args.timeout_sec),
    )
    report = agent.run(underlyings)
    body = agent.to_markdown(report)
    print(body)

    # If LLM_AUTOTRADE is enabled, hand ideas off to the main-loop's
    # queue. The bot's LLMOriginationSignal picks them up on next tick
    # and runs them through the full filter chain + order path.
    import os as _os
    if _os.getenv("LLM_AUTOTRADE", "").strip() in ("1", "true", "yes"):
        try:
            from src.intelligence.llm_autotrade_queue import (
                LLMAutotradeQueue, QueuedIdea,
            )
            q_ideas = []
            for i in report.ideas:
                if i.direction not in ("call", "put"):
                    continue
                if i.confidence not in ("medium", "high"):
                    continue
                qi = QueuedIdea(
                    id=QueuedIdea.make_id(i.symbol, i.direction, i.strike,
                                           i.expiry),
                    ts=report.ts,
                    symbol=i.symbol,
                    direction=i.direction,
                    confidence=i.confidence,
                    strike=i.strike,
                    expiry=i.expiry,
                    entry=i.entry,
                    profit_target=i.profit_target,
                    stop_loss=i.stop_loss,
                    time_horizon=i.time_horizon,
                    rationale=i.rationale,
                    source="research_agent",
                )
                q_ideas.append(qi)
            from src.intelligence.llm_autotrade_queue import write_ideas
            n = write_ideas(q_ideas)
            print(f"[autotrade] queued {n} idea(s) for main loop")
        except Exception as e:                          # noqa: BLE001
            print(f"[autotrade] queue write failed: {e}")

    if not args.no_discord:
        # Post via the multi-channel notifier; title='llm_ideas' so
        # operators can map DISCORD_WEBHOOK_URL_LLM_IDEAS to a
        # dedicated channel if desired (else falls back to catch-all).
        meta = {
            "Model":        report.model or "(none)",
            "Latency":      f"{report.latency_sec:.1f}s",
            "Ideas":        len(report.ideas),
            "Headlines":    report.n_headlines,
            "Underlyings":  ",".join(underlyings),
            "Sources":      ",".join(
                sorted({s for L in report.quote_sources.values() for s in L})
            ),
            "_footer":      "options_research",
        }
        build_notifier().notify(
            body,
            level="info",
            title="llm_ideas",
            meta=meta,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
