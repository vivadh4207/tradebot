"""Nightly macro sweep — 70B LLM reviews today's action + the news
pulse and forecasts tomorrow's scenarios.

Different from run_strategy_audit.py (which reviews the BOT's config
+ journal). This script gives the operator a market-level read:

  - Today's session snapshot (VWAP, regime, VIX, SPY/QQQ daily OHLC)
  - Active political_news headlines (last 24h across all sources)
  - Recent signal decisions from the log
  - Last audit health

Output: a structured paragraph posted to Discord (via the
`edge_report` / catch-all webhook).

Scheduled via systemd timer (see deploy/systemd/tradebot-macro-sweep.*)
to fire at 8pm ET on weekdays. Can be invoked on demand via the
Discord `!macro-sweep` command too.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.intelligence.ollama_client import build_ollama_client


_PROMPT_TEMPLATE = """You are the overnight analyst for a paper-trading
options bot. Review the snapshot below and produce a disciplined,
actionable market read for TOMORROW. No hype, no predictions of exact
prices. Your output:

1. ONE-LINE TAKE (under 120 chars): overall tomorrow read.
2. SCENARIOS (3 bullets): base-case, bull-case, bear-case — each
   with trigger levels referencing specific numbers from the snapshot.
3. RISK FLAGS (up to 4 bullets): what could blow up the plan —
   catalysts, regime-change markers, macro risk from the headlines.
4. BOT GUIDANCE (2-4 bullets): which of its enabled strategies should
   be tightened / loosened / stood down given the read.

Be specific. Reference numbers from the snapshot. Do NOT invent data.
If the snapshot lacks something, say so. Keep the whole reply under
1400 characters so it fits a Discord embed.

SNAPSHOT:
{snapshot}

YOUR REPORT:
"""


def _recent_signal_events(log_path: Path, n: int = 40) -> List[str]:
    if not log_path.exists():
        return []
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            f.seek(max(0, size - 400_000))
            if size > 400_000:
                f.readline()
            text = f.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    wanted = ("exec_chain_pass", "ensemble_skip", "entry", "exit_placed",
              "regime_classified", "vix", "shutdown_signal")
    out: List[str] = []
    for line in reversed(text.splitlines()):
        if any(w in line for w in wanted):
            # Strip ANSI + structlog brackets for compactness.
            import re as _re
            clean = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
            clean = _re.sub(r"\[[a-z]+\s*\]\s*", "", clean)
            out.append(clean[:200])
            if len(out) >= n:
                break
    return list(reversed(out))


def _load_political_headlines(settings) -> List[Dict[str, Any]]:
    try:
        from src.intelligence.political_news import build_political_news_provider
        pol = build_political_news_provider(settings)
        if pol is None:
            return []
        snap = pol.snapshot_for_auditor()
        if isinstance(snap, dict):
            return list(snap.get("headlines", []))[:30]
        if isinstance(snap, list):
            return snap[:30]
    except Exception as e:
        return [{"error": f"political_news_unavailable: {e}"}]
    return []


def _last_audit(n: int = 1) -> Optional[Dict[str, Any]]:
    try:
        from src.intelligence.strategy_auditor import read_recent_audits
        recent = read_recent_audits(n)
        return recent[0] if recent else None
    except Exception:
        return None


def _build_snapshot(settings) -> Dict[str, Any]:
    """Compact everything the 70B should see into one serializable dict."""
    snap: Dict[str, Any] = {
        "now_utc": datetime.now(tz=timezone.utc).isoformat(),
        "universe": (settings.raw.get("universe")
                      if hasattr(settings, "raw") else None),
        "mode": ("live" if (settings.raw.get("execution", {}) or {})
                 .get("live_trading", False) else "paper"),
    }

    # broker_state.json — positions + day pnl
    snap_path = ROOT / "logs" / "broker_state.json"
    if snap_path.exists():
        try:
            data = json.loads(snap_path.read_text())
            snap["positions"] = [
                {"symbol": p.get("symbol"), "qty": p.get("qty"),
                 "avg": p.get("avg_price")}
                for p in (data.get("positions") or [])[:10]
            ]
            snap["cash"] = data.get("cash")
            snap["day_pnl"] = data.get("day_pnl")
        except Exception:
            pass

    # Political news pulse
    snap["political_news"] = _load_political_headlines(settings)

    # Recent signal events from tradebot.out
    snap["recent_signal_events"] = _recent_signal_events(
        ROOT / "logs" / "tradebot.out"
    )

    # Last strategy audit
    la = _last_audit()
    if la:
        snap["last_audit"] = {
            "health": la.get("overall_health"),
            "summary": la.get("summary"),
            "ts": la.get("ts"),
        }

    return snap


def _call_70b(prompt: str, *, model_name: str,
               max_tokens: int, timeout_sec: float) -> Optional[str]:
    """One 70B call via Ollama. Returns None on any failure."""
    client = build_ollama_client()
    # Override client timeout for this long-running call.
    client.cfg.timeout_sec = float(timeout_sec)
    if not client.ping():
        return None
    try:
        return client.generate(
            model=model_name,
            prompt=prompt,
            temperature=0.2,
            max_tokens=max_tokens,
            num_ctx=6144,
            stop=["\n\nSNAPSHOT:", "\n\nYOUR REPORT:"],
        )
    except Exception:
        return None


@alert_on_crash("nightly_macro_sweep", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-discord", action="store_true",
                     help="Print the report; skip Discord post.")
    ap.add_argument("--model", default=None,
                     help="Override LLM_AUDITOR_MODEL for this run.")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--timeout-sec", type=float, default=600.0,
                     help="Ollama call timeout (default 600s; 70B on "
                          "Jetson needs room).")
    args = ap.parse_args()

    s = load_settings(ROOT / "config" / "settings.yaml")
    snap = _build_snapshot(s)

    import os as _os
    model_name = (
        args.model
        or (_os.getenv("LLM_AUDITOR_MODEL", "").strip())
        or "llama3.1:70b"
    )
    prompt = _PROMPT_TEMPLATE.format(
        snapshot=json.dumps(snap, indent=2, default=str)[:8000],
    )

    raw = _call_70b(
        prompt, model_name=model_name,
        max_tokens=int(args.max_tokens),
        timeout_sec=float(args.timeout_sec),
    )
    if not raw:
        print("[!] 70B unavailable or returned empty — aborting sweep.")
        return 2

    report = raw.strip()
    # Sanitize mentions so the LLM can't ping @everyone.
    import re as _re
    report = _re.sub(r"@everyone|@here|<@[!&]?\d+>|<@&\d+>",
                      "[mention-stripped]", report)
    # Cap for Discord.
    if len(report) > 1700:
        report = report[:1690].rstrip() + "… [truncated]"

    print(report)

    if not args.no_discord:
        meta = {
            "Model":        model_name,
            "Tokens":       args.max_tokens,
            "Headlines":    len(snap.get("political_news") or []),
            "Positions":    len(snap.get("positions") or []),
            "_footer":      "macro_sweep",
        }
        build_notifier().notify(
            report,
            level="info",
            title="edge_report",           # routes to edge-reports webhook
            meta=meta,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
