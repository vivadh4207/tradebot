"""Daily pull of CBOE put/call OI ratio for SPY + QQQ. Writes risk state.

Cron it alongside the nightly walkforward report:

  0 19 * * 1-5  $TRADEBOT_PY $TRADEBOT/scripts/fetch_putcall_oi.py

What it does:
  1. Fetches CBOE ETF-only P/C ratio daily close (HTTP scrape from
     the public daily CSV; no API key).
  2. Appends to logs/putcall_history.jsonl.
  3. Computes the risk-state (risk_off bool + multiplier) and writes
     logs/putcall_state.json.
  4. Posts the result to the Discord alerts channel if the risk_off
     flag CHANGES from the prior day (no daily noise if it's stable).

Data source detail: CBOE publishes https://www.cboe.com/us/options/market_statistics/daily/
with downloadable CSVs. Schema/URL drift occasionally; when that
happens the script logs the failure and does NOT trip risk-off
(safer default = keep trading). The issue_reporter will alert us.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.notify.base import build_notifier
from src.notify.issue_reporter import alert_on_crash, report_issue
from src.risk.putcall_oi_switch import (
    PutCallOIConfig, compute_risk_state, read_state,
)


_log = logging.getLogger(__name__)


def _fetch_cboe_etf_pc_ratio() -> float:
    """Fetch today's ETF P/C ratio from CBOE daily stats.

    Returns the numeric ratio or raises. Network + parse failures are
    the caller's to handle (we want the daily script to know so it can
    alert, not to silently write stale data).
    """
    # CBOE publishes a daily stats page with an ETF P/C ratio number.
    # For reliability, we use their published daily CSV feed; if the
    # URL schema changes, the script will alert via issue_reporter.
    import urllib.request
    import csv
    import io

    url = "https://www.cboe.com/us/options/market_statistics/daily/?dt="
    # The daily stats HTML page is the closest thing to a stable public
    # feed. We prefer the JSON one when available.
    json_url = "https://cdn.cboe.com/api/global/us_options/daily_market_statistics.json"
    req = urllib.request.Request(
        json_url,
        headers={"User-Agent": "tradebot/1.0 (putcall_oi fetch)"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    # Traverse whatever schema they publish — defensive; log specifically
    # if the shape changes.
    ratio = None
    # Two common locations we've seen:
    if isinstance(data, dict):
        d = data.get("data") or data
        if isinstance(d, dict):
            # possible keys: "etf_put_call_ratio", "etf_pc_ratio"
            for k in ("etf_put_call_ratio", "etf_pc_ratio",
                      "ETF_PC_RATIO"):
                if k in d:
                    ratio = float(d[k])
                    break
            if ratio is None and "ratios" in d:
                rr = d["ratios"]
                if isinstance(rr, dict):
                    for k in ("etf", "ETF"):
                        if k in rr:
                            ratio = float(rr[k])
                            break
    if ratio is None:
        raise RuntimeError(f"CBOE JSON shape unrecognized; keys: {list(data.keys())[:10] if isinstance(data, dict) else '?'}")
    return float(ratio)


@alert_on_crash("fetch_putcall_oi", rethrow=False)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-discord", action="store_true",
                     help="Print result to stdout; skip Discord alert.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Compute state but don't write files.")
    args = ap.parse_args()

    cfg = PutCallOIConfig()
    history_path = ROOT / cfg.history_path
    state_path   = ROOT / cfg.state_path
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # --- fetch today's ratio ---
    try:
        ratio = _fetch_cboe_etf_pc_ratio()
    except Exception as e:
        report_issue(
            scope="fetch_putcall_oi.cboe",
            message=f"CBOE fetch failed: {type(e).__name__}: {e}",
            exc=e,
            throttle_sec=6 * 3600.0,
        )
        return 1

    today = datetime.now(tz=timezone.utc).date().isoformat()
    entry = {"date": today, "ratio": round(ratio, 3)}
    if not args.dry_run:
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # --- load history + compute state ---
    history = []
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            try:
                history.append(json.loads(line))
            except Exception:
                continue
    # Sort oldest-first
    history.sort(key=lambda r: r.get("date", ""))

    state = compute_risk_state(history, cfg)

    if not args.dry_run:
        prior = read_state(state_path)
        state_path.write_text(json.dumps(state, indent=2))

        # --- notify only if the flag FLIPPED ---
        if not args.no_discord and bool(prior.get("risk_off")) != bool(state.get("risk_off")):
            flavor = "risk-off ACTIVATED" if state["risk_off"] else "risk-off CLEARED"
            build_notifier().notify(
                f"put/call OI switch: {flavor}",
                level="warn" if state["risk_off"] else "info",
                title="risk",
                meta={
                    "risk_off": state["risk_off"],
                    "ratio_latest": state.get("ratio_latest"),
                    "ratio_5d_avg": state.get("ratio_5d_avg"),
                    "prior_5d_avg": state.get("ratio_prior_5d_avg"),
                    "is_rising": state.get("is_rising"),
                    "size_multiplier": state.get("size_multiplier"),
                    "expires_at": state.get("expires_at") or "—",
                    "_footer": f"CBOE EOD • {today}",
                },
            )

    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
