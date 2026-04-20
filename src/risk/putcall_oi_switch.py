"""Macro put/call open-interest risk switch.

Low-cost macro hedging detector. Pulls end-of-day CBOE put/call OI
ratios for SPY and QQQ, computes a 5-day rolling average, and emits
a "risk-off" state when aggregate hedging is intensifying.

Downstream effect: the position sizer reads this state and multiplies
entry size by the `risk_off_size_multiplier` (default 0.7) for
`risk_off_duration_days` (default 3) after the trigger fires.

This is NOT an entry signal. It's a portfolio-level dimmer switch.
The evidence base (per expert review) is thin — 0-20bps Sharpe
contribution if any — but the cost to implement is small and it adds
a real guardrail against trading through broad hedging regimes.

Data source: CBOE publishes P/C ratios for total equities, ETF-only,
index-only. Free daily data, no API key needed. We pull the ETF-only
series since SPY and QQQ are ETFs.

Output: logs/putcall_state.json with {risk_off: bool, entered_at: iso,
ratio_latest: float, ratio_5day_avg: float}. Dashboard surfaces it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_log = logging.getLogger(__name__)


@dataclass
class PutCallOIConfig:
    # Trigger: put OI / call OI must exceed this AND be rising vs 5-day avg.
    risk_off_ratio_threshold: float = 2.0
    lookback_days: int = 5
    risk_off_duration_days: int = 3
    risk_off_size_multiplier: float = 0.70
    history_path: str = "logs/putcall_history.jsonl"
    state_path: str = "logs/putcall_state.json"


def compute_risk_state(
    history: List[Dict[str, Any]],
    cfg: PutCallOIConfig,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Given a sorted-oldest-first list of daily P/C ratio records,
    decide whether today is a risk-off day.

    Each record: {"date": "YYYY-MM-DD", "ratio": float}

    Risk-off triggers when:
      (a) latest ratio > risk_off_ratio_threshold, AND
      (b) 5-day average is rising relative to the prior 5-day window
         (aggregate hedging is intensifying, not just mean-reverting).

    Once triggered, stays on for `risk_off_duration_days` calendar days
    regardless of ratio evolution (the sizer doesn't thrash on-off).
    """
    now = now or datetime.now(tz=timezone.utc)
    if len(history) < cfg.lookback_days:
        return {
            "risk_off": False,
            "reason": "insufficient_history",
            "ratio_latest": history[-1]["ratio"] if history else None,
            "ratio_5d_avg": None,
        }

    last_window = [r["ratio"] for r in history[-cfg.lookback_days:]]
    prior_window = ([r["ratio"] for r in history[-2 * cfg.lookback_days:-cfg.lookback_days]]
                     if len(history) >= 2 * cfg.lookback_days else last_window)
    latest = last_window[-1]
    ratio_5d = sum(last_window) / len(last_window)
    prior_5d = sum(prior_window) / len(prior_window)
    is_rising = ratio_5d > prior_5d

    trigger_today = latest > cfg.risk_off_ratio_threshold and is_rising

    return {
        "risk_off": bool(trigger_today),
        "reason": ("trigger_fired" if trigger_today else "no_trigger"),
        "ratio_latest": round(latest, 3),
        "ratio_5d_avg": round(ratio_5d, 3),
        "ratio_prior_5d_avg": round(prior_5d, 3),
        "is_rising": bool(is_rising),
        "threshold": cfg.risk_off_ratio_threshold,
        "size_multiplier": (cfg.risk_off_size_multiplier if trigger_today else 1.0),
        "entered_at": (now.isoformat() if trigger_today else None),
        "expires_at": ((now + timedelta(days=cfg.risk_off_duration_days)).isoformat()
                        if trigger_today else None),
    }


def read_state(state_path: Path,
                now: Optional[datetime] = None) -> Dict[str, Any]:
    """Load the latest risk state. Returns a safe default if the file
    doesn't exist, is malformed, or the risk_off window has expired."""
    now = now or datetime.now(tz=timezone.utc)
    default = {"risk_off": False, "size_multiplier": 1.0,
               "reason": "no_state_file"}
    if not state_path.exists():
        return default
    try:
        data = json.loads(state_path.read_text())
    except Exception:
        return default

    expires = data.get("expires_at")
    if data.get("risk_off") and expires:
        try:
            exp_dt = datetime.fromisoformat(expires)
            if exp_dt < now:
                # The window has expired — the switch self-clears.
                return {"risk_off": False, "size_multiplier": 1.0,
                         "reason": "window_expired",
                         "prior": data}
        except Exception:
            pass

    # Ensure size_multiplier is present (older state files may not have it)
    if "size_multiplier" not in data:
        data["size_multiplier"] = 1.0
    return data


def current_size_multiplier(
    state_path: Optional[Path] = None,
    cfg: Optional[PutCallOIConfig] = None,
    now: Optional[datetime] = None,
) -> float:
    """Callback used by the position sizer. Defaults to 1.0 if no state
    file or if the risk-off window has expired. Safe to call on every
    tick; reads one file."""
    cfg = cfg or PutCallOIConfig()
    path = state_path or Path(cfg.state_path)
    state = read_state(path, now=now)
    mult = state.get("size_multiplier", 1.0)
    try:
        mult = float(mult)
    except Exception:
        mult = 1.0
    # Clamp to sane range — guard against a malformed state file
    return max(0.1, min(1.5, mult))
