"""Saves tracker — quantifies how much the defensive exits save us.

Every time a "protective" exit fires (green_to_red_killswitch,
profit_lock, chart_lower_highs, vwap_break, support_break, etc), we
log it. A background re-check 30 min and 2h later fetches the
contract's current price and records the delta:

  - Contract kept falling → we SAVED money (cut bleeding)
  - Contract recovered to higher than exit → we LEFT money on the table

Nightly Discord report aggregates both numbers so the operator can see
whether the exit engine is net positive and whether triggers need
tightening or loosening.

Storage: data/exit_saves.jsonl (append-only).
Re-check: simple cron via hourly launchd agent, or called from
  main_loop every N minutes.

Operator: 'good safety net especially on 0DTE we need to act fast'.
This module proves whether the safety net is paying off.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional


_log = logging.getLogger(__name__)
_lock = RLock()


# Which exit reasons count as "defensive" — we only track these.
# Hitting the fixed PT or SL isn't a "save", it's just the trade
# running its normal course. We only care about the NEW intelligence
# the operator asked for: protecting profits when momentum reverses.
_DEFENSIVE_PREFIXES = (
    "profit_lock",
    "profit_floor",
    "support_break",
    "resistance_break",
    "chart_lower_highs",
    "chart_higher_lows",
    "vwap_break",
    "vwap_break_up",
    "green_to_red",
    "llm_urgent_close",
    "llm_urgent_trim",
    "momentum_reversal",
    "volume_dry_up",
)


def _is_defensive(reason: str) -> bool:
    if not reason:
        return False
    head = reason.split(":", 1)[0]
    return head.startswith(_DEFENSIVE_PREFIXES)


def _store_path() -> Path:
    try:
        from ..core.data_paths import data_path
        return Path(data_path("logs/exit_saves.jsonl"))
    except Exception:
        return Path("logs/exit_saves.jsonl")


@dataclass
class SaveRecord:
    ts: float
    symbol: str
    underlying: str
    exit_reason: str
    exit_price: float
    qty: int
    peak_pnl_pct: float
    exit_pnl_pct: float
    dte: int
    # Filled after the re-check runs:
    recheck_30m_price: Optional[float] = None
    recheck_2h_price: Optional[float] = None
    saved_usd_30m: Optional[float] = None
    saved_usd_2h: Optional[float] = None


def record_exit(symbol: str, underlying: str, exit_reason: str,
                 exit_price: float, qty: int, peak_pnl_pct: float,
                 exit_pnl_pct: float, dte: int) -> None:
    """Log a defensive exit. No-op if the exit reason isn't defensive."""
    if not _is_defensive(exit_reason):
        _log.debug("saves_skip_non_defensive reason=%s", exit_reason[:60])
        return
    # Scrub underlying: if caller passed the whole entry-tag blob or an
    # OCC-format option symbol, extract just the ticker.
    import re as _re
    if underlying and ("|" in underlying or "=" in underlying):
        m = _re.search(r"sym=([A-Z]{1,6})\b", underlying)
        underlying = m.group(1) if m else underlying.split("|")[0][:6]
    if underlying == symbol:
        m = _re.match(r"^([A-Z]{1,6})\d{6}[CP]\d{8}$", symbol or "")
        if m:
            underlying = m.group(1)
    try:
        rec = SaveRecord(
            ts=time.time(), symbol=symbol, underlying=underlying,
            exit_reason=exit_reason, exit_price=float(exit_price),
            qty=int(qty), peak_pnl_pct=float(peak_pnl_pct),
            exit_pnl_pct=float(exit_pnl_pct), dte=int(dte),
        )
        path = _store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with path.open("a") as f:
                f.write(json.dumps(asdict(rec)) + "\n")
    except Exception as e:                                  # noqa: BLE001
        _log.info("saves_record_err err=%s", e)


def _read_all() -> List[Dict[str, Any]]:
    path = _store_path()
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with _lock:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _write_all(rows: List[Dict[str, Any]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def recheck_pending(mp) -> int:
    """Walk the saves log, fill in recheck prices for records old
    enough (>30 min) and under 8h (then it's stale news). Returns
    count updated. `mp` is a MultiProvider."""
    rows = _read_all()
    if not rows:
        return 0
    now = time.time()
    updated = 0
    for r in rows:
        age = now - float(r.get("ts", now))
        # Re-check the 30-min mark between 30-180 min after exit
        if 1800 <= age < 10800 and r.get("recheck_30m_price") is None:
            px = _fetch_contract_price(mp, r["symbol"])
            if px is not None and px > 0:
                r["recheck_30m_price"] = px
                r["saved_usd_30m"] = (
                    (float(r["exit_price"]) - px)
                    * int(r["qty"]) * 100
                )
                updated += 1
        # Re-check the 2h mark between 2h-8h after exit
        if 7200 <= age < 28800 and r.get("recheck_2h_price") is None:
            px = _fetch_contract_price(mp, r["symbol"])
            if px is not None and px > 0:
                r["recheck_2h_price"] = px
                r["saved_usd_2h"] = (
                    (float(r["exit_price"]) - px)
                    * int(r["qty"]) * 100
                )
                updated += 1
    if updated:
        _write_all(rows)
    return updated


def _fetch_contract_price(mp, occ_symbol: str) -> Optional[float]:
    """Pull current mid for an OCC option symbol. Returns None on fail."""
    try:
        # Quote path — most providers support options via latest_quote
        q = mp.latest_quote(occ_symbol)
        if q is not None and q.mid and q.mid > 0:
            return float(q.mid)
    except Exception:
        pass
    return None


def summary(since_hours: int = 24) -> Dict[str, Any]:
    """Aggregate saves over the last N hours."""
    rows = _read_all()
    cutoff = time.time() - since_hours * 3600
    recent = [r for r in rows if float(r.get("ts", 0)) >= cutoff]
    saved_30m = sum(float(r.get("saved_usd_30m") or 0)
                      for r in recent if r.get("saved_usd_30m") is not None)
    saved_2h = sum(float(r.get("saved_usd_2h") or 0)
                     for r in recent if r.get("saved_usd_2h") is not None)
    # "Wins" = saves where we did save money; "losses" = we'd have done
    # better holding.
    wins_30m = [r for r in recent
                 if (r.get("saved_usd_30m") or 0) > 0]
    regrets_30m = [r for r in recent
                    if (r.get("saved_usd_30m") or 0) < 0]
    # Top save + top regret for highlight reel
    def _best(lst, key):
        if not lst:
            return None
        return max(lst, key=lambda r: abs(r.get(key) or 0))
    return {
        "window_hours":  since_hours,
        "n_exits":       len(recent),
        "n_rechecked":   sum(1 for r in recent
                              if r.get("recheck_30m_price") is not None),
        "saved_usd_30m": round(saved_30m, 2),
        "saved_usd_2h":  round(saved_2h, 2),
        "n_wins_30m":    len(wins_30m),
        "n_regrets_30m": len(regrets_30m),
        "top_save":      _best(wins_30m, "saved_usd_30m"),
        "top_regret":    _best(regrets_30m, "saved_usd_30m"),
        "by_reason":     _by_reason(recent),
        "by_dte_bucket": _by_dte(recent),
    }


def _by_reason(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = (r.get("exit_reason") or "").split(":")[0]
        if not key:
            continue
        d = out.setdefault(key, {"count": 0, "saved_usd_30m": 0.0})
        d["count"] += 1
        d["saved_usd_30m"] += float(r.get("saved_usd_30m") or 0)
    # Round + sort
    return {k: {"count": v["count"],
                 "saved_usd_30m": round(v["saved_usd_30m"], 2)}
             for k, v in sorted(out.items(),
                                  key=lambda kv: -kv[1]["saved_usd_30m"])}


def _by_dte(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {
        "0dte":  {"count": 0, "saved_usd_30m": 0.0},
        "short": {"count": 0, "saved_usd_30m": 0.0},
        "swing": {"count": 0, "saved_usd_30m": 0.0},
    }
    for r in rows:
        dte = int(r.get("dte", 0) or 0)
        bucket = ("0dte" if dte == 0 else
                   "short" if dte <= 7 else "swing")
        out[bucket]["count"] += 1
        out[bucket]["saved_usd_30m"] += float(r.get("saved_usd_30m") or 0)
    return {k: {"count": v["count"],
                 "saved_usd_30m": round(v["saved_usd_30m"], 2)}
             for k, v in out.items()}
