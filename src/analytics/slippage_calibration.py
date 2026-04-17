"""Continuous slippage calibration — record per-fill predictions vs realized.

Motivation: our `StochasticCostModel` produces a predicted slippage in
bps per fill. If that prediction systematically over- or under-shoots
reality, backtests are misleading and live-trade costs will surprise us.

Approach: every fill writes a JSONL line with:
  - predicted bps + its components (half-spread, size impact, vix impact, noise)
  - fill price + the mid the model saw
  - context (vix, bid/ask, symbol, side, qty)

Analysis script (scripts/calibrate_slippage.py) reads this file weekly
and produces:
  - mean / median / p95 bps actually realized
  - breakdown of components
  - proposed tuning deltas for the model constants

File format: one JSON object per line, append-only, thread-safe.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional


def _iso(dt: Optional[datetime] = None) -> str:
    return (dt or datetime.now(tz=timezone.utc)).isoformat()


class SlippageLogger:
    """Append-only JSONL logger. One line per fill.

    Usage:
        logger = SlippageLogger("logs/slippage_calibration.jsonl")
        logger.record(fill, predicted_bps=3.2, components={...}, mid=500.0, vix=18)
    """

    def __init__(self, path: str = "logs/slippage_calibration.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        *,
        ts: Optional[datetime] = None,
        symbol: str,
        side: str,
        qty: int,
        is_option: bool,
        limit_price: float,
        executed_price: float,
        predicted_bps: float,
        components: Dict[str, float],
        mid: float,
        vix: float,
        tag: str = "",
    ) -> None:
        """Record ONE fill's calibration payload."""
        # Observed bps: how much did the executed price deviate from mid?
        # Sign convention: positive = worse than mid (i.e. we paid more on BUY,
        # received less on SELL).
        if mid > 0:
            slip = (executed_price - mid) if side == "buy" else (mid - executed_price)
            observed_bps = (slip / mid) * 10_000.0
        else:
            observed_bps = 0.0
        row = {
            "ts": _iso(ts),
            "symbol": symbol,
            "side": side,
            "qty": int(qty),
            "is_option": bool(is_option),
            "limit_price": float(limit_price),
            "executed_price": float(executed_price),
            "mid": float(mid),
            "predicted_bps": float(predicted_bps),
            "observed_bps": float(observed_bps),
            "components": {k: float(v) for k, v in components.items()},
            "vix": float(vix),
            "tag": str(tag),
        }
        line = json.dumps(row, separators=(",", ":"))
        # Append-only write with OS-level atomicity for a single write() call
        # that's ≤ PIPE_BUF in size. The line is always small (<2KB).
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


@dataclass
class CalibrationStats:
    n: int
    mean_predicted: float
    mean_observed: float
    median_observed: float
    p95_observed: float
    p99_observed: float
    mean_ratio: float                       # observed / predicted
    per_component_mean: Dict[str, float]    # half_spread_bps, size_impact_bps, ...
    per_symbol_mean: Dict[str, float]       # symbol → mean observed bps
    days_covered: float


@dataclass
class TuningProposal:
    current: Dict[str, float]
    proposed: Dict[str, float]
    rationale: List[str] = field(default_factory=list)


def load_recent(path: str | Path, days: int = 7) -> List[Dict[str, Any]]:
    """Read all rows within the last `days` days."""
    p = Path(path)
    if not p.exists():
        return []
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                ts = datetime.fromisoformat(r["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    rows.append(r)
            except Exception:
                continue
    return rows


def analyze(rows: List[Dict[str, Any]]) -> Optional[CalibrationStats]:
    if not rows:
        return None
    preds = [r.get("predicted_bps", 0.0) for r in rows]
    obs = [r.get("observed_bps", 0.0) for r in rows]
    # Components
    comps: Dict[str, List[float]] = {}
    for r in rows:
        for k, v in (r.get("components") or {}).items():
            comps.setdefault(k, []).append(float(v))
    per_comp = {k: (sum(v) / len(v)) if v else 0.0 for k, v in comps.items()}
    # Per-symbol
    by_sym: Dict[str, List[float]] = {}
    for r in rows:
        by_sym.setdefault(r.get("symbol", "?"), []).append(float(r.get("observed_bps", 0.0)))
    per_sym = {k: (sum(v) / len(v)) if v else 0.0 for k, v in by_sym.items()}
    # Quantiles without numpy (keep this module dep-light)
    sorted_obs = sorted(obs)
    def _q(p: float) -> float:
        if not sorted_obs:
            return 0.0
        idx = min(len(sorted_obs) - 1, max(0, int(round(p * (len(sorted_obs) - 1)))))
        return float(sorted_obs[idx])
    mean_p = sum(preds) / len(preds) if preds else 0.0
    mean_o = sum(obs) / len(obs) if obs else 0.0
    ratio = (mean_o / mean_p) if mean_p > 1e-9 else 0.0
    # Days covered
    if rows:
        first = datetime.fromisoformat(rows[0]["ts"])
        last = datetime.fromisoformat(rows[-1]["ts"])
        days = max((last - first).total_seconds() / 86400.0, 1e-9)
    else:
        days = 0.0
    return CalibrationStats(
        n=len(rows),
        mean_predicted=mean_p,
        mean_observed=mean_o,
        median_observed=float(median(obs)) if obs else 0.0,
        p95_observed=_q(0.95),
        p99_observed=_q(0.99),
        mean_ratio=ratio,
        per_component_mean=per_comp,
        per_symbol_mean=per_sym,
        days_covered=days,
    )


def propose_tuning(stats: CalibrationStats,
                    current: Optional[Dict[str, float]] = None,
                    max_step: float = 0.30) -> TuningProposal:
    """Suggest incremental tweaks to StochasticCostModel constants.

    Rules (conservative — never more than +/- 30% per week):
      - If mean_ratio > 1.2 (model under-predicting), bump
        `half_spread_mult` and `size_impact_coef` up proportionally.
      - If mean_ratio < 0.8 (over-predicting, too pessimistic), back off.
      - If one component dominates abnormally, flag for manual tuning.
    """
    current = current or {
        "half_spread_mult": 1.0,
        "size_impact_coef": 0.25,
        "vix_impact_coef": 0.015,
        "slip_noise_bps": 0.5,
        "slip_floor_bps": 0.5,
    }
    proposed = dict(current)
    rationale: List[str] = []

    if stats.n < 30:
        rationale.append(
            f"insufficient_samples: only {stats.n} fills — keeping current "
            f"constants (need 30+ for meaningful calibration)"
        )
        return TuningProposal(current=current, proposed=proposed, rationale=rationale)

    r = stats.mean_ratio
    if r > 1.2:
        step = min(max_step, (r - 1.0) * 0.5)
        proposed["half_spread_mult"] = round(current["half_spread_mult"] * (1 + step), 3)
        proposed["size_impact_coef"] = round(current["size_impact_coef"] * (1 + step), 3)
        rationale.append(
            f"model under-predicting by {(r - 1) * 100:.1f}% → bump "
            f"half_spread_mult + size_impact_coef by {step * 100:.1f}%"
        )
    elif r < 0.8 and r > 0:
        step = min(max_step, (1.0 - r) * 0.5)
        proposed["half_spread_mult"] = round(current["half_spread_mult"] * (1 - step), 3)
        proposed["size_impact_coef"] = round(current["size_impact_coef"] * (1 - step), 3)
        rationale.append(
            f"model over-predicting by {(1 - r) * 100:.1f}% → back off "
            f"half_spread_mult + size_impact_coef by {step * 100:.1f}%"
        )
    else:
        rationale.append(
            f"mean_observed / mean_predicted = {r:.2f} — within "
            f"[0.8, 1.2] tolerance; no tuning"
        )

    # Dominant-component check: if one component is > 80% of total, flag
    total_comp = sum(stats.per_component_mean.values()) or 1e-9
    for comp, v in stats.per_component_mean.items():
        if v / total_comp > 0.8:
            rationale.append(
                f"WARNING: {comp} accounts for {v / total_comp * 100:.0f}% of "
                f"slippage — review manually"
            )
    return TuningProposal(current=current, proposed=proposed, rationale=rationale)
