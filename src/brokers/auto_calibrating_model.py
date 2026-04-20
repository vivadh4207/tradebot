"""AutoCalibratingCostModel — wraps StochasticCostModel and rewrites
its own constants on a schedule from observed fills.

Flow:
  1. Every fill records a calibration row (JSONL).
  2. On a schedule (hourly / daily / manual), `recalibrate()` reads the
     last N hours of fills, runs `propose_tuning()`, and applies the
     proposal to the wrapped model IF the step size is within guardrails.
  3. Every adjustment is appended to `logs/calibration_history.jsonl`
     for audit; significant shifts push a notifier alert.

Guardrails (non-negotiable):
  - Never move any constant more than 30% in a single recalibration.
  - Never drift more than 2x from baseline across all cycles.
  - If calibration stats show < MIN_SAMPLES fills, no change.
  - Hourly calibration: at most 10% step per hour.
  - Daily: at most 30% step per day.

This is a THIN wrapper — all fill logic delegated to the wrapped model.
Thread-safe: a lock protects the constants during adjustment so no
concurrent fill sees inconsistent state mid-write.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ..core.types import Order
from .slippage_model import StochasticCostModel, FillCost, MarketContext


_log = logging.getLogger(__name__)


class AutoCalibratingCostModel:
    """Drop-in replacement for StochasticCostModel with self-tuning."""

    def __init__(
        self,
        inner: Optional[StochasticCostModel] = None,
        *,
        calibration_path: str = "logs/slippage_calibration.jsonl",
        history_path: str = "logs/calibration_history.jsonl",
        min_samples: int = 30,
        max_step_per_cycle: float = 0.30,
        max_drift_from_baseline: float = 2.0,
        notifier=None,
        alert_on_ratio_above: float = 1.5,
        alert_on_ratio_below: float = 0.5,
    ):
        self._inner = inner or StochasticCostModel()
        self.calibration_path = Path(calibration_path)
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_samples = int(min_samples)
        self.max_step_per_cycle = float(max_step_per_cycle)
        self.max_drift_from_baseline = float(max_drift_from_baseline)
        self.notifier = notifier
        self.alert_hi = float(alert_on_ratio_above)
        self.alert_lo = float(alert_on_ratio_below)
        self._lock = threading.RLock()
        self._last_recalibrated: Optional[datetime] = None

        # Baseline (from settings) — never drift more than max_drift from this
        self._baseline = self._current_constants()

    # --- delegate the hot path to the wrapped model under a read-light lock ---
    def fill(self, order: Order, ctx: MarketContext) -> FillCost:
        with self._lock:
            # The inner model's constants are the only mutable state; hold
            # the lock just long enough to read and dispatch.
            return self._inner.fill(order, ctx)

    def _current_constants(self):
        return {
            "half_spread_mult": self._inner.base_half_spread_mult,
            "size_impact_coef": self._inner.size_impact_coef,
            "vix_impact_coef": self._inner.vix_impact_coef,
            "slip_noise_bps": self._inner.random_noise_bps,
            "slip_floor_bps": self._inner.min_slippage_bps,
        }

    def _apply(self, new_constants: dict) -> dict:
        """Apply constants to the inner model under the lock. Returns the
        diff: {name: (old, new)} for every change."""
        changes = {}
        with self._lock:
            cur = self._current_constants()
            for k, v in new_constants.items():
                if k not in cur:
                    continue
                if abs(cur[k] - v) < 1e-6:
                    continue
                # Guardrail: drift from baseline
                base = self._baseline.get(k, cur[k])
                if base > 0 and (v / base > self.max_drift_from_baseline
                                  or v * self.max_drift_from_baseline < base):
                    _log.warning(
                        "calibration_drift_cap_hit field=%s baseline=%.4f proposed=%.4f",
                        k, base, v,
                    )
                    continue
                changes[k] = (cur[k], v)
            # Actually set attributes
            for k, (old, new) in changes.items():
                attr = {
                    "half_spread_mult": "base_half_spread_mult",
                    "size_impact_coef": "size_impact_coef",
                    "vix_impact_coef": "vix_impact_coef",
                    "slip_noise_bps": "random_noise_bps",
                    "slip_floor_bps": "min_slippage_bps",
                }[k]
                setattr(self._inner, attr, float(new))
        return changes

    def recalibrate(self, lookback_hours: float = 24.0,
                    max_step_per_cycle: Optional[float] = None) -> dict:
        """Read recent fills, compute stats, apply adjustments.

        Returns a dict with: stats, proposal, changes (applied), notes.
        Thread-safe; fails soft (empty result on any error).
        """
        from ..analytics.slippage_calibration import (
            load_recent, analyze, propose_tuning,
        )
        max_step = max_step_per_cycle if max_step_per_cycle is not None else self.max_step_per_cycle
        try:
            days = max(lookback_hours / 24.0, 1.0 / 24.0)
            rows = load_recent(self.calibration_path, days=days)
            stats = analyze(rows)
        except Exception as e:
            _log.warning("recalibrate_read_failed: %s", e)
            return {"error": str(e)}

        if stats is None or stats.n < self.min_samples:
            _log.info("recalibrate_insufficient_samples n=%s min=%d",
                       getattr(stats, "n", 0), self.min_samples)
            return {"stats": None, "changes": {},
                    "notes": f"insufficient_samples n={getattr(stats, 'n', 0)}"}

        cur = self._current_constants()
        proposal = propose_tuning(stats, current=cur, max_step=max_step)
        # Apply bounded by max_step_per_cycle in our guardrails
        changes = self._apply(proposal.proposed)

        # Audit trail
        rec = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "lookback_hours": lookback_hours,
            "n_fills": stats.n,
            "mean_predicted_bps": round(stats.mean_predicted, 3),
            "mean_observed_bps": round(stats.mean_observed, 3),
            "ratio": round(stats.mean_ratio, 3),
            "changes": {k: {"old": round(v[0], 4), "new": round(v[1], 4)}
                         for k, v in changes.items()},
            "rationale": proposal.rationale,
        }
        try:
            with self.history_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

        if stats.mean_ratio >= self.alert_hi or (0 < stats.mean_ratio <= self.alert_lo):
            if self.notifier is not None:
                try:
                    self.notifier.notify(
                        f"Slippage calibration ratio={stats.mean_ratio:.2f} "
                        f"(n={stats.n}, pred={stats.mean_predicted:.1f}bps, "
                        f"obs={stats.mean_observed:.1f}bps). Adjusted "
                        f"{len(changes)} constants.",
                        level="warn", title="calibration",
                    )
                except Exception:
                    pass

        self._last_recalibrated = datetime.now(tz=timezone.utc)
        return rec


def start_calibration_scheduler(
    model: AutoCalibratingCostModel,
    *,
    mode: str = "hourly",
    stop_event: Optional[threading.Event] = None,
) -> Optional[threading.Thread]:
    """Kick off a background thread that recalibrates `model` on a schedule.

    mode:
      'hourly' → every hour, step cap 0.10 per cycle
      'daily'  → every 24h, step cap 0.30 per cycle
      'manual' → no-op (caller calls recalibrate() manually)

    Returns the thread (daemon=True) or None in manual mode.
    """
    if mode not in {"hourly", "daily", "manual"}:
        raise ValueError(f"bad mode: {mode!r}")
    if mode == "manual":
        return None

    stop_event = stop_event or threading.Event()
    interval_sec = 3600.0 if mode == "hourly" else 86400.0
    lookback_hours = 6.0 if mode == "hourly" else 24.0
    max_step = 0.10 if mode == "hourly" else 0.30

    def _loop():
        # First recalibration after one interval (not immediately on startup —
        # avoids reacting to stale yesterday's data on first boot).
        while not stop_event.wait(interval_sec):
            try:
                model.recalibrate(
                    lookback_hours=lookback_hours,
                    max_step_per_cycle=max_step,
                )
            except Exception as e:                 # noqa: BLE001
                _log.warning("scheduled_recalibration_failed: %s", e)

    t = threading.Thread(target=_loop, name=f"calibration-{mode}", daemon=True)
    t.start()
    return t
