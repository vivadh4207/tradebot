"""LLM-autotrade queue — shared file-backed handoff between the
research agent (writer) and the main trading loop's
LLMOriginationSignal (reader).

Design:
  - JSONL file, append-only for writes.
  - Reader uses mtime to skip redundant scans (zero I/O when queue
    hasn't changed).
  - Consumed-idea tracker is a small JSON set (idea_id → ts).
  - A "kill switch" file (logs/llm_autotrade.kill) disables the whole
    feature regardless of LLM_AUTOTRADE env, written by the Discord
    `!llm-autotrade off` command for instant stop.

Safety invariants enforced here (before main loop ever sees an idea):
  - Idea must not be older than max_age_min (default 30m)
  - Idea confidence must be in allowed set (default {medium, high})
  - Idea must reference a REAL integer strike for SPY/QQQ (no .4x)
  - Kill switch absent
  - Daily cap not exceeded
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


_log = logging.getLogger(__name__)

_QUEUE_PATH = "logs/llm_autotrade_queue.jsonl"
_CONSUMED_PATH = "logs/llm_autotrade_consumed.json"
_KILL_SWITCH_PATH = "logs/llm_autotrade.kill"
_COUNTER_PATH = "logs/llm_autotrade_daily_counter.json"
_CAP_OVERRIDE_PATH = "logs/llm_autotrade_cap_override.json"

# Hard safety ceiling the operator cannot exceed even via override.
# Paper mode means no capital at risk, but unbounded autotrade would
# still burn API rate limits + distort backtest interpretation.
_MAX_DAILY_CAP_HARD_LIMIT = 50


# Whole-dollar strikes only for these symbols — rejects LLM hallucinations
# like 716.4. Add more here as you start researching other symbols.
_INTEGER_STRIKE_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA"}


@dataclass
class QueuedIdea:
    """One LLM-proposed trade idea ready for main-loop pickup."""
    id: str                          # content hash
    ts: str                          # ISO
    symbol: str
    direction: str                   # "call" | "put" | "spread"
    confidence: str                  # "low" | "medium" | "high"
    strike: Optional[float] = None
    expiry: Optional[str] = None
    entry: Optional[float] = None
    profit_target: Optional[float] = None
    stop_loss: Optional[float] = None
    time_horizon: str = ""
    rationale: str = ""
    source: str = "research_agent"

    @staticmethod
    def make_id(symbol: str, direction: str, strike: Optional[float],
                 expiry: Optional[str]) -> str:
        key = f"{symbol}|{direction}|{strike}|{expiry or ''}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]


# ----------------------------------------------------------- writer (agent)


def write_ideas(ideas: List[QueuedIdea], data_root: Optional[Path] = None
                ) -> int:
    """Append ideas to the queue. Called by the research agent when
    LLM_AUTOTRADE=1 is set in env. Returns number written.

    Dedupe by id — an agent run that re-proposes the same idea doesn't
    double-enter.
    """
    if not ideas:
        return 0
    path = _resolve_path(_QUEUE_PATH, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Collect existing IDs so we don't re-queue the same idea.
    existing: Set[str] = set()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        existing.add(json.loads(line).get("id", ""))
                    except Exception:
                        continue
        except Exception:
            pass
    n_written = 0
    with path.open("a", encoding="utf-8") as f:
        for idea in ideas:
            if idea.id in existing:
                continue
            f.write(json.dumps(asdict(idea), separators=(",", ":")) + "\n")
            n_written += 1
    return n_written


# ----------------------------------------------------------- reader (signal)


class LLMAutotradeQueue:
    """Reader side. Thread-safe. Called by LLMOriginationSignal.emit()
    on every tick but cheap — uses mtime check + cached parse."""

    def __init__(self,
                 *, data_root: Optional[Path] = None,
                 max_age_min: int = 30,
                 allowed_confidences: Optional[Set[str]] = None,
                 max_trades_per_day: int = 3):
        self._data_root = data_root
        self._max_age_min = int(max_age_min)
        self._allowed = allowed_confidences or {"medium", "high"}
        self._max_per_day = int(max_trades_per_day)
        self._lock = threading.Lock()
        self._mtime_seen = 0.0
        self._cached_rows: List[QueuedIdea] = []
        self._consumed: Set[str] = self._load_consumed()

    @property
    def queue_path(self) -> Path:
        return _resolve_path(_QUEUE_PATH, self._data_root)

    @property
    def consumed_path(self) -> Path:
        return _resolve_path(_CONSUMED_PATH, self._data_root)

    @property
    def kill_switch_path(self) -> Path:
        return _resolve_path(_KILL_SWITCH_PATH, self._data_root)

    @property
    def counter_path(self) -> Path:
        return _resolve_path(_COUNTER_PATH, self._data_root)

    @property
    def cap_override_path(self) -> Path:
        return _resolve_path(_CAP_OVERRIDE_PATH, self._data_root)

    # ---------- gating ----------

    def is_killed(self) -> bool:
        return self.kill_switch_path.exists()

    def set_killed(self, killed: bool) -> None:
        p = self.kill_switch_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if killed:
            p.write_text(datetime.now(tz=timezone.utc).isoformat())
        else:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def daily_counter(self) -> int:
        p = self.counter_path
        if not p.exists():
            return 0
        try:
            d = json.loads(p.read_text())
            today = datetime.now(tz=timezone.utc).date().isoformat()
            if d.get("date") != today:
                return 0
            return int(d.get("count", 0))
        except Exception:
            return 0

    def current_cap(self) -> int:
        """Return the ACTIVE daily cap. Override file wins over the
        config default; capped at _MAX_DAILY_CAP_HARD_LIMIT."""
        p = self.cap_override_path
        if not p.exists():
            return self._max_per_day
        try:
            d = json.loads(p.read_text())
            v = int(d.get("cap", self._max_per_day))
            return max(1, min(_MAX_DAILY_CAP_HARD_LIMIT, v))
        except Exception:
            return self._max_per_day

    def set_cap_override(self, new_cap: Optional[int]) -> int:
        """Persist an override cap for today and beyond. Pass None to
        clear override (go back to config default). Returns the
        effective cap that will apply."""
        p = self.cap_override_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if new_cap is None:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            return self._max_per_day
        new_cap = max(1, min(_MAX_DAILY_CAP_HARD_LIMIT, int(new_cap)))
        p.write_text(json.dumps({"cap": new_cap}))
        return new_cap

    def reset_daily_counter(self) -> None:
        """Zero today's trade counter. Useful when operator bumps the
        cap and wants the new quota to reflect immediately."""
        try:
            self.counter_path.unlink()
        except FileNotFoundError:
            pass

    def _bump_counter(self) -> int:
        p = self.counter_path
        p.parent.mkdir(parents=True, exist_ok=True)
        today = datetime.now(tz=timezone.utc).date().isoformat()
        with self._lock:
            cur = self.daily_counter()
            new = cur + 1
            p.write_text(json.dumps({"date": today, "count": new}))
        return new

    # ---------- read ----------

    def _maybe_refresh(self) -> None:
        p = self.queue_path
        if not p.exists():
            self._cached_rows = []
            return
        try:
            mtime = p.stat().st_mtime
        except Exception:
            return
        if mtime <= self._mtime_seen and self._cached_rows:
            return                                  # unchanged — skip parse
        rows: List[QueuedIdea] = []
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        rows.append(QueuedIdea(**{
                            k: d.get(k) for k in QueuedIdea.__dataclass_fields__
                        }))
                    except Exception:
                        continue
        except Exception:
            return
        self._cached_rows = rows
        self._mtime_seen = mtime

    def _is_fresh(self, idea: QueuedIdea) -> bool:
        try:
            ts = datetime.fromisoformat(idea.ts.replace("Z", "+00:00"))
            age = (datetime.now(tz=timezone.utc) - ts).total_seconds() / 60
            return age <= self._max_age_min
        except Exception:
            return False

    def _is_allowed(self, idea: QueuedIdea) -> bool:
        if idea.confidence not in self._allowed:
            return False
        if idea.direction not in ("call", "put"):
            return False                            # skip spreads for now
        # Reject fractional strikes for whole-dollar symbols.
        if (idea.symbol in _INTEGER_STRIKE_SYMBOLS
                and idea.strike is not None
                and abs(idea.strike - round(idea.strike)) > 0.001):
            _log.info("llm_autotrade_reject_fractional_strike sym=%s k=%s",
                       idea.symbol, idea.strike)
            return False
        return True

    # ---------- public ----------

    def next_idea_for(self, symbol: str) -> Optional[QueuedIdea]:
        """Pop the freshest unconsumed valid idea for `symbol`.

        Returns None if killed, daily cap hit, or nothing eligible.
        Caller gets exclusive right to the idea — a successful pop
        marks it consumed before returning so duplicate ticks don't
        re-pick the same idea."""
        if self.is_killed():
            return None
        if self.daily_counter() >= self.current_cap():
            return None
        with self._lock:
            self._maybe_refresh()
            for idea in self._cached_rows:
                if idea.id in self._consumed:
                    continue
                if idea.symbol != symbol:
                    continue
                if not self._is_fresh(idea):
                    continue
                if not self._is_allowed(idea):
                    self._mark_consumed(idea.id)
                    continue
                self._mark_consumed(idea.id)
                self._bump_counter()
                return idea
        return None

    def peek_state(self) -> Dict[str, Any]:
        """Status view for !llm-autotrade status."""
        with self._lock:
            self._maybe_refresh()
            fresh = [i for i in self._cached_rows
                     if i.id not in self._consumed and self._is_fresh(i)]
        return {
            "killed": self.is_killed(),
            "daily_count": self.daily_counter(),
            "daily_cap": self.current_cap(),
            "daily_cap_default": self._max_per_day,
            "cap_override_active": self.current_cap() != self._max_per_day,
            "queue_fresh": len(fresh),
            "queue_total": len(self._cached_rows),
            "consumed": len(self._consumed),
            "max_age_min": self._max_age_min,
            "allowed_confidences": sorted(self._allowed),
        }

    # ---------- consumed tracking ----------

    def _load_consumed(self) -> Set[str]:
        p = self.consumed_path
        if not p.exists():
            return set()
        try:
            d = json.loads(p.read_text())
            return set(d.get("ids", []))
        except Exception:
            return set()

    def _save_consumed(self) -> None:
        p = self.consumed_path
        p.parent.mkdir(parents=True, exist_ok=True)
        # Keep at most last 1024 consumed IDs
        keep = list(self._consumed)[-1024:]
        p.write_text(json.dumps({"ids": keep}))

    def _mark_consumed(self, idea_id: str) -> None:
        self._consumed.add(idea_id)
        self._save_consumed()


# ----------------------------------------------------------- helpers


def _resolve_path(rel: str, data_root: Optional[Path] = None) -> Path:
    if data_root is not None:
        return Path(data_root) / rel
    try:
        from ..core.data_paths import data_path
        return Path(data_path(rel))
    except Exception:
        return Path(rel)
