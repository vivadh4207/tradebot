"""Live tradebot.out digest — summarize the last N minutes of log lines
into a single Discord-sized message.

Used by scripts/post_log_summary.py (cron/systemd-timer) and by the
!summary Discord command. Intentionally stdlib-only and cheap — this
runs every hour so it can't afford to load pandas.

The digest focuses on what an operator watching the bot wants to see
at a glance:
  - Mode, regime, VIX (current)
  - Positions count
  - What the signal engine has been doing (entries / exits / skip reasons)
  - Error + warning counts (not every line — just grouped)
  - Latest audit score, if any

Format is markdown so it renders cleanly in Discord embeds.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Most recent structured-log line looks like:
#   2026-04-20T15:59:27.221981Z [info     ] event_name  key=val key=val
_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+"
    r"\[(?P<level>\w+)\s*\]\s+"
    r"(?P<event>\S+)\s*(?P<rest>.*)$"
)

# Python-stdlib logging line (WARNING / ERROR) looks like:
#   2026-04-20 11:33:52,665 [WARNING] src.x.y :: event_name key=...
_LEGACY_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2}),\d+\s+"
    r"\[(?P<level>\w+)\]\s+(?P<module>[\w.]+)\s*::\s*(?P<rest>.*)$"
)


@dataclass
class Digest:
    window_minutes: int
    n_lines: int = 0
    n_parsed: int = 0

    # Current state (last seen value)
    last_regime: Optional[str] = None
    last_vix: Optional[float] = None
    last_mode: Optional[str] = None
    last_universe: List[str] = field(default_factory=list)
    last_audit_health: Optional[int] = None
    last_audit_age_min: Optional[int] = None
    positions_count: Optional[int] = None

    # Signal engine activity in the window
    entries_fired: int = 0
    exits_fired: int = 0
    skips_by_reason: Counter = field(default_factory=Counter)
    highest_score: float = 0.0
    highest_score_symbol: str = ""

    # Infrastructure signals
    n_warnings: int = 0
    n_errors: int = 0
    warn_by_topic: Counter = field(default_factory=Counter)
    shutdown_signals: int = 0
    alpaca_network_errors: int = 0

    # For pruning messages that are always present but not useful
    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"**tradebot summary · last {self.window_minutes} min**")

        # -- Current state
        state_bits = []
        if self.last_mode:
            state_bits.append(f"mode={self.last_mode}")
        if self.last_universe:
            state_bits.append(f"universe={','.join(self.last_universe)}")
        if self.last_regime:
            state_bits.append(f"regime={self.last_regime}")
        if self.last_vix is not None:
            state_bits.append(f"vix={self.last_vix:.1f}")
        if self.positions_count is not None:
            state_bits.append(f"positions={self.positions_count}")
        if state_bits:
            lines.append("· " + " · ".join(state_bits))

        # -- Signal activity
        if (self.entries_fired or self.exits_fired or self.skips_by_reason
                or self.highest_score):
            sig_bits = []
            if self.entries_fired:
                sig_bits.append(f"entries={self.entries_fired}")
            if self.exits_fired:
                sig_bits.append(f"exits={self.exits_fired}")
            if self.skips_by_reason:
                top = self.skips_by_reason.most_common(3)
                skip_summary = ", ".join(f"{k}×{v}" for k, v in top)
                total = sum(self.skips_by_reason.values())
                sig_bits.append(f"skips={total} ({skip_summary})")
            if self.highest_score:
                sig_bits.append(
                    f"best_score={self.highest_score:.2f}"
                    + (f" ({self.highest_score_symbol})"
                       if self.highest_score_symbol else "")
                )
            lines.append("· " + " · ".join(sig_bits))

        # -- Infrastructure
        infra_bits = []
        if self.n_errors:
            infra_bits.append(f"errors={self.n_errors}")
        if self.n_warnings:
            top = self.warn_by_topic.most_common(3)
            topic = ", ".join(f"{k}×{v}" for k, v in top) if top else ""
            infra_bits.append(
                f"warnings={self.n_warnings}" + (f" ({topic})" if topic else "")
            )
        if self.shutdown_signals:
            infra_bits.append(f"sigterms={self.shutdown_signals}")
        if self.alpaca_network_errors:
            infra_bits.append(f"alpaca_net_err={self.alpaca_network_errors}")
        if infra_bits:
            lines.append("· " + " · ".join(infra_bits))

        # -- Audit
        if self.last_audit_health is not None:
            age = (f" ({self.last_audit_age_min}m ago)"
                   if self.last_audit_age_min is not None else "")
            lines.append(
                f"· last_audit={self.last_audit_health}/100{age}"
            )

        if self.n_parsed == 0:
            lines.append("· (no parseable events in window)")

        return "\n".join(lines)


def _parse_rest(rest: str) -> Dict[str, str]:
    """Extract key=value pairs from a structlog tail like
    'reason=below_threshold:0.700<0.85 regime=range_lowvol symbol=QQQ'."""
    out: Dict[str, str] = {}
    for m in re.finditer(r"(\w+)=([^\s]+)", rest or ""):
        out[m.group(1)] = m.group(2)
    return out


def _parse_ts(raw: str) -> Optional[datetime]:
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _extract_audit_age(log_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Return (overall_health, minutes_since) for the most recent audit,
    reading strategy_audit.jsonl (sibling of tradebot.out)."""
    audit_path = log_path.parent / "strategy_audit.jsonl"
    if not audit_path.exists():
        return None, None
    try:
        last_line = ""
        with audit_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if not last_line:
            return None, None
        rec = json.loads(last_line)
        health = int(rec.get("overall_health", 0))
        ts_raw = rec.get("ts", "")
        ts = _parse_ts(ts_raw)
        if ts is None:
            return health, None
        mins = int((datetime.now(tz=timezone.utc) - ts).total_seconds() // 60)
        return health, max(0, mins)
    except Exception:
        return None, None


def build_digest(log_path: Path, window_minutes: int = 60) -> Digest:
    """Scan the tail of tradebot.out over the last `window_minutes` and
    return a Digest. Safe on missing file — returns an empty digest."""
    d = Digest(window_minutes=window_minutes)
    if not log_path.exists():
        return d

    cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=window_minutes)

    # Read the last ~2 MB — enough for 1-2 hours of logs at typical rates.
    try:
        size = log_path.stat().st_size
        read_from = max(0, size - 2_000_000)
        with log_path.open("rb") as f:
            f.seek(read_from)
            # discard partial first line if we seeked mid-line
            if read_from > 0:
                f.readline()
            raw = f.read().decode("utf-8", errors="replace")
    except Exception:
        return d

    for line in raw.splitlines():
        d.n_lines += 1
        m = _LINE_RE.match(line)
        legacy = None
        if m is None:
            legacy = _LEGACY_RE.match(line)
            if legacy is None:
                continue
            # Best-effort timestamp for legacy line
            ts_raw = f"{legacy.group('date')}T{legacy.group('time')}+00:00"
            event_parts = (legacy.group("rest") or "").split(None, 1)
            event = event_parts[0] if event_parts else ""
            rest = event_parts[1] if len(event_parts) > 1 else ""
            level = legacy.group("level").lower()
        else:
            ts_raw = m.group("ts")
            event = m.group("event")
            rest = m.group("rest") or ""
            level = m.group("level").lower()

        ts = _parse_ts(ts_raw)
        if ts is None or ts < cutoff:
            continue
        d.n_parsed += 1
        kv = _parse_rest(rest)

        # -- Current-state trackers
        if event == "data_adapter":
            # startup marker
            pass
        elif event in ("regime_updated", "regime_classified"):
            if "regime" in kv:
                d.last_regime = kv["regime"]
        elif event == "vix_update" or event == "vix_snapshot":
            try:
                d.last_vix = float(kv.get("vix", "nan"))
            except Exception:
                pass
        elif event == "positions_snapshot" or event == "broker_snapshot_restored":
            try:
                d.positions_count = int(kv.get("n", "0"))
            except Exception:
                pass

        # -- Signal engine activity
        if event == "exec_chain_pass":
            d.entries_fired += 1
        elif event == "exit_placed" or event == "position_closed":
            d.exits_fired += 1
        elif event == "ensemble_skip":
            # reason looks like 'below_threshold:0.700<0.85'
            reason = kv.get("reason", "unknown")
            # collapse the score so we don't get 1000 buckets
            reason_bucket = reason.split(":", 1)[0]
            d.skips_by_reason[reason_bucket] += 1
            # regime tag
            if "regime" in kv:
                d.last_regime = kv["regime"]
            # try to pick up the score
            m2 = re.search(r"(\d+\.\d+)", reason)
            if m2:
                try:
                    score = float(m2.group(1))
                    if score > d.highest_score:
                        d.highest_score = score
                        d.highest_score_symbol = kv.get("symbol", "")
                except Exception:
                    pass
        elif event == "shutdown_signal":
            d.shutdown_signals += 1

        # -- Infrastructure
        if level in ("warn", "warning"):
            d.n_warnings += 1
            if "alpaca" in (event or "") or "alpaca" in rest:
                d.alpaca_network_errors += 1
                d.warn_by_topic["alpaca"] += 1
            elif "notifier" in (event or "") or "notifier" in rest:
                d.warn_by_topic["notifier"] += 1
            elif "reconcile" in (event or "") or "reconcile" in rest:
                d.warn_by_topic["reconcile"] += 1
            else:
                # bucket by event prefix up to first `_`
                bucket = (event or "warn").split("_", 1)[0]
                d.warn_by_topic[bucket or "warn"] += 1
        elif level == "error":
            d.n_errors += 1

    # Audit enrichment — independent of the log window.
    health, age = _extract_audit_age(log_path)
    d.last_audit_health = health
    d.last_audit_age_min = age
    return d
