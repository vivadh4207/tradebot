"""Per-signal audit log.

One jsonl line per SignalSource.emit() call — whether or not a signal
was actually produced. Captures:

  - source name (momentum, orb, extreme_momentum, tradingview_webhook, ...)
  - symbol
  - timestamp
  - emitted (bool)
  - rationale (if emitted) OR reason_not_emitted
  - confidence (if emitted)
  - side / option_right (if emitted)

Used for post-hoc analysis: which signals fire often but lose, which
block trades unnecessarily, which have degraded over time. The
dashboard can tail the log for a quick live view.

Enabled by setting `TRADEBOT_SIGNAL_AUDIT=1` (off by default to keep
I/O cost zero in the hot path when the operator isn't analyzing).

Not a replacement for the ensemble_decisions journal — this captures
the RAW per-source emit decisions BEFORE the ensemble aggregates them.
The existing journal captures the post-ensemble decision.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_lock = threading.Lock()
_DEFAULT_LOG = "logs/signal_audit.jsonl"


def _enabled() -> bool:
    return os.getenv("TRADEBOT_SIGNAL_AUDIT", "").strip() in ("1", "true", "yes")


def _log_path() -> Path:
    return Path(os.getenv("TRADEBOT_SIGNAL_AUDIT_PATH", _DEFAULT_LOG))


def log_emit(
    source: str,
    symbol: str,
    *,
    emitted: bool,
    confidence: Optional[float] = None,
    rationale: str = "",
    side: Optional[str] = None,
    option_right: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one audit record. Silent when audit is disabled. Never raises."""
    if not _enabled():
        return
    record = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "source": source,
        "symbol": symbol,
        "emitted": bool(emitted),
        "confidence": (round(float(confidence), 4)
                        if confidence is not None else None),
        "rationale": (rationale or "")[:300],
        "side": side,
        "right": option_right,
    }
    if meta:
        # Shallow, JSON-safe copy. Reject non-serializable values.
        safe_meta = {}
        for k, v in meta.items():
            try:
                json.dumps({k: v})
                safe_meta[str(k)] = v
            except Exception:
                safe_meta[str(k)] = str(v)
        record["meta"] = safe_meta

    path = _log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        # Audit failure must never stop the trade loop.
        pass


def audit_source(signal_source):
    """Wrap a SignalSource so its emit() is audited automatically.

    Usage:
        src = audit_source(MomentumSignal())
        sig = src.emit(ctx)   # exactly same behavior, audited behind the scenes

    If auditing is disabled the wrapper is a no-op (zero per-call
    overhead).
    """
    if not _enabled():
        return signal_source

    original_emit = signal_source.emit
    source_name = getattr(signal_source, "name", type(signal_source).__name__)

    def _wrapped(ctx):
        sig = original_emit(ctx)
        try:
            if sig is None:
                log_emit(
                    source=source_name,
                    symbol=getattr(ctx, "symbol", "?"),
                    emitted=False,
                    rationale="",
                )
            else:
                log_emit(
                    source=source_name,
                    symbol=sig.symbol,
                    emitted=True,
                    confidence=sig.confidence,
                    rationale=sig.rationale,
                    side=sig.side.value if hasattr(sig.side, "value") else str(sig.side),
                    option_right=(sig.option_right.value
                                   if sig.option_right and hasattr(sig.option_right, "value")
                                   else str(sig.option_right) if sig.option_right else None),
                    meta=sig.meta,
                )
        except Exception:
            pass
        return sig

    signal_source.emit = _wrapped
    return signal_source


def read_tail(n: int = 100) -> list:
    """Return the last N audit records (most recent last). Used by the
    dashboard to show a live tail. Returns [] if the file is missing."""
    path = _log_path()
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-n:]
        out = []
        for ln in lines:
            if not ln.strip():
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out
    except Exception:
        return []
