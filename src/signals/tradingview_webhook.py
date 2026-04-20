"""TradingView webhook receiver + signal source.

Two pieces in one file:

  1. `ingest()`   — server-side entry called by the FastAPI webhook
     endpoint. Validates the shared secret, validates schema,
     appends one JSON line to `logs/tradingview_queue.jsonl`.

  2. `TradingViewWebhookSignal` — a SignalSource the bot polls on each
     tick. It reads un-consumed alerts from the queue for the current
     symbol, emits a Signal into the ensemble, and marks the alert
     consumed so it's not re-emitted.

Why file-backed: the dashboard (HTTP receiver) and the bot (signal
consumer) are separate processes under launchd/systemd. A file queue
is the simplest correct IPC — no zmq, no redis, no races beyond
"multiple writers occasionally interleave" which jsonl tolerates.

## Expected payload from TradingView

Configure your TradingView alert to POST to:
    http://<host>:8000/webhook/tradingview

With a body shaped like:
    {
      "secret": "<TRADINGVIEW_WEBHOOK_SECRET>",
      "symbol": "SPY",
      "side": "buy"|"sell",
      "reason": "rsi_oversold + vwap_reversion",
      "confidence": 0.7,
      "tf": "5m",
      "spot": 580.25,
      "ts": "{{time}}"
    }

Only `secret`, `symbol`, `side` are required. Everything else is
advisory and fed to the ensemble as metadata.

## Security

This is a public endpoint the instant the dashboard is reachable. The
shared secret check is the ONLY thing between an attacker and the
bot's signal feed. Rotate the secret if it ever leaks.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.types import Signal, Side
from .base import SignalSource, SignalContext


_log = logging.getLogger(__name__)

# Singleton queue path — both writer (dashboard) and reader (bot) use it.
# Override via TRADINGVIEW_QUEUE_PATH for tests / custom deployments.
_DEFAULT_QUEUE_PATH = "logs/tradingview_queue.jsonl"
_queue_lock = threading.Lock()


def _queue_path() -> Path:
    return Path(os.getenv("TRADINGVIEW_QUEUE_PATH",
                           _DEFAULT_QUEUE_PATH))


# ------------------------------------------------------------ ingest side


@dataclass
class IngestResult:
    ok: bool
    status_code: int = 200
    error: Optional[str] = None
    alert_id: Optional[str] = None


def ingest(payload: Dict[str, Any]) -> IngestResult:
    """Server-side: validate + persist a TradingView alert payload.

    Called from the FastAPI webhook handler. Returns a structured
    result so the caller can map to HTTP status cleanly. Never raises.
    """
    secret_expected = (os.getenv("TRADINGVIEW_WEBHOOK_SECRET") or "").strip()
    if not secret_expected:
        return IngestResult(
            ok=False, status_code=503,
            error="TRADINGVIEW_WEBHOOK_SECRET not configured — "
                  "set it in .env and restart the dashboard",
        )
    got = str(payload.get("secret", "")).strip()
    # Constant-time compare — a timing oracle on a secret-length string
    # isn't a huge risk here but the cost of doing it right is zero.
    import hmac
    if not hmac.compare_digest(got, secret_expected):
        return IngestResult(ok=False, status_code=401,
                             error="bad secret")

    symbol = str(payload.get("symbol", "")).strip().upper()
    side_raw = str(payload.get("side", "")).strip().lower()
    if not symbol or side_raw not in ("buy", "sell"):
        return IngestResult(ok=False, status_code=400,
                             error="missing/invalid symbol or side")

    # Allowed universe — refuse alerts for symbols we don't trade.
    # Prevents a noisy alert feed from racing our discipline.
    allowed = _allowed_symbols()
    if allowed and symbol not in allowed:
        return IngestResult(ok=False, status_code=400,
                             error=f"symbol {symbol} not in trading universe {allowed}")

    # Build a canonical record — strip `secret` before we persist.
    alert = {
        "alert_id": f"tv-{int(time.time() * 1000)}",
        "received_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": symbol,
        "side": side_raw,
        "reason": str(payload.get("reason", ""))[:200],
        "confidence": _clamp(float(payload.get("confidence", 0.6)),
                              0.0, 1.0),
        "tf": str(payload.get("tf", ""))[:8],
        "spot": _safe_float(payload.get("spot")),
        "ts": str(payload.get("ts", ""))[:40],
        "raw": {k: v for k, v in payload.items()
                 if k != "secret" and _json_safe(v)},
        "consumed": False,
    }

    path = _queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _queue_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(alert, separators=(",", ":")) + "\n")

    return IngestResult(ok=True, status_code=200, alert_id=alert["alert_id"])


def _allowed_symbols() -> List[str]:
    """Parse the bot's universe out of settings.yaml if present;
    fall back to [SPY, QQQ] if the file can't be read."""
    try:
        from pathlib import Path as _P
        import yaml
        root = _P(__file__).resolve().parents[2]
        with (root / "config" / "settings.yaml").open("r") as f:
            s = yaml.safe_load(f) or {}
        uni = s.get("universe", ["SPY", "QQQ"])
        if isinstance(uni, list):
            return [str(x).upper() for x in uni]
    except Exception:
        pass
    return ["SPY", "QQQ"]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _json_safe(v) -> bool:
    try:
        json.dumps(v)
        return True
    except Exception:
        return False


# ------------------------------------------------------------ signal side


@dataclass
class TradingViewWebhookSignal(SignalSource):
    """Polls the webhook queue and emits unconsumed alerts as Signals.

    One pass per tick per symbol. Idempotent: each alert is consumed
    exactly once (consumed=true written back). Stale alerts (older
    than `ttl_sec`) are also marked consumed without emission so the
    queue doesn't grow unboundedly.
    """
    name: str = "tradingview_webhook"
    ttl_sec: float = 120.0

    def emit(self, ctx: SignalContext) -> Optional[Signal]:
        path = _queue_path()
        if not path.exists():
            return None
        now = time.time()
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        # Read-modify-write is cheap because the queue is small (active
        # signal count is at most a handful at any time).
        with _queue_lock:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                return None

            picked: Optional[Dict[str, Any]] = None
            updated: List[str] = []
            any_change = False
            for ln in lines:
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if rec.get("consumed"):
                    updated.append(ln)
                    continue
                # TTL — drop stale without emitting (TV alerts beyond
                # 2 min are rarely actionable for intraday options)
                rt = rec.get("received_at", "")
                try:
                    age = now - datetime.fromisoformat(rt).timestamp()
                except Exception:
                    age = 0.0
                if age > self.ttl_sec:
                    rec["consumed"] = True
                    rec["consumed_at"] = now_iso
                    rec["consume_reason"] = "expired"
                    updated.append(json.dumps(rec, separators=(",", ":")))
                    any_change = True
                    continue
                if picked is None and rec.get("symbol", "").upper() == ctx.symbol.upper():
                    picked = rec
                    rec = dict(rec, consumed=True,
                               consumed_at=now_iso,
                               consume_reason="emitted")
                    updated.append(json.dumps(rec, separators=(",", ":")))
                    any_change = True
                else:
                    updated.append(ln)

            if any_change:
                # Rewrite file atomically — expired AND emitted records
                # both need to be persisted so we don't re-process them
                # on the next tick.
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text("\n".join(updated) + ("\n" if updated else ""))
                tmp.replace(path)

        if picked is None:
            return None

        side = Side.BUY if picked["side"] == "buy" else Side.SELL
        return Signal(
            source=self.name,
            symbol=picked["symbol"],
            side=side,
            confidence=float(picked.get("confidence", 0.6)),
            rationale=f"tradingview:{picked.get('reason', '')}",
            meta={
                "alert_id": picked.get("alert_id"),
                "tf": picked.get("tf"),
                "spot_at_alert": picked.get("spot"),
                "received_at": picked.get("received_at"),
                "tv_raw": picked.get("raw", {}),
            },
        )
