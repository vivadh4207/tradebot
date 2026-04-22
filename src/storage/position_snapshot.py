"""Position-state persistence for crash reconciliation.

Why: PaperBroker stores positions + cash in memory. On crash+restart we'd
lose that state and potentially re-enter positions the journal shows as
still open, or miss closing positions the broker thinks we hold.

Approach: snapshot the broker state to a small JSON file after every fill
and every mark-to-market. On startup, load the snapshot and restore
positions, cash, day_pnl, total_pnl. Then — crucially — in live mode,
reconcile against the actual broker's `positions()` and halt if there's
a mismatch.

Snapshot path is configured; we use `logs/broker_state.json` by default.
Small (<1 KB typical), atomic write via tmpfile + rename.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_log = logging.getLogger(__name__)


@dataclass
class PositionRecord:
    symbol: str
    qty: int
    avg_price: float
    is_option: bool
    underlying: Optional[str] = None
    strike: Optional[float] = None
    expiry_iso: Optional[str] = None
    right: Optional[str] = None
    multiplier: int = 1
    entry_ts: float = 0.0
    entry_tag: str = ""
    auto_profit_target: Optional[float] = None
    auto_stop_loss: Optional[float] = None
    consecutive_holds: int = 0
    # Trailing-stop state — must persist across restarts or the
    # green-to-red killswitch and profit-lock can't fire on positions
    # held longer than one bot session.
    peak_price: Optional[float] = None
    peak_pnl_pct: Optional[float] = None
    scaled_out: bool = False


@dataclass
class BrokerSnapshot:
    version: int = 1
    saved_at: str = ""
    cash: float = 0.0
    day_pnl: float = 0.0
    total_pnl: float = 0.0
    positions: List[PositionRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "saved_at": self.saved_at,
            "cash": self.cash,
            "day_pnl": self.day_pnl,
            "total_pnl": self.total_pnl,
            "positions": [asdict(p) for p in self.positions],
        }


def save_snapshot(path: str | Path, broker) -> None:
    """Write a broker state snapshot atomically. Silently tolerates I/O errors."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        positions = []
        for pos in broker.positions():
            positions.append(PositionRecord(
                symbol=pos.symbol, qty=pos.qty, avg_price=pos.avg_price,
                is_option=pos.is_option, underlying=pos.underlying,
                strike=pos.strike,
                expiry_iso=pos.expiry.isoformat() if pos.expiry else None,
                right=pos.right.value if pos.right else None,
                multiplier=pos.multiplier,
                entry_ts=pos.entry_ts,
                entry_tag=str((pos.entry_tags or {}).get("tag", "")),
                auto_profit_target=pos.auto_profit_target,
                auto_stop_loss=pos.auto_stop_loss,
                consecutive_holds=int(pos.consecutive_holds),
                peak_price=getattr(pos, "peak_price", None),
                peak_pnl_pct=getattr(pos, "peak_pnl_pct", None),
                scaled_out=bool(getattr(pos, "scaled_out", False)),
            ))
        acct = broker.account()
        snap = BrokerSnapshot(
            saved_at=datetime.now(tz=timezone.utc).isoformat(),
            cash=acct.cash, day_pnl=acct.day_pnl, total_pnl=acct.total_pnl,
            positions=positions,
        )
        # Atomic: write to sibling tmpfile then rename.
        fd, tmp = tempfile.mkstemp(
            prefix=".broker_snap.", suffix=".tmp", dir=str(p.parent)
        )
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(snap.to_dict(), indent=0))
        os.replace(tmp, str(p))
    except Exception as e:
        _log.warning("broker_snapshot_save_failed path=%s err=%s", path, e)


def load_snapshot(path: str | Path) -> Optional[BrokerSnapshot]:
    """Return a BrokerSnapshot or None if the file is missing / corrupt."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        _log.warning("broker_snapshot_parse_failed path=%s err=%s", p, e)
        return None
    try:
        positions = [PositionRecord(**r) for r in data.get("positions", [])]
        return BrokerSnapshot(
            version=int(data.get("version", 1)),
            saved_at=str(data.get("saved_at", "")),
            cash=float(data.get("cash", 0.0)),
            day_pnl=float(data.get("day_pnl", 0.0)),
            total_pnl=float(data.get("total_pnl", 0.0)),
            positions=positions,
        )
    except Exception as e:
        _log.warning("broker_snapshot_corrupt path=%s err=%s", p, e)
        return None


def restore_into_paper_broker(broker, snap: BrokerSnapshot) -> int:
    """Replay a snapshot into a fresh PaperBroker. Returns # positions restored.

    Does NOT re-submit orders; mutates internal dict + cash directly. Caller
    is responsible for holding the broker lock if needed (we take it).
    """
    from ..core.types import Position as Pos, OptionRight

    with getattr(broker, "_lock", _NullCtx()):
        broker._cash = float(snap.cash)
        broker._day_pnl = float(snap.day_pnl)
        broker._total_pnl = float(snap.total_pnl)
        broker._equity = broker._cash + sum(
            r.qty * r.avg_price * r.multiplier for r in snap.positions
        )
        broker._positions = {}
        for r in snap.positions:
            expiry = date.fromisoformat(r.expiry_iso) if r.expiry_iso else None
            right = OptionRight(r.right) if r.right else None
            broker._positions[r.symbol] = Pos(
                symbol=r.symbol, qty=int(r.qty), avg_price=float(r.avg_price),
                is_option=bool(r.is_option), underlying=r.underlying,
                strike=r.strike, expiry=expiry, right=right,
                multiplier=int(r.multiplier), entry_ts=float(r.entry_ts),
                entry_tags={"tag": r.entry_tag} if r.entry_tag else {},
                auto_profit_target=r.auto_profit_target,
                auto_stop_loss=r.auto_stop_loss,
                consecutive_holds=int(r.consecutive_holds),
                peak_price=(float(r.peak_price)
                             if r.peak_price is not None else None),
                peak_pnl_pct=(float(r.peak_pnl_pct)
                                if r.peak_pnl_pct is not None else None),
                scaled_out=bool(r.scaled_out),
            )
    _log.info("broker_snapshot_restored n=%d cash=%.2f", len(snap.positions), snap.cash)
    return len(snap.positions)


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def reconcile_with_live(broker_sim, broker_live) -> Dict[str, Any]:
    """Compare in-memory (sim) book to the live broker's actual positions.

    Returns a structured report:
      {"ok": bool, "missing": [sym, ...], "extra": [sym, ...], "qty_mismatch": [...]}

    `ok=True` means both sides agree. ANY non-empty mismatch list means the
    operator must intervene — do NOT auto-correct silently in live mode.
    """
    live = {p.symbol: p for p in broker_live.positions()}
    sim = {p.symbol: p for p in broker_sim.positions()}
    missing = [s for s in sim if s not in live]        # we think we hold, broker doesn't
    extra = [s for s in live if s not in sim]          # broker holds, we don't
    qty_mismatch = []
    for s, p in sim.items():
        if s in live and int(live[s].qty) != int(p.qty):
            qty_mismatch.append({"symbol": s,
                                  "sim_qty": int(p.qty),
                                  "live_qty": int(live[s].qty)})
    ok = not (missing or extra or qty_mismatch)
    return {"ok": ok, "missing": missing, "extra": extra,
            "qty_mismatch": qty_mismatch,
            "n_sim": len(sim), "n_live": len(live)}
