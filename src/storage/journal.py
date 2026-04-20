"""TradeJournal — persistent local SQLite store for fills, round-trip
trades, equity history, ML predictions, and ensemble decisions.

Single backend (SQLite): zero-setup, file-based, no external
dependencies, no cloud request quotas. A prior version supported
CockroachDB; it was removed because the bot's write frequency
outgrew the free-tier request-unit cap. SQLite on a local disk
handles 10k+ writes/sec without breathing hard — well beyond what
the bot generates.
"""
from __future__ import annotations

import abc
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from ..core.types import Fill, Order, Position, Side


@dataclass
class ClosedTrade:
    symbol: str
    opened_at: datetime
    closed_at: Optional[datetime]
    side: str                 # 'long' | 'short'
    qty: int
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    entry_tag: Optional[str]
    exit_reason: Optional[str]
    is_option: bool


@dataclass
class EnsembleRecord:
    id: Optional[int]
    ts: datetime
    symbol: str
    regime: str
    emitted: bool
    dominant_direction: Optional[str]
    dominant_score: Optional[float]
    opposing_score: Optional[float]
    n_inputs: int
    reason: str
    contributors: Optional[str] = None    # JSON string


@dataclass
class MLPrediction:
    id: Optional[int]
    ts: datetime
    symbol: str
    model: str
    pred_class: int                     # 0=bearish, 1=neutral, 2=bullish
    confidence: float
    p_bearish: float
    p_neutral: float
    p_bullish: float
    horizon_minutes: int
    up_thr: float
    down_thr: float
    entry_price: Optional[float] = None
    forward_return: Optional[float] = None
    true_class: Optional[int] = None
    resolved_at: Optional[datetime] = None


def _schema_sql() -> str:
    here = Path(__file__).parent / "schema.sql"
    return here.read_text()


def _to_utc(ts: float | datetime | None) -> datetime:
    if ts is None:
        return datetime.now(tz=timezone.utc)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


class TradeJournal(abc.ABC):
    """Persist fills, closed trades, and equity snapshots. Backend-agnostic."""

    @abc.abstractmethod
    def init_schema(self) -> None: ...

    @abc.abstractmethod
    def record_fill(self, fill: Fill) -> None: ...

    @abc.abstractmethod
    def record_trade(self, t: ClosedTrade) -> None: ...

    @abc.abstractmethod
    def record_equity(self, ts: datetime, equity: float, cash: float, day_pnl: float) -> None: ...

    @abc.abstractmethod
    def closed_trades(self, since: Optional[datetime] = None) -> List[ClosedTrade]: ...

    @abc.abstractmethod
    def equity_series(self, since: Optional[datetime] = None,
                      limit: int = 5000) -> List[tuple]:
        """Return [(ts_iso_utc, equity, cash, day_pnl), ...] ordered by ts asc."""

    @abc.abstractmethod
    def record_ml_prediction(self, p: MLPrediction) -> int:
        """Insert a prediction. Returns the row id."""

    @abc.abstractmethod
    def unresolved_ml_predictions(self, older_than: datetime,
                                   limit: int = 1000) -> List[MLPrediction]:
        """Predictions whose horizon has passed but `true_class` is NULL."""

    @abc.abstractmethod
    def resolve_ml_prediction(self, prediction_id: int,
                               forward_return: float, true_class: int) -> None:
        """Fill in forward_return / true_class / resolved_at for an id."""

    @abc.abstractmethod
    def resolved_ml_predictions(self, model: Optional[str] = None,
                                 since: Optional[datetime] = None,
                                 limit: int = 20000) -> List[MLPrediction]:
        """Return resolved predictions for analysis."""

    @abc.abstractmethod
    def record_ensemble_decision(self, e: EnsembleRecord) -> int:
        """Persist one EnsembleCoordinator decision. Returns row id."""

    @abc.abstractmethod
    def ensemble_decisions(self, since: Optional[datetime] = None,
                            regime: Optional[str] = None,
                            emitted: Optional[bool] = None,
                            limit: int = 20000) -> List[EnsembleRecord]: ...

    @abc.abstractmethod
    def close(self) -> None: ...


# ------------------------------------------------------------------ SQLite
class SqliteJournal(TradeJournal):
    """File-backed journal. Perfect for paper trading on a laptop."""

    def __init__(self, path: str = "logs/tradebot.sqlite"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(p), isolation_level=None, check_same_thread=False)
        # WAL is preferred (concurrent reads during writes), but some mounts
        # (FUSE, network, sandbox overlays) refuse it with disk I/O error —
        # silently fall back to the default rollback journal in that case.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            self._conn.execute("PRAGMA journal_mode=DELETE;")

    def init_schema(self) -> None:
        # SQLite needs tiny adaptations of the Postgres schema
        sql = _schema_sql()
        sql = sql.replace("BIGSERIAL", "INTEGER")
        sql = sql.replace("TIMESTAMPTZ", "TEXT")
        sql = sql.replace("NUMERIC(18,4)", "REAL")
        sql = sql.replace("NUMERIC(10,6)", "REAL")
        sql = sql.replace("BOOLEAN", "INTEGER")
        for stmt in sql.split(";"):
            s = stmt.strip()
            if s:
                self._conn.execute(s)

    def record_fill(self, fill: Fill) -> None:
        o = fill.order
        self._conn.execute(
            "INSERT INTO fills (ts, symbol, side, qty, price, fee, is_option, tag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_to_utc(fill.ts).isoformat(), o.symbol, o.side.value,
             fill.qty, fill.price, fill.fee, int(o.is_option), o.tag),
        )

    def record_trade(self, t: ClosedTrade) -> None:
        self._conn.execute(
            "INSERT INTO trades (symbol, opened_at, closed_at, side, qty, "
            "entry_price, exit_price, pnl, pnl_pct, entry_tag, exit_reason, is_option) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (t.symbol, _to_utc(t.opened_at).isoformat(),
             _to_utc(t.closed_at).isoformat() if t.closed_at else None,
             t.side, t.qty, t.entry_price, t.exit_price, t.pnl, t.pnl_pct,
             t.entry_tag, t.exit_reason, int(t.is_option)),
        )

    def record_equity(self, ts: datetime, equity: float, cash: float, day_pnl: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO equity_curve (ts, equity, cash, day_pnl) VALUES (?, ?, ?, ?)",
            (_to_utc(ts).isoformat(), equity, cash, day_pnl),
        )

    def closed_trades(self, since: Optional[datetime] = None) -> List[ClosedTrade]:
        cur = self._conn.cursor()
        if since is not None:
            cur.execute(
                "SELECT symbol, opened_at, closed_at, side, qty, entry_price, exit_price, "
                "pnl, pnl_pct, entry_tag, exit_reason, is_option FROM trades "
                "WHERE closed_at IS NOT NULL AND closed_at >= ? ORDER BY closed_at",
                (_to_utc(since).isoformat(),),
            )
        else:
            cur.execute(
                "SELECT symbol, opened_at, closed_at, side, qty, entry_price, exit_price, "
                "pnl, pnl_pct, entry_tag, exit_reason, is_option FROM trades "
                "WHERE closed_at IS NOT NULL ORDER BY closed_at"
            )
        rows = cur.fetchall()
        out: List[ClosedTrade] = []
        for r in rows:
            out.append(ClosedTrade(
                symbol=r[0],
                opened_at=datetime.fromisoformat(r[1]),
                closed_at=datetime.fromisoformat(r[2]) if r[2] else None,
                side=r[3], qty=int(r[4]),
                entry_price=float(r[5]),
                exit_price=float(r[6]) if r[6] is not None else None,
                pnl=float(r[7]) if r[7] is not None else None,
                pnl_pct=float(r[8]) if r[8] is not None else None,
                entry_tag=r[9], exit_reason=r[10],
                is_option=bool(r[11]),
            ))
        return out

    def equity_series(self, since: Optional[datetime] = None,
                      limit: int = 5000) -> List[tuple]:
        cur = self._conn.cursor()
        if since is not None:
            cur.execute(
                "SELECT ts, equity, cash, day_pnl FROM equity_curve "
                "WHERE ts >= ? ORDER BY ts ASC LIMIT ?",
                (_to_utc(since).isoformat(), int(limit)),
            )
        else:
            cur.execute(
                "SELECT ts, equity, cash, day_pnl FROM equity_curve "
                "ORDER BY ts ASC LIMIT ?", (int(limit),)
            )
        return [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in cur.fetchall()]

    # ---- ML predictions ----
    def record_ml_prediction(self, p: MLPrediction) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO ml_predictions (ts, symbol, model, pred_class, confidence, "
            "p_bearish, p_neutral, p_bullish, horizon_minutes, up_thr, down_thr, "
            "entry_price) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (_to_utc(p.ts).isoformat(), p.symbol, p.model, int(p.pred_class),
             float(p.confidence), float(p.p_bearish), float(p.p_neutral),
             float(p.p_bullish), int(p.horizon_minutes),
             float(p.up_thr), float(p.down_thr),
             None if p.entry_price is None else float(p.entry_price)),
        )
        return int(cur.lastrowid)

    def unresolved_ml_predictions(self, older_than: datetime,
                                   limit: int = 1000) -> List[MLPrediction]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, ts, symbol, model, pred_class, confidence, "
            "p_bearish, p_neutral, p_bullish, horizon_minutes, up_thr, down_thr, "
            "entry_price FROM ml_predictions "
            "WHERE resolved_at IS NULL AND ts <= ? ORDER BY ts ASC LIMIT ?",
            (_to_utc(older_than).isoformat(), int(limit)),
        )
        out = []
        for r in cur.fetchall():
            out.append(MLPrediction(
                id=int(r[0]),
                ts=datetime.fromisoformat(r[1]),
                symbol=r[2], model=r[3],
                pred_class=int(r[4]), confidence=float(r[5]),
                p_bearish=float(r[6]), p_neutral=float(r[7]), p_bullish=float(r[8]),
                horizon_minutes=int(r[9]),
                up_thr=float(r[10]), down_thr=float(r[11]),
                entry_price=float(r[12]) if r[12] is not None else None,
            ))
        return out

    def resolve_ml_prediction(self, prediction_id: int,
                               forward_return: float, true_class: int) -> None:
        self._conn.execute(
            "UPDATE ml_predictions SET forward_return=?, true_class=?, resolved_at=? "
            "WHERE id=?",
            (float(forward_return), int(true_class),
             _to_utc(datetime.now(tz=timezone.utc)).isoformat(),
             int(prediction_id)),
        )

    def resolved_ml_predictions(self, model: Optional[str] = None,
                                 since: Optional[datetime] = None,
                                 limit: int = 20000) -> List[MLPrediction]:
        q = ("SELECT id, ts, symbol, model, pred_class, confidence, "
             "p_bearish, p_neutral, p_bullish, horizon_minutes, up_thr, down_thr, "
             "entry_price, forward_return, true_class, resolved_at "
             "FROM ml_predictions WHERE resolved_at IS NOT NULL")
        params: list = []
        if model is not None:
            q += " AND model = ?"
            params.append(model)
        if since is not None:
            q += " AND ts >= ?"
            params.append(_to_utc(since).isoformat())
        q += " ORDER BY ts ASC LIMIT ?"
        params.append(int(limit))
        cur = self._conn.cursor()
        cur.execute(q, tuple(params))
        out = []
        for r in cur.fetchall():
            out.append(MLPrediction(
                id=int(r[0]),
                ts=datetime.fromisoformat(r[1]),
                symbol=r[2], model=r[3],
                pred_class=int(r[4]), confidence=float(r[5]),
                p_bearish=float(r[6]), p_neutral=float(r[7]), p_bullish=float(r[8]),
                horizon_minutes=int(r[9]),
                up_thr=float(r[10]), down_thr=float(r[11]),
                entry_price=float(r[12]) if r[12] is not None else None,
                forward_return=float(r[13]) if r[13] is not None else None,
                true_class=int(r[14]) if r[14] is not None else None,
                resolved_at=datetime.fromisoformat(r[15]) if r[15] else None,
            ))
        return out

    # ---- ensemble ----
    def record_ensemble_decision(self, e: EnsembleRecord) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO ensemble_decisions (ts, symbol, regime, emitted, "
            "dominant_direction, dominant_score, opposing_score, n_inputs, "
            "reason, contributors) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (_to_utc(e.ts).isoformat(), e.symbol, e.regime, int(bool(e.emitted)),
             e.dominant_direction,
             None if e.dominant_score is None else float(e.dominant_score),
             None if e.opposing_score is None else float(e.opposing_score),
             int(e.n_inputs), e.reason, e.contributors),
        )
        return int(cur.lastrowid)

    def ensemble_decisions(self, since: Optional[datetime] = None,
                            regime: Optional[str] = None,
                            emitted: Optional[bool] = None,
                            limit: int = 20000) -> List[EnsembleRecord]:
        q = ("SELECT id, ts, symbol, regime, emitted, dominant_direction, "
             "dominant_score, opposing_score, n_inputs, reason, contributors "
             "FROM ensemble_decisions WHERE 1=1")
        params: list = []
        if since is not None:
            q += " AND ts >= ?"
            params.append(_to_utc(since).isoformat())
        if regime is not None:
            q += " AND regime = ?"
            params.append(regime)
        if emitted is not None:
            q += " AND emitted = ?"
            params.append(1 if emitted else 0)
        q += " ORDER BY ts ASC LIMIT ?"
        params.append(int(limit))
        cur = self._conn.cursor()
        cur.execute(q, tuple(params))
        return [EnsembleRecord(
            id=int(r[0]),
            ts=datetime.fromisoformat(r[1]),
            symbol=r[2], regime=r[3],
            emitted=bool(r[4]),
            dominant_direction=r[5],
            dominant_score=None if r[6] is None else float(r[6]),
            opposing_score=None if r[7] is None else float(r[7]),
            n_inputs=int(r[8]), reason=r[9], contributors=r[10],
        ) for r in cur.fetchall()]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ------------------------------------------------------------------ factory
def build_journal(sqlite_path: str = "logs/tradebot.sqlite",
                  **_ignored_kwargs) -> TradeJournal:
    """Build the journal. Local SQLite file, zero setup.

    `**_ignored_kwargs` absorbs legacy callers that still pass
    `backend=`, `dsn_env_var=`, or `cockroach_schema=` so an old
    scheduled script doesn't crash on the first run after the
    Cockroach rip-out. Those args are no-ops.
    """
    j = SqliteJournal(sqlite_path)
    j.init_schema()
    return j
