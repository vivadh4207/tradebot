"""TradeJournal — persistent store for fills, round-trip trades, and equity.

Two backends:
- SqliteJournal:    default, zero-setup, file-based.
- CockroachJournal: Postgres-wire-compatible, production-grade. Uses
                    `psycopg` (v3) lazily imported.

Both implement the same interface so callers never care which one's behind.
Read path returns `ClosedTrade` rows suitable for computing priors.
"""
from __future__ import annotations

import abc
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import quote

from ..core.types import Fill, Order, Position, Side


def resolve_cockroach_dsn(env: Optional[dict] = None,
                          dsn_env_var: str = "COCKROACH_DSN") -> str:
    """Return a CockroachDB DSN from env.

    Preference:
      1. COCKROACH_DSN if set (used as-is).
      2. Assembled from COCKROACH_HOST / USER / PASSWORD / DATABASE / PORT
         / SSLMODE / SSLROOTCERT / CLUSTER.

    Raises RuntimeError with an actionable message if neither is configured.
    """
    env = os.environ if env is None else env
    dsn = env.get(dsn_env_var, "").strip()
    if dsn:
        return dsn
    host = env.get("COCKROACH_HOST", "").strip()
    user = env.get("COCKROACH_USER", "").strip()
    password = env.get("COCKROACH_PASSWORD", "").strip()
    # Strip placeholder-looking values so we fail loudly instead of connecting
    # to "your-cluster-host" and getting DNS errors.
    for val in (host, user, password):
        if val.startswith("<") and val.endswith(">"):
            raise RuntimeError(
                "CockroachDB env fields contain placeholder values. "
                "Fill in the real values in .env (not .env.example)."
            )
    if not host or not user or not password:
        raise RuntimeError(
            "CockroachDB is not configured. Set COCKROACH_DSN, or fill "
            "COCKROACH_HOST / COCKROACH_USER / COCKROACH_PASSWORD "
            "(and optionally COCKROACH_DATABASE, COCKROACH_PORT, "
            "COCKROACH_SSLMODE, COCKROACH_SSLROOTCERT, COCKROACH_CLUSTER) "
            "in .env."
        )
    port = env.get("COCKROACH_PORT", "26257").strip() or "26257"
    database = env.get("COCKROACH_DATABASE", "tradebot").strip() or "tradebot"
    sslmode = env.get("COCKROACH_SSLMODE", "verify-full").strip() or "verify-full"
    sslrootcert = env.get("COCKROACH_SSLROOTCERT", "").strip()
    cluster = env.get("COCKROACH_CLUSTER", "").strip()

    user_q = quote(user, safe="")
    pw_q = quote(password, safe="")
    params = [f"sslmode={sslmode}"]
    if sslrootcert:
        params.append(f"sslrootcert={quote(sslrootcert, safe='/')}")
    if cluster:
        # multi-tenant Serverless routing
        params.append(f"options={quote('--cluster=' + cluster, safe='')}")
    qs = "&".join(params)
    return f"postgresql://{user_q}:{pw_q}@{host}:{port}/{database}?{qs}"


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

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ------------------------------------------------------------------ CockroachDB
class CockroachJournal(TradeJournal):
    """Postgres-wire journal. Works for CockroachDB, Postgres, Neon, Supabase.

    Connection string examples:
      postgresql://user:pass@host:26257/defaultdb?sslmode=verify-full&sslrootcert=/path/ca.crt
      postgresql://user:pass@free-tier.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full
    """

    def __init__(self, dsn: str):
        try:
            import psycopg
        except ImportError as e:
            raise ImportError(
                "psycopg (v3) is required for CockroachJournal. "
                "Install with: pip install 'psycopg[binary]'"
            ) from e
        self._psycopg = psycopg
        self._conn = psycopg.connect(dsn, autocommit=True)

    def init_schema(self) -> None:
        with self._conn.cursor() as cur:
            for stmt in _schema_sql().split(";"):
                s = stmt.strip()
                if s:
                    cur.execute(s)

    def record_fill(self, fill: Fill) -> None:
        o = fill.order
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO fills (ts, symbol, side, qty, price, fee, is_option, tag) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (_to_utc(fill.ts), o.symbol, o.side.value, fill.qty,
                 fill.price, fill.fee, o.is_option, o.tag),
            )

    def record_trade(self, t: ClosedTrade) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO trades (symbol, opened_at, closed_at, side, qty, "
                "entry_price, exit_price, pnl, pnl_pct, entry_tag, exit_reason, is_option) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (t.symbol, _to_utc(t.opened_at),
                 _to_utc(t.closed_at) if t.closed_at else None,
                 t.side, t.qty, t.entry_price, t.exit_price,
                 t.pnl, t.pnl_pct, t.entry_tag, t.exit_reason, t.is_option),
            )

    def record_equity(self, ts: datetime, equity: float, cash: float, day_pnl: float) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPSERT INTO equity_curve (ts, equity, cash, day_pnl) VALUES (%s,%s,%s,%s)",
                (_to_utc(ts), equity, cash, day_pnl),
            )

    def closed_trades(self, since: Optional[datetime] = None) -> List[ClosedTrade]:
        q = ("SELECT symbol, opened_at, closed_at, side, qty, entry_price, exit_price, "
             "pnl, pnl_pct, entry_tag, exit_reason, is_option FROM trades "
             "WHERE closed_at IS NOT NULL")
        params: tuple = ()
        if since is not None:
            q += " AND closed_at >= %s"
            params = (_to_utc(since),)
        q += " ORDER BY closed_at"
        with self._conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()
        out: List[ClosedTrade] = []
        for r in rows:
            out.append(ClosedTrade(
                symbol=r[0], opened_at=r[1], closed_at=r[2],
                side=r[3], qty=int(r[4]),
                entry_price=float(r[5]),
                exit_price=float(r[6]) if r[6] is not None else None,
                pnl=float(r[7]) if r[7] is not None else None,
                pnl_pct=float(r[8]) if r[8] is not None else None,
                entry_tag=r[9], exit_reason=r[10], is_option=bool(r[11]),
            ))
        return out

    def equity_series(self, since: Optional[datetime] = None,
                      limit: int = 5000) -> List[tuple]:
        q = "SELECT ts, equity, cash, day_pnl FROM equity_curve"
        params: tuple = ()
        if since is not None:
            q += " WHERE ts >= %s"
            params = (_to_utc(since),)
        q += " ORDER BY ts ASC LIMIT %s"
        params = params + (int(limit),)
        with self._conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()
        return [(r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                 float(r[1]), float(r[2]), float(r[3])) for r in rows]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ------------------------------------------------------------------ factory
def build_journal(backend: str = "sqlite",
                  sqlite_path: str = "logs/tradebot.sqlite",
                  dsn_env_var: str = "COCKROACH_DSN") -> TradeJournal:
    """Build a journal from a backend name.

    backend == 'sqlite'     → SqliteJournal(sqlite_path)
    backend == 'cockroach'  → CockroachJournal(os.environ[dsn_env_var])
    """
    backend = backend.lower()
    if backend == "sqlite":
        j = SqliteJournal(sqlite_path)
    elif backend in {"cockroach", "cockroachdb", "postgres", "postgresql"}:
        dsn = resolve_cockroach_dsn(dsn_env_var=dsn_env_var)
        j = CockroachJournal(dsn)
    else:
        raise ValueError(f"Unknown journal backend: {backend}")
    j.init_schema()
    return j
