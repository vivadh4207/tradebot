-- Schema is the same for SQLite and CockroachDB/Postgres with minor syntax
-- tweaks. The Python layer issues the SQL with the correct dialect.

-- Every raw fill the broker reports
CREATE TABLE IF NOT EXISTS fills (
    id           BIGSERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ NOT NULL,
    symbol       TEXT        NOT NULL,
    side         TEXT        NOT NULL,              -- 'buy' | 'sell'
    qty          INTEGER     NOT NULL,
    price        NUMERIC(18,4) NOT NULL,
    fee          NUMERIC(18,4) NOT NULL DEFAULT 0,
    is_option    BOOLEAN     NOT NULL,
    tag          TEXT        NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_fills_symbol_ts ON fills (symbol, ts);

-- Realized round-trips (entry fill + exit fill collapsed into one row)
CREATE TABLE IF NOT EXISTS trades (
    id            BIGSERIAL PRIMARY KEY,
    symbol        TEXT        NOT NULL,
    opened_at     TIMESTAMPTZ NOT NULL,
    closed_at     TIMESTAMPTZ,
    side          TEXT        NOT NULL,             -- 'long' | 'short'
    qty           INTEGER     NOT NULL,
    entry_price   NUMERIC(18,4) NOT NULL,
    exit_price    NUMERIC(18,4),
    pnl           NUMERIC(18,4),                    -- dollars, net of fees
    pnl_pct       NUMERIC(10,6),
    entry_tag     TEXT,
    exit_reason   TEXT,
    is_option     BOOLEAN     NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades (closed_at);

-- Per-tick portfolio equity snapshots
CREATE TABLE IF NOT EXISTS equity_curve (
    ts        TIMESTAMPTZ PRIMARY KEY,
    equity    NUMERIC(18,4) NOT NULL,
    cash      NUMERIC(18,4) NOT NULL,
    day_pnl   NUMERIC(18,4) NOT NULL
);
