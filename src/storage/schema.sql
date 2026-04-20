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

-- ML model predictions for calibration analysis and A/B comparison.
-- `true_class` + `forward_return` + `resolved_at` are filled in by the
-- resolver script after the horizon has passed.
CREATE TABLE IF NOT EXISTS ml_predictions (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    model           TEXT        NOT NULL,            -- e.g. 'lstm-v1'
    pred_class      INTEGER     NOT NULL,            -- 0=bearish, 1=neutral, 2=bullish
    confidence      NUMERIC(10,6) NOT NULL,
    p_bearish       NUMERIC(10,6),
    p_neutral       NUMERIC(10,6),
    p_bullish       NUMERIC(10,6),
    horizon_minutes INTEGER     NOT NULL,            -- for the resolver
    up_thr          NUMERIC(10,6) NOT NULL,
    down_thr        NUMERIC(10,6) NOT NULL,
    entry_price     NUMERIC(18,4),                   -- price at prediction time
    forward_return  NUMERIC(10,6),                   -- (fwd - entry) / entry
    true_class      INTEGER,                          -- same enum as pred_class
    resolved_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_unresolved
  ON ml_predictions (resolved_at, ts) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_ts
  ON ml_predictions (model, ts);

-- Regime-aware ensemble decisions. One row per TradeBot tick-per-symbol
-- where at least one signal fired. Used by analyze_ensemble.py to measure
-- which regime × contributor combinations actually pay off downstream.
CREATE TABLE IF NOT EXISTS ensemble_decisions (
    id                BIGSERIAL PRIMARY KEY,
    ts                TIMESTAMPTZ NOT NULL,
    symbol            TEXT        NOT NULL,
    regime            TEXT        NOT NULL,
    emitted           BOOLEAN     NOT NULL,
    dominant_direction TEXT,
    dominant_score    NUMERIC(10,6),
    opposing_score    NUMERIC(10,6),
    n_inputs          INTEGER     NOT NULL,
    reason            TEXT        NOT NULL,
    contributors      TEXT                                        -- JSON blob
);
CREATE INDEX IF NOT EXISTS idx_ensemble_decisions_ts
  ON ensemble_decisions (ts);
CREATE INDEX IF NOT EXISTS idx_ensemble_decisions_regime
  ON ensemble_decisions (regime, emitted);
