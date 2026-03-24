-- init_db.sql — TimescaleDB schema initialization
-- Run via: make init-db

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE TABLE IF NOT EXISTS lob_snapshots (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT        NOT NULL,
    bids        JSONB       NOT NULL,
    asks        JSONB       NOT NULL,
    mid_price   FLOAT8
);

CREATE TABLE IF NOT EXISTS trades (
    time    TIMESTAMPTZ NOT NULL,
    symbol  TEXT        NOT NULL,
    price   FLOAT8      NOT NULL,
    qty     FLOAT8      NOT NULL,
    side    CHAR(1)     NOT NULL  -- 'B' buy taker, 'S' sell taker
);

-- Convert to hypertables (TimescaleDB partitioning by time)
SELECT create_hypertable('lob_snapshots', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('trades',        'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');

-- Indexes for fast time-window queries
CREATE INDEX IF NOT EXISTS idx_lob_symbol_time  ON lob_snapshots (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, time DESC);

-- Compression policy: compress chunks older than 7 days (90% size reduction)
SELECT add_compression_policy('lob_snapshots', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('trades',        INTERVAL '7 days', if_not_exists => TRUE);
