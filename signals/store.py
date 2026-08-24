"""Batched append-only derived checkpoints with normalized basket state."""

import hashlib
import json
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS basket_snapshots (
    basket_snapshot_id TEXT PRIMARY KEY, timestamp REAL NOT NULL,
    raw_watermark_event_id TEXT, raw_event_ordinal INTEGER NOT NULL,
    eligible_count INTEGER NOT NULL, coherent_window INTEGER NOT NULL,
    contract_window_id TEXT, regime_label TEXT, basket_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id TEXT PRIMARY KEY, timestamp REAL NOT NULL, asset TEXT NOT NULL,
    market_ticker TEXT, contract_open REAL, contract_close REAL, target REAL,
    raw_watermark_event_id TEXT, raw_event_ordinal INTEGER NOT NULL,
    snapshot_kind TEXT NOT NULL DEFAULT 'all', checkpoint_seconds INTEGER,
    scheduled_timestamp REAL, checkpoint_delay_ms REAL,
    eligible INTEGER NOT NULL, excluded_reasons_json TEXT NOT NULL,
    contract_window_id TEXT, basket_snapshot_id TEXT, features_json TEXT NOT NULL,
    FOREIGN KEY(basket_snapshot_id) REFERENCES basket_snapshots(basket_snapshot_id)
);
CREATE INDEX IF NOT EXISTS idx_feature_market_checkpoint
ON feature_snapshots(market_ticker, checkpoint_seconds, timestamp);
CREATE TABLE IF NOT EXISTS contract_outcomes (
    market_ticker TEXT PRIMARY KEY, asset TEXT NOT NULL,
    settlement_timestamp REAL NOT NULL, result TEXT NOT NULL,
    source_event_id TEXT NOT NULL
);
"""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class FeatureStore:
    def __init__(self, path="kalshi_features_v3.db", batch_size=500):
        self.path = path
        self.batch_size = batch_size
        self.connection = None
        self.pending = 0
        self.snapshots_written = 0
        self.baskets_written = 0
        self.outcomes_written = 0

    def open(self):
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(SCHEMA)
        for table in ("feature_snapshots", "basket_snapshots", "contract_outcomes"):
            self.connection.executescript(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_immutable_update BEFORE UPDATE ON {table}
            BEGIN SELECT RAISE(ABORT, 'derived history is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS {table}_immutable_delete BEFORE DELETE ON {table}
            BEGIN SELECT RAISE(ABORT, 'derived history is immutable'); END;
            """)
        self.connection.commit()
        self.connection.execute("BEGIN")
        return self

    def _maybe_flush(self):
        if self.pending >= self.batch_size:
            self.flush()

    def flush(self):
        if self.connection is None or not self.pending:
            return
        self.connection.commit()
        self.pending = 0
        self.connection.execute("BEGIN")

    def append_snapshot(self, snapshot, snapshot_kind=None, checkpoint_seconds=None,
                        scheduled_timestamp=None):
        kind = snapshot_kind or snapshot.get("snapshot_kind", "all")
        checkpoint = checkpoint_seconds if checkpoint_seconds is not None else snapshot.get("checkpoint_seconds")
        scheduled = scheduled_timestamp if scheduled_timestamp is not None else snapshot.get("scheduled_timestamp")
        delay = None if scheduled is None else (snapshot["timestamp"] - scheduled) * 1000
        features = _json(snapshot["features"])
        basket = snapshot["basket"]
        basket_json = _json(basket)
        basket_identity = _json({"timestamp": snapshot["timestamp"],
            "watermark": snapshot["raw_watermark_event_id"],
            "ordinal": snapshot["raw_event_ordinal"], "basket": basket_json})
        basket_id = hashlib.sha256(basket_identity.encode()).hexdigest()
        before = self.connection.total_changes
        self.connection.execute(
            "INSERT OR IGNORE INTO basket_snapshots VALUES (?,?,?,?,?,?,?,?,?)",
            (basket_id, snapshot["timestamp"], snapshot["raw_watermark_event_id"],
             snapshot["raw_event_ordinal"], basket.get("eligible_count", 0),
             int(bool(basket.get("coherent_window"))), basket.get("contract_window_id"),
             snapshot["features"].get("regime_label"), basket_json))
        self.baskets_written += self.connection.total_changes - before
        identity = _json({"timestamp": snapshot["timestamp"], "asset": snapshot["asset"],
            "market": snapshot["market_ticker"], "watermark": snapshot["raw_watermark_event_id"],
            "ordinal": snapshot["raw_event_ordinal"], "kind": kind,
            "checkpoint": checkpoint, "scheduled": scheduled,
            "features": features, "basket_id": basket_id})
        snapshot_id = hashlib.sha256(identity.encode()).hexdigest()
        reasons = snapshot["features"].get("excluded_reasons", [])
        self.connection.execute(
            "INSERT INTO feature_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (snapshot_id, snapshot["timestamp"], snapshot["asset"], snapshot["market_ticker"],
             snapshot["contract_open"], snapshot["contract_close"], snapshot["target"],
             snapshot["raw_watermark_event_id"], snapshot["raw_event_ordinal"], kind,
             checkpoint, scheduled, delay, int(bool(snapshot["features"].get("eligible"))),
             _json(reasons), snapshot["features"].get("contract_window_id"), basket_id, features))
        self.pending += 1
        self.snapshots_written += 1
        self._maybe_flush()
        return snapshot_id

    def append_outcome(self, ticker, asset, timestamp, result, event_id):
        if result not in ("yes", "no"):
            return False
        existing = self.connection.execute(
            "SELECT result FROM contract_outcomes WHERE market_ticker=?", (ticker,)).fetchone()
        if existing:
            if existing[0] != result:
                raise ValueError(f"conflicting immutable outcome for {ticker}")
            return False
        self.connection.execute("INSERT INTO contract_outcomes VALUES (?,?,?,?,?)",
                                (ticker, asset, timestamp, result, event_id))
        self.pending += 1
        self.outcomes_written += 1
        self._maybe_flush()
        return True

    def rows(self):
        self.flush()
        return self.connection.execute(
            "SELECT f.snapshot_id,f.timestamp,f.asset,f.market_ticker,f.contract_open,"
            "f.contract_close,f.target,f.raw_watermark_event_id,f.raw_event_ordinal,"
            "f.features_json,b.basket_json FROM feature_snapshots f "
            "JOIN basket_snapshots b ON b.basket_snapshot_id=f.basket_snapshot_id "
            "ORDER BY f.raw_event_ordinal,f.asset").fetchall()

    def close(self):
        if self.connection:
            try:
                self.flush(); self.connection.commit()
            except Exception:
                self.connection.rollback(); raise
            finally:
                self.connection.close(); self.connection = None

    def __enter__(self): return self.open()
    def __exit__(self, exc_type, *_):
        if exc_type and self.connection:
            self.connection.rollback(); self.connection.close(); self.connection = None
        else:
            self.close()
