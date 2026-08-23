"""Append-only derived snapshots and physically separate outcome labels."""

import hashlib
import json
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    asset TEXT NOT NULL,
    market_ticker TEXT,
    contract_open REAL,
    contract_close REAL,
    target REAL,
    raw_watermark_event_id TEXT,
    raw_event_ordinal INTEGER NOT NULL,
    features_json TEXT NOT NULL,
    basket_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feature_market_time
ON feature_snapshots(market_ticker, timestamp);
CREATE TABLE IF NOT EXISTS contract_outcomes (
    market_ticker TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    settlement_timestamp REAL NOT NULL,
    result TEXT NOT NULL,
    source_event_id TEXT NOT NULL
);
"""


class FeatureStore:
    def __init__(self, path="kalshi_features_v3.db"):
        self.path = path
        self.connection = None

    def open(self):
        self.connection = sqlite3.connect(self.path)
        self.connection.executescript(SCHEMA)
        for table in ("feature_snapshots", "contract_outcomes"):
            self.connection.executescript(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_immutable_update BEFORE UPDATE ON {table}
            BEGIN SELECT RAISE(ABORT, 'derived history is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS {table}_immutable_delete BEFORE DELETE ON {table}
            BEGIN SELECT RAISE(ABORT, 'derived history is immutable'); END;
            """)
        self.connection.commit()
        return self

    def append_snapshot(self, snapshot):
        features = json.dumps(snapshot["features"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        basket = json.dumps(snapshot["basket"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        identity = json.dumps({
            "timestamp": snapshot["timestamp"], "asset": snapshot["asset"],
            "market": snapshot["market_ticker"], "watermark": snapshot["raw_watermark_event_id"],
            "ordinal": snapshot["raw_event_ordinal"], "features": features, "basket": basket,
        }, sort_keys=True, separators=(",", ":"))
        snapshot_id = hashlib.sha256(identity.encode()).hexdigest()
        self.connection.execute(
            "INSERT INTO feature_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (snapshot_id, snapshot["timestamp"], snapshot["asset"], snapshot["market_ticker"],
             snapshot["contract_open"], snapshot["contract_close"], snapshot["target"],
             snapshot["raw_watermark_event_id"], snapshot["raw_event_ordinal"], features, basket))
        self.connection.commit()
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
        self.connection.commit()
        return True

    def rows(self):
        return self.connection.execute(
            "SELECT snapshot_id,timestamp,asset,market_ticker,contract_open,contract_close,"
            "target,raw_watermark_event_id,raw_event_ordinal,features_json,basket_json "
            "FROM feature_snapshots ORDER BY raw_event_ordinal,asset").fetchall()

    def close(self):
        if self.connection:
            self.connection.close(); self.connection = None

    def __enter__(self): return self.open()
    def __exit__(self, *_): self.close()
