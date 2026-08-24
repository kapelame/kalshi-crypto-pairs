"""Append-only SQLite event store for deterministic stream replay."""

import sqlite3
from pathlib import Path

from .events import EVENT_TABLES, RawEvent


COMMON_COLUMNS = """
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    asset TEXT NOT NULL,
    market_ticker TEXT,
    series_ticker TEXT,
    exchange_timestamp TEXT,
    local_receive_timestamp TEXT NOT NULL,
    processing_timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_payload TEXT NOT NULL,
    contract_open_time TEXT,
    contract_close_time TEXT,
    target REAL,
    sequence INTEGER,
    sequence_generation INTEGER
"""


class RawEventStore:
    def __init__(self, path="kalshi_stream_raw.db"):
        self.path = Path(path)
        self.connection = None

    def open(self):
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        for table in sorted(set(EVENT_TABLES.values())):
            self.connection.execute(f"CREATE TABLE IF NOT EXISTS {table} ({COMMON_COLUMNS})")
            columns = {row[1] for row in self.connection.execute(
                f"PRAGMA table_info({table})")}
            if "sequence_generation" not in columns:
                self.connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN sequence_generation INTEGER")
            self.connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_ticker_time "
                f"ON {table}(market_ticker, local_receive_timestamp)")
            self.connection.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_immutable_update
                BEFORE UPDATE ON {table}
                BEGIN SELECT RAISE(ABORT, 'raw events are immutable'); END
            """)
            self.connection.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_immutable_delete
                BEFORE DELETE ON {table}
                BEGIN SELECT RAISE(ABORT, 'raw events are immutable'); END
            """)
        self.connection.commit()
        return self

    def append(self, event: RawEvent):
        if self.connection is None:
            raise RuntimeError("raw event store is not open")
        table = EVENT_TABLES.get(event.event_type)
        if table is None:
            raise ValueError(f"unsupported event type: {event.event_type}")
        values = (
            event.event_id, event.event_type, event.asset, event.market_ticker,
            event.series_ticker, event.exchange_timestamp,
            event.local_receive_timestamp, event.processing_timestamp,
            event.source, event.payload_json(), event.contract_open_time,
            event.contract_close_time, event.target, event.sequence,
            event.sequence_generation,
        )
        self.connection.execute(
            f"INSERT INTO {table} VALUES ({','.join('?' for _ in values)})", values)
        self.connection.commit()

    def count(self, event_type=None):
        if event_type:
            table = EVENT_TABLES[event_type]
            return self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return sum(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                   for table in set(EVENT_TABLES.values()))

    def counts(self):
        return {kind: self.count(kind) for kind in EVENT_TABLES}

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()
