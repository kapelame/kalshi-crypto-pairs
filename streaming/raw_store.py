"""Append-only SQLite event store for deterministic stream replay."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .events import EVENT_TABLES, RawEvent
from .timeutil import iso_utc


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

INGEST_LEDGER = "raw_ingest_log"


@dataclass(frozen=True)
class PersistedEvent:
    ingest_sequence: int | None
    persistence_timestamp: str
    event_table: str
    source_rowid: int


class RawEventStore:
    def __init__(self, path="kalshi_stream_raw.db", enable_ingest=None):
        self.path = Path(path)
        self.connection = None
        self.ingest_enabled = False
        self.enable_ingest = enable_ingest

    def open(self):
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        existing = {row[0] for row in self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        legacy_has_rows = False
        if INGEST_LEDGER not in existing:
            for table in set(EVENT_TABLES.values()) & existing:
                if self.connection.execute(
                        f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                    legacy_has_rows = True
                    break
        default_ingest = INGEST_LEDGER in existing or not legacy_has_rows
        self.ingest_enabled = (default_ingest if self.enable_ingest is None else
                               bool(self.enable_ingest))
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
        if self.ingest_enabled:
            self.connection.execute(f"""
                CREATE TABLE IF NOT EXISTS {INGEST_LEDGER} (
                    ingest_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_table TEXT NOT NULL,
                    source_rowid INTEGER NOT NULL,
                    request_started_at TEXT,
                    persistence_timestamp TEXT NOT NULL,
                    UNIQUE(event_table, source_rowid)
                )
            """)
            self.connection.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {INGEST_LEDGER}_immutable_update
                BEFORE UPDATE ON {INGEST_LEDGER}
                BEGIN SELECT RAISE(ABORT, 'raw ingest entries are immutable'); END
            """)
            self.connection.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {INGEST_LEDGER}_immutable_delete
                BEFORE DELETE ON {INGEST_LEDGER}
                BEGIN SELECT RAISE(ABORT, 'raw ingest entries are immutable'); END
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
        persistence_timestamp = iso_utc()
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            cursor = self.connection.execute(
                f"INSERT INTO {table} VALUES ({','.join('?' for _ in values)})", values)
            source_rowid = cursor.lastrowid
            ingest_sequence = None
            if self.ingest_enabled:
                ledger = self.connection.execute(
                    f"INSERT INTO {INGEST_LEDGER} "
                    "(event_id,event_table,source_rowid,request_started_at,persistence_timestamp) "
                    "VALUES (?,?,?,?,?)",
                    (event.event_id, table, source_rowid, event.request_started_at,
                     persistence_timestamp))
                ingest_sequence = ledger.lastrowid
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return PersistedEvent(ingest_sequence, persistence_timestamp,
                              table, source_rowid)

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
