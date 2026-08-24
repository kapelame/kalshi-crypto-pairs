"""Canonical raw-event ordering shared by replay and live tailing."""

import json
from collections import defaultdict

from streaming.timeutil import parse_timestamp


TABLE_ORDER = [
    "market_lifecycle_events", "contract_reset_events", "ticker_events",
    "trade_events", "orderbook_snapshots", "orderbook_deltas",
    "underlying_price_events",
]
COLUMNS = ("_rowid", "event_id", "event_type", "asset", "market_ticker",
           "series_ticker", "exchange_timestamp", "local_receive_timestamp",
           "processing_timestamp", "source", "raw_payload", "contract_open_time",
           "contract_close_time", "target", "sequence", "sequence_generation")
INGEST_LEDGER = "raw_ingest_log"


def event_order_key(event):
    """Availability order for new captures; timestamp merge for legacy data."""
    if event.get("_ingest_sequence") is not None:
        return (0, event["_ingest_sequence"])
    return (1, parse_timestamp(event["local_receive_timestamp"]),
            parse_timestamp(event["processing_timestamp"]),
            event["_priority"], event["_rowid"])


def event_from_row(row, priority):
    event = dict(zip(COLUMNS, row))
    event["raw_payload"] = json.loads(event["raw_payload"])
    event["_priority"] = priority
    return event


def event_from_persisted(raw_event, persisted):
    """Build the exact canonical event delivered by ingest-ledger replay."""
    table = persisted.event_table
    try:
        priority = TABLE_ORDER.index(table)
    except ValueError as exc:
        raise ValueError(f"unknown persisted event table: {table}") from exc
    event = {
        "_rowid": persisted.source_rowid,
        "event_id": raw_event.event_id,
        "event_type": raw_event.event_type,
        "asset": raw_event.asset,
        "market_ticker": raw_event.market_ticker,
        "series_ticker": raw_event.series_ticker,
        "exchange_timestamp": raw_event.exchange_timestamp,
        "local_receive_timestamp": raw_event.local_receive_timestamp,
        "processing_timestamp": raw_event.processing_timestamp,
        "source": raw_event.source,
        "raw_payload": raw_event.raw_payload,
        "contract_open_time": raw_event.contract_open_time,
        "contract_close_time": raw_event.contract_close_time,
        "target": raw_event.target,
        "sequence": raw_event.sequence,
        "sequence_generation": raw_event.sequence_generation,
        "_priority": priority,
        "_ingest_sequence": persisted.ingest_sequence,
        "_request_started_at": raw_event.request_started_at,
        "_persistence_timestamp": persisted.persistence_timestamp,
    }
    return event


def generation_projection(columns):
    return ("sequence_generation" if "sequence_generation" in columns
            else "NULL AS sequence_generation")


def select_columns(generation):
    return ("rowid,event_id,event_type,asset,market_ticker,series_ticker,"
            "exchange_timestamp,local_receive_timestamp,processing_timestamp,source,"
            "raw_payload,contract_open_time,contract_close_time,target,sequence,"
            f"{generation}")


def has_ingest_ledger(connection):
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (INGEST_LEDGER,)).fetchone() is not None


def table_specs(connection):
    specs = {}
    for priority, table in enumerate(TABLE_ORDER):
        if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,)).fetchone():
            continue
        columns = {row[1] for row in connection.execute(
            f"PRAGMA table_info({table})")}
        specs[table] = (priority, generation_projection(columns))
    return specs


def read_ingest_batch(connection, after_sequence, limit, specs=None):
    """Resolve compact ledger references without duplicating raw payloads."""
    specs = specs or table_specs(connection)
    ledger_rows = connection.execute(
        f"SELECT ingest_sequence,event_id,event_table,source_rowid,"
        f"request_started_at,persistence_timestamp FROM {INGEST_LEDGER} "
        "WHERE ingest_sequence>? ORDER BY ingest_sequence LIMIT ?",
        (after_sequence, limit)).fetchall()
    references = defaultdict(list)
    for _, _, table, rowid, _, _ in ledger_rows:
        references[table].append(rowid)
    resolved = {}
    for table, rowids in references.items():
        if table not in specs:
            raise ValueError(f"ingest ledger references unknown table: {table}")
        priority, generation = specs[table]
        for offset in range(0, len(rowids), 900):
            chunk = rowids[offset:offset + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT {select_columns(generation)} FROM {table} "
                f"WHERE rowid IN ({placeholders})", chunk).fetchall()
            for row in rows:
                resolved[(table, row[0])] = event_from_row(row, priority)
    output = []
    for sequence, event_id, table, rowid, request_started, persisted in ledger_rows:
        event = resolved.get((table, rowid))
        if event is None or event["event_id"] != event_id:
            raise ValueError(
                f"broken ingest ledger reference sequence={sequence} table={table} rowid={rowid}")
        event["_ingest_sequence"] = sequence
        event["_request_started_at"] = request_started
        event["_persistence_timestamp"] = persisted
        output.append(event)
    return output
