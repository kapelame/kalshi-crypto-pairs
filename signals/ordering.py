"""Canonical raw-event ordering shared by replay and live tailing."""

import json

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


def event_order_key(event):
    """Canonical causal key: receive time, processing time, table, rowid."""
    return (parse_timestamp(event["local_receive_timestamp"]),
            parse_timestamp(event["processing_timestamp"]),
            event["_priority"], event["_rowid"])


def event_from_row(row, priority):
    event = dict(zip(COLUMNS, row))
    event["raw_payload"] = json.loads(event["raw_payload"])
    event["_priority"] = priority
    return event


def generation_projection(columns):
    return ("sequence_generation" if "sequence_generation" in columns
            else "NULL AS sequence_generation")


def select_columns(generation):
    return ("rowid,event_id,event_type,asset,market_ticker,series_ticker,"
            "exchange_timestamp,local_receive_timestamp,processing_timestamp,source,"
            "raw_payload,contract_open_time,contract_close_time,target,sequence,"
            f"{generation}")
