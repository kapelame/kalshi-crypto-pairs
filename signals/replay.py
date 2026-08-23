"""Stable merge of Phase 2 raw tables and causal deterministic replay."""

import asyncio
import json
import sqlite3

from streaming.events import EVENT_TABLES
from streaming.timeutil import parse_timestamp

from .engine import SignalEngine


TABLE_ORDER = [
    "market_lifecycle_events", "contract_reset_events", "ticker_events",
    "trade_events", "orderbook_snapshots", "orderbook_deltas",
    "underlying_price_events",
]


class RawEventReader:
    def __init__(self, path):
        self.path = path

    def read(self, market_ticker=None, stop_timestamp=None):
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        events = []
        for priority, table in enumerate(TABLE_ORDER):
            present = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if not present:
                continue
            query = (f"SELECT rowid,event_id,event_type,asset,market_ticker,series_ticker,"
                     f"exchange_timestamp,local_receive_timestamp,processing_timestamp,source,"
                     f"raw_payload,contract_open_time,contract_close_time,target,sequence FROM {table}")
            params = []
            if market_ticker:
                query += " WHERE market_ticker=?"; params.append(market_ticker)
            for row in connection.execute(query, params):
                event = dict(zip(("_rowid", "event_id", "event_type", "asset", "market_ticker",
                    "series_ticker", "exchange_timestamp", "local_receive_timestamp",
                    "processing_timestamp", "source", "raw_payload", "contract_open_time",
                    "contract_close_time", "target", "sequence"), row))
                event["raw_payload"] = json.loads(event["raw_payload"])
                event["_priority"] = priority
                if stop_timestamp and parse_timestamp(event["local_receive_timestamp"]) > parse_timestamp(stop_timestamp):
                    continue
                events.append(event)
        connection.close()
        events.sort(key=lambda event: (
            parse_timestamp(event["local_receive_timestamp"]),
            parse_timestamp(event["processing_timestamp"]),
            event["_priority"], event["_rowid"]))
        return events


class ReplayEngine:
    def __init__(self, raw_db, feature_store=None, signal_engine=None):
        self.reader = RawEventReader(raw_db)
        self.store = feature_store
        self.engine = signal_engine or SignalEngine()

    async def run(self, speed=0, stop_timestamp=None, market_ticker=None, on_snapshot=None):
        previous_time = None
        snapshots = []
        for event in self.reader.read(market_ticker, stop_timestamp):
            current_time = parse_timestamp(event["local_receive_timestamp"]).timestamp()
            if speed and previous_time is not None:
                await asyncio.sleep(max(0, current_time - previous_time) / speed)
            previous_time = current_time
            snapshot = self.engine.process(event)
            if snapshot is not None:
                snapshots.append(snapshot)
                if self.store:
                    self.store.append_snapshot(snapshot)
                if on_snapshot:
                    result = on_snapshot(snapshot)
                    if asyncio.iscoroutine(result): await result
            outcome = extract_outcome(event)
            if outcome and self.store:
                outcome_ticker = (event["raw_payload"].get("previous_ticker")
                                  if event["event_type"] == "contract_reset" else event["market_ticker"])
                self.store.append_outcome(outcome_ticker, event["asset"], current_time,
                                          outcome, event["event_id"])
        return snapshots


def extract_outcome(event):
    if event["event_type"] == "contract_reset":
        return event["raw_payload"].get("previous_settlement")
    if event["event_type"] != "market_lifecycle":
        return None
    msg = event["raw_payload"].get("msg", event["raw_payload"])
    return msg.get("result") or msg.get("market_result")
