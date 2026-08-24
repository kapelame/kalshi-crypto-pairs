"""Bounded-memory deterministic merge and causal replay orchestration."""

import asyncio
import hashlib
import heapq
import inspect
import json
import sqlite3
from dataclasses import dataclass

from .engine import SignalEngine
from .ordering import (TABLE_ORDER, event_from_row, event_order_key,
                       generation_projection, select_columns)
from streaming.timeutil import parse_timestamp


class RawEventReader:
    """Stream a stable k-way merge with one live row per raw table."""
    def __init__(self, path, fetch_size=2000):
        self.path = path
        self.fetch_size = fetch_size

    def read(self, market_ticker=None, stop_timestamp=None, start_timestamp=None):
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        cursors, heap = [], []
        try:
            for priority, table in enumerate(TABLE_ORDER):
                if not connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone():
                    continue
                columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                generation = generation_projection(columns)
                query = f"SELECT {select_columns(generation)} FROM {table}"
                clauses, params = [], []
                if market_ticker:
                    clauses.append("market_ticker=?"); params.append(market_ticker)
                if start_timestamp:
                    clauses.extend(("rowid>=?", "local_receive_timestamp>=?"))
                    params.extend((max(1, self._rowid_bound(
                        connection, table, start_timestamp, upper=False) - 10000),
                        start_timestamp))
                if stop_timestamp:
                    clauses.extend(("rowid<?", "local_receive_timestamp<=?"))
                    params.extend((self._rowid_bound(
                        connection, table, stop_timestamp, upper=True) + 10000,
                        stop_timestamp))
                if clauses:
                    query += " WHERE " + " AND ".join(clauses)
                query += " ORDER BY local_receive_timestamp,processing_timestamp,rowid"
                cursor = connection.execute(query, params)
                cursors.append((priority, cursor))
                row = cursor.fetchone()
                if row:
                    event = self._event(row, priority)
                    heapq.heappush(heap, (event_order_key(event), len(cursors)-1, event))
            while heap:
                _, cursor_index, event = heapq.heappop(heap)
                yield event
                priority, cursor = cursors[cursor_index]
                row = cursor.fetchone()
                if row:
                    following = self._event(row, priority)
                    heapq.heappush(heap, (event_order_key(following), cursor_index, following))
        finally:
            connection.close()

    @staticmethod
    def _event(row, priority):
        return event_from_row(row, priority)

    @staticmethod
    def _rowid_bound(connection, table, timestamp, upper):
        """Binary-search append-ordered rowids without scanning the raw table."""
        bounds = connection.execute(
            f"SELECT COALESCE(MIN(rowid),1),COALESCE(MAX(rowid),0)+1 FROM {table}").fetchone()
        low, high = bounds
        while low < high:
            middle = (low + high) // 2
            row = connection.execute(
                f"SELECT local_receive_timestamp FROM {table} WHERE rowid=?", (middle,)).fetchone()
            if row is None:
                low = middle + 1
                continue
            before = row[0] <= timestamp if upper else row[0] < timestamp
            if before:
                low = middle + 1
            else:
                high = middle
        return low


@dataclass
class ReplayResult:
    events_processed: int = 0
    snapshots_generated: int = 0
    checkpoints_generated: int = 0
    diagnostic_snapshots: int = 0
    outcomes: int = 0
    digest: str = ""
    collected_snapshots: list | None = None

    def __len__(self):
        return self.snapshots_generated

    def __iter__(self):
        return iter(self.collected_snapshots or ())

    def __getitem__(self, item):
        if self.collected_snapshots is None:
            raise TypeError("snapshots were not collected; pass collect_snapshots=True")
        return self.collected_snapshots[item]


class ReplayEngine:
    MODES = {"checkpoints", "diagnostic", "all"}

    def __init__(self, raw_db, feature_store=None, signal_engine=None):
        self.reader = RawEventReader(raw_db)
        self.store = feature_store
        self.engine = signal_engine or SignalEngine()
        self.persisted_checkpoints = set()
        self.last_diagnostic = {}
        self.seen_outcomes = set()

    async def run(self, speed=0, stop_timestamp=None, market_ticker=None,
                  on_snapshot=None, snapshot_mode="checkpoints", diagnostic_hz=1.0,
                  start_timestamp=None, collect_snapshots=False):
        if snapshot_mode not in self.MODES:
            raise ValueError(f"invalid snapshot mode: {snapshot_mode}")
        previous_time = None
        collected = [] if collect_snapshots else None
        digest = hashlib.sha256()
        result = ReplayResult(collected_snapshots=collected)
        try:
            for event in self.reader.read(market_ticker, stop_timestamp, start_timestamp):
                current_time = parse_timestamp(event["local_receive_timestamp"]).timestamp()
                if speed and previous_time is not None:
                    await asyncio.sleep(max(0, current_time - previous_time) / speed)
                previous_time = current_time
                pre_reset_final = None
                if event["event_type"] == "contract_reset":
                    prior_state = self.engine.assets.get(event["asset"])
                    if prior_state and prior_state.contract.ticker:
                        pre_reset_final = self.engine.snapshot(
                            event["asset"], current_time, event["event_id"],
                            emit_checkpoints=False)
                self.engine.process(event, emit_snapshot=False)
                result.events_processed += 1
                emissions = []
                if snapshot_mode == "all":
                    emissions.append((self.engine.snapshot(
                        event["asset"], current_time, event["event_id"],
                        emit_checkpoints=False), "all", None, None))
                if snapshot_mode in {"checkpoints", "diagnostic"}:
                    emissions.extend(self._due_checkpoints(current_time, event["event_id"]))
                if snapshot_mode == "diagnostic":
                    emissions.extend(self._due_diagnostics(
                        current_time, event["event_id"], diagnostic_hz))
                outcome = extract_outcome(event)
                if outcome:
                    outcome_ticker = (event["raw_payload"].get("previous_ticker")
                                      if event["event_type"] == "contract_reset"
                                      else event["market_ticker"])
                    if outcome_ticker not in self.seen_outcomes:
                        self.seen_outcomes.add(outcome_ticker)
                        final = pre_reset_final or self.engine.snapshot(
                            event["asset"], current_time, event["event_id"],
                            emit_checkpoints=False)
                        final["market_ticker"] = outcome_ticker
                        emissions.append((final, "settlement", None, current_time))
                        if self.store:
                            self.store.append_outcome(
                                outcome_ticker, event["asset"], current_time,
                                outcome, event["event_id"])
                        result.outcomes += 1
                for snapshot, kind, checkpoint, scheduled in emissions:
                    await self._emit(snapshot, kind, checkpoint, scheduled, on_snapshot,
                                     collected, digest, result)
                if result.events_processed % 10000 == 0:
                    await asyncio.sleep(0)
        finally:
            if self.store:
                self.store.flush()
        result.digest = digest.hexdigest()
        return result

    def _due_checkpoints(self, timestamp, watermark):
        emitted = []
        basket = None
        for asset, state in self.engine.assets.items():
            if not state.contract.ticker or state.contract.open_time is None:
                continue
            for checkpoint in self.engine.config.checkpoints:
                key = (state.contract.ticker, checkpoint)
                scheduled = state.contract.open_time + checkpoint
                if key not in self.persisted_checkpoints and timestamp >= scheduled:
                    if basket is None:
                        basket = self.engine.basket_features(timestamp)
                    state.checkpoints_emitted.add(checkpoint)
                    snapshot = self.engine.snapshot(asset, timestamp, watermark,
                                                    emit_checkpoints=False, basket=basket)
                    snapshot["features"][f"checkpoint_{checkpoint}s"] = True
                    self.persisted_checkpoints.add(key)
                    emitted.append((snapshot, "checkpoint", checkpoint, scheduled))
        return emitted

    def _due_diagnostics(self, timestamp, watermark, hz):
        interval = 1.0 / hz if hz > 0 else 1.0
        emitted = []
        basket = None
        for asset, state in self.engine.assets.items():
            if not state.contract.ticker:
                continue
            if timestamp - self.last_diagnostic.get(asset, float("-inf")) >= interval:
                self.last_diagnostic[asset] = timestamp
                if basket is None:
                    basket = self.engine.basket_features(timestamp)
                emitted.append((self.engine.snapshot(
                    asset, timestamp, watermark, emit_checkpoints=False, basket=basket),
                    "diagnostic", None, timestamp))
        return emitted

    async def _emit(self, snapshot, kind, checkpoint, scheduled, callback,
                    collected, digest, result):
        logical_record = {"snapshot": snapshot, "snapshot_kind": kind,
                          "checkpoint_seconds": checkpoint,
                          "scheduled_timestamp": scheduled}
        logical = json.dumps(logical_record, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
        digest.update(len(logical).to_bytes(8, "big")); digest.update(logical)
        result.snapshots_generated += 1
        if kind == "checkpoint": result.checkpoints_generated += 1
        if kind == "diagnostic": result.diagnostic_snapshots += 1
        if collected is not None: collected.append(snapshot)
        if self.store:
            self.store.append_snapshot(snapshot, kind, checkpoint, scheduled)
        if callback:
            returned = callback(snapshot)
            if inspect.isawaitable(returned): await returned


def extract_outcome(event):
    if event["event_type"] == "contract_reset":
        return event["raw_payload"].get("previous_settlement")
    if event["event_type"] != "market_lifecycle":
        return None
    msg = event["raw_payload"].get("msg", event["raw_payload"])
    return msg.get("result") or msg.get("market_result")
