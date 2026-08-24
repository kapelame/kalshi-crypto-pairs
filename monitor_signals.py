#!/usr/bin/env python3
"""Incremental read-only signal monitor; contains no order/execution path."""

import argparse
import asyncio
import os
import sqlite3
import time
from collections import deque
from datetime import timedelta

from signals.engine import SignalEngine
from signals.ordering import (TABLE_ORDER, event_from_row, event_order_key,
                              generation_projection, has_ingest_ledger,
                              read_ingest_batch, select_columns, table_specs)
from streaming.timeutil import parse_timestamp, utc_now


# REST observations capture receive time before an HTTP request whose enforced
# total timeout is 15 seconds. Five seconds cover SQLite's default busy timeout.
# A row beyond this processing-time boundary cannot later introduce an earlier
# receive timestamp under the recorder's synchronous append contract.
DEFAULT_CAUSAL_LATENESS_SECONDS = 20.0


def fmt(value, signed=False):
    if value is None:
        return "---"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


class IncrementalRawTail:
    """Rowid-paginated live merge behind a producer-safe causal watermark."""
    def __init__(self, path, batch_size=100,
                 causal_lateness_seconds=DEFAULT_CAUSAL_LATENESS_SECONDS,
                 clock=utc_now, max_buffer_events=250_000):
        self.path = str(path)
        self.batch_size = batch_size
        self.causal_lateness_seconds = causal_lateness_seconds
        self.clock = clock
        self.max_buffer_events = max_buffer_events
        self.rowids = {table: 0 for table in TABLE_ORDER}
        self.processing_cursors = {table: None for table in TABLE_ORDER}
        self.pending = []
        self.backlog = 0
        self.buffer_high_water = 0
        self.events_read = 0
        self.caught_up = False
        self.connection = None
        self.table_specs = None
        self.ingest_mode = False
        self.latest_ingest_sequence = None
        self.state_persistence_timestamp = None

    def _open(self):
        if self.connection is None:
            self.connection = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
            specs = []
            for priority, table in enumerate(TABLE_ORDER):
                if not self.connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone():
                    continue
                columns = {row[1] for row in self.connection.execute(
                    f"PRAGMA table_info({table})")}
                generation = generation_projection(columns)
                specs.append((priority, table, generation))
            self.table_specs = specs
            self.ingest_mode = has_ingest_ledger(self.connection)
            self.ingest_specs = table_specs(self.connection)
        return self.connection

    def _processing_frontier(self, connection):
        """Latest committed processing time under the single-writer contract."""
        maxima, latest_rowids = [], {}
        for _, table, _ in self.table_specs:
            value = connection.execute(
                f"SELECT rowid,processing_timestamp FROM {table} "
                "ORDER BY rowid DESC LIMIT 1").fetchone()
            if value:
                latest_rowids[table] = value[0]
                maxima.append(parse_timestamp(value[1]))
            else:
                latest_rowids[table] = 0
        # Wall time is also a lower processing bound for work that has not yet
        # been constructed. The causal-lateness contract accounts for REST work
        # that captured receive time before that point.
        return max([self.clock(), *maxima]), latest_rowids

    def read_batch(self, final=False):
        connection = self._open()
        if self.ingest_mode:
            return self._read_ingest_batch(connection)
        fetched, unseen_processing = [], []
        batch_constrained = False
        try:
            connection.execute("BEGIN")
            committed_frontier, latest_rowids = self._processing_frontier(connection)
            for priority, table, generation in self.table_specs:
                available = max(0, self.max_buffer_events -
                                len(self.pending) - len(fetched))
                limit = min(self.batch_size, available)
                rows = ([] if limit == 0 else connection.execute(
                    f"SELECT {select_columns(generation)} FROM {table} "
                    "WHERE rowid>? ORDER BY rowid LIMIT ?",
                    (self.rowids[table], limit)).fetchall())
                if rows:
                    previous = self.processing_cursors[table]
                    for row in rows:
                        processed = parse_timestamp(row[8])
                        if previous is not None and processed < previous:
                            raise ValueError(
                                f"{table} processing timestamps are not append-monotonic")
                        previous = processed
                    self.processing_cursors[table] = previous
                    self.rowids[table] = rows[-1][0]
                for row in rows:
                    fetched.append(event_from_row(row, priority))
                following = connection.execute(
                    f"SELECT processing_timestamp FROM {table} "
                    "WHERE rowid>? ORDER BY rowid LIMIT 1",
                    (self.rowids[table],)).fetchone()
                if following:
                    batch_constrained = True
                    unseen_processing.append(parse_timestamp(following[0]))
                elif committed_frontier is not None:
                    # Any future append is constructed after the latest commit:
                    # RawEventStore.append is synchronous and single-writer.
                    unseen_processing.append(committed_frontier)
            connection.rollback()
        except Exception:
            connection.rollback()
            raise
        self.pending.extend(fetched)
        self.pending.sort(key=event_order_key)
        if final and batch_constrained:
            # The caller has declared the source static, but rows beyond this
            # source-row batch may still sort before the rows just fetched.
            ready = []
        elif final:
            ready, self.pending = self.pending, []
        else:
            frontier = (None if not unseen_processing else
                        min(unseen_processing) - timedelta(
                            seconds=self.causal_lateness_seconds))
            split = 0
            while (frontier is not None and split < len(self.pending) and
                   parse_timestamp(self.pending[split]["local_receive_timestamp"]) < frontier):
                split += 1
            ready, self.pending = self.pending[:split], self.pending[split:]
        self.events_read += len(ready)
        self.caught_up = not batch_constrained
        self.backlog = sum(
            max(0, latest_rowids.get(table, 0) - self.rowids[table])
            for table in self.rowids)
        self.buffer_high_water = max(self.buffer_high_water, len(self.pending))
        return ready

    def _read_ingest_batch(self, connection):
        after = self.latest_ingest_sequence or 0
        try:
            connection.execute("BEGIN")
            events = read_ingest_batch(
                connection, after, self.batch_size, self.ingest_specs)
            latest = connection.execute(
                "SELECT COALESCE(MAX(ingest_sequence),0) FROM raw_ingest_log").fetchone()[0]
            connection.rollback()
        except Exception:
            connection.rollback()
            raise
        if events:
            self.latest_ingest_sequence = events[-1]["_ingest_sequence"]
        self.events_read += len(events)
        self.backlog = max(0, latest - (self.latest_ingest_sequence or 0))
        self.pending.clear()
        self.caught_up = len(events) < self.batch_size and self.backlog == 0
        return events

    def drain(self):
        """Drain a database the caller has established is no longer being written."""
        output = []
        while True:
            events = self.read_batch(final=True)
            output.extend(events)
            if not events and not self.pending:
                return output

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class LiveSignalMonitor:
    def __init__(self, path, batch_size=100, engine=None,
                 causal_lateness_seconds=DEFAULT_CAUSAL_LATENESS_SECONDS):
        self.tail = IncrementalRawTail(path, batch_size, causal_lateness_seconds)
        self.engine = engine or SignalEngine()
        self.state_timestamp = None
        self.raw_watermark = None
        self.events_processed = 0
        self.caught_up = False
        self.started_at = time.monotonic()
        self.processing_samples = deque(maxlen=10000)
        self.maximum_backlog = 0
        self.catch_up_lag_ms = None
        self.latest_ingest_sequence = None
        self.persist_to_engine_samples = deque(maxlen=10000)
        self.last_engine_wall_timestamp = None
        self.render_lag_ms = None

    def consume_once(self):
        events = self.tail.read_batch()
        self._consume(events)
        return len(events)

    def _consume(self, events):
        for event in events:
            started = time.perf_counter()
            engine_wall = time.time()
            self.engine.process(event, emit_snapshot=False)
            self.processing_samples.append((time.perf_counter() - started) * 1000)
            self.state_timestamp = event["local_receive_timestamp"]
            self.raw_watermark = event["event_id"]
            self.latest_ingest_sequence = event.get("_ingest_sequence")
            persisted = event.get("_persistence_timestamp")
            if persisted:
                self.state_persistence_timestamp = persisted
                self.persist_to_engine_samples.append(max(
                    0.0, (engine_wall - parse_timestamp(persisted).timestamp()) * 1000))
            self.last_engine_wall_timestamp = engine_wall
            self.events_processed += 1
        self.caught_up = self.tail.caught_up
        self.maximum_backlog = max(self.maximum_backlog, self.tail.backlog)
        if self.state_timestamp:
            state_basis = self.state_persistence_timestamp or self.state_timestamp
            self.catch_up_lag_ms = max(0.0, (time.time() -
                parse_timestamp(state_basis).timestamp()) * 1000)

    async def consume_once_async(self):
        events = await asyncio.to_thread(self.tail.read_batch)
        for offset in range(0, len(events), 128):
            self._consume(events[offset:offset + 128])
            await asyncio.sleep(0)
        if not events:
            self.caught_up = self.tail.caught_up
        return len(events)

    def coherent_snapshots(self):
        if self.engine.last_event_time is None:
            return {}, {}
        basket = self.engine.basket_features(self.engine.last_event_time)
        snapshots = {}
        for asset in self.engine.assets:
            snapshot = self.engine.snapshot(asset, self.engine.last_event_time,
                                            self.raw_watermark, emit_checkpoints=False,
                                            basket=basket)
            snapshot["basket"] = basket
            snapshots[asset] = snapshot
        return snapshots, basket

    def close(self):
        self.tail.close()


def format_monitor(monitor):
    snapshots, basket = monitor.coherent_snapshots()
    elapsed = max(time.monotonic() - monitor.started_at, 1e-9)
    rate = monitor.events_processed / elapsed
    samples = sorted(monitor.processing_samples)
    persist_samples = sorted(monitor.persist_to_engine_samples)
    def percentile(fraction):
        return None if not samples else samples[min(len(samples)-1, int(len(samples)*fraction))]
    def persist_percentile(fraction):
        return (None if not persist_samples else
                persist_samples[min(len(persist_samples)-1,
                                    int(len(persist_samples)*fraction))])
    monitor.render_lag_ms = (None if monitor.last_engine_wall_timestamp is None else
                             max(0.0, (time.time() -
                                      monitor.last_engine_wall_timestamp) * 1000))
    lines = [f"STATE timestamp={monitor.state_timestamp or '---'} "
             f"watermark={monitor.raw_watermark or '---'} "
             f"ingest_sequence={monitor.latest_ingest_sequence or '---'} "
             f"caught_up={'yes' if monitor.caught_up else 'no'}",
             f"FLOW processed={monitor.events_processed} rate={rate:.0f}/s "
             f"source_backlog={monitor.tail.backlog} "
             f"processing_backlog={monitor.tail.backlog + len(monitor.tail.pending)} "
             f"max_backlog={monitor.maximum_backlog} "
             f"reorder_buffer={len(monitor.tail.pending)} "
             f"buffer_hwm={monitor.tail.buffer_high_water} "
             f"state_lag_ms={fmt(monitor.catch_up_lag_ms)} "
             f"render_lag_ms={fmt(monitor.render_lag_ms)} "
             f"persist_engine_p50={fmt(persist_percentile(.50))} "
             f"p95={fmt(persist_percentile(.95))} p99={fmt(persist_percentile(.99))} "
             f"proc_ms_p50={fmt(percentile(.50))} proc_p95={fmt(percentile(.95))} "
             f"proc_p99={fmt(percentile(.99))}",
             "ASSET TICKER WINDOW STATUS TARGET Q_AGE B_AGE ROLLOVER ELIGIBLE REASONS"]
    for asset in ("BTC", "ETH", "SOL", "XRP", "DOGE"):
        snapshot = snapshots.get(asset)
        f = {} if snapshot is None else snapshot["features"]
        ticker = "---" if snapshot is None else snapshot.get("market_ticker") or "---"
        reasons = ",".join(f.get("excluded_reasons", [])) or "---"
        lines.append(
            f"{asset:<5} {ticker:<29} {str(f.get('contract_window_id') or '---'):<20} "
            f"{str(f.get('market_status') or '---'):<9} {fmt(None if snapshot is None else snapshot.get('target')):>8} "
            f"{fmt(f.get('quote_age_seconds')):>6} {fmt(f.get('book_age_seconds')):>6} "
            f"{str(f.get('rollover_state') or '---'):<8} "
            f"{'yes' if f.get('eligible') else 'no ':<3} {reasons}")
    regime = (next(iter(snapshots.values()))["features"].get("regime_label")
              if snapshots else "---")
    lines += ["", f"ELIGIBLE: {basket.get('eligible_count', 0)}/{basket.get('expected_count', 5)}",
              f"COHERENT WINDOW: {'yes' if basket.get('coherent_window') else 'no'}",
              f"WINDOW: {basket.get('contract_window_id') or '---'}",
              f"BREADTH: {basket.get('breadth_direction', '---')}",
              f"REGIME: {regime}"]
    return "\n".join(lines)


def render(source):
    """Compatibility formatter for one-shot callers and small tests."""
    if isinstance(source, LiveSignalMonitor):
        return format_monitor(source)
    monitor = LiveSignalMonitor(source, batch_size=1000)
    try:
        while monitor.consume_once():
            pass
        return format_monitor(monitor)
    finally:
        monitor.close()


async def run(args):
    monitor = LiveSignalMonitor(args.db, args.batch_size)
    started = time.monotonic()
    next_redraw = started
    try:
        while True:
            await monitor.consume_once_async()
            now = time.monotonic()
            if now >= next_redraw:
                print("\033[2J\033[H" + format_monitor(monitor), flush=True)
                next_redraw = now + args.refresh
            if args.duration and now - started >= args.duration:
                break
            await asyncio.sleep(0 if not monitor.caught_up else min(args.refresh, .05))
    finally:
        monitor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="kalshi_stream_raw.db")
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--refresh", type=float, default=.25)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    if not os.path.exists(args.db):
        raise SystemExit(f"Raw database not found: {args.db}")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nMonitor stopped")


if __name__ == "__main__":
    main()
