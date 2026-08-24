#!/usr/bin/env python3
"""Incremental read-only signal monitor; contains no order/execution path."""

import argparse
import asyncio
import json
import os
import sqlite3
import time

from signals.engine import SignalEngine
from signals.replay import TABLE_ORDER

COLUMNS = ("_rowid", "event_id", "event_type", "asset", "market_ticker",
           "series_ticker", "exchange_timestamp", "local_receive_timestamp",
           "processing_timestamp", "source", "raw_payload", "contract_open_time",
           "contract_close_time", "target", "sequence", "sequence_generation")


def fmt(value, signed=False):
    if value is None:
        return "---"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


class IncrementalRawTail:
    """Bounded per-table reads with a conservative causal merge frontier."""
    def __init__(self, path, batch_size=100):
        self.path = str(path)
        self.batch_size = batch_size
        self.rowids = {table: 0 for table in TABLE_ORDER}
        self.pending = []
        self.events_read = 0
        self.caught_up = False

    def read_batch(self):
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        fetched, constraining = [], []
        try:
            connection.execute("BEGIN")
            for priority, table in enumerate(TABLE_ORDER):
                present = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
                if not present:
                    continue
                columns = {row[1] for row in connection.execute(
                    f"PRAGMA table_info({table})")}
                generation = ("sequence_generation" if "sequence_generation" in columns
                              else "NULL AS sequence_generation")
                rows = connection.execute(
                    f"SELECT rowid,event_id,event_type,asset,market_ticker,series_ticker,"
                    f"exchange_timestamp,local_receive_timestamp,processing_timestamp,source,"
                    f"raw_payload,contract_open_time,contract_close_time,target,sequence,"
                    f"{generation} "
                    f"FROM {table} WHERE rowid>? ORDER BY rowid LIMIT ?",
                    (self.rowids[table], self.batch_size)).fetchall()
                if rows:
                    self.rowids[table] = rows[-1][0]
                if len(rows) == self.batch_size:
                    constraining.append(rows[-1][7])
                for row in rows:
                    event = dict(zip(COLUMNS, row))
                    event["raw_payload"] = json.loads(event["raw_payload"])
                    event["_priority"] = priority
                    fetched.append(event)
            connection.rollback()
        finally:
            connection.close()
        self.pending.extend(fetched)
        self.pending.sort(key=lambda event: (
            event["local_receive_timestamp"], event["processing_timestamp"],
            event["_priority"], event["_rowid"]))
        frontier = min(constraining) if constraining else None
        if frontier is None:
            ready, self.pending = self.pending, []
        else:
            split = 0
            while split < len(self.pending) and self.pending[split]["local_receive_timestamp"] <= frontier:
                split += 1
            ready, self.pending = self.pending[:split], self.pending[split:]
        self.events_read += len(ready)
        self.caught_up = not constraining and not self.pending
        return ready


class LiveSignalMonitor:
    def __init__(self, path, batch_size=100, engine=None):
        self.tail = IncrementalRawTail(path, batch_size)
        self.engine = engine or SignalEngine()
        self.state_timestamp = None
        self.raw_watermark = None
        self.events_processed = 0
        self.caught_up = False

    def consume_once(self):
        events = self.tail.read_batch()
        self._consume(events)
        return len(events)

    def _consume(self, events):
        for event in events:
            self.engine.process(event)
            self.state_timestamp = event["local_receive_timestamp"]
            self.raw_watermark = event["event_id"]
            self.events_processed += 1
        self.caught_up = self.tail.caught_up

    async def consume_once_async(self):
        events = await asyncio.to_thread(self.tail.read_batch)
        for offset in range(0, len(events), 8):
            self._consume(events[offset:offset + 8])
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
                                            self.raw_watermark, emit_checkpoints=False)
            snapshot["basket"] = basket
            snapshots[asset] = snapshot
        return snapshots, basket

    def close(self):
        return None


def format_monitor(monitor):
    snapshots, basket = monitor.coherent_snapshots()
    lines = [f"STATE timestamp={monitor.state_timestamp or '---'} "
             f"watermark={monitor.raw_watermark or '---'} caught_up={'yes' if monitor.caught_up else 'no'}",
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
    while monitor.consume_once():
        pass
    return format_monitor(monitor)


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
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if not os.path.exists(args.db):
        raise SystemExit(f"Raw database not found: {args.db}")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nMonitor stopped")


if __name__ == "__main__":
    main()
