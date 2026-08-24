#!/usr/bin/env python3
"""Read-only SignalEngine/replay benchmark; contains no trading path."""

import argparse
import asyncio
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import resource
import time

from signals.engine import SignalEngine
from signals.replay import ReplayEngine
from signals.store import FeatureStore


class ProfilingSignalEngine(SignalEngine):
    def __init__(self):
        super().__init__()
        self.event_mix = Counter()
        self.family_nanoseconds = defaultdict(int)

    def process(self, event, emit_snapshot=True):
        kind = event["event_type"]
        started = time.perf_counter_ns()
        try:
            return super().process(event, emit_snapshot)
        finally:
            self.event_mix[kind] += 1
            self.family_nanoseconds[kind] += time.perf_counter_ns() - started


async def run(args):
    if not args.no_persist and Path(args.features_db).exists():
        raise SystemExit(f"Refusing to overwrite existing benchmark DB: {args.features_db}")
    store = None if args.no_persist else FeatureStore(
        args.features_db, batch_size=args.batch_size).open()
    engine = ProfilingSignalEngine()
    started = time.perf_counter()
    try:
        result = await ReplayEngine(args.db, store, engine).run(
            speed=0, start_timestamp=args.start, stop_timestamp=args.stop,
            market_ticker=args.market, snapshot_mode=args.snapshot_mode,
            diagnostic_hz=args.diagnostic_hz)
    finally:
        if store: store.close()
    elapsed = time.perf_counter() - started
    output_bytes = 0
    if store:
        output_bytes = sum(path.stat().st_size for path in Path(args.features_db).parent.glob(
            Path(args.features_db).name + "*"))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = rss if __import__("sys").platform == "darwin" else rss * 1024
    report = {
        "raw_events": result.events_processed,
        "elapsed_seconds": elapsed,
        "events_per_second": result.events_processed / elapsed if elapsed else None,
        "event_mix": dict(engine.event_mix),
        "feature_family_seconds": {
            kind: nanoseconds / 1e9 for kind, nanoseconds in engine.family_nanoseconds.items()},
        "peak_rss_bytes": peak_bytes,
        "snapshots_generated": result.snapshots_generated,
        "snapshots_persisted": 0 if store is None else store.snapshots_written,
        "checkpoint_count": result.checkpoints_generated,
        "diagnostic_snapshot_count": result.diagnostic_snapshots,
        "database_bytes_written": output_bytes,
        "digest": result.digest,
        "snapshot_mode": args.snapshot_mode,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--start")
    parser.add_argument("--stop")
    parser.add_argument("--market")
    parser.add_argument("--features-db", default="benchmark_features.db")
    parser.add_argument("--snapshot-mode", choices=("checkpoints", "diagnostic", "all"),
                        default="checkpoints")
    parser.add_argument("--diagnostic-hz", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.db):
        raise SystemExit(f"Raw database not found: {args.db}")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Benchmark stopped; pending feature batches were flushed")


if __name__ == "__main__":
    main()
