#!/usr/bin/env python3
"""Replay immutable raw events into an append-only deterministic feature DB."""

import argparse
import asyncio
from pathlib import Path

from signals.replay import ReplayEngine
from signals.store import FeatureStore


async def run(args):
    if args.features_db and Path(args.features_db).exists():
        raise SystemExit(f"Refusing to overwrite existing derived DB: {args.features_db}")
    store = FeatureStore(args.features_db, args.batch_size).open() if args.features_db else None
    try:
        result = await ReplayEngine(args.db, store).run(
            speed=args.speed, stop_timestamp=args.stop_at, market_ticker=args.market,
            start_timestamp=args.start_at, snapshot_mode=args.snapshot_mode,
            diagnostic_hz=args.diagnostic_hz)
    finally:
        if store: store.close()
    print(f"Replayed {result.events_processed} events; generated "
          f"{result.snapshots_generated} snapshots ({result.checkpoints_generated} checkpoints) "
          f"into {args.features_db or 'no database'}; digest={result.digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Phase 2 raw SQLite database")
    parser.add_argument("--features-db", default="kalshi_features_v3.db")
    parser.add_argument("--speed", type=float, default=0,
                        help="0=max speed; otherwise original timing divided by multiplier")
    parser.add_argument("--stop-at", help="inclusive UTC ISO timestamp")
    parser.add_argument("--start-at", help="inclusive UTC ISO timestamp")
    parser.add_argument("--market", help="replay one exact market ticker")
    parser.add_argument("--snapshot-mode", choices=("checkpoints", "diagnostic", "all"),
                        default="checkpoints")
    parser.add_argument("--diagnostic-hz", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Replay stopped; pending feature batches were flushed")


if __name__ == "__main__": main()
