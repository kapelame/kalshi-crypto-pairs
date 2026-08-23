#!/usr/bin/env python3
"""Replay immutable raw events into an append-only deterministic feature DB."""

import argparse
import asyncio
from pathlib import Path

from signals.replay import ReplayEngine
from signals.store import FeatureStore


async def run(args):
    if Path(args.features_db).exists():
        raise SystemExit(f"Refusing to overwrite existing derived DB: {args.features_db}")
    with FeatureStore(args.features_db) as store:
        snapshots = await ReplayEngine(args.db, store).run(
            speed=args.speed, stop_timestamp=args.stop_at, market_ticker=args.market)
        print(f"Replayed {len(snapshots)} feature snapshots into {args.features_db}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Phase 2 raw SQLite database")
    parser.add_argument("--features-db", default="kalshi_features_v3.db")
    parser.add_argument("--speed", type=float, default=0,
                        help="0=max speed; otherwise original timing divided by multiplier")
    parser.add_argument("--stop-at", help="inclusive UTC ISO timestamp")
    parser.add_argument("--market", help="replay one exact market ticker")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__": main()
