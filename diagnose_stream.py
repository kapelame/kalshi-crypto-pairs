#!/usr/bin/env python3
"""Run a time-limited read-only five-asset WebSocket data diagnostic."""

import argparse
import asyncio
import logging
import statistics
import time

from streaming.auth import CredentialError, load_credentials
from streaming.recorder import StreamRecorder


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


async def run(args):
    load_credentials(args.env_file)
    recorder = StreamRecorder(db_path=args.db, dotenv_path=args.env_file)
    started = time.monotonic()
    await recorder.run(duration=args.duration)
    elapsed = time.monotonic() - started
    summary = recorder.summary(elapsed)
    latencies = summary.pop("latencies_ms")
    print("\n60-second streaming diagnostic" if args.duration == 60 else
          f"\n{args.duration:g}-second streaming diagnostic")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if latencies:
        print("receive_latency_ms: "
              f"p50={percentile(latencies, .5):.3f} "
              f"p95={percentile(latencies, .95):.3f} "
              f"p99={percentile(latencies, .99):.3f} "
              f"mean={statistics.fmean(latencies):.3f}")
    else:
        print("receive_latency_ms: unavailable (exchange timestamp absent)")
    print(f"stale_markets: {[a for a, s in recorder.health.assets.items() if not s.valid()]}")
    print(recorder.health.render())
    count = recorder.health.healthy_count()
    print(f"\n{count}/5 healthy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--db", default="kalshi_stream_diagnostic.db")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run(args))
    except CredentialError as exc:
        raise SystemExit(f"WebSocket credentials unavailable: {exc}")
    except KeyboardInterrupt:
        print("Stopped cleanly by user")


if __name__ == "__main__":
    main()
