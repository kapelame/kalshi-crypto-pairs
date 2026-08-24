#!/usr/bin/env python3
"""Synthetic read-only direct dispatcher burst benchmark; no trading imports."""

import argparse
import asyncio
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from streaming.dispatcher import CausalIngestDispatcher, LiveSignalService
from streaming.events import RawEvent
from streaming.raw_store import RawEventStore


def event(index):
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return RawEvent(
        event_type="orderbook_delta", asset="BTC",
        market_ticker="KXBTC15M-BENCH", source="synthetic_benchmark",
        raw_payload={
            "type": "orderbook_delta", "sid": 2, "seq": index + 1,
            "msg": {"market_ticker": "KXBTC15M-BENCH", "side": "yes",
                    "price_dollars": ".49", "delta_fp": "1"}},
        local_receive_timestamp=now, processing_timestamp=now,
        sequence=index + 1, sequence_generation=1,
        event_id=f"burst-{index}")


async def benchmark(rate, duration, queue_size):
    count = int(rate * duration)
    with tempfile.TemporaryDirectory() as directory:
        store = RawEventStore(Path(directory) / "burst.db").open()
        service = LiveSignalService()
        dispatcher = CausalIngestDispatcher(
            store, service, max_queue=queue_size,
            unhealthy_threshold=max(1, queue_size * 3 // 4))
        await dispatcher.start()
        started = time.perf_counter()
        tasks = []
        for index in range(count):
            due = started + index / rate
            delay = due - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(dispatcher.publish(event(index))))
        input_done = time.perf_counter()
        await asyncio.gather(*tasks)
        drained = time.perf_counter()
        summary = dispatcher.summary()
        await dispatcher.stop()
        store.close()
    return {
        "target_rate": rate, "duration_seconds": duration,
        "input_events": count, "processed_events": service.events_processed,
        "input_elapsed_seconds": input_done - started,
        "drain_seconds": drained - input_done,
        "achieved_processing_rate": count / (drained - started),
        "max_queue": dispatcher.max_queue_depth,
        "dropped_events": dispatcher.dropped_events,
        "ordering_violations": service.ordering_violations,
        **{key: value for key, value in summary.items()
           if key.endswith("_ms")},
    }


async def run(args):
    rows = []
    for rate in args.rates:
        rows.append(await benchmark(rate, args.duration, args.queue_size))
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=int, default=[3000, 5000, 10000])
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--queue-size", type=int, default=100_000)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
