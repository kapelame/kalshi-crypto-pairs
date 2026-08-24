#!/usr/bin/env python3
"""Read-only collector with direct event-driven SignalEngine state delivery."""

import argparse
import asyncio
import json
import logging
import os
import time

from streaming.auth import CredentialError, load_credentials
from streaming.recorder import StreamRecorder


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="kalshi_direct_raw.db")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--render", type=float, default=.1,
                        help="terminal redraw seconds; does not clock feature updates")
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def display(recorder):
    service = recorder.live_service
    snapshots, basket = service.coherent_snapshots()
    state_age = (None if service.state_available_at is None else
                 max(0.0, (time.time() - service.state_available_at) * 1000))
    lines = [
        f"DIRECT ingest_sequence={service.latest_ingest_sequence or '---'} "
        f"watermark={service.raw_watermark or '---'} "
        f"state_timestamp={service.state_timestamp or '---'} "
        f"state_age_ms={'---' if state_age is None else f'{state_age:.1f}'}",
        f"FLOW queue={recorder.dispatcher.queue.qsize()} "
        f"max_queue={recorder.dispatcher.max_queue_depth} "
        f"delivered={recorder.dispatcher.events_delivered} "
        f"dropped={recorder.dispatcher.dropped_events} "
        f"health={recorder.dispatcher.health}",
        "ASSET  TICKER                         ELIGIBLE  REASONS",
    ]
    for asset in ("BTC", "ETH", "SOL", "XRP", "DOGE"):
        snapshot = snapshots.get(asset)
        features = {} if snapshot is None else snapshot["features"]
        ticker = "---" if snapshot is None else snapshot.get("market_ticker") or "---"
        reasons = ",".join(features.get("excluded_reasons", [])) or "---"
        lines.append(f"{asset:<6} {ticker:<30} "
                     f"{'yes' if features.get('eligible') else 'no ':<8} {reasons}")
    lines.extend(("", f"ELIGIBLE: {basket.get('eligible_count', 0)}/5",
                  f"COHERENT WINDOW: {'yes' if basket.get('coherent_window') else 'no'}"))
    return "\n".join(lines)


async def run(args):
    load_credentials(args.env_file)
    recorder = StreamRecorder(db_path=args.db, dotenv_path=args.env_file)

    async def renderer():
        while not recorder.stop_event.is_set():
            await asyncio.sleep(args.render)
            print("\033[2J\033[H" + display(recorder), flush=True)

    render_task = None if args.no_render else asyncio.create_task(renderer())
    started = time.monotonic()
    try:
        await recorder.run(duration=args.duration)
    finally:
        if render_task:
            render_task.cancel()
            await asyncio.gather(render_task, return_exceptions=True)
        summary = recorder.summary(max(time.monotonic() - started, 1e-9))
        print("\nDirect live summary")
        print(json.dumps({key: value for key, value in summary.items()
                          if not key.endswith("latencies_ms")}, indent=2))


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run(arguments()))
    except CredentialError as exc:
        raise SystemExit(f"WebSocket credentials unavailable: {exc}")
    except KeyboardInterrupt:
        print("Stopped cleanly by user")


if __name__ == "__main__":
    main()
