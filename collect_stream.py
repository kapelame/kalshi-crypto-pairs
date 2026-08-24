#!/usr/bin/env python3
"""Safe read-only five-asset Kalshi stream recorder. Contains no order API code."""

import argparse
import asyncio
import logging
import time

from streaming.auth import CredentialError, load_credentials
from streaming.recorder import StreamRecorder


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="kalshi_stream_raw.db")
    parser.add_argument("--env-file", default=".env")
    return parser.parse_args()


async def run(args):
    load_credentials(args.env_file)
    recorder = StreamRecorder(db_path=args.db, dotenv_path=args.env_file)
    started = time.monotonic()
    try:
        await recorder.run()
    finally:
        elapsed = time.monotonic() - started
        print("\nCollection summary")
        for key, value in recorder.summary(elapsed).items():
            if not key.endswith("latencies_ms"):
                print(f"{key}: {value}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = arguments()
    try:
        asyncio.run(run(args))
    except CredentialError as exc:
        raise SystemExit(f"WebSocket credentials unavailable: {exc}")
    except KeyboardInterrupt:
        print("Stopped cleanly by user")


if __name__ == "__main__":
    main()
