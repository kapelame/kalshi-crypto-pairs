#!/usr/bin/env python3
"""Read-only terminal monitor derived from an actively appended raw database."""

import argparse
import asyncio
import os
import time

from signals.engine import SignalEngine
from signals.replay import RawEventReader


def fmt(value, signed=False):
    if value is None: return "---"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def render(raw_db):
    engine = SignalEngine(); latest = {}
    for event in RawEventReader(raw_db).read():
        snapshot = engine.process(event)
        if snapshot: latest[snapshot["asset"]] = snapshot
    lines = ["ASSET   P_UP    dP5    dP30    ACC30   DIST_Z   BOOK_IMB TRADE_IMB  REMAIN"]
    for asset in ("BTC", "ETH", "SOL", "XRP", "DOGE"):
        snapshot = latest.get(asset); f = {} if snapshot is None else snapshot["features"]
        lines.append(f"{asset:<5} {fmt(f.get('midpoint_up_probability')):>6} "
            f"{fmt(f.get('prob_change_5s'),True):>7} {fmt(f.get('prob_change_30s'),True):>7} "
            f"{fmt(f.get('prob_acceleration_30s'),True):>8} "
            f"{fmt(f.get('volatility_adjusted_target_distance'),True):>8} "
            f"{fmt(f.get('book_top3_imbalance'),True):>8} "
            f"{fmt(f.get('trade_30s_imbalance'),True):>9} "
            f"{fmt(f.get('seconds_to_contract_close')):>7}")
    basket = next(iter(latest.values()))["basket"] if latest else {}
    regime = next(iter(latest.values()))["features"].get("regime_label") if latest else "---"
    lines += ["", f"BREADTH: {basket.get('breadth_direction','---')}",
              f"MEAN V30: {fmt(basket.get('mean_probability_velocity'),True)}",
              f"DISPERSION: {basket.get('dispersion_band','---')} "
              f"({fmt(basket.get('normalized_dispersion_score'))})",
              f"BTC FACTOR: {fmt(basket.get('btc_probability_minus_alt_basket'),True)}",
              f"REGIME: {regime}"]
    return "\n".join(lines)


async def run(args):
    started = time.monotonic()
    while True:
        print("\033[2J\033[H" + render(args.db), flush=True)
        if args.duration and time.monotonic() - started >= args.duration: break
        await asyncio.sleep(args.refresh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="kalshi_stream_raw.db")
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--refresh", type=float, default=1)
    args = parser.parse_args()
    if not os.path.exists(args.db): raise SystemExit(f"Raw database not found: {args.db}")
    try: asyncio.run(run(args))
    except KeyboardInterrupt: print("\nMonitor stopped")


if __name__ == "__main__": main()
