#!/usr/bin/env python3
"""Read-only diagnostic for one current 15-minute Kalshi market."""

import argparse
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from kalshi_api import parse_markets_response, parse_orderbook_response


BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = {
    "BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M",
    "XRP": "KXXRP15M", "DOGE": "KXDOGE15M",
}


def get_json(path, params=None):
    url = f"{BASE}{path}"
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def fmt_price(value):
    return "null" if value is None else f"{value:.4f}c"


def fmt_count(value):
    return "null" if value is None else str(value)


def best(levels):
    return max(levels, key=lambda level: level[0]) if levels else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=SERIES, default="BTC")
    args = parser.parse_args()
    markets = parse_markets_response(get_json(
        "/markets", {"series_ticker": SERIES[args.asset], "status": "open", "limit": 1}))
    if not markets:
        raise RuntimeError(f"Kalshi returned no open {SERIES[args.asset]} market")
    market = markets[0]
    book = parse_orderbook_response(get_json(
        f"/markets/{market['ticker']}/orderbook", {"depth": 10}))

    print(f"ticker: {market['ticker']}")
    print(f"status: {market['status']}")
    print(f"strike: {market.get('floor_strike')}")
    print(f"yes bid: {fmt_price(market['yes_bid'])}")
    print(f"yes ask: {fmt_price(market['yes_ask'])}")
    print(f"no bid: {fmt_price(market['no_bid'])}")
    print(f"no ask: {fmt_price(market['no_ask'])}")
    print(f"last price: {fmt_price(market['last_price'])}")
    print(f"volume: {fmt_count(market['volume'])}")
    print(f"open interest: {fmt_count(market['open_interest'])}")
    print(f"order-book best YES bid: {best(book['yes'])}")
    print(f"order-book best NO bid: {best(book['no'])}")


if __name__ == "__main__":
    main()
