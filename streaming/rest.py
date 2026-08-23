"""Read-only REST discovery, recovery snapshots, and underlying observations."""

import asyncio
import json
from urllib.parse import quote

import aiohttp

from kalshi_api import parse_market, parse_markets_response, parse_orderbook_response
from . import ASSET_SERIES
from .events import RawEvent
from .timeutil import iso_utc


REST_BASE = "https://external-api.kalshi.com/trade-api/v2"
COINBASE_URL = "https://api.coinbase.com/v2/exchange-rates?currency=USD"


class RestDataClient:
    def __init__(self, session=None):
        self.session = session
        self._owns_session = session is None

    async def open(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"Accept": "application/json"})
        return self

    async def close(self):
        if self._owns_session and self.session and not self.session.closed:
            await self.session.close()

    async def _get(self, path):
        if self.session is None:
            await self.open()
        async with self.session.get(f"{REST_BASE}{path}") as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"Kalshi REST HTTP {response.status}: {body[:300]}")
            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Kalshi REST returned malformed JSON") from exc

    async def discover_asset(self, asset):
        series = ASSET_SERIES[asset]
        payload = await self._get(
            f"/markets?series_ticker={quote(series)}&status=open&limit=1")
        markets = parse_markets_response(payload)
        if not markets:
            raise RuntimeError(f"no open market for {asset} ({series})")
        return markets[0]

    async def discover_all(self):
        results = await asyncio.gather(
            *(self.discover_asset(asset) for asset in ASSET_SERIES))
        return dict(zip(ASSET_SERIES, results))

    async def market(self, ticker):
        payload = await self._get(f"/markets/{quote(ticker)}")
        return parse_market(payload.get("market"))

    async def orderbook(self, ticker, depth=100):
        payload = await self._get(f"/markets/{quote(ticker)}/orderbook?depth={depth}")
        return payload, parse_orderbook_response(payload)

    async def underlying_events(self, markets):
        received = iso_utc()
        async with self.session.get(COINBASE_URL) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"Coinbase HTTP {response.status}: {body[:300]}")
            payload = json.loads(body)
        rates = payload.get("data", {}).get("rates", {})
        events = []
        for asset, market in markets.items():
            raw_rate = rates.get(asset)
            price = None if raw_rate is None else 1.0 / float(raw_rate)
            raw = {"currency": asset, "usd_price": price,
                   "coinbase_rate": raw_rate, "response": payload}
            events.append(RawEvent(
                event_type="underlying_price", asset=asset,
                market_ticker=market["ticker"], series_ticker=ASSET_SERIES[asset],
                exchange_timestamp=None, local_receive_timestamp=received,
                processing_timestamp=iso_utc(), source="coinbase_rest",
                raw_payload=raw, contract_open_time=market.get("open_time"),
                contract_close_time=market.get("close_time"),
                target=market.get("floor_strike")))
        return events
