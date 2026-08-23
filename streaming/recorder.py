"""Read-only orchestration: REST bootstrap/recovery plus WebSocket raw recording."""

import asyncio
import logging
from collections import Counter

from . import ASSET_SERIES
from .events import RawEvent, StreamSchemaError, make_ws_event
from .health import HealthMonitor
from .raw_store import RawEventStore
from .rest import RestDataClient
from .timeutil import exchange_timestamp, iso_utc, receive_latency_ms
from .websocket import KalshiWebSocketClient


LOGGER = logging.getLogger("kalshi.streaming.recorder")
LIFECYCLE_TYPES = {
    "market_lifecycle", "market_lifecycle_v2", "market_created", "created",
    "activated", "deactivated", "close_date_updated", "determined", "settled",
    "metadata_updated", "event_lifecycle",
}


def build_reset_event(asset, old_market, new_market, prior_final, received_at=None):
    received = received_at or iso_utc()
    payload = {
        "previous_ticker": old_market["ticker"],
        "new_ticker": new_market["ticker"],
        "previous_settlement": (None if prior_final is None else
                                prior_final.get("result") or None),
        "new_target": new_market.get("floor_strike"),
        "opening_timestamp": new_market.get("open_time"),
        "prior_contract_final_market_state": prior_final,
    }
    return RawEvent(
        event_type="contract_reset", asset=asset,
        market_ticker=new_market["ticker"], series_ticker=ASSET_SERIES[asset],
        exchange_timestamp=new_market.get("open_time"),
        local_receive_timestamp=received, processing_timestamp=iso_utc(),
        source="kalshi_rest_rollover", raw_payload=payload,
        contract_open_time=new_market.get("open_time"),
        contract_close_time=new_market.get("close_time"),
        target=new_market.get("floor_strike"))


class StreamRecorder:
    def __init__(self, db_path="kalshi_stream_raw.db", discovery_seconds=5,
                 underlying_seconds=2, dotenv_path=".env"):
        self.store = RawEventStore(db_path)
        self.rest = RestDataClient()
        self.health = HealthMonitor()
        self.discovery_seconds = discovery_seconds
        self.underlying_seconds = underlying_seconds
        self.dotenv_path = dotenv_path
        self.markets = {}
        self.ws = None
        self.stop_event = asyncio.Event()
        self.counts_by_type = Counter()
        self.counts_by_asset = Counter()
        self.receive_latencies = []
        self.stale_markets = set()

    def _append(self, event):
        self.store.append(event)
        self.counts_by_type[event.event_type] += 1
        self.counts_by_asset[event.asset] += 1
        latency = receive_latency_ms(event)
        if latency is not None:
            self.receive_latencies.append(latency)

    def _append_rest_ticker(self, asset, market, received):
        self._append(RawEvent(
            event_type="ticker", asset=asset, market_ticker=market["ticker"],
            series_ticker=ASSET_SERIES[asset],
            exchange_timestamp=market.get("updated_time"),
            local_receive_timestamp=received, processing_timestamp=iso_utc(),
            source="kalshi_rest_snapshot", raw_payload=market,
            contract_open_time=market.get("open_time"),
            contract_close_time=market.get("close_time"),
            target=market.get("floor_strike")))

    async def bootstrap(self):
        self.store.open()
        await self.rest.open()
        self.markets = await self.rest.discover_all()
        for asset, market in self.markets.items():
            received = iso_utc()
            raw_book, book = await self.rest.orderbook(market["ticker"])
            self.health.seed_market(asset, market, book, received)
            self._append_rest_ticker(asset, market, received)
            self._append(RawEvent(
                event_type="market_lifecycle", asset=asset,
                market_ticker=market["ticker"], series_ticker=ASSET_SERIES[asset],
                exchange_timestamp=market.get("updated_time"),
                local_receive_timestamp=received, processing_timestamp=iso_utc(),
                source="kalshi_rest_discovery", raw_payload=market,
                contract_open_time=market.get("open_time"),
                contract_close_time=market.get("close_time"),
                target=market.get("floor_strike")))
            self._append(RawEvent(
                event_type="orderbook_snapshot", asset=asset,
                market_ticker=market["ticker"], series_ticker=ASSET_SERIES[asset],
                exchange_timestamp=None, local_receive_timestamp=received,
                processing_timestamp=iso_utc(), source="kalshi_rest_recovery",
                raw_payload=raw_book, contract_open_time=market.get("open_time"),
                contract_close_time=market.get("close_time"),
                target=market.get("floor_strike")))

    async def on_ws_state(self, connected):
        self.health.set_connected(connected)

    async def on_ws_message(self, payload, received_at, sequence_healthy):
        kind = payload.get("type")
        if kind in ("subscribed", "ok"):
            return
        if kind == "error":
            LOGGER.error("Kalshi WebSocket error: %s", payload.get("msg"))
            return
        msg = payload.get("msg", {})
        ticker = msg.get("market_ticker")
        asset = self.health.ticker_to_asset.get(ticker)
        if asset is None:
            return
        market = self.markets[asset]
        try:
            if kind in ("ticker", "trade", "orderbook_snapshot", "orderbook_delta"):
                self.health.update_ws(payload, received_at, sequence_healthy)
                event = make_ws_event(payload, asset, ASSET_SERIES[asset], market,
                                      received_at)
            elif kind in LIFECYCLE_TYPES:
                event = RawEvent(
                    event_type="market_lifecycle", asset=asset,
                    market_ticker=ticker, series_ticker=ASSET_SERIES[asset],
                    exchange_timestamp=exchange_timestamp(payload),
                    local_receive_timestamp=received_at,
                    processing_timestamp=iso_utc(), source="kalshi_websocket",
                    raw_payload=payload, contract_open_time=market.get("open_time"),
                    contract_close_time=market.get("close_time"),
                    target=market.get("floor_strike"), sequence=payload.get("seq"))
            else:
                return
            self._append(event)
            if not sequence_healthy and kind in ("orderbook_snapshot", "orderbook_delta"):
                raw_book, book = await self.rest.orderbook(ticker)
                recovered_at = iso_utc()
                self.health.seed_market(asset, market, book, recovered_at)
                self.health.assets[asset].sequence_healthy = False
                self._append(RawEvent(
                    event_type="orderbook_snapshot", asset=asset,
                    market_ticker=ticker, series_ticker=ASSET_SERIES[asset],
                    exchange_timestamp=None, local_receive_timestamp=recovered_at,
                    processing_timestamp=iso_utc(), source="kalshi_rest_recovery",
                    raw_payload=raw_book, contract_open_time=market.get("open_time"),
                    contract_close_time=market.get("close_time"),
                    target=market.get("floor_strike")))
                if self.ws:
                    await self.ws.request_reconnect(b"sequence gap")
        except StreamSchemaError as exc:
            LOGGER.error("WebSocket schema failure for %s: %s", ticker, exc)

    async def discovery_loop(self):
        while not self.stop_event.is_set():
            await asyncio.sleep(self.discovery_seconds)
            try:
                latest = await self.rest.discover_all()
                changed = False
                for asset, new_market in latest.items():
                    old = self.markets[asset]
                    if old["ticker"] == new_market["ticker"]:
                        continue
                    changed = True
                    try:
                        prior_final = await self.rest.market(old["ticker"])
                    except Exception as exc:
                        LOGGER.warning("prior settlement lookup failed for %s: %s",
                                       old["ticker"], exc)
                        prior_final = None
                    received = iso_utc()
                    self._append(build_reset_event(
                        asset, old, new_market, prior_final, received))
                    raw_book, book = await self.rest.orderbook(new_market["ticker"])
                    self.health.seed_market(asset, new_market, book, received)
                    self._append_rest_ticker(asset, new_market, received)
                    self._append(RawEvent(
                        event_type="orderbook_snapshot", asset=asset,
                        market_ticker=new_market["ticker"],
                        series_ticker=ASSET_SERIES[asset], exchange_timestamp=None,
                        local_receive_timestamp=received,
                        processing_timestamp=iso_utc(), source="kalshi_rest_recovery",
                        raw_payload=raw_book,
                        contract_open_time=new_market.get("open_time"),
                        contract_close_time=new_market.get("close_time"),
                        target=new_market.get("floor_strike")))
                    self.markets[asset] = new_market
                if changed and self.ws:
                    await self.ws.replace_tickers(
                        [market["ticker"] for market in self.markets.values()])
            except Exception:
                LOGGER.exception("REST discovery/rollover check failed")

    async def underlying_loop(self):
        while not self.stop_event.is_set():
            try:
                for event in await self.rest.underlying_events(self.markets):
                    self._append(event)
            except Exception:
                LOGGER.exception("underlying reference-price poll failed")
            await asyncio.sleep(self.underlying_seconds)

    async def health_loop(self):
        while not self.stop_event.is_set():
            await asyncio.sleep(10)
            for asset, state in self.health.assets.items():
                age = state.orderbook_freshness_ms()
                if age is None or age <= 30_000:
                    continue
                try:
                    market = await self.rest.discover_asset(asset)
                    if market["ticker"] != self.markets[asset]["ticker"]:
                        continue  # discovery_loop owns rollover/reset ordering
                    raw_book, book = await self.rest.orderbook(market["ticker"])
                    received = iso_utc()
                    self.health.seed_market(asset, market, book, received)
                    self._append_rest_ticker(asset, market, received)
                    self._append(RawEvent(
                        event_type="orderbook_snapshot", asset=asset,
                        market_ticker=market["ticker"],
                        series_ticker=ASSET_SERIES[asset], exchange_timestamp=None,
                        local_receive_timestamp=received,
                        processing_timestamp=iso_utc(), source="kalshi_rest_fallback",
                        raw_payload=raw_book,
                        contract_open_time=market.get("open_time"),
                        contract_close_time=market.get("close_time"),
                        target=market.get("floor_strike")))
                except Exception:
                    LOGGER.exception("REST stale-market fallback failed for %s", asset)
            LOGGER.info("stream health\n%s", self.health.render())

    async def run(self, duration=None):
        await self.bootstrap()
        self.ws = KalshiWebSocketClient(
            [market["ticker"] for market in self.markets.values()],
            self.on_ws_message, self.on_ws_state, dotenv_path=self.dotenv_path)
        tasks = [asyncio.create_task(self.ws.run_forever()),
                 asyncio.create_task(self.discovery_loop()),
                 asyncio.create_task(self.underlying_loop()),
                 asyncio.create_task(self.health_loop())]
        try:
            if duration is None:
                await self.stop_event.wait()
            else:
                await asyncio.wait_for(self.stop_event.wait(), duration)
        except asyncio.TimeoutError:
            pass
        finally:
            await self.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        self.stop_event.set()
        if self.ws:
            await self.ws.stop()
        await self.rest.close()
        self.store.close()

    def summary(self, elapsed_seconds):
        rate = sum(self.counts_by_type.values()) / elapsed_seconds if elapsed_seconds else 0
        return {
            "events_by_type": dict(self.counts_by_type),
            "events_by_asset": dict(self.counts_by_asset),
            "message_rate": rate,
            "missing_sequences": 0 if self.ws is None else self.ws.sequence.gaps,
            "reconnects": 0 if self.ws is None else self.ws.reconnects,
            "latencies_ms": list(self.receive_latencies),
        }
