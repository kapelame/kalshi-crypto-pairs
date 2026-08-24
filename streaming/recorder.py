"""Read-only orchestration: REST bootstrap/recovery plus WebSocket raw recording."""

import asyncio
import logging
from collections import Counter

from . import ASSET_SERIES
from .events import RawEvent, StreamSchemaError, make_ws_event
from .contracts import ContractRegistry
from .dispatcher import CausalIngestDispatcher, LiveSignalService
from .health import HealthMonitor
from .raw_store import RawEventStore
from .rest import RestDataClient
from .timeutil import (exchange_timestamp, iso_utc, persistence_latency_ms,
                       receive_latency_ms, request_latency_ms)
from .websocket import KalshiWebSocketClient


LOGGER = logging.getLogger("kalshi.streaming.recorder")
LIFECYCLE_TYPES = {
    "market_lifecycle", "market_lifecycle_v2", "market_created", "created",
    "activated", "deactivated", "close_date_updated", "determined", "settled",
    "metadata_updated", "event_lifecycle",
}


def build_reset_event(asset, old_market, new_market, prior_final, received_at=None,
                      request_started_at=None):
    received = received_at or iso_utc()
    payload = {
        "previous_ticker": old_market["ticker"],
        "new_ticker": new_market["ticker"],
        "previous_settlement": (None if prior_final is None else
                                prior_final.get("result") or None),
        "new_target": new_market.get("floor_strike"),
        "new_status": new_market.get("status"),
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
        target=new_market.get("floor_strike"),
        request_started_at=request_started_at)


class StreamRecorder:
    def __init__(self, db_path="kalshi_stream_raw.db", discovery_seconds=5,
                 underlying_seconds=2, dotenv_path=".env", live_service=None,
                 dispatcher_queue_size=100_000):
        self.store = RawEventStore(db_path)
        self.rest = RestDataClient()
        self.registry = ContractRegistry()
        self.health = HealthMonitor(self.registry)
        self.discovery_seconds = discovery_seconds
        self.underlying_seconds = underlying_seconds
        self.dotenv_path = dotenv_path
        self.markets = {}
        self.ws = None
        self.stop_event = asyncio.Event()
        self.counts_by_type = Counter()
        self.counts_by_asset = Counter()
        self.receive_latencies = []
        self.request_latencies = []
        self.persistence_latencies = []
        self.stale_markets = set()
        self.live_service = live_service or LiveSignalService()
        self.dispatcher = CausalIngestDispatcher(
            self.store, self.live_service, max_queue=dispatcher_queue_size,
            unhealthy_threshold=max(1, dispatcher_queue_size // 2))

    async def _append(self, event):
        envelope = await self.dispatcher.publish(event)
        persisted_timestamp = envelope.event["_persistence_timestamp"]
        self.counts_by_type[event.event_type] += 1
        self.counts_by_asset[event.asset] += 1
        latency = receive_latency_ms(event)
        if latency is not None:
            self.receive_latencies.append(latency)
        request_latency = request_latency_ms(event)
        if request_latency is not None:
            self.request_latencies.append(request_latency)
        persistence_latency = persistence_latency_ms(
            event, persisted_timestamp)
        if persistence_latency is not None:
            self.persistence_latencies.append(persistence_latency)
        return envelope

    async def _append_rest_ticker(self, asset, market, received, request_started_at=None):
        await self._append(RawEvent(
            event_type="ticker", asset=asset, market_ticker=market["ticker"],
            series_ticker=ASSET_SERIES[asset],
            exchange_timestamp=market.get("updated_time"),
            local_receive_timestamp=received, processing_timestamp=iso_utc(),
            source="kalshi_rest_snapshot", raw_payload=market,
            contract_open_time=market.get("open_time"),
            contract_close_time=market.get("close_time"),
            target=market.get("floor_strike"),
            request_started_at=request_started_at))

    async def bootstrap(self):
        self.store.open()
        await self.dispatcher.start(catch_up=True)
        await self.rest.open()
        discoveries = await self.rest.discover_all(timed=True)
        self.markets = {asset: observation.value
                        for asset, observation in discoveries.items()}
        for asset, market in self.markets.items():
            discovery = discoveries[asset]
            book_observation = await self.rest.orderbook(market["ticker"], timed=True)
            raw_book, book = book_observation.value
            self.health.seed_market(
                asset, market, book, book_observation.response_received_at)
            await self._append_rest_ticker(
                asset, market, discovery.response_received_at,
                discovery.request_started_at)
            await self._append(RawEvent(
                event_type="market_lifecycle", asset=asset,
                market_ticker=market["ticker"], series_ticker=ASSET_SERIES[asset],
                exchange_timestamp=market.get("updated_time"),
                local_receive_timestamp=discovery.response_received_at,
                processing_timestamp=iso_utc(),
                source="kalshi_rest_discovery", raw_payload=market,
                contract_open_time=market.get("open_time"),
                contract_close_time=market.get("close_time"),
                target=market.get("floor_strike"),
                request_started_at=discovery.request_started_at))
            await self._append(RawEvent(
                event_type="orderbook_snapshot", asset=asset,
                market_ticker=market["ticker"], series_ticker=ASSET_SERIES[asset],
                exchange_timestamp=None,
                local_receive_timestamp=book_observation.response_received_at,
                processing_timestamp=iso_utc(), source="kalshi_rest_recovery",
                raw_payload=raw_book, contract_open_time=market.get("open_time"),
                contract_close_time=market.get("close_time"),
                target=market.get("floor_strike"),
                request_started_at=book_observation.request_started_at))

    async def on_ws_state(self, connected):
        self.health.set_connected(connected)

    async def on_ws_message(self, payload, received_at, sequence_healthy,
                            sequence_generation=None):
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
        try:
            record = self.registry.record(ticker)
            if record is None:
                return
            market = record.as_market()
            if kind in ("ticker", "trade", "orderbook_snapshot", "orderbook_delta"):
                self.health.update_ws(payload, received_at, sequence_healthy)
                event = make_ws_event(payload, asset, ASSET_SERIES[asset], market,
                                      received_at, sequence_generation)
            elif kind in LIFECYCLE_TYPES:
                record = self.health.apply_lifecycle(asset, ticker, payload, received_at)
                market = record.as_market()
                if self.registry.expected_by_asset.get(asset) == ticker:
                    self.markets[asset] = {**self.markets[asset], **market}
                event = RawEvent(
                    event_type="market_lifecycle", asset=asset,
                    market_ticker=ticker, series_ticker=ASSET_SERIES[asset],
                    exchange_timestamp=exchange_timestamp(payload),
                    local_receive_timestamp=received_at,
                    processing_timestamp=iso_utc(), source="kalshi_websocket",
                    raw_payload=payload, contract_open_time=market.get("open_time"),
                    contract_close_time=market.get("close_time"),
                    target=market.get("floor_strike"), sequence=payload.get("seq"),
                    sequence_generation=sequence_generation)
            else:
                return
            await self._append(event)
            if not sequence_healthy:
                if kind in ("orderbook_snapshot", "orderbook_delta"):
                    book_observation = await self.rest.orderbook(ticker, timed=True)
                    raw_book, book = book_observation.value
                    recovered_at = book_observation.response_received_at
                    self.health.seed_market(asset, market, book, recovered_at)
                    self.health.assets[asset].sequence_healthy = False
                    await self._append(RawEvent(
                        event_type="orderbook_snapshot", asset=asset,
                        market_ticker=ticker, series_ticker=ASSET_SERIES[asset],
                        exchange_timestamp=None, local_receive_timestamp=recovered_at,
                        processing_timestamp=iso_utc(), source="kalshi_rest_recovery",
                        raw_payload=raw_book, contract_open_time=market.get("open_time"),
                        contract_close_time=market.get("close_time"),
                        target=market.get("floor_strike"),
                        request_started_at=book_observation.request_started_at))
                if self.ws:
                    await self.ws.request_reconnect(b"sequence gap")
        except StreamSchemaError as exc:
            LOGGER.error("WebSocket schema failure for %s: %s", ticker, exc)

    async def discovery_loop(self):
        while not self.stop_event.is_set():
            await asyncio.sleep(self.discovery_seconds)
            try:
                changed = False
                for asset in ASSET_SERIES:
                    try:
                        discovery = await self.rest.discover_asset(asset, timed=True)
                        new_market = discovery.value
                    except Exception as exc:
                        LOGGER.warning("REST rollover discovery pending for %s: %s", asset, exc)
                        continue
                    old = self.markets[asset]
                    if old["ticker"] == new_market["ticker"]:
                        # Same ticker metadata can improve after activation (notably target).
                        received = discovery.response_received_at
                        self.registry.update(asset, new_market["ticker"], new_market,
                                             source="kalshi_rest_discovery", timestamp=received)
                        self.health.refresh_market_metadata(asset, new_market, received)
                        merged = self.registry.record(new_market["ticker"]).as_market()
                        # Retain quote/volume fields from the newest REST payload too.
                        self.markets[asset] = {**new_market, **merged}
                        continue
                    changed = True
                    try:
                        prior_observation = await self.rest.market(old["ticker"], timed=True)
                        prior_final = prior_observation.value
                        reset_received = prior_observation.response_received_at
                    except Exception as exc:
                        LOGGER.warning("prior settlement lookup failed for %s: %s",
                                       old["ticker"], exc)
                        prior_final = None
                        reset_received = iso_utc()
                    await self._append(build_reset_event(
                        asset, old, new_market, prior_final, reset_received,
                        discovery.request_started_at))
                    book_observation = await self.rest.orderbook(
                        new_market["ticker"], timed=True)
                    raw_book, book = book_observation.value
                    self.health.seed_market(
                        asset, new_market, book, book_observation.response_received_at)
                    await self._append_rest_ticker(
                        asset, new_market, discovery.response_received_at,
                        discovery.request_started_at)
                    await self._append(RawEvent(
                        event_type="orderbook_snapshot", asset=asset,
                        market_ticker=new_market["ticker"],
                        series_ticker=ASSET_SERIES[asset], exchange_timestamp=None,
                        local_receive_timestamp=book_observation.response_received_at,
                        processing_timestamp=iso_utc(), source="kalshi_rest_recovery",
                        raw_payload=raw_book,
                        contract_open_time=new_market.get("open_time"),
                        contract_close_time=new_market.get("close_time"),
                        target=new_market.get("floor_strike"),
                        request_started_at=book_observation.request_started_at))
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
                    await self._append(event)
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
                    discovery = await self.rest.discover_asset(asset, timed=True)
                    market = discovery.value
                    if market["ticker"] != self.markets[asset]["ticker"]:
                        continue  # discovery_loop owns rollover/reset ordering
                    book_observation = await self.rest.orderbook(
                        market["ticker"], timed=True)
                    raw_book, book = book_observation.value
                    received = book_observation.response_received_at
                    self.health.seed_market(asset, market, book, received)
                    await self._append_rest_ticker(
                        asset, market, discovery.response_received_at,
                        discovery.request_started_at)
                    await self._append(RawEvent(
                        event_type="orderbook_snapshot", asset=asset,
                        market_ticker=market["ticker"],
                        series_ticker=ASSET_SERIES[asset], exchange_timestamp=None,
                        local_receive_timestamp=received,
                        processing_timestamp=iso_utc(), source="kalshi_rest_fallback",
                        raw_payload=raw_book,
                        contract_open_time=market.get("open_time"),
                        contract_close_time=market.get("close_time"),
                        target=market.get("floor_strike"),
                        request_started_at=book_observation.request_started_at))
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
            self.stop_event.set()
            if self.ws:
                await self.ws.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.rest.close()
            await self.dispatcher.stop()
            self.store.close()

    async def stop(self):
        self.stop_event.set()
        if self.ws:
            await self.ws.stop()
        await self.rest.close()
        await self.dispatcher.stop()
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
            "request_latencies_ms": list(self.request_latencies),
            "persistence_latencies_ms": list(self.persistence_latencies),
            "direct_dispatch": self.dispatcher.summary(),
        }
