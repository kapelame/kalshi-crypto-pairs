import asyncio
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from pathlib import Path

from signals.engine import SignalEngine
from signals.replay import RawEventReader
from streaming.dispatcher import (CausalIngestDispatcher, LiveSignalService,
                                  RAW_PERSISTENCE_UNHEALTHY)
from streaming.events import RawEvent
from streaming.raw_store import RawEventStore


BASE = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def iso(offset=0):
    return (BASE + timedelta(seconds=offset)).isoformat(timespec="microseconds")


def raw(index, kind="underlying_price", asset="BTC", receive=None):
    receive = iso(index / 1000) if receive is None else receive
    payload = {"usd_price": 100 + index / 1000}
    ticker = "KXBTC15M-DIRECT"
    if kind == "orderbook_delta":
        payload = {"type": kind, "sid": 2, "seq": index + 1,
                   "msg": {"market_ticker": ticker, "side": "yes",
                           "price_dollars": ".49", "delta_fp": "1"}}
    return RawEvent(
        event_type=kind, asset=asset, market_ticker=ticker, source="test",
        raw_payload=payload, local_receive_timestamp=receive,
        processing_timestamp=receive, event_id=f"event-{index}")


class DirectDispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "raw.db"
        self.store = RawEventStore(self.path).open()
        self.service = LiveSignalService()
        self.dispatcher = CausalIngestDispatcher(
            self.store, self.service, max_queue=20_000,
            unhealthy_threshold=10_000)
        await self.dispatcher.start()

    async def asyncTearDown(self):
        await self.dispatcher.stop()
        self.store.close()
        self.temp.cleanup()

    async def test_exactly_once_and_persisted_live_envelope_identity(self):
        envelopes = await asyncio.gather(
            *(self.dispatcher.publish(raw(index)) for index in range(100)))
        self.assertEqual([e.ingest_sequence for e in envelopes], list(range(1, 101)))
        self.assertEqual(self.dispatcher.events_delivered, 100)
        self.assertEqual(self.dispatcher.dropped_events, 0)
        replay = list(RawEventReader(self.path).read())
        live = [envelope.event for envelope in envelopes]
        projection = lambda event: (
            event["_ingest_sequence"], event["event_id"], event["event_type"],
            event["raw_payload"], event["local_receive_timestamp"])
        self.assertEqual(list(map(projection, live)), list(map(projection, replay)))

    async def test_persistence_failure_never_reaches_engine(self):
        self.store.connection.execute("""
            CREATE TRIGGER reject_direct BEFORE INSERT ON raw_ingest_log
            BEGIN SELECT RAISE(ABORT, 'persistence failed'); END
        """)
        self.store.connection.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            await self.dispatcher.publish(raw(1))
        self.assertEqual(self.service.events_processed, 0)
        self.assertEqual(self.dispatcher.health, RAW_PERSISTENCE_UNHEALTHY)
        self.assertEqual(self.store.count("underlying_price"), 0)

    async def test_live_and_replay_final_state_and_order_match(self):
        for index in range(50):
            await self.dispatcher.publish(raw(index))
        replay_service = LiveSignalService()
        replay_service.catch_up(self.path)
        self.assertEqual(list(self.service.consumed_identity),
                         list(replay_service.consumed_identity))
        self.assertEqual(self.service.digest, replay_service.digest)
        self.assertEqual(self.service.engine.event_count,
                         replay_service.engine.event_count)
        self.assertEqual(self.service.engine.assets["BTC"].price.values,
                         replay_service.engine.assets["BTC"].price.values)

    async def _burst(self, total):
        started = time.perf_counter()
        await asyncio.gather(
            *(self.dispatcher.publish(raw(index, "orderbook_delta"))
              for index in range(total)))
        elapsed = time.perf_counter() - started
        self.assertEqual(self.dispatcher.events_delivered, total)
        self.assertEqual(self.dispatcher.dropped_events, 0)
        self.assertLessEqual(self.dispatcher.max_queue_depth, 20_000)
        self.assertGreater(total / elapsed, 300)

    async def test_3k_per_second_synthetic_burst(self):
        await self._burst(3000)

    async def test_5k_per_second_synthetic_burst(self):
        await self._burst(5000)

    async def test_10k_per_second_synthetic_burst(self):
        await self._burst(10000)


class HandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_catchup_to_direct_handoff_has_no_gap_or_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handoff.db"
            store = RawEventStore(path).open()
            for index in range(25):
                store.append(raw(index))
            service = LiveSignalService()
            dispatcher = CausalIngestDispatcher(store, service)
            start_task = asyncio.create_task(dispatcher.start(catch_up=True))
            await asyncio.sleep(0)
            publish_task = asyncio.create_task(dispatcher.publish(raw(25)))
            await start_task
            await publish_task
            await dispatcher.publish(raw(26))
            await dispatcher.stop()
            self.assertEqual([sequence for sequence, _ in service.consumed_identity],
                             list(range(1, 28)))
            self.assertEqual(len(service.consumed_identity),
                             len(set(service.consumed_identity)))
            store.close()


class GuardTests(unittest.TestCase):
    def test_signal_engine_causal_guard_remains(self):
        engine = SignalEngine()
        first = raw(1).__dict__ | {"_ingest_sequence": 2}
        second = raw(2).__dict__ | {"_ingest_sequence": 1}
        engine.process(first, emit_snapshot=False)
        with self.assertRaisesRegex(ValueError, "raw events are not in causal order"):
            engine.process(second, emit_snapshot=False)

    def test_latency_instrumentation_uses_observation_availability(self):
        service = LiveSignalService()
        available = datetime.now(timezone.utc)
        event = raw(1, receive=available.isoformat(timespec="microseconds")).__dict__ | {
            "_ingest_sequence": 1}
        base = available.timestamp()
        with patch("streaming.dispatcher.time.time",
                   side_effect=[base + .0015, base + .0025]):
            service.consume(event)
        self.assertAlmostEqual(service.engine_latency_ms[-1], 1.5, places=3)
        self.assertAlmostEqual(service.state_latency_ms[-1], 2.5, places=3)


class ReconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_sequence_generation_bootstraps_without_direct_order_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RawEventStore(Path(directory) / "reconnect.db").open()
            service = LiveSignalService()
            dispatcher = CausalIngestDispatcher(store, service)
            await dispatcher.start()
            ticker = "KXBTC15M-DIRECT"

            def snapshot(index, generation):
                timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                return RawEvent(
                    event_type="orderbook_snapshot", asset="BTC",
                    market_ticker=ticker, source="kalshi_websocket",
                    raw_payload={"type": "orderbook_snapshot", "sid": 2, "seq": 1,
                                 "msg": {"market_ticker": ticker,
                                         "yes_dollars_fp": [[".49", "2"]],
                                         "no_dollars_fp": [[".51", "2"]]}},
                    local_receive_timestamp=timestamp,
                    processing_timestamp=timestamp, sequence=1,
                    sequence_generation=generation, event_id=f"snapshot-{index}")

            await dispatcher.publish(snapshot(1, 1))
            await dispatcher.publish(snapshot(2, 2))
            await dispatcher.stop()
            self.assertEqual(service.ordering_violations, 0)
            self.assertEqual(service.engine.last_ingest_sequence, 2)
            store.close()


if __name__ == "__main__":
    unittest.main()
