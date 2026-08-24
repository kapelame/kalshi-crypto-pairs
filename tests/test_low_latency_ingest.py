import asyncio
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from monitor_signals import IncrementalRawTail, LiveSignalMonitor
from signals.engine import SignalEngine
from signals.replay import RawEventReader
from streaming.events import RawEvent
from streaming.raw_store import INGEST_LEDGER, RawEventStore
from streaming.rest import RestDataClient
from streaming.timeutil import parse_timestamp


BASE = datetime(2026, 8, 24, 4, 0, tzinfo=timezone.utc)


def iso(offset):
    return (BASE + timedelta(seconds=offset)).isoformat(timespec="microseconds")


KINDS = (
    "market_lifecycle", "contract_reset", "ticker", "trade",
    "orderbook_snapshot", "orderbook_delta", "underlying_price",
)


def event(kind, offset, event_id=None, request_started_at=None):
    ticker = "KXBTC15M-TEST"
    payload = {"market_ticker": ticker}
    if kind == "ticker":
        payload.update({"status": "active", "yes_bid_dollars": ".49",
                        "yes_ask_dollars": ".51", "no_bid_dollars": ".49",
                        "no_ask_dollars": ".51"})
    elif kind == "trade":
        payload = {"type": "trade", "sid": 3, "seq": 1,
                   "msg": {"market_ticker": ticker, "count_fp": "1",
                           "yes_price_dollars": ".50"}}
    elif kind == "orderbook_snapshot":
        payload = {"orderbook_fp": {"yes_dollars": [[".49", "2"]],
                                     "no_dollars": [[".49", "2"]]}}
    elif kind == "orderbook_delta":
        payload = {"type": kind, "sid": 2, "seq": 1,
                   "msg": {"market_ticker": ticker, "side": "yes",
                           "price_dollars": ".49", "delta_fp": "1"}}
    elif kind == "underlying_price":
        payload = {"usd_price": 100.0}
    elif kind == "contract_reset":
        payload = {"previous_ticker": ticker + "-OLD", "new_ticker": ticker,
                   "new_target": 100.0, "new_status": "active"}
    return RawEvent(
        event_type=kind, asset="BTC", market_ticker=ticker, source="test",
        raw_payload=payload, series_ticker="TEST", exchange_timestamp=iso(offset),
        local_receive_timestamp=iso(offset), processing_timestamp=iso(offset + .001),
        contract_open_time=iso(0), contract_close_time=iso(900), target=100.0,
        request_started_at=request_started_at,
        event_id=event_id or f"{kind}-{offset}")


class GlobalIngestStoreTests(unittest.TestCase):
    def test_monotonic_unique_across_all_tables_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.db"
            with RawEventStore(path) as store:
                receipts = [store.append(event(kind, index / 10))
                            for index, kind in enumerate(KINDS)]
            self.assertEqual([r.ingest_sequence for r in receipts], list(range(1, 8)))
            with RawEventStore(path) as restarted:
                receipt = restarted.append(event("ticker", 1, "after-restart"))
                self.assertEqual(receipt.ingest_sequence, 8)
                rows = restarted.connection.execute(
                    f"SELECT ingest_sequence,event_id FROM {INGEST_LEDGER} "
                    "ORDER BY ingest_sequence").fetchall()
                self.assertEqual(len(rows), len({row[0] for row in rows}))
                self.assertEqual([row[0] for row in rows], list(range(1, 9)))

    def test_raw_row_and_ledger_are_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.db"
            with RawEventStore(path) as store:
                store.connection.execute(f"""
                    CREATE TRIGGER reject_ingest BEFORE INSERT ON {INGEST_LEDGER}
                    BEGIN SELECT RAISE(ABORT, 'test rollback'); END
                """)
                store.connection.commit()
                with self.assertRaises(sqlite3.IntegrityError):
                    store.append(event("ticker", 0))
                self.assertEqual(store.connection.execute(
                    "SELECT COUNT(*) FROM ticker_events").fetchone()[0], 0)
                self.assertEqual(store.connection.execute(
                    f"SELECT COUNT(*) FROM {INGEST_LEDGER}").fetchone()[0], 0)

    def test_request_start_is_separate_and_response_availability_orders_ingest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.db"
            rest = event("underlying_price", .120, "rest",
                         request_started_at=iso(0))
            ws = event("orderbook_delta", .050, "ws")
            with RawEventStore(path) as store:
                store.append(ws)     # REST is still in flight.
                store.append(rest)   # Complete response becomes available later.
            replayed = list(RawEventReader(path).read())
            self.assertEqual([row["event_id"] for row in replayed], ["ws", "rest"])
            self.assertEqual(replayed[1]["_request_started_at"], iso(0))
            self.assertGreaterEqual(replayed[1]["local_receive_timestamp"],
                                    replayed[1]["_request_started_at"])


class HttpAvailabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_coinbase_receive_timestamp_is_after_complete_response(self):
        response_completed = None

        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def text(self):
                nonlocal response_completed
                await asyncio.sleep(.01)
                response_completed = datetime.now(timezone.utc)
                return json.dumps({"data": {"rates": {"BTC": "0.01"}}})

        class Session:
            def get(self, _):
                return Response()

        client = RestDataClient(session=Session())
        market = {"BTC": {"ticker": "BTC-T", "open_time": iso(0),
                          "close_time": iso(900), "floor_strike": 100}}
        events = await client.underlying_events(market)
        observation = events[0]
        self.assertLess(parse_timestamp(observation.request_started_at),
                        parse_timestamp(observation.local_receive_timestamp))
        self.assertGreaterEqual(parse_timestamp(observation.local_receive_timestamp),
                                response_completed)

    async def test_async_completion_order_controls_ingest_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.db"
            with RawEventStore(path) as store:
                async def produce(name, delay, kind):
                    await asyncio.sleep(delay)
                    store.append(event(kind, delay, event_id=name,
                                       request_started_at=iso(0) if name == "rest" else None))

                await asyncio.gather(
                    produce("rest", .12, "underlying_price"),
                    produce("ws", .05, "orderbook_delta"),
                    produce("trade", .07, "trade"))
            self.assertEqual(
                [row["event_id"] for row in RawEventReader(path).read()],
                ["ws", "trade", "rest"])


class NewSchemaDeliveryTests(unittest.TestCase):
    def make_db(self, path, count=100):
        with RawEventStore(path) as store:
            for index in range(count):
                kind = KINDS[index % len(KINDS)]
                store.append(event(kind, index / 1000,
                                   event_id=f"event-{index}"))

    def test_monitor_tails_ingest_sequence_immediately_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.db"
            self.make_db(path, 250)
            tail = IncrementalRawTail(path, batch_size=37)
            consumed = []
            while True:
                batch = tail.read_batch()
                consumed.extend(batch)
                if not batch:
                    break
            sequences = [row["_ingest_sequence"] for row in consumed]
            self.assertEqual(sequences, list(range(1, 251)))
            self.assertEqual(len(sequences), len(set(sequences)))
            self.assertEqual(tail.causal_lateness_seconds, 20.0)  # legacy-only setting
            self.assertEqual(len(tail.pending), 0)
            self.assertEqual(tail.backlog, 0)
            tail.close()

    def test_replay_and_live_delivery_are_identical_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.db"
            self.make_db(path, 70)
            first = list(RawEventReader(path, fetch_size=11).read())
            second = list(RawEventReader(path, fetch_size=13).read())
            first_identity = [(e["_ingest_sequence"], e["event_id"]) for e in first]
            self.assertEqual(first_identity,
                             [(e["_ingest_sequence"], e["event_id"]) for e in second])
            tail = IncrementalRawTail(path, batch_size=9)
            live = []
            while True:
                batch = tail.read_batch()
                live.extend(batch)
                if not batch:
                    break
            self.assertEqual(first_identity,
                             [(e["_ingest_sequence"], e["event_id"]) for e in live])
            tail.close()

    def test_high_rate_book_burst_has_no_reorder_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.db"
            with RawEventStore(path) as store:
                for index in range(5000):
                    store.append(event("orderbook_delta", index / 10000,
                                       event_id=f"book-{index}"))
            tail = IncrementalRawTail(path, batch_size=1000)
            total = 0
            while True:
                rows = tail.read_batch()
                total += len(rows)
                if not rows:
                    break
            self.assertEqual(total, 5000)
            self.assertEqual(tail.buffer_high_water, 0)
            tail.close()

    def test_ingest_guard_rejects_decreasing_sequence(self):
        engine = SignalEngine()
        first = event("underlying_price", 0).__dict__ | {
            "_ingest_sequence": 2, "_persistence_timestamp": iso(.01)}
        second = event("underlying_price", 1).__dict__ | {
            "_ingest_sequence": 1, "_persistence_timestamp": iso(1.01)}
        engine.process(first, emit_snapshot=False)
        with self.assertRaisesRegex(ValueError, "raw events are not in causal order"):
            engine.process(second, emit_snapshot=False)

    def test_legacy_database_keeps_timestamp_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            with RawEventStore(path, enable_ingest=False) as store:
                store.append(event("orderbook_delta", .2, "later"))
                store.append(event("trade", .1, "earlier"))
            rows = list(RawEventReader(path).read())
            self.assertEqual([row["event_id"] for row in rows], ["earlier", "later"])
            self.assertTrue(all(row.get("_ingest_sequence") is None for row in rows))


if __name__ == "__main__":
    unittest.main()
