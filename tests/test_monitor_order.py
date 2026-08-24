import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from monitor_signals import IncrementalRawTail
from signals.engine import SignalEngine
from signals.ordering import event_order_key
from signals.replay import RawEventReader
from streaming.events import RawEvent
from streaming.raw_store import RawEventStore


BASE = datetime(2026, 8, 24, 2, 54, 9, tzinfo=timezone.utc)


def stamp(offset):
    return (BASE + timedelta(seconds=offset)).isoformat(timespec="microseconds")


def raw(kind, offset, processing=None, asset="BTC", event_id=None, sequence=None):
    ticker = f"KX{asset}15M-TEST"
    payload = {"market_ticker": ticker}
    if kind == "orderbook_delta":
        payload = {"type": kind, "sid": 2, "seq": sequence or 1,
                   "msg": {"market_ticker": ticker, "side": "yes",
                           "price_dollars": "0.50", "delta_fp": "1"}}
    elif kind == "trade":
        payload = {"type": kind, "sid": 3, "seq": sequence or 1,
                   "msg": {"market_ticker": ticker, "yes_price_dollars": "0.5",
                           "count_fp": "1"}}
    elif kind == "contract_reset":
        payload = {"previous_ticker": ticker + "-OLD", "new_ticker": ticker,
                   "new_target": 100.0, "new_status": "active"}
    return RawEvent(
        event_id=event_id or f"{kind}-{offset}-{processing}", event_type=kind,
        asset=asset, market_ticker=ticker, series_ticker="TEST",
        exchange_timestamp=stamp(offset), local_receive_timestamp=stamp(offset),
        processing_timestamp=stamp(offset if processing is None else processing),
        source="test", raw_payload=payload, contract_open_time=stamp(-9),
        contract_close_time=stamp(891), target=100.0, sequence=sequence,
        sequence_generation=1 if sequence is not None else None)


class MutableClock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class LiveTailOrderingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "raw.db"
        # Phase 3.2.1 regressions explicitly exercise the legacy timestamp path.
        self.store = RawEventStore(self.path, enable_ingest=False).open()

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_actual_forensic_failure_pattern_is_buffered(self):
        clock = MutableClock(BASE + timedelta(seconds=.64))
        tail = IncrementalRawTail(self.path, batch_size=128,
                                  causal_lateness_seconds=.20, clock=clock)
        newer = raw("orderbook_delta", .639445, .639568,
                    event_id="dd7bc7de-fa1c-4cb8-85bc-6c9f79e8ab75", sequence=31707)
        self.store.append(newer)
        self.assertEqual(tail.read_batch(), [])
        delayed = raw("underlying_price", .553601, .669565,
                      event_id="dc950b30-9a23-4cd1-8360-fe1e5638b09c")
        self.store.append(delayed)
        clock.value = BASE + timedelta(seconds=.90)
        events = tail.read_batch()
        self.assertEqual([event["event_id"] for event in events],
                         [delayed.event_id, newer.event_id])
        tail.close()

    def test_cross_table_poll_and_128_batch_boundary(self):
        clock = MutableClock(BASE + timedelta(seconds=10))
        for index in range(128):
            self.store.append(raw("orderbook_delta", index / 1000,
                                  index / 1000 + .001, event_id=f"book-{index}",
                                  sequence=index + 1))
        self.store.append(raw("trade", .095, .101, event_id="trade-before-end",
                              sequence=1))
        tail = IncrementalRawTail(self.path, batch_size=128,
                                  causal_lateness_seconds=.2, clock=clock)
        events = tail.drain()
        self.assertEqual(events, sorted(events, key=event_order_key))
        self.assertLess([e["event_id"] for e in events].index("trade-before-end"),
                        [e["event_id"] for e in events].index("book-127"))
        tail.close()

    def test_identical_timestamp_ties_match_table_priority_then_rowid(self):
        same = .5
        self.store.append(raw("orderbook_delta", same, same, event_id="book-1", sequence=1))
        self.store.append(raw("trade", same, same, event_id="trade", sequence=1))
        self.store.append(raw("orderbook_delta", same, same, event_id="book-2", sequence=2))
        tail = IncrementalRawTail(self.path, batch_size=1)
        events = tail.drain()
        self.assertEqual([e["event_id"] for e in events], ["trade", "book-1", "book-2"])
        tail.close()

    def test_catchup_live_exactly_once_and_order_equivalent_to_replay(self):
        for index in range(12):
            kind = "trade" if index % 3 == 0 else "orderbook_delta"
            self.store.append(raw(kind, index / 10, index / 10 + .01,
                                  event_id=f"initial-{index}", sequence=index + 1))
        tail = IncrementalRawTail(self.path, batch_size=5)
        consumed = []
        while True:
            batch = tail.read_batch(final=True)
            consumed.extend(batch)
            if not batch and not tail.pending:
                break
        for index in range(5):
            self.store.append(raw("orderbook_delta", 2 + index / 10,
                                  2.01 + index / 10, event_id=f"live-{index}",
                                  sequence=20 + index))
        consumed.extend(tail.drain())
        reference = list(RawEventReader(self.path).read())
        identities = [(e["_priority"], e["_rowid"]) for e in consumed]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(identities,
                         [(e["_priority"], e["_rowid"]) for e in reference])
        tail.close()

    def test_concurrent_cross_table_append_is_ordered_and_exactly_once(self):
        # Use current event time so the reorder watermark, not static age, does the work.
        now = datetime.now(timezone.utc)
        clock = MutableClock(now)
        tail = IncrementalRawTail(self.path, batch_size=17,
                                  causal_lateness_seconds=.10, clock=clock)
        written = []

        def append_event(kind, receive_ms, process_ms, index):
            event = raw(kind, receive_ms / 1000, process_ms / 1000,
                        event_id=f"concurrent-{index}", sequence=index + 1)
            # Shift the synthetic BASE timestamps to the live clock epoch.
            delta = now - BASE
            event = replace(
                event,
                local_receive_timestamp=(datetime.fromisoformat(
                    event.local_receive_timestamp) + delta).isoformat(timespec="microseconds"),
                processing_timestamp=(datetime.fromisoformat(
                    event.processing_timestamp) + delta).isoformat(timespec="microseconds"))
            self.store.append(event)
            written.append(event.event_id)

        schedule = [("orderbook_delta", 60, 61), ("trade", 50, 70),
                    ("orderbook_delta", 80, 81), ("contract_reset", 40, 90)]
        for index, args in enumerate(schedule):
            append_event(*args, index)
            clock.value = now + timedelta(milliseconds=65 + index * 10)
            tail.read_batch()
        clock.value = now + timedelta(seconds=.30)
        consumed = tail.read_batch() + tail.drain()
        ids = [event["event_id"] for event in consumed]
        self.assertEqual(set(ids), set(written))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(consumed, sorted(consumed, key=event_order_key))
        self.assertLessEqual(tail.buffer_high_water, len(written))
        tail.close()

    def test_async_writer_and_reader_preserve_exactly_once_order(self):
        now = datetime.now(timezone.utc)
        written = []
        schedule = [("orderbook_delta", 30, 31), ("trade", 20, 40),
                    ("orderbook_delta", 50, 51), ("contract_reset", 10, 60)]

        def writer():
            with RawEventStore(self.path, enable_ingest=False) as store:
                for index, (kind, receive_ms, process_ms) in enumerate(schedule):
                    event = raw(kind, receive_ms / 1000, process_ms / 1000,
                                event_id=f"threaded-{index}", sequence=index + 1)
                    delta = now - BASE
                    event = replace(
                        event,
                        local_receive_timestamp=(datetime.fromisoformat(
                            event.local_receive_timestamp) + delta).isoformat(
                                timespec="microseconds"),
                        processing_timestamp=(datetime.fromisoformat(
                            event.processing_timestamp) + delta).isoformat(
                                timespec="microseconds"))
                    store.append(event)
                    written.append(event.event_id)
                    time.sleep(.01)

        tail = IncrementalRawTail(self.path, batch_size=2,
                                  causal_lateness_seconds=.10)
        thread = threading.Thread(target=writer)
        thread.start()
        consumed = []
        while thread.is_alive():
            consumed.extend(tail.read_batch())
            time.sleep(.003)
        thread.join()
        time.sleep(.11)
        consumed.extend(tail.read_batch())
        consumed.extend(tail.drain())
        ids = [event["event_id"] for event in consumed]
        self.assertEqual(set(ids), set(written))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(consumed, sorted(consumed, key=event_order_key))
        tail.close()

    def test_signal_engine_guard_remains(self):
        engine = SignalEngine()
        first = raw("underlying_price", 2, 2).__dict__
        second = raw("underlying_price", 1, 1).__dict__
        engine.process(first, emit_snapshot=False)
        with self.assertRaisesRegex(ValueError, "raw events are not in causal order"):
            engine.process(second, emit_snapshot=False)

    def test_reorder_buffer_has_hard_bound_and_backpressures_source(self):
        for index in range(30):
            self.store.append(raw("orderbook_delta", index / 1000,
                                  index / 1000, event_id=f"bounded-{index}",
                                  sequence=index + 1))
        clock = MutableClock(BASE + timedelta(milliseconds=10))
        tail = IncrementalRawTail(
            self.path, batch_size=20, causal_lateness_seconds=10,
            clock=clock, max_buffer_events=7)
        tail.read_batch()
        self.assertLessEqual(len(tail.pending), 7)
        self.assertGreater(tail.backlog, 0)
        tail.close()


if __name__ == "__main__":
    unittest.main()
