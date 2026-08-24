import asyncio
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from signals.incremental import TradeWindowSet
from signals.replay import RawEventReader, ReplayEngine, TABLE_ORDER, event_order_key
from signals.store import FeatureStore
from streaming import ASSET_SERIES
from streaming.events import RawEvent
from streaming.raw_store import RawEventStore


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def iso(seconds):
    return (BASE + timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def raw(kind, asset, seconds, payload, ticker=None, event_id=None):
    return RawEvent(
        kind, asset, ticker or f"{asset}-T", "test", payload,
        series_ticker=ASSET_SERIES[asset], exchange_timestamp=iso(seconds),
        local_receive_timestamp=iso(seconds), processing_timestamp=iso(seconds),
        contract_open_time=iso(0), contract_close_time=iso(900), target=100,
        event_id=event_id or f"{kind}-{asset}-{seconds}")


class IncrementalAggregateTests(unittest.TestCase):
    def test_trade_windows_match_brute_force(self):
        trades = [(0, 2, "yes"), (3, 5, None), (7, 4, "no"),
                  (18, 8, "yes"), (31, 3, "no"), (59, 6, "yes")]
        windows = TradeWindowSet((5, 15, 30, 60))
        for timestamp, size, direction in trades:
            windows.append(timestamp, size, direction)
        now = 61
        actual = windows.features(now)
        for window in (5, 15, 30, 60):
            selected = [item for item in trades if item[0] >= now-window]
            sizes = [item[1] for item in selected]
            known = [item for item in selected if item[2] in ("yes", "no")]
            yes = sum(item[1] for item in known if item[2] == "yes")
            no = sum(item[1] for item in known if item[2] == "no")
            recent = sum(item[1] for item in selected if item[0] >= now-window/2)/(window/2)
            earlier = sum(item[1] for item in selected if item[0] < now-window/2)/(window/2)
            prefix = f"trade_{window}s_"
            self.assertEqual(actual[prefix+"count"], len(selected))
            self.assertAlmostEqual(actual[prefix+"contracts"], sum(sizes))
            self.assertEqual(actual[prefix+"max_size"], max(sizes) if sizes else None)
            self.assertEqual(actual[prefix+"yes_aggressive"], yes if known else None)
            self.assertEqual(actual[prefix+"no_aggressive"], no if known else None)
            self.assertAlmostEqual(actual[prefix+"volume_acceleration"], recent-earlier)


class StreamingReplayTests(unittest.IsolatedAsyncioTestCase):
    def make_ordered_db(self, path):
        events = [
            raw("ticker", "BTC", 0, {"status": "active", "yes_bid_dollars": ".49",
                "yes_ask_dollars": ".51", "no_bid_dollars": ".49",
                "no_ask_dollars": ".51"}, event_id="a"),
            raw("underlying_price", "BTC", .05, {"usd_price": 100}, event_id="b"),
            raw("trade", "BTC", .1, {"msg": {"market_ticker": "BTC-T",
                "count_fp": "2", "yes_price_dollars": ".5"}}, event_id="c"),
            raw("orderbook_snapshot", "BTC", .2, {"orderbook_fp": {
                "yes_dollars": [[".49", "2"]], "no_dollars": [[".49", "3"]]}},
                event_id="d"),
        ]
        with RawEventStore(path) as store:
            for event in events: store.append(event)
        return events

    def test_streaming_kway_merge_matches_reference_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"raw.db"
            self.make_ordered_db(path)
            streamed = list(RawEventReader(path).read())
            reference = sorted(streamed, key=event_order_key)
            self.assertEqual([e["event_id"] for e in streamed],
                             [e["event_id"] for e in reference])

    async def test_checkpoint_first_event_after_schedule_and_imperfect_state(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory)/"raw.db"
            feature_path = Path(directory)/"features.db"
            events = [
                raw("ticker", "BTC", 0, {"status": "active",
                    "yes_bid_dollars": ".49", "yes_ask_dollars": ".51",
                    "no_bid_dollars": ".49", "no_ask_dollars": ".51"}),
                raw("orderbook_snapshot", "BTC", .1, {"orderbook_fp": {
                    "yes_dollars": [[".49", "2"]], "no_dollars": [[".49", "3"]]}}),
                raw("underlying_price", "BTC", 29.9, {"usd_price": 100}),
                raw("underlying_price", "BTC", 30.2, {"usd_price": 101}),
                raw("underlying_price", "BTC", 31, {"usd_price": 102}),
            ]
            with RawEventStore(raw_path) as raw_store:
                for event in events: raw_store.append(event)
            with FeatureStore(feature_path, batch_size=100) as feature_store:
                result = await ReplayEngine(raw_path, feature_store).run()
            self.assertEqual(result.checkpoints_generated, 1)
            connection = sqlite3.connect(feature_path)
            row = connection.execute(
                "SELECT checkpoint_seconds,scheduled_timestamp,timestamp,"
                "checkpoint_delay_ms,eligible,excluded_reasons_json "
                "FROM feature_snapshots WHERE snapshot_kind='checkpoint'").fetchone()
            connection.close()
            self.assertEqual(row[0], 30)
            self.assertAlmostEqual(row[2]-row[1], .2)
            self.assertAlmostEqual(row[3], 200, places=3)
            self.assertEqual(row[4], 0)
            self.assertIn("STALE_QUOTE", json.loads(row[5]))

    async def test_checkpoint_digest_is_deterministic_and_memory_collection_is_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"raw.db"
            self.make_ordered_db(path)
            first = await ReplayEngine(path).run(snapshot_mode="checkpoints")
            second = await ReplayEngine(path).run(snapshot_mode="checkpoints")
            self.assertEqual(first.digest, second.digest)
            self.assertIsNone(first.collected_snapshots)

    async def test_persistence_modes_default_to_sparse_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"raw.db"
            events = [raw("ticker", "BTC", 0, {"status": "active",
                "yes_bid_dollars": ".49", "yes_ask_dollars": ".51",
                "no_bid_dollars": ".49", "no_ask_dollars": ".51"})]
            for index in range(100):
                events.append(raw("orderbook_delta", "BTC", .01 + index/100,
                    {"msg": {"market_ticker": "BTC-T", "side": "yes",
                     "price_dollars": ".49", "delta_fp": "1"}},
                    event_id=f"delta-{index}"))
            with RawEventStore(path) as store:
                for event in events: store.append(event)
            checkpoints = await ReplayEngine(path).run()
            diagnostics = await ReplayEngine(path).run(
                snapshot_mode="diagnostic", diagnostic_hz=1)
            all_rows = await ReplayEngine(path).run(snapshot_mode="all")
            self.assertEqual(checkpoints.snapshots_generated, 0)
            self.assertLessEqual(diagnostics.diagnostic_snapshots, 2)
            self.assertEqual(all_rows.snapshots_generated, len(events))


if __name__ == "__main__": unittest.main()
