import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from monitor_signals import LiveSignalMonitor, format_monitor
from signals.engine import SignalEngine
from streaming import ASSET_SERIES
from streaming.contracts import ContractRegistry, RolloverState
from streaming.events import RawEvent
from streaming.health import HealthMonitor
from streaming.raw_store import RawEventStore
from streaming.recorder import StreamRecorder


BASE = datetime(2026, 8, 23, 21, 15, tzinfo=timezone.utc)


def iso(seconds):
    return (BASE + timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def raw(kind, asset, seconds, payload, ticker, target=100.0, opened=0, closed=900,
        event_id=None, source="test"):
    return {
        "event_id": event_id or f"{kind}-{asset}-{seconds}", "event_type": kind,
        "asset": asset, "market_ticker": ticker, "series_ticker": ASSET_SERIES[asset],
        "exchange_timestamp": iso(seconds), "local_receive_timestamp": iso(seconds),
        "processing_timestamp": iso(seconds), "source": source, "raw_payload": payload,
        "contract_open_time": iso(opened), "contract_close_time": iso(closed),
        "target": target, "sequence": 1,
    }


def quote(asset, seconds, ticker, probability=.7, target=100.0, opened=0, closed=900):
    bid, ask = probability - .01, probability + .01
    payload = {"ticker": ticker, "status": "active",
               "yes_bid_dollars": str(bid), "yes_ask_dollars": str(ask),
               "no_bid_dollars": str(1-ask), "no_ask_dollars": str(1-bid),
               "last_price_dollars": str(probability), "volume_fp": "1.0",
               "open_interest_fp": "2.0"}
    return raw("ticker", asset, seconds, payload, ticker, target, opened, closed)


def book(asset, seconds, ticker, target=100.0, opened=0, closed=900):
    payload = {"orderbook_fp": {"yes_dollars": [["0.60", "10"]],
                                "no_dollars": [["0.30", "10"]]}}
    return raw("orderbook_snapshot", asset, seconds, payload, ticker, target,
               opened, closed, source="kalshi_rest_recovery")


def reset(asset, seconds, old, new, target, opened=900, closed=1800):
    return raw("contract_reset", asset, seconds, {
        "previous_ticker": old, "new_ticker": new, "new_target": target,
        "new_status": "active", "previous_settlement": "yes"},
        new, target, opened, closed)


class RegistryAndHealthTests(unittest.TestCase):
    def test_same_ticker_null_target_is_repaired_without_later_null_erasure(self):
        registry = ContractRegistry()
        registry.update("DOGE", "D", {"status": "active", "floor_strike": None}, expected=True)
        self.assertIsNone(registry.record("D").target)
        registry.update("DOGE", "D", {"floor_strike": .093782})
        registry.update("DOGE", "D", {"floor_strike": None})
        self.assertEqual(registry.record("D").target, .093782)

    def test_metadata_updated_repairs_only_its_own_ticker(self):
        registry = ContractRegistry()
        registry.update("BTC", "OLD", {"status": "closed", "floor_strike": 77000})
        registry.update("BTC", "NEW", {"status": "active", "floor_strike": None}, expected=True)
        registry.apply_lifecycle("BTC", "NEW", {"type": "market_lifecycle_v2", "msg": {
            "market_ticker": "NEW", "event_type": "metadata_updated", "floor_strike": 78013.52}})
        self.assertEqual(registry.record("NEW").target, 78013.52)
        self.assertEqual(registry.record("OLD").target, 77000)

    def test_old_settlement_never_inherits_new_metadata(self):
        registry = ContractRegistry()
        registry.update("ETH", "OLD", {"status": "active", "floor_strike": 2449.47})
        registry.update("ETH", "NEW", {"status": "active", "floor_strike": 2464.95}, expected=True)
        old = registry.apply_lifecycle("ETH", "OLD", {"msg": {
            "market_ticker": "OLD", "event_type": "settled", "result": "yes"}})
        self.assertEqual(old.target, 2449.47)
        self.assertEqual(old.result, "yes")
        self.assertEqual(registry.record("NEW").target, 2464.95)

    def test_closed_market_and_pending_rollover_are_ineligible(self):
        monitor = HealthMonitor()
        market = {"ticker": "SOL-OLD", "status": "closed", "floor_strike": 95.5,
                  "open_time": iso(0), "close_time": iso(900), "yes_bid": 45,
                  "yes_ask": 46, "no_bid": 54, "no_ask": 55, "volume": 1,
                  "open_interest": 1}
        monitor.seed_market("SOL", market, book={"yes": [], "no": []}, received_at=iso(901))
        decision = monitor.assets["SOL"].eligibility(iso(901))
        self.assertFalse(decision.eligible)
        self.assertIn("CLOSED_MARKET", decision.reasons)
        self.assertEqual(monitor.assets["SOL"].rollover_state, RolloverState.PENDING)

    def test_same_ticker_health_metadata_refresh_repairs_target(self):
        monitor = HealthMonitor()
        market = {"ticker": "DOGE-X", "status": "active", "floor_strike": None,
                  "open_time": iso(0), "close_time": iso(900), "yes_bid": 45,
                  "yes_ask": 46, "no_bid": 54, "no_ask": 55, "volume": 1,
                  "open_interest": 1}
        monitor.seed_market("DOGE", market, book={"yes": [], "no": []}, received_at=iso(1))
        self.assertIn("MISSING_TARGET", monitor.assets["DOGE"].eligibility(iso(1)).reasons)
        monitor.refresh_market_metadata("DOGE", {**market, "floor_strike": .093782}, iso(2))
        self.assertEqual(monitor.assets["DOGE"].target, .093782)
        self.assertTrue(monitor.assets["DOGE"].eligibility(iso(2)).eligible)


class RecorderNormalizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_lifecycle_normalized_from_old_ticker_registry(self):
        recorder = StreamRecorder()
        captured = []
        async def capture(event):
            captured.append(event)
        recorder._append = capture
        old = {"ticker": "ETH-OLD", "status": "active", "floor_strike": 2449.47,
               "open_time": iso(0), "close_time": iso(900), "yes_bid": 45,
               "yes_ask": 46, "no_bid": 54, "no_ask": 55, "volume": 1,
               "open_interest": 1}
        new = {**old, "ticker": "ETH-NEW", "floor_strike": 2464.95,
               "open_time": iso(900), "close_time": iso(1800)}
        recorder.health.seed_market("ETH", old, book={}, received_at=iso(899))
        recorder.health.seed_market("ETH", new, book={}, received_at=iso(901))
        recorder.markets["ETH"] = new
        await recorder.on_ws_message({"type": "market_lifecycle_v2", "seq": 1, "msg": {
            "market_ticker": "ETH-OLD", "event_type": "settled"}}, iso(902), True)
        self.assertEqual(captured[0].market_ticker, "ETH-OLD")
        self.assertEqual(captured[0].target, 2449.47)
        self.assertEqual(captured[0].contract_open_time, old["open_time"])


class SignalCoherenceTests(unittest.TestCase):
    def seed_old(self, engine):
        for asset in ASSET_SERIES:
            ticker = f"{asset}-OLD"
            engine.process(quote(asset, 899, ticker, opened=0, closed=900))
            engine.process(book(asset, 899, ticker, opened=0, closed=900))

    def test_reset_finalizes_old_state_before_new_metadata(self):
        engine = SignalEngine()
        engine.process(quote("BTC", 0, "BTC-OLD", .2, target=90))
        engine.process(book("BTC", 0, "BTC-OLD", target=90))
        engine.assets["BTC"].contract.result = "no"
        snapshot = engine.process(reset("BTC", 900, "BTC-OLD", "BTC-NEW", 110))
        self.assertEqual(snapshot["features"]["prior_result"], "yes")
        self.assertEqual(engine.registry.record("BTC-OLD").target, 90)
        self.assertEqual(snapshot["target"], 110)
        self.assertEqual(len(engine.assets["BTC"].probability.values), 0)
        self.assertEqual(engine.assets["BTC"].trades, [])
        self.assertEqual(engine.assets["BTC"].bid_book, {})

    def test_late_old_ticker_quote_cannot_contaminate_new_contract(self):
        engine = SignalEngine()
        engine.process(quote("BTC", 0, "BTC-OLD", .2, target=90))
        engine.process(book("BTC", 0, "BTC-OLD", target=90))
        engine.process(reset("BTC", 900, "BTC-OLD", "BTC-NEW", 110))
        engine.process(quote("BTC", 901, "BTC-NEW", .7, target=110,
                             opened=900, closed=1800))
        before = engine.assets["BTC"].yes_bid
        engine.process(quote("BTC", 902, "BTC-OLD", .1, target=90,
                             opened=0, closed=900))
        self.assertEqual(engine.assets["BTC"].yes_bid, before)
        self.assertEqual(engine.assets["BTC"].contract.ticker, "BTC-NEW")

    def test_stale_quote_excludes_velocity_and_basket(self):
        engine = SignalEngine()
        engine.process(quote("BTC", 0, "BTC", .6))
        engine.process(book("BTC", 0, "BTC"))
        engine.process(quote("BTC", 5, "BTC", .7))
        stale = engine.process(raw("underlying_price", "BTC", 20,
                                   {"usd_price": 101}, "BTC"))
        self.assertIsNone(stale["features"]["prob_velocity_5s"])
        self.assertNotIn("BTC", stale["basket"]["eligible_assets"])
        self.assertIn("STALE_QUOTE", stale["basket"]["excluded_reasons"]["BTC"])

    def test_sequence_gap_excludes_until_fresh_snapshot(self):
        engine = SignalEngine()
        first = quote("BTC", 0, "BTC")
        engine.process(first)
        snapshot_event = book("BTC", .1, "BTC")
        snapshot_event["raw_payload"].update({"sid": 2, "seq": 1})
        engine.process(snapshot_event)
        gap = raw("orderbook_delta", "BTC", 1, {"type": "orderbook_delta",
                  "sid": 2, "seq": 3, "msg": {"market_ticker": "BTC",
                  "side": "yes", "price_dollars": ".60", "delta_fp": "1"}},
                  "BTC")
        broken = engine.process(gap)
        self.assertIn("SEQUENCE_UNHEALTHY",
                      broken["basket"]["excluded_reasons"]["BTC"])
        recovered = book("BTC", 1.1, "BTC")
        recovered["raw_payload"].update({"sid": 2, "seq": 4})
        fixed = engine.process(recovered)
        self.assertNotIn("SEQUENCE_UNHEALTHY",
                         fixed["basket"]["excluded_reasons"].get("BTC", []))

    def test_reconnect_book_and_trade_restarts_do_not_poison_sequence(self):
        engine = SignalEngine()
        for index, asset in enumerate(ASSET_SERIES, 1):
            timestamp = .02 * index
            engine.process(quote(asset, timestamp, asset))
            initial = book(asset, timestamp + .01, asset)
            initial["raw_payload"].update({"sid": 2, "seq": index})
            initial["sequence_generation"] = 1
            engine.process(initial)
        self.assertEqual(engine.snapshot("BTC", .1, "before")["basket"]["eligible_count"], 5)

        after = None
        for index, asset in enumerate(ASSET_SERIES, 1):
            bootstrap = book(asset, .2 + .01 * index, asset)
            bootstrap["raw_payload"].update({"sid": 2, "seq": index})
            bootstrap["sequence_generation"] = 2
            after = engine.process(bootstrap)
        trade = raw("trade", "BTC", .3, {"type": "trade", "sid": 3, "seq": 1,
                    "msg": {"market_ticker": "BTC", "yes_price_dollars": ".70",
                    "count_fp": "1"}}, "BTC")
        trade["sequence_generation"] = 2
        after = engine.process(trade)
        self.assertEqual(after["basket"]["eligible_count"], 5)
        self.assertNotIn("SEQUENCE_UNHEALTHY",
                         after["basket"]["excluded_reasons"].get("BTC", []))

    def test_trade_gap_requires_generation_recovery_not_same_generation_book(self):
        engine = SignalEngine()
        engine.process(quote("BTC", 0, "BTC"))
        initial = book("BTC", .1, "BTC")
        initial["raw_payload"].update({"sid": 2, "seq": 1})
        initial["sequence_generation"] = 1
        engine.process(initial)
        for second, sequence in ((.2, 1), (.3, 3)):
            trade = raw("trade", "BTC", second, {"type": "trade", "sid": 3,
                        "seq": sequence, "msg": {"market_ticker": "BTC",
                        "yes_price_dollars": ".7", "count_fp": "1"}}, "BTC")
            trade["sequence_generation"] = 1
            broken = engine.process(trade)
        self.assertIn("SEQUENCE_UNHEALTHY", broken["features"]["excluded_reasons"])
        same_generation = book("BTC", .4, "BTC")
        same_generation["raw_payload"].update({"sid": 2, "seq": 2})
        same_generation["sequence_generation"] = 1
        still_broken = engine.process(same_generation)
        self.assertIn("SEQUENCE_UNHEALTHY",
                      still_broken["features"]["excluded_reasons"])
        recovered = book("BTC", .5, "BTC")
        recovered["raw_payload"].update({"sid": 2, "seq": 1})
        recovered["sequence_generation"] = 2
        fixed = engine.process(recovered)
        self.assertNotIn("SEQUENCE_UNHEALTHY", fixed["features"]["excluded_reasons"])

    def test_530_forensic_regression(self):
        engine = SignalEngine()
        self.seed_old(engine)
        schedule = [("BTC", 906.9, None), ("DOGE", 907.0, None),
                    ("ETH", 927.3, 2475.65), ("XRP", 927.4, 1.5352)]
        for asset, second, target in schedule:
            engine.process(reset(asset, second, f"{asset}-OLD", f"{asset}-NEW", target))
            engine.process(quote(asset, second+.01, f"{asset}-NEW", target=target,
                                 opened=900, closed=1800))
            engine.process(book(asset, second+.02, f"{asset}-NEW", target=target,
                                opened=900, closed=1800))
        at_30 = engine.process(raw("underlying_price", "ETH", 930,
                                   {"usd_price": 2470}, "ETH-NEW", 2475.65, 900, 1800))
        self.assertLess(at_30["basket"]["eligible_count"], 5)
        self.assertNotEqual(at_30["features"]["regime_label"], "SYNC_UP_ACCELERATING")
        self.assertIn("MISSING_TARGET", at_30["basket"]["excluded_reasons"]["BTC"])
        self.assertIn("WINDOW_MISMATCH", at_30["basket"]["excluded_reasons"]["SOL"])
        self.assertIn("eligible", at_30["basket"]["breadth_direction"])
        self.assertNotIn("5/5", at_30["basket"]["breadth_direction"])
        self.assertIsNone(engine.snapshot("BTC", (BASE + timedelta(seconds=930)).timestamp(),
                                          "audit")["features"]["absolute_target_distance"])

        for asset, second, target in (("BTC", 932, 78013.52), ("DOGE", 933, .094325)):
            engine.process(raw("market_lifecycle", asset, second, {"msg": {
                "market_ticker": f"{asset}-NEW", "event_type": "metadata_updated",
                "floor_strike": target}}, f"{asset}-NEW", None, 900, 1800))
        engine.process(reset("SOL", 937.7, "SOL-OLD", "SOL-NEW", 96.0812))
        for index, asset in enumerate(ASSET_SERIES):
            target = {"BTC": 78013.52, "DOGE": .094325, "ETH": 2475.65,
                      "XRP": 1.5352, "SOL": 96.0812}[asset]
            second = 939 + index * .01
            engine.process(quote(asset, second, f"{asset}-NEW", target=target,
                                 opened=900, closed=1800))
            final = engine.process(book(asset, second+.001, f"{asset}-NEW", target=target,
                                        opened=900, closed=1800))
        self.assertEqual(final["basket"]["eligible_count"], 5)
        self.assertTrue(final["basket"]["coherent_window"])
        self.assertEqual({engine.assets[a].contract.window_id for a in ASSET_SERIES},
                         {iso(900).replace("+00:00", "Z").replace(".000000", "")})
        for asset in ASSET_SERIES:
            self.assertTrue(all(t >= (BASE + timedelta(seconds=900)).timestamp()
                                for t, _ in engine.assets[asset].probability.values))


class MonitorTests(unittest.TestCase):
    def make_db(self, path):
        with RawEventStore(path) as store:
            for index, asset in enumerate(ASSET_SERIES):
                item = quote(asset, index / 100, f"{asset}-T")
                store.append(RawEvent(**{key: value for key, value in item.items()
                                        if key != "_rowid"}))
                item = book(asset, index / 100 + .001, f"{asset}-T")
                store.append(RawEvent(**{key: value for key, value in item.items()
                                        if key != "_rowid"}))

    def test_monitor_consumes_once_and_uses_one_watermark(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.db"
            self.make_db(path)
            monitor = LiveSignalMonitor(path, batch_size=100)
            self.assertEqual(monitor.consume_once(), 10)
            count = monitor.events_processed
            self.assertEqual(monitor.consume_once(), 0)
            self.assertEqual(monitor.events_processed, count)
            snapshots, _ = monitor.coherent_snapshots()
            self.assertEqual({s["raw_watermark_event_id"] for s in snapshots.values()},
                             {monitor.raw_watermark})
            output = format_monitor(monitor)
            self.assertIn("watermark=", output)
            self.assertIn("ELIGIBLE: 5/5", output)
            monitor.close()

    @unittest.skipIf(os.name == "nt", "POSIX SIGINT test")
    def test_ctrl_c_shutdown_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.db"
            self.make_db(path)
            process = subprocess.Popen(
                [sys.executable, "monitor_signals.py", "--db", str(path),
                 "--refresh", ".25"], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, cwd=Path(__file__).parents[1])
            time.sleep(.2)
            started = time.monotonic()
            process.send_signal(signal.SIGINT)
            process.wait(timeout=1)
            self.assertLess(time.monotonic() - started, 1)


if __name__ == "__main__":
    unittest.main()
