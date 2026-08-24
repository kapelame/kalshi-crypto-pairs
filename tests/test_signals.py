import asyncio
import json
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from signals.config import SignalConfig
from signals.engine import SignalEngine, canonical_probability, classify_regime, reset_regime_label
from signals.export import checkpoint_rows
from signals.history import TimeHistory
from signals.replay import ReplayEngine
from signals.store import FeatureStore
from streaming import ASSET_SERIES
from streaming.events import RawEvent
from streaming.raw_store import RawEventStore
from monitor_signals import render


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def iso(seconds): return (BASE + timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def event(kind, asset, seconds, payload, ticker=None, event_id=None, target=100.0,
          opened=0, closed=900, source="test"):
    return {
        "event_id": event_id or f"{kind}-{asset}-{seconds}", "event_type": kind,
        "asset": asset, "market_ticker": ticker or f"{ASSET_SERIES[asset]}-TEST",
        "series_ticker": ASSET_SERIES[asset], "exchange_timestamp": iso(seconds),
        "local_receive_timestamp": iso(seconds), "processing_timestamp": iso(seconds),
        "source": source, "raw_payload": payload, "contract_open_time": iso(opened),
        "contract_close_time": iso(closed), "target": target, "sequence": None,
    }


def ticker(asset, seconds, probability, spread=.02, **kwargs):
    bid, ask = probability - spread/2, probability + spread/2
    payload = {"ticker": f"{ASSET_SERIES[asset]}-TEST", "status": "active",
               "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}",
               "no_bid_dollars": f"{1-ask:.4f}", "no_ask_dollars": f"{1-bid:.4f}",
               "last_price_dollars": f"{probability:.4f}", "volume_fp": "0.00",
               "open_interest_fp": "10.00"}
    return event("ticker", asset, seconds, payload, **kwargs)


def underlying(asset, seconds, price, **kwargs):
    return event("underlying_price", asset, seconds, {"usd_price": price}, **kwargs)


def book(asset, seconds, **kwargs):
    payload = {"orderbook_fp": {"yes_dollars": [["0.40", "10"]],
                                "no_dollars": [["0.55", "10"]]}}
    return event("orderbook_snapshot", asset, seconds, payload,
                 source="kalshi_rest_recovery", **kwargs)


class ProbabilityHistoryTests(unittest.TestCase):
    def test_canonical_probability_and_crossed_missing(self):
        probability, spread, bid, ask = canonical_probability(.49, .53, .47, .51)
        self.assertEqual((probability, spread, bid, ask), (.51, .040000000000000036, .49, .53))
        self.assertEqual(canonical_probability(None, .53, .47, .51)[:2], (None, None))
        self.assertEqual(canonical_probability(.60, .55, .45, .40)[:2], (None, None))

    def test_time_based_velocity_uses_elapsed_time_not_message_count(self):
        history = TimeHistory()
        history.append(0, .51); history.append(7, .55); history.append(30, .67)
        self.assertAlmostEqual(history.velocity(30, 30), (.67-.51)/30)
        self.assertAlmostEqual(history.change(30, 5), .12)  # last state at/before t-5 is t=7

    def test_probability_acceleration_and_missing_history(self):
        engine = SignalEngine()
        first = engine.process(ticker("BTC", 0, .51))
        self.assertIsNone(first["features"]["prob_velocity_5s"])
        engine.process(ticker("BTC", 5, .55))
        engine.process(ticker("BTC", 10, .62))
        final = engine.process(ticker("BTC", 15, .72))
        self.assertGreater(final["features"]["prob_velocity_5s"], 0)
        self.assertGreater(final["features"]["prob_acceleration_5s"], 0)

    def test_stale_quote_never_produces_probability(self):
        engine = SignalEngine(SignalConfig(quote_stale_seconds=2))
        engine.process(ticker("BTC", 0, .6))
        snapshot = engine.process(underlying("BTC", 3, 101))
        self.assertFalse(snapshot["features"]["quote_fresh"])
        self.assertIsNone(snapshot["features"]["midpoint_up_probability"])


class CrossAssetAndRegimeTests(unittest.TestCase):
    def seeded(self, probabilities, second=0):
        engine = SignalEngine()
        snapshots = []
        for asset, probability in zip(ASSET_SERIES, probabilities):
            engine.process(ticker(asset, second, probability))
            snapshots.append(engine.process(book(asset, second)))
        return engine, snapshots[-1]

    def test_breadth_dispersion_and_btc_residual(self):
        engine, snapshot = self.seeded([.70, .72, .69, .68, .71])
        basket = snapshot["basket"]
        self.assertEqual(basket["assets_above_0_50"], 5)
        self.assertEqual(basket["breadth_direction"], "5/5 UP")
        self.assertLess(basket["probability_stddev"], .02)
        eth = engine.snapshot("ETH", BASE.timestamp(), "w")
        self.assertAlmostEqual(eth["features"]["alt_probability_minus_btc"], .02)
        _, fragmented = self.seeded([.70, .40, .51, .80, .30])
        self.assertGreater(fragmented["basket"]["probability_stddev"], basket["probability_stddev"])

    def test_deterministic_regime_labels(self):
        _, snapshot = self.seeded([.7]*5)
        self.assertEqual(classify_regime(snapshot["basket"], SignalConfig()),
                         "SYNC_UP_DECELERATING")
        self.assertEqual(reset_regime_label("DOWN", snapshot["basket"]),
                         "REVERSAL_UP_CANDIDATE")
        self.assertEqual(reset_regime_label("UP", snapshot["basket"]),
                         "RESET_UP_CANDIDATE")


class UnderlyingBookTradeTests(unittest.TestCase):
    def test_returns_volatility_target_distance_and_no_future_price(self):
        engine = SignalEngine()
        engine.process(underlying("BTC", 0, 100))
        engine.process(underlying("BTC", 5, 101))
        before = engine.process(underlying("BTC", 10, 102))
        frozen = deepcopy(before)
        self.assertAlmostEqual(before["features"]["price_return_5s"], 1/101)
        self.assertEqual(before["features"]["absolute_target_distance"], 2)
        self.assertIsNotNone(before["features"]["volatility_adjusted_target_distance"])
        engine.process(underlying("BTC", 20, 150))
        self.assertEqual(before, frozen)

    def test_orderbook_features_and_stale_guard(self):
        engine = SignalEngine(SignalConfig(book_stale_seconds=2))
        payload = {"orderbook_fp": {"yes_dollars": [["0.40","10"],["0.39","5"],["0.38","5"]],
                                    "no_dollars": [["0.55","5"],["0.54","5"],["0.53","5"]]}}
        fresh = engine.process(event("orderbook_snapshot", "BTC", 0, payload,
                                     source="kalshi_rest_recovery"))
        self.assertAlmostEqual(fresh["features"]["book_l1_imbalance"], 1/3)
        self.assertEqual(fresh["features"]["book_bid_depth"], 20)
        frozen = deepcopy(fresh)
        stale = engine.process(underlying("BTC", 3, 101))
        self.assertIsNone(stale["features"]["book_top3_imbalance"])
        self.assertEqual(fresh, frozen)

    def test_trade_flow_windows_and_unknown_direction(self):
        engine = SignalEngine()
        known = {"type":"trade","msg":{"market_ticker":"T","trade_id":"1",
                 "yes_price_dollars":"0.6","no_price_dollars":"0.4","count_fp":"10","taker_side":"yes"}}
        unknown = {"type":"trade","msg":{"market_ticker":"T","trade_id":"2",
                   "yes_price_dollars":"0.6","no_price_dollars":"0.4","count_fp":"5"}}
        engine.process(event("trade", "BTC", 0, known))
        snapshot = engine.process(event("trade", "BTC", 4, unknown))
        self.assertEqual(snapshot["features"]["trade_5s_count"], 2)
        self.assertEqual(snapshot["features"]["trade_5s_contracts"], 15)
        self.assertEqual(snapshot["features"]["trade_5s_yes_aggressive"], 10)
        empty = SignalEngine().process(event("trade", "BTC", 0, unknown))
        self.assertIsNone(empty["features"]["trade_5s_imbalance"])


class PriorResetCheckpointTests(unittest.TestCase):
    def test_prior_window_and_reset_transition(self):
        engine = SignalEngine()
        engine.process(ticker("BTC", 0, .2))
        engine.process(underlying("BTC", 0, 100))
        engine.process(ticker("BTC", 10, .1))
        engine.process(underlying("BTC", 10, 90))
        reset_payload = {"previous_ticker":"OLD","new_ticker":"NEW",
                         "previous_settlement":"no","new_target":95,
                         "prior_contract_final_market_state":{"result":"no"}}
        reset = event("contract_reset", "BTC", 900, reset_payload, ticker="NEW",
                      target=95, opened=900, closed=1800)
        snapshot = engine.process(reset)
        self.assertEqual(snapshot["features"]["prior_result"], "no")
        self.assertAlmostEqual(snapshot["features"]["prior_return"], -.1)
        checkpoint = engine.process(ticker("BTC", 931, .7, ticker="NEW", target=95,
                                           opened=900, closed=1800))
        self.assertTrue(checkpoint["features"]["checkpoint_30s"])
        later = engine.process(ticker("BTC", 932, .71, ticker="NEW", target=95,
                                      opened=900, closed=1800))
        self.assertFalse(later["features"]["checkpoint_30s"])


class ReplayLeakageTests(unittest.IsolatedAsyncioTestCase):
    def create_raw(self, path):
        raw_events = [
            RawEvent("ticker", "BTC", "T", "test", ticker("BTC",0,.5)["raw_payload"],
                     series_ticker="KXBTC15M", exchange_timestamp=iso(0),
                     local_receive_timestamp=iso(0), processing_timestamp=iso(0),
                     contract_open_time=iso(0), contract_close_time=iso(900), target=100,
                     event_id="e1"),
            RawEvent("ticker", "BTC", "T", "test", ticker("BTC",5,.6)["raw_payload"],
                     series_ticker="KXBTC15M", exchange_timestamp=iso(5),
                     local_receive_timestamp=iso(5), processing_timestamp=iso(5),
                     contract_open_time=iso(0), contract_close_time=iso(900), target=100,
                     event_id="e2"),
            RawEvent("ticker", "BTC", "T", "test", ticker("BTC",10,.9)["raw_payload"],
                     series_ticker="KXBTC15M", exchange_timestamp=iso(10),
                     local_receive_timestamp=iso(10), processing_timestamp=iso(10),
                     contract_open_time=iso(0), contract_close_time=iso(900), target=100,
                     event_id="e3"),
        ]
        with RawEventStore(path) as store:
            for item in raw_events: store.append(item)
        return raw_events

    async def test_replay_determinism_live_equivalence_and_stop_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)/"raw.db"; raw_events = self.create_raw(raw)
            first = await ReplayEngine(raw).run(speed=0)
            second = await ReplayEngine(raw).run(speed=0)
            self.assertEqual(first, second)
            direct_engine = SignalEngine()
            direct = []
            for item in raw_events:
                direct.append(direct_engine.process({
                    **item.__dict__, "raw_payload": item.raw_payload}))
            self.assertEqual(first, direct)
            stopped = await ReplayEngine(raw).run(speed=0, stop_timestamp=iso(5))
            self.assertEqual(len(stopped), 2)
            self.assertAlmostEqual(stopped[-1]["features"]["midpoint_up_probability"], .6)

    async def test_settlement_joined_only_after_frozen_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory)/"features.db"
            with FeatureStore(db) as store:
                snapshot = SignalEngine().process(ticker("BTC", 31, .6))
                store.append_snapshot(snapshot)
                self.assertNotIn("outcome", snapshot["features"])
                before = checkpoint_rows(store.connection)
                self.assertIsNone(before[0]["outcome"])
                store.append_outcome("KXBTC15M-TEST", "BTC", BASE.timestamp()+901, "yes", "settle")
                after = checkpoint_rows(store.connection)
                self.assertEqual(after[0]["outcome"], "yes")
                self.assertEqual(after[0]["features"], before[0]["features"])

    async def test_future_trade_and_reset_do_not_change_past_snapshot(self):
        engine = SignalEngine()
        trade_payload = {"type":"trade","msg":{"market_ticker":"T","trade_id":"a",
                         "yes_price_dollars":"0.6","no_price_dollars":"0.4",
                         "count_fp":"2","taker_side":"yes"}}
        past = engine.process(event("trade", "BTC", 10, trade_payload))
        frozen = deepcopy(past)
        future_trade = deepcopy(trade_payload); future_trade["msg"]["trade_id"] = "b"
        engine.process(event("trade", "BTC", 20, future_trade))
        engine.process(event("contract_reset", "BTC", 900, {
            "previous_ticker":"T", "new_ticker":"N", "previous_settlement":"yes"},
            ticker="N", opened=900, closed=1800))
        self.assertEqual(past, frozen)

    async def test_five_asset_monitor_render(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.db"
            with RawEventStore(raw) as store:
                for index, asset in enumerate(ASSET_SERIES):
                    item = ticker(asset, index / 100, .55 + index / 100)
                    store.append(RawEvent(
                        "ticker", asset, item["market_ticker"], "test", item["raw_payload"],
                        series_ticker=ASSET_SERIES[asset], exchange_timestamp=item["exchange_timestamp"],
                        local_receive_timestamp=item["local_receive_timestamp"],
                        processing_timestamp=item["processing_timestamp"],
                        contract_open_time=item["contract_open_time"],
                        contract_close_time=item["contract_close_time"], target=100,
                        event_id=item["event_id"]))
            output = render(raw)
            for asset in ASSET_SERIES: self.assertIn(asset, output)
            self.assertIn("BREADTH:", output)


if __name__ == "__main__": unittest.main()
