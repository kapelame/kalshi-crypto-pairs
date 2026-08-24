import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from streaming import ASSET_SERIES
from streaming.auth import WS_SIGN_PATH, build_auth_headers
from streaming.events import (
    RawEvent, StreamSchemaError, make_ws_event, parse_orderbook_delta,
    parse_orderbook_snapshot, parse_ticker, parse_trade,
)
from streaming.health import HealthMonitor
from streaming.raw_store import RawEventStore
from streaming.recorder import build_reset_event
from streaming.rest import RestDataClient
from streaming.timeutil import exchange_timestamp, processing_latency_ms, receive_latency_ms
from streaming.websocket import KalshiWebSocketClient, SequenceValidator, subscription_messages
from collector import ASSETS as COLLECTOR_ASSETS, DataStore


TICKER = {"type": "ticker", "sid": 11, "msg": {
    "market_ticker": "KXDOGE15M-26AUG231600-00",
    "price_dollars": "0.4800", "yes_bid_dollars": "0.4500",
    "yes_ask_dollars": "0.5300", "volume_fp": "0.00",
    "open_interest_fp": "20422.50", "yes_bid_size_fp": "300.00",
    "yes_ask_size_fp": "150.00", "last_trade_size_fp": "25.00",
    "ts": 1669149841, "ts_ms": 1669149841123,
    "time": "2022-11-22T20:44:01.123456Z"}}

TRADE = {"type": "trade", "sid": 12, "seq": 4, "msg": {
    "trade_id": "trade-1", "market_ticker": TICKER["msg"]["market_ticker"],
    "yes_price_dollars": "0.3600", "no_price_dollars": "0.6400",
    "count_fp": "136.25", "taker_side": "no", "ts_ms": 1669149841000}}

SNAPSHOT = {"type": "orderbook_snapshot", "sid": 2, "seq": 2, "msg": {
    "market_ticker": TICKER["msg"]["market_ticker"],
    "yes_dollars_fp": [["0.0800", "300.00"]],
    "no_dollars_fp": [["0.5400", "20.50"]]}}

DELTA = {"type": "orderbook_delta", "sid": 2, "seq": 3, "msg": {
    "market_ticker": TICKER["msg"]["market_ticker"],
    "price_dollars": "0.9600", "delta_fp": "-54.25", "side": "yes",
    "ts_ms": 1669149841000}}


class ParserTests(unittest.TestCase):
    def test_doge_and_ticker_parsing_preserves_zero(self):
        self.assertEqual(ASSET_SERIES["DOGE"], "KXDOGE15M")
        parsed = parse_ticker(TICKER)
        self.assertEqual(parsed["volume_fp"], 0.0)
        self.assertIsNone(parse_ticker({"type": "ticker", "msg": {
            "market_ticker": "T"}})["volume_fp"])

    def test_trade_parsing(self):
        parsed = parse_trade(TRADE)
        self.assertEqual(parsed["count_fp"], 136.25)
        self.assertEqual(parsed["yes_price_dollars"], .36)

    def test_orderbook_snapshot_and_delta(self):
        self.assertEqual(parse_orderbook_snapshot(SNAPSHOT)["yes_dollars_fp"],
                         [(.08, 300.0)])
        self.assertEqual(parse_orderbook_delta(DELTA)["delta_fp"], -54.25)
        with self.assertRaises(StreamSchemaError):
            parse_orderbook_delta({**DELTA, "msg": {**DELTA["msg"], "side": "bad"}})

    def test_exchange_timestamp_uses_high_precision_time(self):
        self.assertEqual(exchange_timestamp(TICKER), "2022-11-22T20:44:01.123456Z")
        event = make_ws_event(TICKER, "DOGE", "KXDOGE15M", received_at=
                              "2022-11-22T20:44:01.223456+00:00")
        self.assertEqual(event.exchange_timestamp, "2022-11-22T20:44:01.123456Z")
        self.assertAlmostEqual(receive_latency_ms(event), 100.0)
        self.assertIsNotNone(processing_latency_ms(event))


class AuthAndConnectionTests(unittest.IsolatedAsyncioTestCase):
    def test_authentication_message_construction(self):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        from cryptography.hazmat.primitives import serialization
        pem = key.private_bytes(serialization.Encoding.PEM,
                                serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
        headers = build_auth_headers("key-id", pem, 1700000000123)
        signature = __import__("base64").b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
        key.public_key().verify(
            signature, f"1700000000123GET{WS_SIGN_PATH}".encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        self.assertEqual(headers["KALSHI-ACCESS-KEY"], "key-id")

    def test_sequence_gap_detection(self):
        validator = SequenceValidator()
        self.assertTrue(validator.observe(2, 10))
        self.assertTrue(validator.observe(2, 11))
        self.assertFalse(validator.observe(2, 13))
        self.assertEqual(validator.gaps, 1)

    async def test_resubscribe_builds_same_read_only_commands(self):
        class FakeSocket:
            def __init__(self): self.sent = []
            async def send_str(self, value): self.sent.append(json.loads(value))
        client = KalshiWebSocketClient(["A", "B"], lambda *_: None)
        first, second = FakeSocket(), FakeSocket()
        await client._subscribe(first)
        await client._subscribe(second)
        self.assertEqual(first.sent, second.sent)
        self.assertEqual(first.sent, subscription_messages(["A", "B"]))
        self.assertTrue(first.sent[1]["params"]["use_yes_price"])
        self.assertTrue(all(message["cmd"] == "subscribe" for message in first.sent))


class RestDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_doge_discovery_uses_verified_series(self):
        client = RestDataClient(session=object())
        seen = []
        async def fake_get(path):
            seen.append(path)
            return {"markets": [{"ticker": "KXDOGE15M-X", "status": "active",
                "yes_bid_dollars": "0.0000", "volume_fp": "0.00"}]}
        client._get = fake_get
        market = await client.discover_asset("DOGE")
        self.assertIn("series_ticker=KXDOGE15M", seen[0])
        self.assertEqual(market["ticker"], "KXDOGE15M-X")
        self.assertEqual(market["yes_bid"], 0.0)

    async def test_discovery_rejects_closed_payload_even_from_open_query(self):
        client = RestDataClient(session=object())
        async def fake_get(_path):
            return {"markets": [{"ticker": "KXSOL15M-OLD", "status": "closed"}]}
        client._get = fake_get
        with self.assertRaisesRegex(RuntimeError, "non-active"):
            await client.discover_asset("SOL")


class StoreRolloverHealthTests(unittest.TestCase):
    def test_legacy_collector_migrates_for_doge_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE features (ts TEXT, ts_unix REAL)")
            connection.execute("INSERT INTO features VALUES ('old', 1.0)")
            connection.commit()
            connection.close()
            store = DataStore(str(path))
            store.init()
            columns = {row[1] for row in store._conn.execute("PRAGMA table_info(features)")}
            self.assertIn("DOGE", COLLECTOR_ASSETS)
            self.assertIn("doge_ticker", columns)
            self.assertEqual(store.row_count(), 1)
            store.close()

    def test_raw_store_append_only_and_null_vs_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.db"
            with RawEventStore(path) as store:
                event = RawEvent("ticker", "DOGE", "T", "test",
                                 {"missing": None, "zero": 0})
                store.append(event)
                row = store.connection.execute(
                    "SELECT raw_payload FROM ticker_events").fetchone()[0]
                self.assertEqual(json.loads(row), {"missing": None, "zero": 0})
                with self.assertRaises(sqlite3.IntegrityError):
                    store.append(event)
                with self.assertRaises(sqlite3.IntegrityError):
                    store.connection.execute("DELETE FROM ticker_events")

    def test_contract_rollover_event(self):
        old = {"ticker": "OLD", "open_time": "2026-08-23T19:30:00Z"}
        new = {"ticker": "NEW", "open_time": "2026-08-23T19:45:00Z",
               "close_time": "2026-08-23T20:00:00Z", "floor_strike": .092}
        event = build_reset_event("DOGE", old, new,
                                  {"ticker": "OLD", "status": "finalized", "result": "yes"})
        self.assertEqual(event.exchange_timestamp, "2026-08-23T19:45:00Z")
        self.assertEqual(event.raw_payload["previous_settlement"], "yes")
        self.assertEqual(event.target, .092)

    def test_five_asset_health(self):
        monitor = HealthMonitor()
        now = "2026-08-23T20:00:00.000000+00:00"
        for index, asset in enumerate(ASSET_SERIES):
            market = {"ticker": f"{ASSET_SERIES[asset]}-{index}",
                      "status": "active", "open_time": "2026-08-23T20:00:00Z",
                      "close_time": "2026-08-23T20:15:00Z",
                      "floor_strike": float(index + 1), "yes_bid": 45.0,
                      "yes_ask": 46.0, "no_bid": 54.0, "no_ask": 55.0,
                      "last_price": 45.5, "volume": 0.0, "open_interest": 1.0}
            monitor.seed_market(asset, market, book={"yes": [], "no": []},
                                received_at=now)
        self.assertEqual(monitor.healthy_count(now), 5)
        self.assertTrue(monitor.all_healthy(now))
        monitor.assets["DOGE"].yes_bid = None
        self.assertEqual(monitor.healthy_count(now), 4)


if __name__ == "__main__":
    unittest.main()
