import unittest

from kalshi_api import (
    KalshiSchemaError, parse_market, parse_markets_response,
    parse_orderbook_response,
)


MARKET = {
    "ticker": "KXBTC15M-26AUG231515-15",
    "status": "active",
    "yes_bid_dollars": "0.0260",
    "yes_ask_dollars": "0.0270",
    "no_bid_dollars": "0.9730",
    "no_ask_dollars": "0.9740",
    "last_price_dollars": "0.0260",
    "volume_fp": "1902956.48",
    "open_interest_fp": "393675.22",
}


class MarketParserTests(unittest.TestCase):
    def test_current_fixed_point_fields_and_units(self):
        market = parse_market(MARKET)
        self.assertEqual(market["yes_bid"], 2.6)
        self.assertEqual(market["yes_ask"], 2.7)
        self.assertEqual(market["no_bid"], 97.3)
        self.assertEqual(market["no_ask"], 97.4)
        self.assertEqual(market["last_price"], 2.6)
        self.assertEqual(market["volume"], 1902956.48)
        self.assertEqual(market["open_interest"], 393675.22)

    def test_missing_values_remain_none(self):
        market = parse_market({"ticker": "T", "status": "active"})
        for field in ("yes_bid", "yes_ask", "no_bid", "no_ask",
                      "last_price", "volume", "open_interest"):
            self.assertIsNone(market[field])

    def test_legitimate_zero_is_not_missing(self):
        market = parse_market({
            "ticker": "T", "status": "active",
            "yes_bid_dollars": "0.0000", "volume_fp": "0.00",
            "open_interest_fp": "0.00",
        })
        self.assertEqual(market["yes_bid"], 0.0)
        self.assertEqual(market["volume"], 0.0)
        self.assertEqual(market["open_interest"], 0.0)

    def test_malformed_response_rejected(self):
        with self.assertRaises(KalshiSchemaError):
            parse_markets_response({"markets": {}})
        with self.assertRaises(KalshiSchemaError):
            parse_market({**MARKET, "yes_bid_dollars": 0.026})


class OrderbookParserTests(unittest.TestCase):
    def test_current_orderbook_schema_and_units(self):
        book = parse_orderbook_response({"orderbook_fp": {
            "yes_dollars": [["0.0260", "281.00"]],
            "no_dollars": [["0.9730", "12.50"]],
        }})
        self.assertEqual(book["yes"], [(2.6, 281.0)])
        self.assertEqual(book["no"], [(97.3, 12.5)])

    def test_old_or_malformed_orderbook_rejected(self):
        with self.assertRaises(KalshiSchemaError):
            parse_orderbook_response({"orderbook": {"yes": [], "no": []}})
        with self.assertRaises(KalshiSchemaError):
            parse_orderbook_response({"orderbook_fp": {
                "yes_dollars": [["0.50"]], "no_dollars": []}})


if __name__ == "__main__":
    unittest.main()
