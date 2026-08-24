"""Lossless raw-event envelope and documented WebSocket payload validation."""

import json
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from .timeutil import exchange_timestamp, iso_utc


class StreamSchemaError(ValueError):
    pass


EVENT_TABLES = {
    "market_lifecycle": "market_lifecycle_events",
    "contract_reset": "contract_reset_events",
    "ticker": "ticker_events",
    "trade": "trade_events",
    "orderbook_snapshot": "orderbook_snapshots",
    "orderbook_delta": "orderbook_deltas",
    "underlying_price": "underlying_price_events",
}


@dataclass(frozen=True)
class RawEvent:
    event_type: str
    asset: str
    market_ticker: str | None
    source: str
    raw_payload: dict
    series_ticker: str | None = None
    exchange_timestamp: str | None = None
    local_receive_timestamp: str = field(default_factory=iso_utc)
    processing_timestamp: str = field(default_factory=iso_utc)
    contract_open_time: str | None = None
    contract_close_time: str | None = None
    target: float | None = None
    sequence: int | None = None
    sequence_generation: int | None = None
    request_started_at: str | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def payload_json(self):
        return json.dumps(self.raw_payload, separators=(",", ":"),
                          sort_keys=True, ensure_ascii=False)


def _decimal_string(value, field_name, *, signed=False):
    if not isinstance(value, str):
        raise StreamSchemaError(f"{field_name} must be a fixed-point string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise StreamSchemaError(f"invalid {field_name}") from exc
    if not result.is_finite() or (not signed and result < 0):
        raise StreamSchemaError(f"invalid {field_name}")
    return float(result)


def _envelope(payload, expected_type):
    if not isinstance(payload, dict) or payload.get("type") != expected_type:
        raise StreamSchemaError(f"expected {expected_type} envelope")
    msg = payload.get("msg")
    if not isinstance(msg, dict) or not isinstance(msg.get("market_ticker"), str):
        raise StreamSchemaError(f"{expected_type}.msg.market_ticker is required")
    return msg


def parse_ticker(payload):
    msg = _envelope(payload, "ticker")
    result = dict(msg)
    for field_name in ("price_dollars", "yes_bid_dollars", "yes_ask_dollars"):
        result[field_name] = (None if msg.get(field_name) is None else
                              _decimal_string(msg[field_name], field_name))
    for field_name in ("volume_fp", "open_interest_fp", "yes_bid_size_fp",
                       "yes_ask_size_fp", "last_trade_size_fp"):
        result[field_name] = (None if msg.get(field_name) is None else
                              _decimal_string(msg[field_name], field_name))
    return result


def parse_trade(payload):
    msg = _envelope(payload, "trade")
    if not isinstance(msg.get("trade_id"), str):
        raise StreamSchemaError("trade_id is required")
    result = dict(msg)
    for field_name in ("yes_price_dollars", "no_price_dollars", "count_fp"):
        result[field_name] = _decimal_string(msg.get(field_name), field_name)
    return result


def _levels(value, field_name):
    if not isinstance(value, list):
        raise StreamSchemaError(f"{field_name} must be an array")
    result = []
    for level in value:
        if not isinstance(level, list) or len(level) != 2:
            raise StreamSchemaError(f"{field_name} level must be [price, quantity]")
        result.append((_decimal_string(level[0], f"{field_name}.price"),
                       _decimal_string(level[1], f"{field_name}.quantity")))
    return result


def parse_orderbook_snapshot(payload):
    msg = _envelope(payload, "orderbook_snapshot")
    return {**msg,
            "yes_dollars_fp": _levels(msg.get("yes_dollars_fp"), "yes_dollars_fp"),
            "no_dollars_fp": _levels(msg.get("no_dollars_fp"), "no_dollars_fp")}


def parse_orderbook_delta(payload):
    msg = _envelope(payload, "orderbook_delta")
    if msg.get("side") not in ("yes", "no"):
        raise StreamSchemaError("delta side must be yes or no")
    return {**msg,
            "price_dollars": _decimal_string(msg.get("price_dollars"), "price_dollars"),
            "delta_fp": _decimal_string(msg.get("delta_fp"), "delta_fp", signed=True)}


def make_ws_event(payload, asset, series_ticker, market=None, received_at=None,
                  sequence_generation=None):
    event_type = payload.get("type")
    if event_type not in EVENT_TABLES:
        raise StreamSchemaError(f"unsupported event type: {event_type}")
    msg = payload.get("msg", {})
    market = market or {}
    received_at = received_at or iso_utc()
    return RawEvent(
        event_type=event_type, asset=asset,
        market_ticker=msg.get("market_ticker"), series_ticker=series_ticker,
        exchange_timestamp=exchange_timestamp(payload),
        local_receive_timestamp=received_at, processing_timestamp=iso_utc(),
        source="kalshi_websocket", raw_payload=payload,
        contract_open_time=market.get("open_time"),
        contract_close_time=market.get("close_time"),
        target=market.get("floor_strike"), sequence=payload.get("seq"),
        sequence_generation=sequence_generation)
