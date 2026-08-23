"""Parsing helpers for Kalshi's public fixed-point market-data API."""

from decimal import Decimal, InvalidOperation


class KalshiSchemaError(ValueError):
    """Raised when a successful response does not match the documented schema."""


def _decimal(value, field, *, minimum=None, maximum=None):
    if value is None:
        return None
    if not isinstance(value, str):
        raise KalshiSchemaError(f"{field} must be a fixed-point string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise KalshiSchemaError(f"{field} is not a decimal string: {value!r}") from exc
    if not parsed.is_finite():
        raise KalshiSchemaError(f"{field} must be finite")
    if minimum is not None and parsed < minimum:
        raise KalshiSchemaError(f"{field} is below {minimum}")
    if maximum is not None and parsed > maximum:
        raise KalshiSchemaError(f"{field} is above {maximum}")
    return parsed


def _price_cents(market, field):
    value = _decimal(market.get(field), field, minimum=Decimal("0"),
                     maximum=Decimal("1"))
    return float(value * 100) if value is not None else None


def _count(payload, field):
    value = _decimal(payload.get(field), field, minimum=Decimal("0"))
    return float(value) if value is not None else None


def parse_market(market):
    """Return collector units: prices in cents and counts as contract floats."""
    if not isinstance(market, dict):
        raise KalshiSchemaError("market must be an object")
    ticker = market.get("ticker")
    status = market.get("status")
    if not isinstance(ticker, str) or not ticker:
        raise KalshiSchemaError("market.ticker must be a non-empty string")
    if not isinstance(status, str) or not status:
        raise KalshiSchemaError("market.status must be a non-empty string")

    parsed = dict(market)
    parsed.update({
        "yes_bid": _price_cents(market, "yes_bid_dollars"),
        "yes_ask": _price_cents(market, "yes_ask_dollars"),
        "no_bid": _price_cents(market, "no_bid_dollars"),
        "no_ask": _price_cents(market, "no_ask_dollars"),
        "last_price": _price_cents(market, "last_price_dollars"),
        "volume": _count(market, "volume_fp"),
        "open_interest": _count(market, "open_interest_fp"),
    })
    return parsed


def parse_markets_response(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("markets"), list):
        raise KalshiSchemaError("response.markets must be an array")
    return [parse_market(market) for market in payload["markets"]]


def parse_orderbook_response(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("orderbook_fp"), dict):
        raise KalshiSchemaError("response.orderbook_fp must be an object")
    book = payload["orderbook_fp"]
    result = {}
    for side in ("yes_dollars", "no_dollars"):
        levels = book.get(side)
        if not isinstance(levels, list):
            raise KalshiSchemaError(f"orderbook_fp.{side} must be an array")
        parsed_levels = []
        for index, level in enumerate(levels):
            if not isinstance(level, list) or len(level) != 2:
                raise KalshiSchemaError(f"{side}[{index}] must be [price, quantity]")
            price = _decimal(level[0], f"{side}[{index}].price",
                             minimum=Decimal("0"), maximum=Decimal("1"))
            quantity = _decimal(level[1], f"{side}[{index}].quantity",
                                minimum=Decimal("0"))
            parsed_levels.append((float(price * 100), float(quantity)))
        result["yes" if side.startswith("yes") else "no"] = parsed_levels
    return result
