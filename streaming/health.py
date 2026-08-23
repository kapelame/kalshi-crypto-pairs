"""Five-asset streaming health state; no feature or strategy calculations."""

from dataclasses import dataclass

from . import ASSET_SERIES
from .events import parse_orderbook_delta, parse_orderbook_snapshot, parse_ticker, parse_trade
from .timeutil import iso_utc, latency_ms


@dataclass
class AssetHealth:
    active_ticker: str | None = None
    target: float | None = None
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    latest_trade: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    orderbook_timestamp: str | None = None
    last_event_timestamp: str | None = None
    websocket_connected: bool = False
    sequence_healthy: bool = True
    rest_fallback_active: bool = True

    @property
    def spread(self):
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    def orderbook_freshness_ms(self, now=None):
        return latency_ms(self.orderbook_timestamp, now or iso_utc())

    def last_event_age_ms(self, now=None):
        return latency_ms(self.last_event_timestamp, now or iso_utc())

    def valid(self, now=None, stale_seconds=30):
        book_age = self.orderbook_freshness_ms(now)
        return (self.active_ticker is not None and self.target is not None and
                self.yes_bid is not None and self.yes_ask is not None and
                self.no_bid is not None and self.no_ask is not None and
                self.volume is not None and self.open_interest is not None and
                book_age is not None and book_age <= stale_seconds * 1000)


class HealthMonitor:
    def __init__(self):
        self.assets = {asset: AssetHealth() for asset in ASSET_SERIES}
        self.ticker_to_asset = {}

    def set_connected(self, connected):
        for state in self.assets.values():
            state.websocket_connected = connected

    def seed_market(self, asset, market, book=None, received_at=None):
        now = received_at or iso_utc()
        state = self.assets[asset]
        state.active_ticker = market["ticker"]
        state.target = market.get("floor_strike")
        state.yes_bid = _cents(market.get("yes_bid"))
        state.yes_ask = _cents(market.get("yes_ask"))
        state.no_bid = _cents(market.get("no_bid"))
        state.no_ask = _cents(market.get("no_ask"))
        state.latest_trade = _cents(market.get("last_price"))
        state.volume = market.get("volume")
        state.open_interest = market.get("open_interest")
        state.last_event_timestamp = now
        if book is not None:
            state.orderbook_timestamp = now
        state.rest_fallback_active = True
        self.ticker_to_asset[market["ticker"]] = asset

    def update_ws(self, payload, received_at, sequence_healthy=True):
        msg = payload.get("msg", {})
        asset = self.ticker_to_asset.get(msg.get("market_ticker"))
        if asset is None:
            return None
        state = self.assets[asset]
        state.last_event_timestamp = received_at
        kind = payload.get("type")
        if kind == "orderbook_snapshot":
            state.sequence_healthy = sequence_healthy
        else:
            state.sequence_healthy = state.sequence_healthy and sequence_healthy
        if kind == "ticker":
            parsed = parse_ticker(payload)
            state.latest_trade = parsed["price_dollars"]
            state.yes_bid = parsed["yes_bid_dollars"]
            state.yes_ask = parsed["yes_ask_dollars"]
            state.no_bid = (None if state.yes_ask is None else 1 - state.yes_ask)
            state.no_ask = (None if state.yes_bid is None else 1 - state.yes_bid)
            state.volume = parsed["volume_fp"]
            state.open_interest = parsed["open_interest_fp"]
            state.rest_fallback_active = False
        elif kind == "trade":
            state.latest_trade = parse_trade(payload)["yes_price_dollars"]
        elif kind == "orderbook_snapshot":
            parse_orderbook_snapshot(payload)
            state.orderbook_timestamp = received_at
            state.rest_fallback_active = False
        elif kind == "orderbook_delta":
            parse_orderbook_delta(payload)
            state.orderbook_timestamp = received_at
            state.rest_fallback_active = False
        return asset

    def healthy_count(self, now=None, stale_seconds=30):
        return sum(state.valid(now, stale_seconds) for state in self.assets.values())

    def all_healthy(self, now=None, stale_seconds=30):
        return self.healthy_count(now, stale_seconds) == len(self.assets)

    def render(self, now=None):
        now = now or iso_utc()
        lines = []
        for asset, state in self.assets.items():
            valid = state.valid(now)
            lines.append(
                f"{asset:<4} {'valid' if valid else 'stale'} "
                f"ticker={state.active_ticker or '---'} target={_fmt(state.target)} "
                f"Y={_fmt(state.yes_bid)}/{_fmt(state.yes_ask)} "
                f"N={_fmt(state.no_bid)}/{_fmt(state.no_ask)} "
                f"last={_fmt(state.latest_trade)} spread={_fmt(state.spread)} "
                f"vol={_fmt(state.volume)} oi={_fmt(state.open_interest)} "
                f"book_age_ms={_fmt(state.orderbook_freshness_ms(now))} "
                f"event_age_ms={_fmt(state.last_event_age_ms(now))} "
                f"ws={'yes' if state.websocket_connected else 'no'} "
                f"seq={'yes' if state.sequence_healthy else 'no'} "
                f"rest={'yes' if state.rest_fallback_active else 'no'}")
        lines.append(f"VALID: {self.healthy_count(now)}/{len(self.assets)}")
        return "\n".join(lines)


def _cents(value):
    return None if value is None else value / 100


def _fmt(value):
    return "None" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
