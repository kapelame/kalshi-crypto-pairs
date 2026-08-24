"""Five-asset streaming health state; no feature or strategy calculations."""

from dataclasses import dataclass

from . import ASSET_SERIES
from .contracts import ContractRegistry, RolloverState, evaluate_eligibility
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
    quote_timestamp: str | None = None
    last_event_timestamp: str | None = None
    websocket_connected: bool = False
    sequence_healthy: bool = True
    rest_fallback_active: bool = True
    market_status: str | None = None
    contract_open_time: str | None = None
    contract_close_time: str | None = None
    contract_window_id: str | None = None
    rollover_state: RolloverState = RolloverState.CURRENT
    expected_ticker: str | None = None

    @property
    def spread(self):
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    def orderbook_freshness_ms(self, now=None):
        return latency_ms(self.orderbook_timestamp, now or iso_utc())

    def last_event_age_ms(self, now=None):
        return latency_ms(self.last_event_timestamp, now or iso_utc())

    def eligibility(self, now=None, quote_stale_seconds=10, book_stale_seconds=30,
                    expected_window_id=None):
        quote_age = latency_ms(self.quote_timestamp, now or iso_utc())
        book_age = self.orderbook_freshness_ms(now)
        quote_usable = (self.yes_bid is not None and self.yes_ask is not None and
                        self.no_bid is not None and self.no_ask is not None and
                        quote_age is not None and quote_age <= quote_stale_seconds * 1000)
        book_usable = book_age is not None and book_age <= book_stale_seconds * 1000
        return evaluate_eligibility(
            ticker=self.active_ticker, expected_ticker=self.expected_ticker,
            status=self.market_status, target=self.target,
            open_time=self.contract_open_time, close_time=self.contract_close_time,
            rollover_state=self.rollover_state, quote_fresh=quote_usable,
            book_fresh=book_usable, sequence_healthy=self.sequence_healthy,
            expected_window_id=expected_window_id, evaluation_timestamp=now or iso_utc())

    def valid(self, now=None, stale_seconds=30):
        return self.eligibility(now, stale_seconds, stale_seconds).eligible


class HealthMonitor:
    def __init__(self, registry=None):
        self.registry = registry or ContractRegistry()
        self.assets = {asset: AssetHealth() for asset in ASSET_SERIES}
        self.ticker_to_asset = {}

    def set_connected(self, connected):
        for state in self.assets.values():
            state.websocket_connected = connected

    def seed_market(self, asset, market, book=None, received_at=None, expected=True):
        now = received_at or iso_utc()
        record = self.registry.update(asset, market["ticker"], market,
                                      source="rest", timestamp=now)
        if expected:
            self.registry.expect(asset, market["ticker"], pending=True)
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
        state.quote_timestamp = now
        if book is not None:
            state.orderbook_timestamp = now
        state.rest_fallback_active = True
        self.ticker_to_asset[market["ticker"]] = asset
        self._sync_contract(asset, record)
        self._resolve_if_ready(asset, now)

    def _sync_contract(self, asset, record=None):
        state = self.assets[asset]
        record = record or self.registry.current(asset)
        state.expected_ticker = self.registry.expected_by_asset.get(asset)
        state.rollover_state = self.registry.rollover_state(asset)
        if record is None:
            return
        state.active_ticker = record.ticker
        state.market_status = record.status
        state.target = record.target
        state.contract_open_time = record.open_time
        state.contract_close_time = record.close_time
        state.contract_window_id = record.window_id

    def _resolve_if_ready(self, asset, now=None):
        state = self.assets[asset]
        # Evaluate all requirements except the rollover-state marker itself.
        prior = state.rollover_state
        state.rollover_state = RolloverState.RESOLVED
        decision = state.eligibility(now)
        state.rollover_state = prior
        if not decision.reasons or decision.reasons == ("ROLLOVER_PENDING",):
            self.registry.mark_resolved(asset)
        elif "ROLLOVER_PENDING" in decision.reasons:
            self.registry.rollover_by_asset[asset] = RolloverState.PENDING
        self._sync_contract(asset)

    def apply_lifecycle(self, asset, ticker, payload, received_at=None):
        now = received_at or iso_utc()
        record = self.registry.apply_lifecycle(
            asset, ticker, payload, source="kalshi_websocket", timestamp=now)
        if self.registry.expected_by_asset.get(asset) == ticker:
            msg = payload.get("msg", payload)
            if (msg.get("event_type") or payload.get("type")) in {
                    "settled", "determined", "deactivated"}:
                self.registry.rollover_by_asset[asset] = RolloverState.PENDING
            self._sync_contract(asset, record)
            self._resolve_if_ready(asset, now)
        return record

    def refresh_market_metadata(self, asset, market, received_at=None):
        now = received_at or iso_utc()
        record = self.registry.update(asset, market["ticker"], market,
                                      source="rest", timestamp=now)
        if self.registry.expected_by_asset.get(asset) == market["ticker"]:
            self._sync_contract(asset, record)
            self._resolve_if_ready(asset, now)
        return record

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
            state.quote_timestamp = received_at
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
        self._resolve_if_ready(asset, received_at)
        return asset

    def healthy_count(self, now=None, stale_seconds=30):
        expected_window = self._expected_window_id()
        return sum(state.eligibility(now, stale_seconds, stale_seconds,
                                     expected_window).eligible
                   for state in self.assets.values())

    def all_healthy(self, now=None, stale_seconds=30):
        return self.healthy_count(now, stale_seconds) == len(self.assets)

    def render(self, now=None):
        now = now or iso_utc()
        lines = []
        expected_window = self._expected_window_id()
        for asset, state in self.assets.items():
            decision = state.eligibility(now, expected_window_id=expected_window)
            valid = decision.eligible
            status = "VALID" if valid else "+".join(decision.reasons)
            lines.append(
                f"{asset:<4} {status} "
                f"ticker={state.active_ticker or '---'} target={_fmt(state.target)} "
                f"Y={_fmt(state.yes_bid)}/{_fmt(state.yes_ask)} "
                f"N={_fmt(state.no_bid)}/{_fmt(state.no_ask)} "
                f"last={_fmt(state.latest_trade)} spread={_fmt(state.spread)} "
                f"vol={_fmt(state.volume)} oi={_fmt(state.open_interest)} "
                f"book_age_ms={_fmt(state.orderbook_freshness_ms(now))} "
                f"event_age_ms={_fmt(state.last_event_age_ms(now))} "
                f"ws={'yes' if state.websocket_connected else 'no'} "
                f"seq={'yes' if state.sequence_healthy else 'no'} "
                f"rest={'yes' if state.rest_fallback_active else 'no'} "
                f"window={state.contract_window_id or '---'} "
                f"rollover={state.rollover_state.value}")
        lines.append(f"VALID: {self.healthy_count(now)}/{len(self.assets)}")
        return "\n".join(lines)

    def _expected_window_id(self):
        windows = [state.contract_window_id for state in self.assets.values()
                   if state.contract_window_id is not None]
        return max(windows) if windows else None


def _cents(value):
    return None if value is None else value / 100


def _fmt(value):
    return "None" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
