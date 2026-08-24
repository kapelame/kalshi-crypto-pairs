"""Mutable in-memory state derived solely from immutable past raw events."""

from dataclasses import dataclass, field

from .history import TimeHistory
from .incremental import TradeWindowSet
from streaming.contracts import RolloverState


@dataclass
class ContractState:
    ticker: str | None = None
    open_time: float | None = None
    close_time: float | None = None
    target: float | None = None
    probability_values: list = field(default_factory=list)
    price_values: list = field(default_factory=list)
    result: str | None = None
    status: str | None = None
    window_id: str | None = None


@dataclass
class AssetState:
    asset: str
    probability: TimeHistory
    velocity_histories: dict
    price: TimeHistory
    trades: list = field(default_factory=list)
    bid_book: dict = field(default_factory=dict)
    ask_book: dict = field(default_factory=dict)
    bid_prices: list = field(default_factory=list)
    ask_prices: list = field(default_factory=list)
    cached_book_features: dict = field(default_factory=dict)
    cached_price_features: dict = field(default_factory=dict)
    trade_windows: TradeWindowSet | None = None
    quote_time: float | None = None
    book_time: float | None = None
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    last_trade: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    contract: ContractState = field(default_factory=ContractState)
    prior_window: dict = field(default_factory=dict)
    checkpoints_emitted: set = field(default_factory=set)
    expected_ticker: str | None = None
    rollover_state: RolloverState = RolloverState.CURRENT
    sequence_healthy: bool = True

    @classmethod
    def create(cls, asset, config):
        return cls(
            asset=asset,
            probability=TimeHistory(config.history_retention_seconds),
            velocity_histories={window: TimeHistory(config.history_retention_seconds)
                                for window in config.velocity_windows},
            price=TimeHistory(config.history_retention_seconds),
            trade_windows=TradeWindowSet(config.trade_windows))
