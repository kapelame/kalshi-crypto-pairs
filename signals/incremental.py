"""Incremental causal aggregates used by the live and replay signal engine."""

from collections import deque
from dataclasses import dataclass, field

from .mathutil import safe_ratio


@dataclass
class _TradeWindow:
    seconds: float
    recent: deque = field(default_factory=deque)
    earlier: deque = field(default_factory=deque)
    maximum: deque = field(default_factory=deque)
    total: float = 0.0
    recent_total: float = 0.0
    earlier_total: float = 0.0
    yes: float = 0.0
    no: float = 0.0
    known: int = 0
    count: int = 0

    def append(self, timestamp, size, direction, ordinal):
        item = (timestamp, size, direction, ordinal)
        self.recent.append(item)
        self.total += size
        self.recent_total += size
        self.count += 1
        if direction == "yes":
            self.yes += size; self.known += 1
        elif direction == "no":
            self.no += size; self.known += 1
        while self.maximum and self.maximum[-1][0] <= size:
            self.maximum.pop()
        self.maximum.append((size, ordinal, timestamp))
        self.advance(timestamp)

    def advance(self, timestamp):
        midpoint = timestamp - self.seconds / 2
        cutoff = timestamp - self.seconds
        while self.recent and self.recent[0][0] < midpoint:
            item = self.recent.popleft()
            self.recent_total -= item[1]
            self.earlier_total += item[1]
            self.earlier.append(item)
        while self.earlier and self.earlier[0][0] < cutoff:
            _, size, direction, ordinal = self.earlier.popleft()
            self.earlier_total -= size
            self.total -= size
            self.count -= 1
            if direction == "yes":
                self.yes -= size; self.known -= 1
            elif direction == "no":
                self.no -= size; self.known -= 1
        while self.maximum and self.maximum[0][2] < cutoff:
            self.maximum.popleft()

    def features(self, timestamp):
        self.advance(timestamp)
        prefix = f"trade_{int(self.seconds)}s_"
        average = None if not self.count else self.total / self.count
        imbalance = None if not self.known else safe_ratio(
            self.yes - self.no, self.yes + self.no)
        half = self.seconds / 2
        return {
            prefix + "count": self.count,
            prefix + "contracts": self.total,
            prefix + "yes_aggressive": self.yes if self.known else None,
            prefix + "no_aggressive": self.no if self.known else None,
            prefix + "imbalance": imbalance,
            prefix + "average_size": average,
            prefix + "max_size": self.maximum[0][0] if self.maximum else None,
            prefix + "volume_acceleration": (
                self.recent_total / half - self.earlier_total / half),
        }


class TradeWindowSet:
    def __init__(self, windows):
        self.windows = {window: _TradeWindow(float(window)) for window in windows}
        self.ordinal = 0

    def append(self, timestamp, size, direction):
        self.ordinal += 1
        for window in self.windows.values():
            window.append(timestamp, size, direction, self.ordinal)

    def features(self, timestamp):
        result = {}
        for window in self.windows.values():
            result.update(window.features(timestamp))
        return result


def top_book_features(bid_book, ask_book, bid_prices, ask_prices):
    bids = [(price, bid_book[price]) for price in reversed(bid_prices[-5:])]
    asks = [(price, ask_book[price]) for price in ask_prices[:5]]

    def imbalance(levels):
        bid = sum(qty for _, qty in bids[:levels])
        ask = sum(qty for _, qty in asks[:levels])
        return safe_ratio(bid - ask, bid + ask)

    bid_depth = sum(qty for _, qty in bids)
    ask_depth = sum(qty for _, qty in asks)
    slope = None
    if len(bids) >= 2 and len(asks) >= 2:
        slope = (bids[0][0] - bids[-1][0]) + (asks[-1][0] - asks[0][0])
    return {
        "book_l1_imbalance": imbalance(1),
        "book_top3_imbalance": imbalance(3),
        "book_top5_imbalance": imbalance(5),
        "book_bid_depth": bid_depth,
        "book_ask_depth": ask_depth,
        "book_depth_ratio": safe_ratio(bid_depth, ask_depth),
        "book_slope": slope,
    }
