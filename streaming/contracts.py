"""Ticker-keyed contract metadata, rollover state, and shared eligibility policy."""

from dataclasses import dataclass, field
from enum import Enum

from .timeutil import parse_timestamp


ACTIVE_STATUSES = frozenset({"active", "open"})


class RolloverState(str, Enum):
    CURRENT = "CURRENT"
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"


def contract_window_id(open_time, close_time=None):
    """Return the canonical UTC quarter-hour opening timestamp."""
    if open_time is None:
        return None
    opened = parse_timestamp(open_time)
    if opened.second or opened.microsecond or opened.minute % 15:
        return None
    if close_time is not None:
        closed = parse_timestamp(close_time)
        if (closed - opened).total_seconds() != 900:
            return None
    return opened.strftime("%Y-%m-%dT%H:%M:00Z")


@dataclass
class MarketRecord:
    asset: str
    ticker: str
    status: str | None = None
    target: float | None = None
    open_time: str | None = None
    close_time: str | None = None
    result: str | None = None
    source: str | None = None
    source_timestamp: str | None = None
    updated_at: str | None = None

    @property
    def window_id(self):
        return contract_window_id(self.open_time, self.close_time)

    @property
    def active(self):
        return (self.status or "").lower() in ACTIVE_STATUSES

    def as_market(self):
        return {
            "ticker": self.ticker, "status": self.status,
            "floor_strike": self.target, "open_time": self.open_time,
            "close_time": self.close_time, "result": self.result,
        }


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reasons: tuple[str, ...]
    contract_window_id: str | None


def evaluate_eligibility(*, ticker, expected_ticker, status, target,
                         open_time, close_time, rollover_state,
                         quote_fresh, book_fresh, sequence_healthy,
                         expected_window_id=None, evaluation_timestamp=None):
    reasons = []
    window_id = contract_window_id(open_time, close_time)
    if not ticker or not expected_ticker or ticker != expected_ticker:
        reasons.append("WRONG_TICKER")
    if (status or "").lower() not in ACTIVE_STATUSES:
        reasons.append("CLOSED_MARKET")
    expired = False
    if close_time is not None and evaluation_timestamp is not None:
        expired = parse_timestamp(evaluation_timestamp) >= parse_timestamp(close_time)
        if expired:
            reasons.append("CLOSED_MARKET")
    if rollover_state != RolloverState.RESOLVED:
        reasons.append("ROLLOVER_PENDING")
    elif expired:
        reasons.append("ROLLOVER_PENDING")
    if target is None:
        reasons.append("MISSING_TARGET")
    if not quote_fresh:
        reasons.append("STALE_QUOTE")
    if not book_fresh:
        reasons.append("STALE_BOOK")
    if not sequence_healthy:
        reasons.append("SEQUENCE_UNHEALTHY")
    if window_id is None or (expected_window_id is not None and window_id != expected_window_id):
        reasons.append("WINDOW_MISMATCH")
    return Eligibility(not reasons, tuple(dict.fromkeys(reasons)), window_id)


class ContractRegistry:
    """Metadata is merged only into the record identified by its own ticker."""
    def __init__(self):
        self.by_ticker = {}
        self.expected_by_asset = {}
        self.rollover_by_asset = {}

    def record(self, ticker):
        return self.by_ticker.get(ticker)

    def current(self, asset):
        return self.record(self.expected_by_asset.get(asset))

    def rollover_state(self, asset):
        return self.rollover_by_asset.get(asset, RolloverState.CURRENT)

    def expect(self, asset, ticker, *, pending=True):
        changed = self.expected_by_asset.get(asset) != ticker
        self.expected_by_asset[asset] = ticker
        if changed or pending:
            self.rollover_by_asset[asset] = RolloverState.PENDING
        return changed

    def update(self, asset, ticker, metadata, *, source=None, timestamp=None,
               expected=False):
        if not ticker:
            raise ValueError("market ticker is required")
        record = self.by_ticker.get(ticker)
        if record is None:
            record = self.by_ticker[ticker] = MarketRecord(asset=asset, ticker=ticker)
        elif record.asset != asset:
            raise ValueError(f"ticker {ticker} already belongs to {record.asset}")
        values = {
            "status": metadata.get("status"),
            "target": metadata.get("floor_strike", metadata.get("target")),
            "open_time": metadata.get("open_time"),
            "close_time": metadata.get("close_time"),
            "result": metadata.get("result") or metadata.get("market_result"),
        }
        for name, value in values.items():
            # Incomplete payloads never erase authoritative known metadata.
            if value is not None and value != "":
                setattr(record, name, float(value) if name == "target" else value)
        record.source = source or record.source
        record.source_timestamp = timestamp or record.source_timestamp
        record.updated_at = timestamp or record.updated_at
        if expected:
            self.expect(asset, ticker, pending=True)
        return record

    def apply_lifecycle(self, asset, ticker, payload, *, source=None, timestamp=None):
        msg = payload.get("msg", payload)
        patch = dict(msg)
        event_type = msg.get("event_type") or payload.get("type")
        if event_type in {"settled", "determined", "deactivated"}:
            patch["status"] = "settled" if event_type == "settled" else "closed"
        elif event_type in {"activated", "created"} and not patch.get("status"):
            patch["status"] = "active"
        return self.update(asset, ticker, patch, source=source, timestamp=timestamp)

    def mark_resolved(self, asset):
        self.rollover_by_asset[asset] = RolloverState.RESOLVED
