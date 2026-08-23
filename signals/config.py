"""Documented, configurable descriptive thresholds and windows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    probability_windows: tuple = (1, 5, 15, 30, 60, 120)
    velocity_windows: tuple = (1, 5, 15, 30, 60)
    acceleration_windows: tuple = (5, 15, 30)
    price_windows: tuple = (1, 5, 15, 30, 60, 120, 300)
    trade_windows: tuple = (5, 15, 30, 60)
    checkpoints: tuple = (30, 60, 120, 180, 300, 600)
    neutral_band: float = 0.02
    synchronization_velocity_tolerance: float = 0.01
    synchronized_min_assets: int = 4
    low_dispersion_threshold: float = 0.08
    acceleration_epsilon: float = 0.0001
    quote_stale_seconds: float = 10.0
    book_stale_seconds: float = 10.0
    history_retention_seconds: float = 1200.0
    realized_volatility_window: float = 300.0
